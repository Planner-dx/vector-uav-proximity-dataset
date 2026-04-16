"""
PINN训练脚本：两种物理约束变体 + 三模型对比
PINN-A: 多项式经验模型
PINN-B: Garofano-Soldado (2024) 势流半经验模型
"""
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# 复用baseline
from train_mlp_baseline import ProximityMLP, prepare_data, assign_class, CLASS_NAMES

# ============ 平台参数常量 ============
DATA_DIR  = Path(__file__).parent / "02_dataset"   # 输入: 训练数据
MLP_DIR   = Path(__file__).parent / "03_mlp"       # 输入: MLP模型 & scaler
PINN_DIR  = Path(__file__).parent / "04_pinn"      # 输出: PINN模型 & 图表 & 日志
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

R = 0.08              # 螺旋桨半径 [m]
ALPHA_DEG = 30.0      # 电机倾斜角 [degrees]
ALPHA_RAD = np.radians(ALPHA_DEG)
BETA_DEG = 30.0
BETA_RAD = np.radians(BETA_DEG)
DSHOT_OGE = 830.0     # 远离地面时电机DShot基线值
C_FRAME = 0.10        # 机架中心宽度 [m]
JK = 2.2              # 喷泉效应系数

# 训练超参
LR = 1e-3
EPOCHS = 200
BATCH_SIZE = 64
PATIENCE = 20
CLS_LOSS_WEIGHT = 0.5
PHYS_LOSS_WEIGHT = 0.1

# 分类阈值（与baseline一致）
SAFE_THRESHOLD = 5.0
WARNING_THRESHOLD = 3.0

# DShot特征在19维特征中的索引: motor_mean_0..5 → 12..17
MOTOR_FEAT_IDX = list(range(12, 18))

# ============ 电机坐标计算 ============
MOTOR_DISTANCES = [0.30, 0.20, 0.30, 0.20, 0.30, 0.20]
ANGLES_DEG = [120, 180, 240, 300, 0, 60]

def compute_motor_positions():
    """计算不等边六边形布局的6个电机坐标（以几何中心为原点）"""
    positions = [(0.0, 0.0)]  # M1 at origin
    for dist, ang in zip(MOTOR_DISTANCES, ANGLES_DEG):
        prev = positions[-1]
        dx = dist * np.cos(np.radians(ang))
        dy = dist * np.sin(np.radians(ang))
        positions.append((prev[0] + dx, prev[1] + dy))
    positions = positions[:6]  # M1~M6
    # 平移到几何中心
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    centered = [(p[0] - cx, p[1] - cy) for p in positions]
    return centered

MOTOR_POSITIONS = compute_motor_positions()

# 对角电机距离 (M1-M4, M2-M5, M3-M6)
def compute_opposite_distances(positions):
    """计算三对对角电机距离"""
    dists = []
    for i in range(3):
        j = i + 3
        dx = positions[j][0] - positions[i][0]
        dy = positions[j][1] - positions[i][1]
        dists.append(np.sqrt(dx**2 + dy**2))
    return dists

OPPOSITE_DISTS = compute_opposite_distances(MOTOR_POSITIONS)
D_OPPOSITE = np.mean(OPPOSITE_DISTS)  # 平均对角距离
RD = 2 * D_OPPOSITE - C_FRAME         # 喷泉效应有效距离

# Physics Loss B 的区间参数
# Section 1 (z/R ≤ 2.5): Kc=0, Kj=0.6, Kh=0.6
FT_S1 = 0.0 + 0.6 * np.cos(ALPHA_RAD) + 0.6 * np.sin(ALPHA_RAD)
# Section 3 (z/R > 3.0): Kc=1, Kj=0, Kh=0
FT_S3 = 1.0

# Section 2 过渡区系数 (Table III)
d0, d1, d2 = -0.0468, 0.0356, 0.0283
r0, r1, r2 = 1.1498, -0.0591, -0.0327
A_TRANS = d0 + d1 * np.cos(ALPHA_RAD) * np.sin(BETA_RAD) + d2 * np.sin(ALPHA_RAD) * np.cos(BETA_RAD)
B_TRANS = r0 + r1 * np.cos(ALPHA_RAD) * np.sin(BETA_RAD) + r2 * np.sin(ALPHA_RAD) * np.cos(BETA_RAD)

# ============ 打印几何参数 ============
def print_geometry():
    print("=" * 60)
    print("平台几何参数")
    print("=" * 60)
    print(f"  螺旋桨半径 R = {R} m")
    print(f"  倾斜角 α = β = {ALPHA_DEG}°")
    print(f"  DShot OGE基线 = {DSHOT_OGE}")
    print(f"  喷泉系数 Jk = {JK}")
    print(f"  机架中心宽度 C_FRAME = {C_FRAME} m")
    print(f"\n  6个电机坐标 (x, y) [m]:")
    for i, (x, y) in enumerate(MOTOR_POSITIONS):
        print(f"    M{i+1}: ({x:+.4f}, {y:+.4f})")
    print(f"\n  对角电机距离:")
    for i in range(3):
        print(f"    M{i+1}-M{i+4}: {OPPOSITE_DISTS[i]:.4f} m")
    print(f"    平均 d_opposite = {D_OPPOSITE:.4f} m")
    print(f"    rd = 2*d_opp - C_frame = {RD:.4f} m")
    print(f"\n  Physics B 区间参数:")
    print(f"    Section 1 ft = {FT_S1:.4f}")
    print(f"    Section 3 ft = {FT_S3:.4f}")
    print(f"    Section 2 A_trans = {A_TRANS:.4f}, B_trans = {B_TRANS:.4f}")

# ============ 加载Scaler ============
def load_scaler():
    with open(MLP_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return (torch.tensor(scaler['mean'], dtype=torch.float32),
            torch.tensor(scaler['std'], dtype=torch.float32))

# ============ T_ratio实际值 ============
def compute_T_ratio_actual(X_norm, scaler_mean, scaler_std):
    """从标准化特征反算实际DShot，计算T_ratio_actual = (DSHOT_OGE / DShot_mean)^2"""
    motor_raw = X_norm[:, 12:18] * scaler_std[12:18] + scaler_mean[12:18]
    dshot_mean = motor_raw.mean(dim=1)
    # 避免除零
    dshot_mean = dshot_mean.clamp(min=100.0)
    T_ratio = (DSHOT_OGE / dshot_mean) ** 2
    return T_ratio

# ============ Physics Loss A: 多项式经验模型 ============
def compute_physics_loss_A(z_pred, X_norm, scaler_mean, scaler_std):
    """
    T_ratio_A = -0.5025 * (1/(z/R))^2 + 0.6503 * (1/(z/R)) + 0.9040
    """
    T_ratio_actual = compute_T_ratio_actual(X_norm, scaler_mean, scaler_std)

    mask = z_pred > 1.0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=z_pred.device, requires_grad=True)

    z_r = z_pred[mask]
    inv_zr = 1.0 / z_r
    T_ratio_pred = -0.5025 * inv_zr**2 + 0.6503 * inv_zr + 0.9040

    return F.mse_loss(T_ratio_pred, T_ratio_actual[mask])

# ============ Physics Loss B: 势流半经验模型 ============
# 预计算电机间相对距离平方 (6x6)
_REL_DIST_SQ = np.zeros((6, 6))
for _i in range(6):
    for _j in range(6):
        dx = MOTOR_POSITIONS[_j][0] - MOTOR_POSITIONS[_i][0]
        dy = MOTOR_POSITIONS[_j][1] - MOTOR_POSITIONS[_i][1]
        _REL_DIST_SQ[_i, _j] = dx**2 + dy**2
REL_DIST_SQ = torch.tensor(_REL_DIST_SQ, dtype=torch.float32)

def compute_physics_loss_B(z_pred, X_norm, scaler_mean, scaler_std):
    """
    Garofano-Soldado (2024) 势流半经验模型
    Section 1 (z/R ≤ 2.5): T = 1/(1 - avg_dv/ft_s1 - dv_f)
    Section 2 (2.5 < z/R ≤ 3.0): T = A_trans * z/R + B_trans
    Section 3 (z/R > 3.0): T = 1/(1 - avg_dv/ft_s3 - dv_f)
    """
    T_ratio_actual = compute_T_ratio_actual(X_norm, scaler_mean, scaler_std)

    mask = z_pred > 1.0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=z_pred.device, requires_grad=True)

    z_r = z_pred[mask]          # z/R
    z_act = z_r * R             # 实际高度 [m]
    T_act = T_ratio_actual[mask]
    N = z_r.shape[0]

    # 每个旋翼受到的总诱导速度变化（包含自身image + 其他5个旋翼image）
    # rel_dist_sq: (6, 6), z_act: (N,)
    rel_sq = REL_DIST_SQ.to(z_pred.device)  # (6, 6)
    z_act_sq4 = 4.0 * z_act**2              # (N,)

    # 对每个旋翼i，计算所有6个旋翼j的贡献之和
    # denom_ij = (rel_sq[i,j] + 4*z_act^2)^1.5  →  shape (6, 6, N)
    # 用广播计算
    rel_sq_exp = rel_sq.unsqueeze(-1)         # (6, 6, 1)
    z_sq4_exp = z_act_sq4.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
    denom = (rel_sq_exp + z_sq4_exp) ** 1.5   # (6, 6, N)
    denom = denom.clamp(min=1e-12)

    z_act_exp = z_act.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
    dv_all = (R**2 / 2.0) * z_act_exp / denom    # (6, 6, N)

    # 每个旋翼i受到的总诱导速度变化 = sum over j
    dv_per_rotor = dv_all.sum(dim=1)              # (6, N)
    # 6旋翼平均
    avg_dv = dv_per_rotor.mean(dim=0)             # (N,)

    # 喷泉效应
    dv_f = 2.0 * R**2 * JK * z_act / (RD**2 + z_act_sq4).clamp(min=1e-12) ** 1.5

    # Section 1: z/R ≤ 2.5
    denom_s1 = (1.0 - avg_dv / FT_S1 - dv_f).clamp(min=0.01)
    T_s1 = 1.0 / denom_s1

    # Section 3: z/R > 3.0
    denom_s3 = (1.0 - avg_dv / FT_S3 - dv_f).clamp(min=0.01)
    T_s3 = 1.0 / denom_s3

    # Section 2: 2.5 < z/R ≤ 3.0 (线性过渡)
    T_s2 = A_TRANS * z_r + B_TRANS

    # 选择区间
    T_ratio_pred = torch.where(z_r <= 2.5, T_s1,
                    torch.where(z_r <= 3.0, T_s2, T_s3))

    return F.mse_loss(T_ratio_pred, T_act)

# ============ 训练函数 ============
def train_pinn(model_name, physics_loss_fn, train_data, val_data, scaler_mean, scaler_std):
    """通用PINN训练函数"""
    print(f"\n{'=' * 60}")
    print(f"训练 {model_name}")
    print(f"{'=' * 60}")

    X_train, y_train, c_train = [t.to(DEVICE) for t in train_data]
    X_val, y_val, c_val = [t.to(DEVICE) for t in val_data]
    s_mean = scaler_mean.to(DEVICE)
    s_std = scaler_std.to(DEVICE)

    train_ds = TensorDataset(X_train, y_train, c_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ProximityMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse_fn = nn.MSELoss()
    ce_fn = nn.CrossEntropyLoss()

    log = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_mae': [],
           'val_acc': [], 'physics_loss': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        epoch_loss_sum = 0.0
        epoch_phys_sum = 0.0
        n_batches = 0

        for xb, yb, cb in train_loader:
            z_pred, cls_logits = model(xb)
            loss_data = mse_fn(z_pred, yb)
            loss_cls = ce_fn(cls_logits, cb)
            loss_phys = physics_loss_fn(z_pred, xb, s_mean, s_std)
            loss = loss_data + CLS_LOSS_WEIGHT * loss_cls + PHYS_LOSS_WEIGHT * loss_phys

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            epoch_phys_sum += loss_phys.item()
            n_batches += 1

        train_loss = epoch_loss_sum / n_batches
        train_phys = epoch_phys_sum / n_batches

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            z_pred_v, cls_logits_v = model(X_val)
            val_mse = mse_fn(z_pred_v, y_val).item()
            val_ce = ce_fn(cls_logits_v, c_val).item()
            val_phys = physics_loss_fn(z_pred_v, X_val, s_mean, s_std).item()
            val_loss = val_mse + CLS_LOSS_WEIGHT * val_ce + PHYS_LOSS_WEIGHT * val_phys
            val_mae = (z_pred_v - y_val).abs().mean().item()
            val_acc = (cls_logits_v.argmax(1) == c_val).float().mean().item()

        log['epoch'].append(epoch)
        log['train_loss'].append(train_loss)
        log['val_loss'].append(val_loss)
        log['val_mae'].append(val_mae)
        log['val_acc'].append(val_acc)
        log['physics_loss'].append(train_phys)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1 or patience_counter == PATIENCE:
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | phys={train_phys:.4f} | "
                  f"val_loss={val_loss:.4f} | mae={val_mae:.3f} | "
                  f"acc={val_acc:.3f} | pat={patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, log

# ============ 评估函数 ============
def evaluate_model(model, test_data, model_name):
    """评估单个模型，返回指标dict"""
    X_test, y_test, c_test = [t.to(DEVICE) for t in test_data]
    model.eval()
    with torch.no_grad():
        z_pred, cls_logits = model(X_test)

    z_np = z_pred.cpu().numpy()
    y_np = y_test.cpu().numpy()
    c_pred = cls_logits.argmax(1).cpu().numpy()
    c_true = c_test.cpu().numpy()

    errors = z_np - y_np
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt((errors**2).mean()))
    acc = float((c_pred == c_true).mean())

    # 分区MAE
    danger_mask = y_np <= 3.0
    warning_mask = (y_np > 3.0) & (y_np <= 5.0)
    safe_mask = y_np > 5.0

    mae_danger = float(np.abs(errors[danger_mask]).mean()) if danger_mask.sum() > 0 else 0.0
    mae_warning = float(np.abs(errors[warning_mask]).mean()) if warning_mask.sum() > 0 else 0.0
    mae_safe = float(np.abs(errors[safe_mask]).mean()) if safe_mask.sum() > 0 else 0.0

    cm = confusion_matrix(c_true, c_pred, labels=[0, 1, 2])

    return {
        'name': model_name,
        'mae': mae, 'rmse': rmse, 'acc': acc,
        'mae_danger': mae_danger, 'mae_warning': mae_warning, 'mae_safe': mae_safe,
        'z_pred': z_np, 'y_true': y_np,
        'errors': errors,
        'c_pred': c_pred, 'c_true': c_true,
        'cm': cm,
    }

# ============ 对比可视化 ============
def plot_comparison(results_list, logs_a, logs_b):
    """生成所有对比图"""
    names = [r['name'] for r in results_list]
    colors = ['#2196F3', '#FF9800', '#4CAF50']  # blue, orange, green

    # --- 1. comparison_bar.png: MAE/RMSE/Accuracy 柱状图 ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    metrics = [('MAE', 'mae'), ('RMSE', 'rmse'), ('Accuracy', 'acc')]
    for ax, (label, key) in zip(axes, metrics):
        vals = [r[key] for r in results_list]
        bars = ax.bar(names, vals, color=colors, width=0.5, edgecolor='white')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_ylabel(label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(PINN_DIR / 'comparison_bar.png', dpi=150)
    plt.close(fig)
    print("  Saved: comparison_bar.png")

    # --- 2. comparison_by_zone.png: 三区间×三模型分组柱状图 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    zones = ['Danger\n(z/R≤3)', 'Warning\n(3<z/R≤5)', 'Safe\n(z/R>5)']
    zone_keys = ['mae_danger', 'mae_warning', 'mae_safe']
    x = np.arange(len(zones))
    width = 0.25
    for i, r in enumerate(results_list):
        vals = [r[k] for k in zone_keys]
        bars = ax.bar(x + i * width, vals, width, label=r['name'], color=colors[i],
                      edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Zone')
    ax.set_ylabel('MAE')
    ax.set_title('MAE by Height Zone', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(zones)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(PINN_DIR / 'comparison_by_zone.png', dpi=150)
    plt.close(fig)
    print("  Saved: comparison_by_zone.png")

    # --- 3 & 4. PINN-A/B prediction scatter ---
    for r, color in zip(results_list[1:], colors[1:]):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(r['y_true'], r['z_pred'], s=8, alpha=0.5, c=color)
        lim = [min(r['y_true'].min(), r['z_pred'].min()) - 0.5,
               max(r['y_true'].max(), r['z_pred'].max()) + 0.5]
        ax.plot(lim, lim, 'r--', linewidth=1.5, label='y = x')
        ax.set_xlabel('True z/R')
        ax.set_ylabel('Predicted z/R')
        ax.set_title(f"{r['name']}: Prediction vs Truth\n(MAE={r['mae']:.3f}, RMSE={r['rmse']:.3f})")
        ax.legend()
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f"{'pinn_a' if 'A' in r['name'] else 'pinn_b'}_prediction.png"
        fig.savefig(PINN_DIR / fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")

    # --- 5. all_models_error_vs_height.png ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for r, c in zip(results_list, colors):
        ax.scatter(r['y_true'], r['errors'], s=6, alpha=0.4, c=c, label=r['name'])
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(3.0, color='red', linestyle='--', alpha=0.5, label='Danger/Warning')
    ax.axvline(5.0, color='orange', linestyle='--', alpha=0.5, label='Warning/Safe')
    ax.set_xlabel('True z/R')
    ax.set_ylabel('Error (pred - true)')
    ax.set_title('All Models: Error vs Height', fontsize=13, fontweight='bold')
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PINN_DIR / 'all_models_error_vs_height.png', dpi=150)
    plt.close(fig)
    print("  Saved: all_models_error_vs_height.png")

    # --- 6. physics_loss_comparison.png ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(logs_a['epoch'], logs_a['physics_loss'], 'o-', color=colors[1],
            markersize=2, label='PINN-A Physics Loss')
    ax.plot(logs_b['epoch'], logs_b['physics_loss'], 's-', color=colors[2],
            markersize=2, label='PINN-B Physics Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Physics Loss')
    ax.set_title('Physics Loss During Training', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PINN_DIR / 'physics_loss_comparison.png', dpi=150)
    plt.close(fig)
    print("  Saved: physics_loss_comparison.png")

    # --- 混淆矩阵 (PINN-A 和 PINN-B) ---
    for r in results_list[1:]:
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(r['cm'], display_labels=CLASS_NAMES)
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f"{r['name']} Confusion Matrix (Acc={r['acc']:.3f})")
        fig.tight_layout()
        tag = 'pinn_a' if 'A' in r['name'] else 'pinn_b'
        fig.savefig(PINN_DIR / f'{tag}_confusion_matrix.png', dpi=150)
        plt.close(fig)


# ============ Main ============
if __name__ == '__main__':
    print_geometry()

    # 加载数据
    train_data, val_data, test_data = prepare_data()
    scaler_mean, scaler_std = load_scaler()

    # --- 加载MLP Baseline ---
    print(f"\n{'=' * 60}")
    print("加载 MLP Baseline 权重")
    print(f"{'=' * 60}")
    baseline_model = ProximityMLP().to(DEVICE)
    baseline_model.load_state_dict(torch.load(MLP_DIR / 'mlp_baseline.pth',
                                               map_location=DEVICE, weights_only=True))
    baseline_model.eval()
    print("  Loaded: mlp_baseline.pth")

    # --- 训练 PINN-A ---
    pinn_a_model, log_a = train_pinn(
        "PINN-A (Polynomial)", compute_physics_loss_A,
        train_data, val_data, scaler_mean, scaler_std
    )
    torch.save(pinn_a_model.state_dict(), PINN_DIR / 'pinn_a_model.pth')
    print("  Saved: pinn_a_model.pth")

    # --- 训练 PINN-B ---
    pinn_b_model, log_b = train_pinn(
        "PINN-B (Potential Flow)", compute_physics_loss_B,
        train_data, val_data, scaler_mean, scaler_std
    )
    torch.save(pinn_b_model.state_dict(), PINN_DIR / 'pinn_b_model.pth')
    print("  Saved: pinn_b_model.pth")

    # --- 合并训练日志 ---
    log_df = pd.DataFrame({
        'epoch': log_a['epoch'],
        'pinn_a_train_loss': log_a['train_loss'],
        'pinn_a_val_loss': log_a['val_loss'],
        'pinn_a_val_mae': log_a['val_mae'],
        'pinn_a_val_acc': log_a['val_acc'],
        'pinn_a_physics_loss': log_a['physics_loss'],
    })
    log_b_df = pd.DataFrame({
        'epoch': log_b['epoch'],
        'pinn_b_train_loss': log_b['train_loss'],
        'pinn_b_val_loss': log_b['val_loss'],
        'pinn_b_val_mae': log_b['val_mae'],
        'pinn_b_val_acc': log_b['val_acc'],
        'pinn_b_physics_loss': log_b['physics_loss'],
    })
    merged_log = pd.merge(log_df, log_b_df, on='epoch', how='outer')
    merged_log.to_csv(PINN_DIR / 'pinn_training_log.csv', index=False)
    print(f"\n  Saved: pinn_training_log.csv")

    # --- 三模型评估 ---
    print(f"\n{'=' * 60}")
    print("Step 3: 三模型对比评估")
    print(f"{'=' * 60}")

    r_baseline = evaluate_model(baseline_model, test_data, 'MLP Baseline')
    r_pinn_a = evaluate_model(pinn_a_model, test_data, 'PINN-A (Poly)')
    r_pinn_b = evaluate_model(pinn_b_model, test_data, 'PINN-B (PotFlow)')
    results = [r_baseline, r_pinn_a, r_pinn_b]

    # 打印对比表格
    header = f"{'Metric':<22} | {'MLP Baseline':>13} | {'PINN-A (Poly)':>13} | {'PINN-B (PotFlow)':>16}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    rows = [
        ('MAE (overall)', 'mae'),
        ('RMSE (overall)', 'rmse'),
        ('Classification Acc', 'acc'),
        ('MAE (z/R≤3 danger)', 'mae_danger'),
        ('MAE (3<z/R≤5 warn)', 'mae_warning'),
        ('MAE (z/R>5 safe)', 'mae_safe'),
    ]
    for label, key in rows:
        vals = [r[key] for r in results]
        best_idx = vals.index(min(vals)) if key != 'acc' else vals.index(max(vals))
        parts = []
        for i, v in enumerate(vals):
            s = f"{v:.4f}"
            if i == best_idx:
                s = f"*{v:.4f}*"
            parts.append(s)
        w = [13, 13, 16]
        print(f"  {label:<20} | {parts[0]:>{w[0]}} | {parts[1]:>{w[1]}} | {parts[2]:>{w[2]}}")
    print(sep)

    # 保存 comparison_results.json
    comp_json = {}
    for r in results:
        comp_json[r['name']] = {
            'mae': r['mae'], 'rmse': r['rmse'], 'accuracy': r['acc'],
            'mae_danger': r['mae_danger'], 'mae_warning': r['mae_warning'],
            'mae_safe': r['mae_safe'],
        }
    with open(PINN_DIR / 'comparison_results.json', 'w') as f:
        json.dump(comp_json, f, indent=2)
    print(f"\n  Saved: comparison_results.json")

    # --- 画图 ---
    print(f"\n{'=' * 60}")
    print("生成对比图表")
    print(f"{'=' * 60}")
    plot_comparison(results, log_a, log_b)

    print(f"\nDone! All outputs saved to {PINN_DIR}")
