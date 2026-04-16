"""
MLP Baseline: 从传感器特征预测无人机离地高度 z/R
多任务学习：回归(z/R) + 分类(safe/warning/danger)
"""
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# ============ 常量 ============
DATA_DIR  = Path(__file__).parent / "02_dataset"   # 输入: 训练数据
MODEL_DIR = Path(__file__).parent / "03_mlp"       # 输出: 模型权重 & 归一化参数 & 图表
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 分类阈值
SAFE_THRESHOLD = 5.0      # z/R > 5  → safe(0)
WARNING_THRESHOLD = 3.0   # 3 < z/R ≤ 5 → warning(1)
                           # z/R ≤ 3 → danger(2)
CLASS_NAMES = ['safe', 'warning', 'danger']

# 训练超参
LR = 1e-3
EPOCHS = 200
BATCH_SIZE = 64
PATIENCE = 20
CLS_LOSS_WEIGHT = 0.5

# 划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15

# ============ 模型定义 ============
class ProximityMLP(nn.Module):
    """
    多任务MLP：回归预测z/R + 分类safe/warning/danger
    结构: 19 → 64 → ReLU → 32 → ReLU → 16 → ReLU → 1(回归)
                                              ↘ 3(分类)
    """
    def __init__(self, in_dim=19, num_classes=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.reg_head = nn.Linear(16, 1)        # 回归头
        self.cls_head = nn.Linear(16, num_classes)  # 分类头

    def forward(self, x):
        h = self.backbone(x)
        z_pred = self.reg_head(h).squeeze(-1)    # (B,)
        cls_logits = self.cls_head(h)            # (B, num_classes)
        return z_pred, cls_logits


def assign_class(y):
    """根据z/R值分配类别标签"""
    labels = torch.full_like(y, 2, dtype=torch.long)  # default: danger
    labels[y > WARNING_THRESHOLD] = 1                  # warning
    labels[y > SAFE_THRESHOLD] = 0                     # safe
    return labels


# ============ Step 1: 数据清洗与划分 ============
def detect_segments(y, min_gap=5):
    """
    检测悬停段：基于z/R的显著跳变来分割。
    返回每个样本所属段的编号。
    min_gap: 相邻样本z/R差值超过此值视为新段（用于检测大跳变）。
    对于阶梯悬停，用z/R四舍五入到整数级别做聚类更稳健。
    """
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y
    # 用高度的整数级别作为段标识，再检测变化点
    level = np.round(y_np).astype(int)
    seg_ids = np.zeros(len(y_np), dtype=int)
    seg_id = 0
    for i in range(1, len(y_np)):
        if level[i] != level[i - 1]:
            seg_id += 1
        seg_ids[i] = seg_id
    return seg_ids


def prepare_data():
    print("=" * 60)
    print("Step 1: 数据清洗与划分")
    print("=" * 60)

    data = torch.load(DATA_DIR / 'dataset.pt', weights_only=True)
    X, y = data['X'], data['y']
    print(f"  原始样本数: {len(y)}")

    # 过滤 z/R < 0
    mask = y >= 0
    X, y = X[mask], y[mask]
    print(f"  过滤z/R<0后: {len(y)}")

    # 检测悬停段
    seg_ids = detect_segments(y)
    unique_segs = np.unique(seg_ids)
    n_segs = len(unique_segs)
    print(f"  检测到 {n_segs} 个悬停/过渡段")

    # 计算每段的平均高度，用于打印
    seg_info = []
    for s in unique_segs:
        smask = seg_ids == s
        seg_info.append((s, smask.sum(), y[smask].mean().item()))
    print(f"  各段 [seg_id: n_samples, mean_z/R]:")
    for sid, cnt, mz in seg_info:
        print(f"    seg {sid:2d}: {cnt:4d} samples, z/R={mz:.1f}")

    # 分段交错划分：按段序号 mod 7 分配 (5:1:1 ≈ 71%:14%:14%)
    # 每7段中 0,1,2,3,4→train, 5→val, 6→test
    # 这样相邻段不会同时出现在train和val/test中，避免泄漏
    # 同时各高度层都有机会被分配到各集合
    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)

    for s in unique_segs:
        smask = seg_ids == s
        mod = s % 7
        if mod <= 4:
            train_mask |= smask
        elif mod == 5:
            val_mask |= smask
        else:  # mod == 6
            test_mask |= smask

    # 如果val或test为空（段数太少），回退到简单3-fold分配
    if val_mask.sum() == 0 or test_mask.sum() == 0:
        print("  段数不足，回退到 mod 3 分配 (train/val/test)")
        train_mask[:] = False; val_mask[:] = False; test_mask[:] = False
        for s in unique_segs:
            smask = seg_ids == s
            mod = s % 3
            if mod == 0:
                train_mask |= smask
            elif mod == 1:
                val_mask |= smask
            else:
                test_mask |= smask

    X_all = X.clone()
    y_all = y.clone()

    # 特征标准化（仅用训练集统计量）
    X_train_raw = X_all[train_mask]
    mean = X_train_raw.mean(dim=0)
    std = X_train_raw.std(dim=0)
    std[std < 1e-8] = 1.0
    X_norm = (X_all - mean) / std

    # 保存scaler
    scaler = {'mean': mean.numpy(), 'std': std.numpy()}
    with open(MODEL_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved: scaler.pkl (基于训练集计算)")

    X_train, y_train = X_norm[train_mask], y_all[train_mask]
    X_val, y_val = X_norm[val_mask], y_all[val_mask]
    X_test, y_test = X_norm[test_mask], y_all[test_mask]

    # 分类标签
    c_train = assign_class(y_train)
    c_val = assign_class(y_val)
    c_test = assign_class(y_test)

    print(f"\n  训练集: {len(y_train)}  验证集: {len(y_val)}  测试集: {len(y_test)}")
    for name, yy, c in [('训练', y_train, c_train), ('验证', y_val, c_val), ('测试', y_test, c_test)]:
        counts = [(c == i).sum().item() for i in range(3)]
        print(f"    {name}集: z/R=[{yy.min():.1f}, {yy.max():.1f}]  "
              f"safe={counts[0]}, warning={counts[1]}, danger={counts[2]}")

    return (X_train, y_train, c_train), (X_val, y_val, c_val), (X_test, y_test, c_test)


# ============ Step 3: 训练 ============
def train_model(train_data, val_data):
    print(f"\n{'=' * 60}")
    print("Step 3: 训练")
    print(f"{'=' * 60}")
    print(f"  Device: {DEVICE}")

    X_train, y_train, c_train = [t.to(DEVICE) for t in train_data]
    X_val, y_val, c_val = [t.to(DEVICE) for t in val_data]

    train_ds = TensorDataset(X_train, y_train, c_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ProximityMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse_fn = nn.MSELoss()
    ce_fn = nn.CrossEntropyLoss()

    # 日志
    log = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for xb, yb, cb in train_loader:
            z_pred, cls_logits = model(xb)
            loss = mse_fn(z_pred, yb) + CLS_LOSS_WEIGHT * ce_fn(cls_logits, cb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        train_loss = train_loss_sum / n_batches

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            z_pred_v, cls_logits_v = model(X_val)
            val_mse = mse_fn(z_pred_v, y_val).item()
            val_ce = ce_fn(cls_logits_v, c_val).item()
            val_loss = val_mse + CLS_LOSS_WEIGHT * val_ce
            val_mae = (z_pred_v - y_val).abs().mean().item()
            val_acc = (cls_logits_v.argmax(1) == c_val).float().mean().item()

        log['epoch'].append(epoch)
        log['train_loss'].append(train_loss)
        log['val_loss'].append(val_loss)
        log['val_mae'].append(val_mae)
        log['val_acc'].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1 or patience_counter == PATIENCE:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_mae={val_mae:.3f} | "
                  f"val_acc={val_acc:.3f} | patience={patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # 恢复最佳模型
    model.load_state_dict(best_state)

    # 保存模型权重
    torch.save(best_state, MODEL_DIR / 'mlp_baseline.pth')
    print(f"  Model saved: mlp_baseline.pth")

    # 保存训练日志
    pd.DataFrame(log).to_csv(MODEL_DIR / 'training_log.csv', index=False)
    print(f"  Log saved: training_log.csv")

    return model, log


# ============ Step 4: 评估与可视化 ============
def evaluate_and_plot(model, test_data, log):
    print(f"\n{'=' * 60}")
    print("Step 4: 评估与可视化")
    print(f"{'=' * 60}")

    X_test, y_test, c_test = [t.to(DEVICE) for t in test_data]

    model.eval()
    with torch.no_grad():
        z_pred, cls_logits = model(X_test)

    z_pred_np = z_pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    c_pred_np = cls_logits.argmax(1).cpu().numpy()
    c_test_np = c_test.cpu().numpy()

    # 指标
    errors = z_pred_np - y_test_np
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    cls_acc = (c_pred_np == c_test_np).mean()

    print(f"\n  测试集指标:")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    分类 Accuracy = {cls_acc:.4f}")

    # --- 图1: Loss曲线 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log['epoch'], log['train_loss'], 'b-', label='Train Loss')
    ax.plot(log['epoch'], log['val_loss'], 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / 'loss_curve.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: loss_curve.png")

    # --- 图2: Prediction vs Truth ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test_np, z_pred_np, s=8, alpha=0.5, c='steelblue')
    lim = [min(y_test_np.min(), z_pred_np.min()) - 0.5,
           max(y_test_np.max(), z_pred_np.max()) + 0.5]
    ax.plot(lim, lim, 'r--', linewidth=1.5, label='y = x')
    ax.set_xlabel('True z/R')
    ax.set_ylabel('Predicted z/R')
    ax.set_title(f'Prediction vs Truth (MAE={mae:.3f}, RMSE={rmse:.3f})')
    ax.legend()
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / 'prediction_vs_truth.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: prediction_vs_truth.png")

    # --- 图3: Error vs Height ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_test_np, errors, s=8, alpha=0.5, c='coral')
    ax.axhline(0, color='k', linewidth=0.8)
    # 分段统计
    bin_edges = np.arange(0, y_test_np.max() + 1, 1.0)
    bin_centers, bin_mae, bin_std = [], [], []
    for i in range(len(bin_edges) - 1):
        mask = (y_test_np >= bin_edges[i]) & (y_test_np < bin_edges[i + 1])
        if mask.sum() > 2:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_mae.append(np.abs(errors[mask]).mean())
            bin_std.append(errors[mask].std())
    ax.errorbar(bin_centers, [0] * len(bin_centers), yerr=bin_std,
                fmt='s', color='darkred', markersize=6, capsize=4,
                label='±1σ per z/R bin')
    ax.set_xlabel('True z/R')
    ax.set_ylabel('Prediction Error (pred - true)')
    ax.set_title('Error Distribution vs Height')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / 'error_vs_height.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: error_vs_height.png")

    # --- 图4: 混淆矩阵 ---
    cm = confusion_matrix(c_test_np, c_pred_np, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix (Acc={cls_acc:.3f})')
    fig.tight_layout()
    fig.savefig(MODEL_DIR / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: confusion_matrix.png")

    # 每类precision/recall
    print(f"\n  分类详细:")
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        total_true = cm[i, :].sum()
        total_pred = cm[:, i].sum()
        prec = tp / total_pred if total_pred > 0 else 0
        rec = tp / total_true if total_true > 0 else 0
        print(f"    {name:8s}: precision={prec:.3f}, recall={rec:.3f}, "
              f"support={total_true}")


# ============ Main ============
if __name__ == '__main__':
    train_data, val_data, test_data = prepare_data()
    model, log = train_model(train_data, val_data)
    evaluate_and_plot(model, test_data, log)
    print(f"\nDone! All outputs saved to {MODEL_DIR}")
