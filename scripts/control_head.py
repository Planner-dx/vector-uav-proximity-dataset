"""
Control Head — 矢量推力无人机近壁避障控制输出头
Pipeline Step 4: PINN模型推理 → 速度缩放因子 alpha

方案C: 回归连续映射(主通道) + 分类安全兜底(override) + 非对称EMA平滑
"""
import logging
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# 尝试从训练脚本导入模型类
try:
    from train_mlp_baseline import ProximityMLP
except ImportError:
    # Fallback: 定义兼容的模型类（与训练脚本完全一致）
    import torch.nn as nn
    class ProximityMLP(nn.Module):
        """Fallback模型定义，与train_mlp_baseline.py中完全一致"""
        def __init__(self, in_dim=19, num_classes=3):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
            )
            self.reg_head = nn.Linear(16, 1)
            self.cls_head = nn.Linear(16, num_classes)

        def forward(self, x):
            h = self.backbone(x)
            z_pred = self.reg_head(h).squeeze(-1)
            cls_logits = self.cls_head(h)
            return z_pred, cls_logits

logger = logging.getLogger(__name__)


class ControlHead:
    """
    近壁避障控制输出头
    输入 19维传感器特征 → 输出速度缩放因子 alpha in [0, 1]

    方案C策略:
      1) 回归主通道: z/R → alpha 平滑sigmoid映射
      2) 分类兜底: danger高置信度时强制alpha=0
      3) 非对称EMA: 减速快响应, 加速慢恢复
    """

    # ==================== 映射阈值 ====================
    ZR_FULL_STOP = 2.5       # z/R ≤ 此值 → alpha = 0 (完全停止)
    ZR_FULL_SPEED = 5.5      # z/R ≥ 此值 → alpha = 1 (正常飞行)
    ZR_SIGMOID_CENTER = 4.0  # sigmoid中心点
    SIGMOID_STEEPNESS = 3.0  # sigmoid斜率 (越大过渡越陡)

    # ==================== 分类兜底阈值 ====================
    DANGER_OVERRIDE_CONF = 0.8   # danger置信度 > 此值 → 强制alpha=0
    SAFE_BOOST_CONF = 0.9        # safe置信度 > 此值 且 z/R > 4.0 → 允许alpha提升
    SAFE_BOOST_ZR_MIN = 4.0      # safe boost生效的最低z/R
    SAFE_BOOST_ALPHA = 0.85      # safe boost时alpha下限

    # ==================== EMA平滑参数 ====================
    EMA_DECEL = 0.5   # 减速方向EMA系数 (alpha变小, 快响应)
    EMA_ACCEL = 0.2   # 加速方向EMA系数 (alpha变大, 慢恢复)

    # ==================== 分类标签 ====================
    CLASS_NAMES = ['safe', 'warning', 'danger']

    def __init__(self, model_path, scaler_path, model_type='pinn_a', device='cpu'):
        """
        Args:
            model_path: .pth模型权重路径
            scaler_path: scaler.pkl路径
            model_type: 'pinn_a' 或 'pinn_b', 仅用于日志标识
            device: 'cpu' 或 'cuda'
        """
        self.device = torch.device(device)
        self.model_type = model_type

        # 加载模型
        self.model = ProximityMLP(in_dim=19, num_classes=3)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"ControlHead loaded model: {model_path} (type={model_type})")

        # 加载scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        self.scaler_mean = torch.tensor(scaler['mean'], dtype=torch.float32, device=self.device)
        self.scaler_std = torch.tensor(scaler['std'], dtype=torch.float32, device=self.device)
        logger.info(f"ControlHead loaded scaler: {scaler_path}")

        # EMA状态
        self._alpha_prev = None

    def reset(self):
        """重置EMA状态，新飞行开始时调用"""
        self._alpha_prev = None
        logger.info("ControlHead EMA state reset")

    @staticmethod
    def alpha_from_zr(z_r: float, zr_stop=None, zr_full=None,
                      center=None, steepness=None) -> float:
        """
        纯函数: z/R → alpha 的连续映射

        使用 smoothstep 实现平滑过渡:
          z/R ≤ zr_stop  → 0.0
          z/R ≥ zr_full  → 1.0
          中间用 sigmoid 曲线平滑过渡

        Args:
            z_r: 归一化离地高度
            zr_stop: 完全停止阈值 (默认用类属性)
            zr_full: 正常飞行阈值 (默认用类属性)
            center: sigmoid中心点 (默认用类属性)
            steepness: sigmoid斜率 (默认用类属性)

        Returns:
            alpha in [0, 1]
        """
        if zr_stop is None:
            zr_stop = ControlHead.ZR_FULL_STOP
        if zr_full is None:
            zr_full = ControlHead.ZR_FULL_SPEED
        if center is None:
            center = ControlHead.ZR_SIGMOID_CENTER
        if steepness is None:
            steepness = ControlHead.SIGMOID_STEEPNESS

        if z_r <= zr_stop:
            return 0.0
        if z_r >= zr_full:
            return 1.0

        # Sigmoid映射, 归一化到 [zr_stop, zr_full] 区间
        x = steepness * (z_r - center)
        sig = 1.0 / (1.0 + np.exp(-x))

        # 将sigmoid值重新映射确保边界连续性
        # sig(zr_stop) 和 sig(zr_full) 处的值
        sig_lo = 1.0 / (1.0 + np.exp(-steepness * (zr_stop - center)))
        sig_hi = 1.0 / (1.0 + np.exp(-steepness * (zr_full - center)))

        # 线性重映射到 [0, 1]
        alpha = (sig - sig_lo) / (sig_hi - sig_lo)
        return float(np.clip(alpha, 0.0, 1.0))

    def _apply_ema(self, alpha_new: float) -> float:
        """非对称EMA平滑: 减速快, 加速慢"""
        if self._alpha_prev is None:
            self._alpha_prev = alpha_new
            return alpha_new

        if alpha_new < self._alpha_prev:
            # 减速 → 快响应
            ema_coeff = self.EMA_DECEL
        else:
            # 加速 → 慢恢复
            ema_coeff = self.EMA_ACCEL

        alpha_smooth = ema_coeff * alpha_new + (1.0 - ema_coeff) * self._alpha_prev
        self._alpha_prev = alpha_smooth
        return alpha_smooth

    @torch.no_grad()
    def compute_alpha(self, features_19d: np.ndarray) -> dict:
        """
        核心推理函数, 单次调用

        Args:
            features_19d: 19维原始特征（未标准化）, shape (19,) 或 (1, 19)

        Returns:
            dict {
                'alpha': float,           # 最终速度缩放因子 [0,1] (EMA后)
                'alpha_raw': float,       # 未经EMA平滑的alpha
                'z_r_pred': float,        # 回归预测的z/R
                'classification': str,    # 'safe'/'warning'/'danger'
                'class_confidence': float,# 分类置信度
                'override_active': bool,  # 是否触发了安全兜底
            }
        """
        # --- 输入预处理 ---
        feat = np.asarray(features_19d, dtype=np.float32).flatten()
        assert feat.shape[0] == 19, f"Expected 19 features, got {feat.shape[0]}"
        x = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 标准化
        x_norm = (x - self.scaler_mean) / self.scaler_std

        # --- 模型推理 ---
        z_pred, cls_logits = self.model(x_norm)
        z_r = float(z_pred[0].cpu())
        probs = F.softmax(cls_logits[0], dim=0).cpu().numpy()

        cls_idx = int(probs.argmax())
        cls_name = self.CLASS_NAMES[cls_idx]
        cls_conf = float(probs[cls_idx])

        # --- 回归主通道: z/R → alpha ---
        alpha_reg = self.alpha_from_zr(z_r)

        # --- 分类安全兜底 ---
        override_active = False
        alpha_raw = alpha_reg

        # Danger override: 高置信danger → 强制停止
        if cls_idx == 2 and cls_conf > self.DANGER_OVERRIDE_CONF:
            alpha_raw = 0.0
            override_active = True
            logger.warning(
                f"DANGER OVERRIDE: cls={cls_name}({cls_conf:.2f}), "
                f"z/R={z_r:.2f}, alpha forced 0.0"
            )

        # Safe boost: 高置信safe且z/R合理 → 防止回归低估
        elif cls_idx == 0 and cls_conf > self.SAFE_BOOST_CONF \
                and z_r > self.SAFE_BOOST_ZR_MIN:
            if alpha_raw < self.SAFE_BOOST_ALPHA:
                alpha_raw = self.SAFE_BOOST_ALPHA
                override_active = True
                logger.info(
                    f"SAFE BOOST: cls={cls_name}({cls_conf:.2f}), "
                    f"z/R={z_r:.2f}, alpha boosted to {self.SAFE_BOOST_ALPHA}"
                )

        # --- EMA平滑 ---
        alpha_smooth = self._apply_ema(alpha_raw)

        # 日志: 区间切换检测
        if self._alpha_prev is not None:
            prev = self._alpha_prev
            if prev > 0.5 and alpha_smooth <= 0.5:
                logger.warning(f"ZONE CHANGE: entering danger/warning zone, "
                               f"alpha {prev:.2f} → {alpha_smooth:.2f}")
            elif prev <= 0.5 and alpha_smooth > 0.5:
                logger.info(f"ZONE CHANGE: entering safe zone, "
                            f"alpha {prev:.2f} → {alpha_smooth:.2f}")

        return {
            'alpha': float(np.clip(alpha_smooth, 0.0, 1.0)),
            'alpha_raw': float(np.clip(alpha_raw, 0.0, 1.0)),
            'z_r_pred': z_r,
            'classification': cls_name,
            'class_confidence': cls_conf,
            'override_active': override_active,
        }


# ============================================================
# 验证与可视化
# ============================================================
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from train_mlp_baseline import prepare_data

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    DATA_DIR    = Path(__file__).parent
    MODEL_PATH  = DATA_DIR / "04_pinn" / "pinn_a_model.pth"
    SCALER_PATH = DATA_DIR / "03_mlp"  / "scaler.pkl"
    OUT_DIR     = DATA_DIR / "05_control_head"

    print("=" * 60)
    print("ControlHead 验证与分析")
    print("=" * 60)

    # --- 加载数据 ---
    _, _, test_data = prepare_data()
    X_test, y_test, c_test = test_data
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()

    # 反标准化得到原始特征 (compute_alpha 需要原始特征)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    s_mean = scaler['mean']
    s_std = scaler['std']
    # X_test已经是标准化后的, 反标准化
    X_test_raw = X_test_np * s_std + s_mean

    # --- 初始化 ControlHead ---
    ctrl = ControlHead(MODEL_PATH, SCALER_PATH, model_type='pinn_a')

    # --- 跑一遍测试集 ---
    print(f"\n推理测试集: {len(y_test_np)} samples")
    results = []
    ctrl.reset()
    for i in range(len(y_test_np)):
        r = ctrl.compute_alpha(X_test_raw[i])
        r['y_true'] = float(y_test_np[i])
        r['sample_idx'] = i
        results.append(r)

    # 提取数组
    alphas = np.array([r['alpha'] for r in results])
    alphas_raw = np.array([r['alpha_raw'] for r in results])
    zr_preds = np.array([r['z_r_pred'] for r in results])
    overrides = np.array([r['override_active'] for r in results])
    y_true = np.array([r['y_true'] for r in results])

    # --- 分区统计 ---
    print(f"\n{'=' * 60}")
    print("各区间 alpha 统计")
    print(f"{'=' * 60}")
    zones = [
        ('Danger  (z/R_true ≤ 3)',   y_true <= 3.0),
        ('Warning (3 < z/R_true ≤ 5)', (y_true > 3.0) & (y_true <= 5.0)),
        ('Safe    (z/R_true > 5)',    y_true > 5.0),
        ('All',                        np.ones(len(y_true), dtype=bool)),
    ]

    header = f"{'Zone':<30} | {'Count':>5} | {'Mean':>6} | {'Min':>6} | {'Max':>6} | {'Override%':>9}"
    print(header)
    print("-" * len(header))
    for name, mask in zones:
        if mask.sum() == 0:
            print(f"  {name:<28} | {'N/A':>5}")
            continue
        a = alphas[mask]
        ov_pct = overrides[mask].sum() / mask.sum() * 100
        print(f"  {name:<28} | {mask.sum():5d} | {a.mean():6.3f} | "
              f"{a.min():6.3f} | {a.max():6.3f} | {ov_pct:8.1f}%")

    # --- 图1 & 图2 ---
    print(f"\n生成分析图...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # ===== 图1: z/R vs alpha 映射曲线 + 数据散点 =====
    ax = axes[0]

    # 理论曲线
    zr_range = np.linspace(0, 20, 500)
    alpha_theory = np.array([ControlHead.alpha_from_zr(z) for z in zr_range])
    ax.plot(zr_range, alpha_theory, 'k-', linewidth=2.5, label='Mapping curve', zorder=5)

    # 区间背景色
    ax.axvspan(0, ControlHead.ZR_FULL_STOP, alpha=0.15, color='red', label='Stop zone')
    ax.axvspan(ControlHead.ZR_FULL_STOP, ControlHead.ZR_FULL_SPEED,
               alpha=0.10, color='orange', label='Transition zone')
    ax.axvspan(ControlHead.ZR_FULL_SPEED, 20, alpha=0.08, color='green', label='Full speed zone')

    # 数据散点: 按override着色
    normal_mask = ~overrides
    ax.scatter(zr_preds[normal_mask], alphas_raw[normal_mask],
               s=12, alpha=0.5, c='steelblue', label=f'Normal ({normal_mask.sum()})')
    if overrides.sum() > 0:
        ax.scatter(zr_preds[overrides], alphas_raw[overrides],
                   s=30, alpha=0.8, c='red', marker='x', linewidths=1.5,
                   label=f'Override ({overrides.sum()})')

    # 阈值标注
    for zr_val, lbl in [(ControlHead.ZR_FULL_STOP, 'STOP'),
                         (ControlHead.ZR_SIGMOID_CENTER, 'CENTER'),
                         (ControlHead.ZR_FULL_SPEED, 'FULL')]:
        ax.axvline(zr_val, color='gray', linestyle=':', alpha=0.6)
        ax.text(zr_val, 1.05, lbl, ha='center', fontsize=8, color='gray')

    ax.set_xlabel('z/R (predicted)', fontsize=11)
    ax.set_ylabel('alpha (raw, before EMA)', fontsize=11)
    ax.set_title('Control Mapping: z/R → alpha', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.5, 20)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc='center right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # ===== 图2: 时间序列 raw vs smoothed =====
    ax = axes[1]
    t = np.arange(len(alphas))

    ax.plot(t, alphas_raw, '-', color='lightcoral', linewidth=0.8,
            alpha=0.7, label='alpha_raw')
    ax.plot(t, alphas, '-', color='darkblue', linewidth=1.5,
            label='alpha_smoothed (EMA)')

    # 真实z/R归一化到[0,1]显示参考
    y_norm = y_true / y_true.max()
    ax.plot(t, y_norm, '--', color='gray', linewidth=0.8, alpha=0.5,
            label='z/R_true (normalized)')

    # override标记
    if overrides.sum() > 0:
        ov_idx = np.where(overrides)[0]
        ax.scatter(ov_idx, alphas[ov_idx], s=20, c='red', marker='v',
                   zorder=5, label='Override')

    ax.set_xlabel('Sample index (time order)', fontsize=11)
    ax.set_ylabel('alpha', fontsize=11)
    ax.set_title(f'EMA Smoothing Effect (decel={ControlHead.EMA_DECEL}, '
                 f'accel={ControlHead.EMA_ACCEL})', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUT_DIR / 'control_head_analysis.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # --- 推理速度测试 ---
    import time
    dummy = X_test_raw[0]
    ctrl.reset()
    # 预热
    for _ in range(10):
        ctrl.compute_alpha(dummy)
    # 计时
    n_iters = 1000
    t0 = time.perf_counter()
    for _ in range(n_iters):
        ctrl.compute_alpha(dummy)
    elapsed = (time.perf_counter() - t0) / n_iters * 1000
    print(f"\n  单次推理延迟: {elapsed:.2f} ms  ({'OK' if elapsed < 5 else 'SLOW'}, target < 5ms)")

    print(f"\nDone!")
