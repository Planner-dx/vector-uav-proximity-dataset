"""
PX4 ULG飞行日志 → PINN训练数据集处理脚本
矢量推力六旋翼 阶梯悬停实验
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pyulog import ULog
from pathlib import Path

# ============ 参数 ============
ULG_FILE = Path(__file__).parent / "01_raw_data" / "log_56_2026-3-17-18-24-10.ulg"
R = 0.08  # 螺旋桨半径 m
RESAMPLE_HZ = 100  # 重采样频率
WIN_LEN = 50  # 窗口长度 (0.5s)
WIN_STEP = 10  # 步长 (0.1s)
OUT_DIR = Path(__file__).parent / "02_dataset"

# ============ Step 1: ULG解析与数据对齐 ============
print("=" * 60)
print("Step 1: 解析ULG并对齐数据")
print("=" * 60)

ulog = ULog(str(ULG_FILE))

def topic_to_df(ulog, topic_name):
    """将ULog topic转为DataFrame，时间戳转为秒"""
    for d in ulog.data_list:
        if d.name == topic_name:
            df = pd.DataFrame(d.data)
            df['timestamp'] = df['timestamp'] / 1e6  # μs → s
            return df
    raise ValueError(f"Topic '{topic_name}' not found in ULG file")

# 解析4个topic
df_act = topic_to_df(ulog, 'actuator_outputs')
df_sensor = topic_to_df(ulog, 'sensor_combined')
df_pos = topic_to_df(ulog, 'vehicle_local_position')
df_hover = topic_to_df(ulog, 'hover_thrust_estimate')

print(f"  actuator_outputs:        {len(df_act)} samples")
print(f"  sensor_combined:         {len(df_sensor)} samples")
print(f"  vehicle_local_position:  {len(df_pos)} samples")
print(f"  hover_thrust_estimate:   {len(df_hover)} samples")

# 确定公共时间范围
t_start = max(df_act['timestamp'].min(), df_sensor['timestamp'].min(),
              df_pos['timestamp'].min(), df_hover['timestamp'].min())
t_end = min(df_act['timestamp'].max(), df_sensor['timestamp'].max(),
            df_pos['timestamp'].max(), df_hover['timestamp'].max())

# 统一时间轴 (100Hz)
dt = 1.0 / RESAMPLE_HZ
t_unified = np.arange(t_start, t_end, dt)
print(f"  公共时间范围: {t_end - t_start:.1f}s, 重采样点数: {len(t_unified)}")

# 最近邻插值对齐
def nearest_interp(t_target, t_src, values):
    """最近邻插值"""
    idx = np.searchsorted(t_src, t_target)
    idx = np.clip(idx, 1, len(t_src) - 1)
    left = idx - 1
    right = idx
    mask = (t_target - t_src[left]) < (t_src[right] - t_target)
    idx = np.where(mask, left, right)
    return values[idx]

t_src_act = df_act['timestamp'].values
t_src_sensor = df_sensor['timestamp'].values
t_src_pos = df_pos['timestamp'].values
t_src_hover = df_hover['timestamp'].values

# 电机DShot output[0]~output[5]
motor_cols = [f'output[{i}]' for i in range(6)]
motors = np.column_stack([
    nearest_interp(t_unified, t_src_act, df_act[c].values) for c in motor_cols
])

# 加速度 & 陀螺仪
acc_cols = [f'accelerometer_m_s2[{i}]' for i in range(3)]
gyro_cols = [f'gyro_rad[{i}]' for i in range(3)]
acc = np.column_stack([
    nearest_interp(t_unified, t_src_sensor, df_sensor[c].values) for c in acc_cols
])
gyro = np.column_stack([
    nearest_interp(t_unified, t_src_sensor, df_sensor[c].values) for c in gyro_cols
])

# 高度: height = -z, z_over_R = height / R
z_raw = nearest_interp(t_unified, t_src_pos, df_pos['z'].values)
height = -z_raw
z_over_R = height / R

# hover_thrust
hover_thrust = nearest_interp(t_unified, t_src_hover, df_hover['hover_thrust'].values)

# 相对时间（从0开始）
t_rel = t_unified - t_unified[0]

print(f"  高度范围: {height.min():.3f} ~ {height.max():.3f} m")
print(f"  z/R 范围: {z_over_R.min():.2f} ~ {z_over_R.max():.2f}")

# ============ Step 2: 滑动窗口特征提取 ============
print("\n" + "=" * 60)
print("Step 2: 滑动窗口特征提取")
print("=" * 60)

n_samples = len(t_unified)
features_list = []
labels_list = []
window_centers = []

for start in range(0, n_samples - WIN_LEN + 1, WIN_STEP):
    end = start + WIN_LEN
    center = start + WIN_LEN // 2

    # 加速度: 均值(3) + 标准差(3)
    acc_win = acc[start:end]
    acc_mean = acc_win.mean(axis=0)
    acc_std = acc_win.std(axis=0)

    # 陀螺仪: 均值(3) + 标准差(3)
    gyro_win = gyro[start:end]
    gyro_mean = gyro_win.mean(axis=0)
    gyro_std = gyro_win.std(axis=0)

    # 电机DShot: 均值(6)
    motor_win = motors[start:end]
    motor_mean = motor_win.mean(axis=0)

    # hover_thrust: 均值(1)
    ht_mean = hover_thrust[start:end].mean()

    # 拼接 19维特征
    feat = np.concatenate([acc_mean, acc_std, gyro_mean, gyro_std, motor_mean, [ht_mean]])
    features_list.append(feat)

    # 标签: 窗口中心的 z_over_R
    labels_list.append(z_over_R[center])
    window_centers.append(t_rel[center])

X = np.array(features_list)
y = np.array(labels_list)
t_centers = np.array(window_centers)

print(f"  样本数: {X.shape[0]}")
print(f"  特征维度: {X.shape[1]}")

# ============ Step 3: 输出 ============
print("\n" + "=" * 60)
print("Step 3: 保存数据集")
print("=" * 60)

# 列名
feat_names = (
    [f'acc_mean_{ax}' for ax in ['x', 'y', 'z']] +
    [f'acc_std_{ax}' for ax in ['x', 'y', 'z']] +
    [f'gyro_mean_{ax}' for ax in ['x', 'y', 'z']] +
    [f'gyro_std_{ax}' for ax in ['x', 'y', 'z']] +
    [f'motor_mean_{i}' for i in range(6)] +
    ['hover_thrust_mean']
)

# CSV
df_out = pd.DataFrame(X, columns=feat_names)
df_out['z_over_R'] = y
csv_path = OUT_DIR / 'dataset.csv'
df_out.to_csv(csv_path, index=False)
print(f"  CSV saved: {csv_path} ({len(df_out)} rows x {df_out.shape[1]} cols)")

# PyTorch .pt
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
pt_path = OUT_DIR / 'dataset.pt'
torch.save({'X': X_tensor, 'y': y_tensor}, pt_path)
print(f"  PT  saved: {pt_path}")

# 统计信息
print(f"\n{'=' * 60}")
print("数据集统计")
print(f"{'=' * 60}")
print(f"  样本数:     {len(y)}")
print(f"  特征维度:   {X.shape[1]}")
print(f"  z/R 最小值: {y.min():.2f}")
print(f"  z/R 最大值: {y.max():.2f}")
print(f"  z/R 均值:   {y.mean():.2f}")
print(f"  z/R 标准差: {y.std():.2f}")

# 各高度段样本数
bins = np.arange(np.floor(y.min()), np.ceil(y.max()) + 1, 1.0)
if len(bins) < 2:
    bins = np.linspace(y.min(), y.max(), 10)
counts, edges = np.histogram(y, bins=bins)
print(f"\n  z/R 分布直方图:")
for i in range(len(counts)):
    bar = '█' * min(counts[i] // 5, 40)
    print(f"    [{edges[i]:6.1f}, {edges[i+1]:6.1f}): {counts[i]:5d} {bar}")

# ============ 画总览图 ============
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle('ULG Flight Data Overview', fontsize=14, fontweight='bold')

# 1. 高度时间序列
axes[0].plot(t_rel, height, 'b-', linewidth=0.5)
axes[0].set_ylabel('Height (m)')
axes[0].set_title('Height vs Time')
axes[0].grid(True, alpha=0.3)

# 2. 加速度z轴
axes[1].plot(t_rel, acc[:, 2], 'r-', linewidth=0.5)
axes[1].set_ylabel('Acc Z (m/s²)')
axes[1].set_title('Accelerometer Z-axis')
axes[1].grid(True, alpha=0.3)

# 3. 电机均值
motor_avg = motors.mean(axis=1)
axes[2].plot(t_rel, motor_avg, 'g-', linewidth=0.5)
axes[2].set_ylabel('Motor DShot Avg')
axes[2].set_title('Average Motor Command')
axes[2].grid(True, alpha=0.3)

# 4. hover_thrust
axes[3].plot(t_rel, hover_thrust, 'm-', linewidth=0.5)
axes[3].set_ylabel('Hover Thrust')
axes[3].set_xlabel('Time (s)')
axes[3].set_title('Hover Thrust Estimate')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = OUT_DIR / 'data_overview.png'
plt.savefig(fig_path, dpi=150)
print(f"\n  Overview plot saved: {fig_path}")
plt.close()

print("\nDone!")
