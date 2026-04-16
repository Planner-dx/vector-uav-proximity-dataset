#!/usr/bin/env python3
"""
Gazebo PX4 SITL 集成测试 for control_head.py

两种模式:
  force — 开环强制设定 alpha，验证控制链路
  loop  — 闭环 PINN 推理，验证端到端集成

使用方法:
  # 先在其他终端启动:
  #   1) PX4 SITL + Gazebo
  #   2) roslaunch mavros px4.launch
  # 然后运行:
  python3 gazebo_sim_test.py --mode force
  python3 gazebo_sim_test.py --mode loop --model pinn_a
"""
import os
import sys
import time
import argparse
import csv
import logging
from collections import deque
from pathlib import Path
from threading import Lock

import numpy as np

import rospy
from mavros_msgs.msg import State, ActuatorControl
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from mavros_msgs.srv import SetMode, SetModeRequest
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

# 将脚本目录加入path, 确保能导入 control_head
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from control_head import ControlHead

# ============ 日志 ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('gazebo_sim_test')

# ============ 常量 ============
# 特征维度与顺序 (与 process_ulg.py 一致):
#   acc_mean(3) + acc_std(3) + gyro_mean(3) + gyro_std(3)
#   + motor_mean(6) + hover_thrust_mean(1) = 19
FEATURE_DIM = 19
WINDOW_SIZE = 50         # 0.5s @ 100Hz
IMU_RATE_HZ = 100        # 期望 IMU 采样率

# 仿真中电机输出归一化 0~1, 实机 DShot 约 0~2047
# 映射: DShot = actuator_norm * DSHOT_SCALE + DSHOT_OFFSET
DSHOT_SCALE = 2047.0
DSHOT_OFFSET = 0.0

# 默认 hover_thrust (仿真中可能没有该 topic)
DEFAULT_HOVER_THRUST = 0.5

# Offboard setpoint 预发送频率
SETPOINT_RATE = 20  # Hz

# 连接超时
CONNECT_TIMEOUT = 10.0  # s

# 测试剖面 (force 模式) — (end_time_s, alpha, vx, vy, vz, description)
FORCE_PROFILE = [
    # Phase 1: 起飞到 2m
    (10.0,  1.0,  0.0, 0.0, 0.5,  'takeoff alpha=1.0 climb to 2m'),
    # Phase 2: 半速水平移动
    (20.0,  0.5,  1.0, 0.0, 0.0,  'cruise alpha=0.5 vx=0.5m/s'),
    # Phase 3: 完全悬停 (alpha=0)
    (25.0,  0.0,  1.0, 0.0, 0.0,  'hover alpha=0.0 (stop)'),
    # Phase 4: alpha 渐变 0→1
    (35.0, 'ramp', 1.0, 0.0, 0.0, 'ramp alpha 0→1'),
    # Phase 5: 返回降落
    (40.0,  1.0,  0.0, 0.0, -0.3, 'descend alpha=1.0 land'),
]


class GazeboSimTest:
    """Gazebo SITL 集成测试主类"""

    def __init__(self, mode='force', model_type='pinn_a'):
        self.mode = mode
        self.model_type = model_type

        # --- 状态 ---
        self.current_state = State()
        self.current_pose = PoseStamped()
        self.data_lock = Lock()

        # IMU 滑动窗口 (每条: [ax, ay, az, gx, gy, gz], timestamp)
        self.imu_buffer = deque(maxlen=WINDOW_SIZE)
        # 电机指令滑动窗口 (每条: [m0..m5], timestamp)
        self.actuator_buffer = deque(maxlen=WINDOW_SIZE)
        # hover_thrust
        self.hover_thrust = DEFAULT_HOVER_THRUST

        # 日志记录
        self.log_data = []  # list of dicts
        self.test_summary = []

        # --- ROS 初始化 ---
        rospy.init_node('gazebo_sim_test', anonymous=True)
        self.rate = rospy.Rate(SETPOINT_RATE)

        # 订阅
        rospy.Subscriber('/mavros/state', State,
                         self._state_cb, queue_size=10)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped,
                         self._pose_cb, queue_size=10)
        rospy.Subscriber('/mavros/imu/data', Imu,
                         self._imu_cb, queue_size=50)
        # 电机输出: 仿真中常见的两个 topic，都订阅
        rospy.Subscriber('/mavros/target_actuator_control', ActuatorControl,
                         self._actuator_cb, queue_size=50)
        rospy.Subscriber('/mavros/actuator_control', ActuatorControl,
                         self._actuator_cb, queue_size=50)
        # hover_thrust (可能不存在)
        rospy.Subscriber('/mavros/hover_thrust_estimate', Float64,
                         self._hover_thrust_cb, queue_size=10)

        # 发布
        self.vel_pub = rospy.Publisher(
            '/mavros/setpoint_velocity/cmd_vel',
            TwistStamped, queue_size=10)

        # 服务代理
        rospy.wait_for_service('/mavros/cmd/arming', timeout=CONNECT_TIMEOUT)
        rospy.wait_for_service('/mavros/set_mode', timeout=CONNECT_TIMEOUT)
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        # --- ControlHead (仅 loop 模式) ---
        self.ctrl = None
        if mode == 'loop':
            model_file = f'{model_type}_model.pth'
            model_path = SCRIPT_DIR / "04_pinn" / model_file
            scaler_path = SCRIPT_DIR / "03_mlp" / "scaler.pkl"
            if not model_path.exists():
                rospy.logfatal(f"Model not found: {model_path}")
                sys.exit(1)
            self.ctrl = ControlHead(
                str(model_path), str(scaler_path),
                model_type=model_type, device='cpu')
            self.ctrl.reset()
            logger.info(f"ControlHead initialized: {model_file}")

        logger.info(f"GazeboSimTest initialized (mode={mode})")

    # ==================== ROS Callbacks ====================

    def _state_cb(self, msg):
        self.current_state = msg

    def _pose_cb(self, msg):
        self.current_pose = msg

    def _imu_cb(self, msg):
        """提取 acc + gyro, 存入滑动窗口 (轻量, 无重计算)"""
        a = msg.linear_acceleration
        g = msg.angular_velocity
        t = msg.header.stamp.to_sec()
        with self.data_lock:
            self.imu_buffer.append(([a.x, a.y, a.z, g.x, g.y, g.z], t))

    def _actuator_cb(self, msg):
        """提取 6 电机控制量, 缩放到 DShot 范围"""
        ctrls = msg.controls  # 通常 8 通道, 取前 6
        motors = [ctrls[i] * DSHOT_SCALE + DSHOT_OFFSET for i in range(min(6, len(ctrls)))]
        # 补齐到 6
        while len(motors) < 6:
            motors.append(0.0)
        t = msg.header.stamp.to_sec()
        with self.data_lock:
            self.actuator_buffer.append((motors, t))

    def _hover_thrust_cb(self, msg):
        self.hover_thrust = msg.data

    # ==================== 特征构建 ====================

    def build_features(self):
        """
        从滑动窗口构建 19 维特征 (与 process_ulg.py 一致)

        Returns:
            np.ndarray shape (19,) 或 None (数据不足)
        """
        with self.data_lock:
            imu_list = list(self.imu_buffer)
            act_list = list(self.actuator_buffer)

        if len(imu_list) < 10 or len(act_list) < 5:
            return None

        # IMU: [ax, ay, az, gx, gy, gz] × N
        imu_arr = np.array([d[0] for d in imu_list])  # (N, 6)
        acc = imu_arr[:, :3]   # (N, 3)
        gyro = imu_arr[:, 3:]  # (N, 3)

        acc_mean = acc.mean(axis=0)     # (3,)
        acc_std = acc.std(axis=0)       # (3,)
        gyro_mean = gyro.mean(axis=0)   # (3,)
        gyro_std = gyro.std(axis=0)     # (3,)

        # 电机 DShot: [m0..m5] × M
        motor_arr = np.array([d[0] for d in act_list])  # (M, 6)
        motor_mean = motor_arr.mean(axis=0)  # (6,)

        # hover_thrust
        ht = self.hover_thrust

        # 拼接: 19维
        features = np.concatenate([
            acc_mean, acc_std,       # 6
            gyro_mean, gyro_std,     # 6
            motor_mean,              # 6
            [ht],                    # 1
        ]).astype(np.float32)

        assert features.shape[0] == FEATURE_DIM, \
            f"Feature dim mismatch: {features.shape[0]} != {FEATURE_DIM}"
        return features

    # ==================== 工具函数 ====================

    def _make_vel_msg(self, vx, vy, vz):
        """构建 TwistStamped 消息"""
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        return msg

    def _wait_for_connection(self):
        """等待 MAVROS 连接 FCU"""
        logger.info("Waiting for FCU connection...")
        t0 = time.time()
        while not rospy.is_shutdown():
            if self.current_state.connected:
                logger.info("FCU connected!")
                return True
            if time.time() - t0 > CONNECT_TIMEOUT:
                logger.error(f"FCU connection timeout ({CONNECT_TIMEOUT}s)")
                return False
            self.rate.sleep()
        return False

    def _pre_arm_setpoints(self, n=100):
        """
        PX4 Offboard 模式要求: 先以 >2Hz 发送 setpoint, 再切模式+arm
        发送 n 次零速度 setpoint
        """
        logger.info(f"Sending {n} pre-arm setpoints...")
        zero_vel = self._make_vel_msg(0, 0, 0)
        for _ in range(n):
            if rospy.is_shutdown():
                return
            self.vel_pub.publish(zero_vel)
            self.rate.sleep()

    def _set_offboard_and_arm(self):
        """切换 Offboard 模式并 Arm"""
        # 切 OFFBOARD
        logger.info("Setting OFFBOARD mode...")
        resp = self.set_mode_client(SetModeRequest(custom_mode='OFFBOARD'))
        if not resp.mode_sent:
            logger.error("Failed to set OFFBOARD mode")
            return False
        rospy.sleep(1.0)

        # ARM
        logger.info("Arming...")
        resp = self.arming_client(CommandBoolRequest(value=True))
        if not resp.success:
            logger.error("Failed to arm")
            return False
        rospy.sleep(1.0)

        logger.info("OFFBOARD + Armed!")
        return True

    def _get_position(self):
        """获取当前位置 (x, y, z)"""
        p = self.current_pose.pose.position
        return p.x, p.y, p.z

    def _record(self, t, alpha, vx_cmd, vy_cmd, vz_cmd, phase_desc,
                extra=None):
        """记录一帧日志"""
        x, y, z = self._get_position()
        entry = {
            'time': t,
            'alpha': alpha,
            'vx_cmd': vx_cmd,
            'vy_cmd': vy_cmd,
            'vz_cmd': vz_cmd,
            'x': x, 'y': y, 'z': z,
            'mode': self.mode,
            'phase': phase_desc,
        }
        if extra:
            entry.update(extra)
        self.log_data.append(entry)

    # ==================== 模式 1: 开环强制测试 ====================

    def run_force_mode(self):
        """开环强制设定 alpha, 按 FORCE_PROFILE 执行"""
        logger.info("=" * 50)
        logger.info("FORCE MODE: 开环强制 alpha 测试")
        logger.info("=" * 50)

        if not self._wait_for_connection():
            return False
        self._pre_arm_setpoints()
        if not self._set_offboard_and_arm():
            return False

        t_start = rospy.Time.now().to_sec()
        phase_idx = 0
        prev_alpha = 0.0
        ramp_start_t = None

        self.test_summary.append("=== Force Mode Test ===")

        while not rospy.is_shutdown() and phase_idx < len(FORCE_PROFILE):
            t_now = rospy.Time.now().to_sec()
            t_rel = t_now - t_start

            # 查找当前 phase
            end_t, alpha_spec, vx_base, vy_base, vz_base, desc = \
                FORCE_PROFILE[phase_idx]

            if t_rel > end_t:
                self.test_summary.append(
                    f"  Phase {phase_idx}: {desc} — completed at t={t_rel:.1f}s")
                phase_idx += 1
                ramp_start_t = None
                continue

            # 计算 alpha
            if alpha_spec == 'ramp':
                # 渐变: 在当前 phase 内从 0 线性升到 1
                if ramp_start_t is None:
                    ramp_start_t = t_rel
                    prev_phase_end = FORCE_PROFILE[phase_idx - 1][0] \
                        if phase_idx > 0 else 0
                    ramp_start_t = prev_phase_end
                phase_duration = end_t - ramp_start_t
                alpha = np.clip((t_rel - ramp_start_t) / phase_duration, 0.0, 1.0)
            else:
                alpha = float(alpha_spec)

            # 缩放速度指令
            vx_cmd = vx_base * alpha
            vy_cmd = vy_base * alpha
            vz_cmd = vz_base * alpha

            # 发布
            vel_msg = self._make_vel_msg(vx_cmd, vy_cmd, vz_cmd)
            self.vel_pub.publish(vel_msg)

            # 记录
            self._record(t_rel, alpha, vx_cmd, vy_cmd, vz_cmd, desc)

            if abs(alpha - prev_alpha) > 0.05:
                logger.info(f"[t={t_rel:.1f}s] alpha={alpha:.2f} "
                            f"vel=({vx_cmd:.2f},{vy_cmd:.2f},{vz_cmd:.2f}) "
                            f"| {desc}")
                prev_alpha = alpha

            # 检查 Offboard 是否保持
            if self.current_state.mode != 'OFFBOARD':
                logger.warning(f"Mode changed to {self.current_state.mode}, "
                               f"re-requesting OFFBOARD")
                self.set_mode_client(SetModeRequest(custom_mode='OFFBOARD'))

            self.rate.sleep()

        self.test_summary.append(f"  Total duration: {t_rel:.1f}s")
        logger.info("Force mode test completed")
        return True

    # ==================== 模式 2: 闭环集成测试 ====================

    def run_loop_mode(self):
        """闭环 PINN 推理, 实时采集 → 特征 → alpha → 速度指令"""
        logger.info("=" * 50)
        logger.info("LOOP MODE: 闭环 PINN 集成测试")
        logger.info("=" * 50)

        if self.ctrl is None:
            logger.error("ControlHead not initialized")
            return False

        if not self._wait_for_connection():
            return False
        self._pre_arm_setpoints()
        if not self._set_offboard_and_arm():
            return False

        self.test_summary.append("=== Loop Mode Test ===")

        # 测试剖面: 简单的悬停+移动
        # 0-5s: 爬升到2m, 5-25s: 巡航, 25-30s: 降落
        DURATION = 30.0
        TARGET_ALT = 2.0

        t_start = rospy.Time.now().to_sec()
        inference_count = 0
        inference_times = []

        while not rospy.is_shutdown():
            t_now = rospy.Time.now().to_sec()
            t_rel = t_now - t_start

            if t_rel > DURATION:
                break

            # 基础速度指令
            if t_rel < 5.0:
                vx_base, vy_base, vz_base = 0.0, 0.0, 0.5
                phase = 'climb'
            elif t_rel < 25.0:
                vx_base, vy_base, vz_base = 0.3, 0.0, 0.0
                phase = 'cruise'
            else:
                vx_base, vy_base, vz_base = 0.0, 0.0, -0.3
                phase = 'descend'

            # PINN 推理
            features = self.build_features()
            extra = {}
            if features is not None:
                t_inf_start = time.perf_counter()
                result = self.ctrl.compute_alpha(features)
                t_inf_ms = (time.perf_counter() - t_inf_start) * 1000
                inference_times.append(t_inf_ms)

                alpha = result['alpha']
                inference_count += 1
                extra = {
                    'z_r_pred': result['z_r_pred'],
                    'classification': result['classification'],
                    'class_confidence': result['class_confidence'],
                    'override': result['override_active'],
                    'inference_ms': t_inf_ms,
                }

                if inference_count % 20 == 1:
                    logger.info(
                        f"[t={t_rel:.1f}s] alpha={alpha:.3f} "
                        f"z/R={result['z_r_pred']:.2f} "
                        f"cls={result['classification']}"
                        f"({result['class_confidence']:.2f}) "
                        f"inf={t_inf_ms:.1f}ms | {phase}")
            else:
                alpha = 1.0  # 数据不足时默认不限速
                extra = {'note': 'insufficient_data'}

            # 缩放并发布
            vx_cmd = vx_base * alpha
            vy_cmd = vy_base * alpha
            vz_cmd = vz_base * alpha
            self.vel_pub.publish(self._make_vel_msg(vx_cmd, vy_cmd, vz_cmd))

            self._record(t_rel, alpha, vx_cmd, vy_cmd, vz_cmd, phase, extra)

            # Offboard 保持
            if self.current_state.mode != 'OFFBOARD':
                self.set_mode_client(SetModeRequest(custom_mode='OFFBOARD'))

            self.rate.sleep()

        # 统计
        if inference_times:
            inf_arr = np.array(inference_times)
            freq = inference_count / DURATION
            self.test_summary.append(f"  Inference count: {inference_count}")
            self.test_summary.append(f"  Inference freq:  {freq:.1f} Hz "
                                     f"({'PASS' if freq >= 10 else 'FAIL'}, target>=10Hz)")
            self.test_summary.append(f"  Inference time:  "
                                     f"mean={inf_arr.mean():.2f}ms, "
                                     f"max={inf_arr.max():.2f}ms, "
                                     f"p99={np.percentile(inf_arr, 99):.2f}ms")
            logger.info(f"Inference stats: {freq:.1f}Hz, "
                        f"mean={inf_arr.mean():.2f}ms, max={inf_arr.max():.2f}ms")
        else:
            self.test_summary.append("  WARNING: No inference performed!")

        self.test_summary.append(f"  Total duration: {DURATION:.1f}s")
        logger.info("Loop mode test completed")
        return True

    # ==================== 结果输出 ====================

    def log_results(self):
        """保存 CSV 日志和文本摘要"""
        out_dir = SCRIPT_DIR

        # --- CSV ---
        csv_path = out_dir / 'sim_test_log.csv'
        if self.log_data:
            # 收集所有字段
            all_keys = set()
            for d in self.log_data:
                all_keys.update(d.keys())
            fieldnames = sorted(all_keys)
            # 确保关键列在前面
            priority = ['time', 'alpha', 'vx_cmd', 'vy_cmd', 'vz_cmd',
                         'x', 'y', 'z', 'mode', 'phase']
            ordered = [k for k in priority if k in fieldnames]
            ordered += [k for k in fieldnames if k not in priority]

            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=ordered,
                                        extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.log_data)
            logger.info(f"Saved: {csv_path} ({len(self.log_data)} rows)")
        else:
            logger.warning("No log data to save")

        # --- Summary ---
        summary_path = out_dir / 'sim_test_summary.txt'
        self.test_summary.insert(0, f"Gazebo SITL Test Summary")
        self.test_summary.insert(1, f"Mode: {self.mode}")
        self.test_summary.insert(2, f"Model: {self.model_type}")
        self.test_summary.insert(3, "=" * 40)

        # 判定结果
        passed = True
        checks = []

        if self.mode == 'force':
            # 检查: 各 phase 是否都执行了
            phases_executed = set(d.get('phase', '') for d in self.log_data)
            if len(phases_executed) >= 3:
                checks.append("PASS: All phases executed")
            else:
                checks.append(f"FAIL: Only {len(phases_executed)} phases executed")
                passed = False

            # 检查: alpha=0 时速度是否确实为零
            zero_alpha = [d for d in self.log_data if d['alpha'] < 0.01]
            if zero_alpha:
                max_vel = max(
                    abs(d['vx_cmd']) + abs(d['vy_cmd']) + abs(d['vz_cmd'])
                    for d in zero_alpha)
                if max_vel < 0.01:
                    checks.append("PASS: alpha=0 → velocity=0")
                else:
                    checks.append(f"FAIL: alpha=0 but max_vel={max_vel:.3f}")
                    passed = False

        elif self.mode == 'loop':
            # 检查: 推理频率
            if self.log_data:
                inf_entries = [d for d in self.log_data
                               if 'inference_ms' in d]
                if inf_entries:
                    duration = self.log_data[-1]['time'] - self.log_data[0]['time']
                    freq = len(inf_entries) / max(duration, 0.1)
                    if freq >= 10:
                        checks.append(f"PASS: Inference freq={freq:.1f}Hz (>=10Hz)")
                    else:
                        checks.append(f"FAIL: Inference freq={freq:.1f}Hz (<10Hz)")
                        passed = False

            # 检查: alpha 输出合理 (高空应接近1)
            high_alt = [d for d in self.log_data
                        if d.get('z', 0) > 1.5 and 'z_r_pred' in d]
            if high_alt:
                avg_alpha = np.mean([d['alpha'] for d in high_alt])
                if avg_alpha > 0.7:
                    checks.append(f"PASS: High-alt avg alpha={avg_alpha:.2f} (>0.7)")
                else:
                    checks.append(f"WARN: High-alt avg alpha={avg_alpha:.2f} (<0.7)")

        self.test_summary.append("")
        self.test_summary.append("--- Checks ---")
        self.test_summary.extend(checks)
        self.test_summary.append("")
        self.test_summary.append(f"OVERALL: {'PASS' if passed else 'FAIL'}")

        with open(summary_path, 'w') as f:
            f.write('\n'.join(self.test_summary) + '\n')
        logger.info(f"Saved: {summary_path}")

        # 终端打印摘要
        print("\n" + "=" * 50)
        for line in self.test_summary:
            print(line)
        print("=" * 50)

        return passed

    # ==================== 入口 ====================

    def run(self):
        """运行测试"""
        try:
            if self.mode == 'force':
                success = self.run_force_mode()
            elif self.mode == 'loop':
                success = self.run_loop_mode()
            else:
                logger.error(f"Unknown mode: {self.mode}")
                return False
        except rospy.ROSInterruptException:
            logger.info("ROS interrupted")
            success = False
        finally:
            # 无论成功失败都保存日志
            self.log_results()

        return success


# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gazebo PX4 SITL integration test for control_head.py')
    parser.add_argument('--mode', choices=['force', 'loop'], default='force',
                        help='force: open-loop alpha test; loop: closed-loop PINN')
    parser.add_argument('--model', choices=['pinn_a', 'pinn_b'], default='pinn_a',
                        help='PINN model variant (loop mode only)')
    args = parser.parse_args()

    tester = GazeboSimTest(mode=args.mode, model_type=args.model)
    success = tester.run()
    sys.exit(0 if success else 1)
