#!/usr/bin/env python3
"""
Scenario 2: PINN 引导的水平前飞 GE 悬停

流程: 起飞至 1.25m → 稳定悬停 5s → 以 0.2m/s 沿 +X 方向前飞
     → Stage 1: alpha < 0.8 (warning zone) → 停止水平运动，原地悬停
     → Stage 2: alpha < 0.15 持续 1.5s (danger zone) → GE 确认，等待人工降落

高度说明:
  正方体高度 1m，飞行高度 1.25m → 飞越正方体时 z_above_cube = 0.25m → z/R ≈ 3.1
  飞越地面时 z/R ≈ 15.6 (safe，不会误触发)

前进方向: +X (local frame)，请确保 +X 方向前方有足够空间放置正方体

运行方式:
  python3 scenario2_horizontal_ge.py --model pinn_a

安全说明:
  GE 确认后仅悬停，不自动降落，需人工切换 LAND/HOLD 模式
  超出 MAX_FORWARD_DIST 未检测到 GE → 自动切换 AUTO.LOITER
"""
import sys
import time
import csv
import argparse
from collections import deque
from pathlib import Path
from threading import Lock

import numpy as np
import rospy
from mavros_msgs.msg import State, ActuatorControl
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from control_head import ControlHead

# ── 飞行参数 ──────────────────────────────────────────────────────────────────
TAKEOFF_ALT       = 1.25  # 飞行高度 [m]，保证飞越 1m 正方体时 z/R ≈ 3.1
HOVER_STABLE_TIME = 5.0   # 起飞后稳定悬停时间 [s]
V_FORWARD         = 0.2   # 前飞速度 [m/s]
MAX_FORWARD_DIST  = 6.0   # 最大前飞距离安全限制 [m]

# 两段响应阈值
WARNING_ALPHA     = 0.80  # Stage 1: alpha 低于此值 → 停止水平运动
GE_CONFIRM_ALPHA  = 0.15  # Stage 2: alpha 低于此值持续 GE_CONFIRM_TIME s → 悬停等待
GE_CONFIRM_TIME   = 1.5   # 危险区 GE 确认所需持续时间 [s]

# ── 系统参数 ──────────────────────────────────────────────────────────────────
REACH_TOL        = 0.10
CONNECT_TIMEOUT  = 15.0
ARM_TIMEOUT      = 15.0
WINDOW_SIZE      = 50
DSHOT_SCALE      = 2047.0
SETPOINT_RATE    = 20
DT               = 1.0 / SETPOINT_RATE
DEFAULT_HOVER_THRUST = 0.5

# ── 仿真 GE 注入参数 (--sim 模式专用，实机不使用) ─────────────────────────────
# 正方体参数 (需与 spawn_cube.py 保持一致)
SIM_CUBE_HEIGHT    = 1.0   # 正方体高度 [m]
SIM_CUBE_HALF      = 0.5   # 正方体半宽 [m] (1m × 1m 底面)
SIM_GE_RANGE       = 10.0  # GE 影响范围 = SIM_GE_RANGE × R
# 过渡区宽度: 无人机进入正方体边缘时 GE 逐渐增强
SIM_TRANSITION_W   = 0.20  # [m], 越小过渡越陡


class HorizontalGEHover:
    def __init__(self, model_type: str = "pinn_a",
                 sim_mode: bool = False, cube_x: float = 3.0, cube_y: float = 0.0):
        self.model_type   = model_type
        self.sim_mode     = sim_mode
        self.cube_x       = cube_x   # 正方体中心 X (仅 sim 用)
        self.cube_y       = cube_y   # 正方体中心 Y (仅 sim 用)
        self.fcu_state    = State()
        self.pose         = PoseStamped()
        self.lock         = Lock()
        self.imu_buf      = deque(maxlen=WINDOW_SIZE)
        self.act_buf      = deque(maxlen=WINDOW_SIZE)
        self.hover_thrust = DEFAULT_HOVER_THRUST
        self.log_data     = []

        rospy.init_node("scenario2_horizontal_ge", anonymous=True)
        self.rate = rospy.Rate(SETPOINT_RATE)

        # 订阅
        rospy.Subscriber("/mavros/state",
                         State, self._cb_state, queue_size=10)
        rospy.Subscriber("/mavros/local_position/pose",
                         PoseStamped, self._cb_pose, queue_size=10)
        rospy.Subscriber("/mavros/imu/data",
                         Imu, self._cb_imu, queue_size=50)
        rospy.Subscriber("/mavros/target_actuator_control",
                         ActuatorControl, self._cb_act, queue_size=50)
        rospy.Subscriber("/mavros/actuator_control",
                         ActuatorControl, self._cb_act, queue_size=50)
        rospy.Subscriber("/mavros/hover_thrust_estimate",
                         Float64, self._cb_ht, queue_size=10)

        # 发布 & 服务
        self.pub_pos  = rospy.Publisher(
            "/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        rospy.wait_for_service("/mavros/cmd/arming",  timeout=CONNECT_TIMEOUT)
        rospy.wait_for_service("/mavros/set_mode",    timeout=CONNECT_TIMEOUT)
        self.svc_arm  = rospy.ServiceProxy("/mavros/cmd/arming",  CommandBool)
        self.svc_mode = rospy.ServiceProxy("/mavros/set_mode",    SetMode)

        # 加载模型
        model_path  = SCRIPT_DIR / "04_pinn" / f"{model_type}_model.pth"
        scaler_path = SCRIPT_DIR / "03_mlp" / "scaler.pkl"
        self.ctrl = ControlHead(
            str(model_path), str(scaler_path), model_type=model_type)
        self.ctrl.reset()
        rospy.loginfo(f"Model loaded: {model_type}")

    # ── ROS 回调 ──────────────────────────────────────────────────────────────
    def _cb_state(self, m): self.fcu_state = m
    def _cb_pose(self, m):  self.pose = m
    def _cb_ht(self, m):    self.hover_thrust = m.data

    def _cb_imu(self, m):
        a, g = m.linear_acceleration, m.angular_velocity
        with self.lock:
            self.imu_buf.append([a.x, a.y, a.z, g.x, g.y, g.z])

    def _cb_act(self, m):
        ctrls  = list(m.controls)
        motors = [ctrls[i] * DSHOT_SCALE if i < len(ctrls) else 0.0
                  for i in range(6)]
        with self.lock:
            self.act_buf.append(motors)

    # ── 工具 ──────────────────────────────────────────────────────────────────
    def _xyz(self):
        p = self.pose.pose.position
        return p.x, p.y, p.z

    def _alt(self):
        return self._xyz()[2]

    def _pub(self, x: float, y: float, z: float):
        msg = PoseStamped()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        self.pub_pos.publish(msg)

    def _keep_offboard(self):
        if self.fcu_state.mode != "OFFBOARD":
            self.svc_mode(SetModeRequest(custom_mode="OFFBOARD"))

    def _build_features(self):
        """构建 19 维传感器特征向量（与训练集一致）"""
        with self.lock:
            imu = list(self.imu_buf)
            act = list(self.act_buf)
        if len(imu) < 10 or len(act) < 5:
            return None
        imu_arr   = np.array(imu, dtype=np.float32)  # (N, 6)
        acc, gyro = imu_arr[:, :3], imu_arr[:, 3:]
        motor_arr = np.array(act, dtype=np.float32)   # (M, 6)
        return np.concatenate([
            acc.mean(0), acc.std(0),          # 6
            gyro.mean(0), gyro.std(0),        # 6
            motor_arr.mean(0),                # 6
            [self.hover_thrust],              # 1
        ]).astype(np.float32)                 # = 19

    def _inject_ge_sim(self, feat: np.ndarray) -> np.ndarray:
        """
        Gazebo 仿真专用: 根据当前位置/高度模拟 GE 效应。
        正方体上方: z_above = z_abs - CUBE_HEIGHT (较小, GE 强)
        正方体之外: z_above = z_abs              (较大, 无 GE)
        过渡区: 用 x 方向距离平滑插值 effect_strength, 模拟逐渐进入的过程。
        """
        cx, cy, cz = self._xyz()

        # X 方向平滑过渡系数 (0=未到正方体, 1=完全在正方体上方)
        dx = cx - (self.cube_x - SIM_CUBE_HALF)           # 到正方体左边缘距离
        x_blend = float(np.clip(dx / SIM_TRANSITION_W, 0.0, 1.0))

        # 有效地面高度: 在正方体上方时为 CUBE_HEIGHT, 否则为 0
        in_y = abs(cy - self.cube_y) <= (SIM_CUBE_HALF + SIM_TRANSITION_W)
        surface_z = SIM_CUBE_HEIGHT * x_blend if in_y else 0.0

        z_above   = max(cz - surface_z, 0.01)
        effect    = max(0.0, 1.0 - z_above / (SIM_GE_RANGE * R))
        if effect < 1e-3:
            return feat

        m = feat.copy()
        m[12:18] *= (1.0 - 0.85 * effect)   # 电机 DShot 均值下降
        m[3:6]   *= (1.0 + 40.0 * effect)   # acc std 上升
        m[9:12]  *= (1.0 + 15.0 * effect)   # gyro std 上升
        m[18]    *= (1.0 - 0.70 * effect)   # hover_thrust 下降
        return m

    def _infer(self) -> dict:
        feat = self._build_features()
        if feat is None:
            return {"alpha": 1.0, "alpha_raw": 1.0, "z_r_pred": 99.0,
                    "classification": "safe", "class_confidence": 0.0,
                    "override_active": False}
        if self.sim_mode:
            feat = self._inject_ge_sim(feat)
        return self.ctrl.compute_alpha(feat)

    @staticmethod
    def _bar(a: float, w: int = 20) -> str:
        n = int(round(float(np.clip(a, 0, 1)) * w))
        return "#" * n + "." * (w - n)

    def _log(self, stage: str, x_cmd: float, res: dict):
        ax, ay, az = self._xyz()
        self.log_data.append({
            "stage":    stage,
            "x":        round(ax, 4),
            "y":        round(ay, 4),
            "alt":      round(az, 4),
            "x_cmd":    round(x_cmd, 4),
            "z_r":      round(res["z_r_pred"], 2),
            "alpha":    round(res["alpha"], 4),
            "cls":      res["classification"],
            "conf":     round(res["class_confidence"], 3),
            "override": res["override_active"],
        })

    def _save_log(self):
        if not self.log_data:
            return
        path = SCRIPT_DIR / "06_simulation" / "scenario2_log.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.log_data[0].keys()))
            w.writeheader()
            w.writerows(self.log_data)
        rospy.loginfo(f"Log saved: {path}")

    # ── 连接 & 解锁 ───────────────────────────────────────────────────────────
    def _wait_connect(self) -> bool:
        rospy.loginfo("Waiting for FCU connection ...")
        deadline = time.time() + CONNECT_TIMEOUT
        while not rospy.is_shutdown():
            if self.fcu_state.connected:
                rospy.loginfo("FCU connected")
                return True
            if time.time() > deadline:
                rospy.logerr("FCU connection timeout")
                return False
            self.rate.sleep()
        return False

    def _arm_offboard(self, x: float, y: float, z: float) -> bool:
        rospy.loginfo("Pre-streaming setpoints ...")
        for _ in range(100):
            if rospy.is_shutdown():
                return False
            self._pub(x, y, z)
            self.rate.sleep()

        deadline = time.time() + ARM_TIMEOUT
        last_req = 0.0
        while not rospy.is_shutdown() and time.time() < deadline:
            self._pub(x, y, z)
            now = time.time()
            if now - last_req > 1.0:
                if self.fcu_state.mode != "OFFBOARD":
                    self.svc_mode(SetModeRequest(custom_mode="OFFBOARD"))
                elif not self.fcu_state.armed:
                    self.svc_arm(CommandBoolRequest(value=True))
                last_req = now
            if self.fcu_state.mode == "OFFBOARD" and self.fcu_state.armed:
                rospy.loginfo("OFFBOARD + Armed OK")
                return True
            self.rate.sleep()

        rospy.logerr(f"Arm/OFFBOARD failed: mode={self.fcu_state.mode}, "
                     f"armed={self.fcu_state.armed}")
        return False

    # ── 飞行阶段 ──────────────────────────────────────────────────────────────
    def _phase_takeoff(self) -> bool:
        x, y, _ = self._xyz()
        rospy.loginfo(f"[TAKEOFF] climbing to {TAKEOFF_ALT:.2f}m ...")
        while not rospy.is_shutdown():
            self._pub(x, y, TAKEOFF_ALT)
            self._keep_offboard()
            if abs(self._alt() - TAKEOFF_ALT) < REACH_TOL:
                rospy.loginfo(f"[TAKEOFF] reached {self._alt():.2f}m")
                return True
            self.rate.sleep()
        return False

    def _phase_hover_stable(self) -> bool:
        x, y, z = self._xyz()
        rospy.loginfo(f"[STABILIZE] holding {HOVER_STABLE_TIME:.0f}s at z={z:.2f}m")
        t0 = time.time()
        while not rospy.is_shutdown():
            self._pub(x, y, z)
            self._keep_offboard()
            if time.time() - t0 >= HOVER_STABLE_TIME:
                return True
            self.rate.sleep()
        return False

    def _phase_forward(self):
        """
        前飞阶段，两段响应：

        Stage 1 (warning):  alpha < WARNING_ALPHA (0.80)
            → 停止水平运动，原地悬停，继续检测 GE

        Stage 2 (danger):   alpha < GE_CONFIRM_ALPHA (0.15) 持续 GE_CONFIRM_TIME s
            → GE 确认，返回当前位置供下一阶段悬停

        前进方向: +X (local frame)
        """
        x0, y0, _ = self._xyz()
        x_cmd      = x0
        ge_timer   = 0.0
        stage1_triggered = False

        rospy.loginfo(
            f"[FORWARD] start x={x0:.2f}m  V={V_FORWARD}m/s  "
            f"max_dist={MAX_FORWARD_DIST:.1f}m")
        rospy.loginfo(
            f"[FORWARD] Stage1 threshold: α<{WARNING_ALPHA}  "
            f"Stage2 threshold: α<{GE_CONFIRM_ALPHA} for {GE_CONFIRM_TIME}s")

        while not rospy.is_shutdown():
            res       = self._infer()
            alpha     = res["alpha"]
            z_r       = res["z_r_pred"]
            cls_n     = res["classification"]
            cls_c     = res["class_confidence"]
            cx, cy, _ = self._xyz()
            dist      = x_cmd - x0

            # ── Stage 1: warning → 停止水平运动 ─────────────────────────────
            if alpha < WARNING_ALPHA:
                if not stage1_triggered:
                    rospy.logwarn(
                        f"[FORWARD] *** STAGE 1: GE WARNING ***  "
                        f"α={alpha:.2f}  x={cx:.2f}m  z/R={z_r:.1f}  "
                        f"Stopping horizontal motion.")
                    stage1_triggered = True
                # x_cmd 不再推进，保持当前位置

            else:
                # 未触发 Stage 1: 正常前进
                if dist < MAX_FORWARD_DIST:
                    x_cmd += V_FORWARD * DT
                else:
                    rospy.logwarn(
                        f"[FORWARD] MAX_DIST={MAX_FORWARD_DIST:.1f}m reached "
                        f"without GE detection — check cube position or model")
                    return False, cx, cy, TAKEOFF_ALT

            self._pub(x_cmd, y0, TAKEOFF_ALT)
            self._keep_offboard()

            rospy.loginfo(
                f"[FORWARD] x={cx:.2f}m  x_cmd={x_cmd:.2f}m  dist={dist:.2f}m  "
                f"z/R={z_r:.1f}  {cls_n}({cls_c:.2f})  "
                f"α={alpha:.2f} {self._bar(alpha)}"
                + ("  [STOPPED]" if stage1_triggered else "")
            )
            self._log("FORWARD", x_cmd, res)

            # ── Stage 2: danger → GE 确认计时 ───────────────────────────────
            if alpha < GE_CONFIRM_ALPHA:
                ge_timer += DT
            else:
                ge_timer = 0.0

            if ge_timer >= GE_CONFIRM_TIME:
                rospy.logwarn(
                    f"[FORWARD] *** STAGE 2: GE DANGER CONFIRMED ***  "
                    f"x={cx:.2f}m  z/R={z_r:.1f}  Entering hover.")
                return True, cx, cy, TAKEOFF_ALT

            self.rate.sleep()
        return False, 0.0, 0.0, TAKEOFF_ALT

    def _phase_wait_manual_land(self, x: float, y: float, z: float):
        """
        GE 确认后悬停，等待操作员手动降落。
        持续输出提示，不执行自动 land 指令。
        按 Ctrl-C 或人工切换模式后退出。
        """
        rospy.logwarn("=" * 60)
        rospy.logwarn("[GE_HOVER] Platform GE confirmed! Hovering in place.")
        rospy.logwarn("  >>> 请操作员手动切换 LAND 或 HOLD 模式 <<<")
        rospy.logwarn(f"  Hovering at  x={x:.2f}m  y={y:.2f}m  z={z:.2f}m")
        rospy.logwarn("=" * 60)

        while not rospy.is_shutdown():
            # 如果操作员已经切换了模式，退出循环
            if self.fcu_state.mode not in ("OFFBOARD",):
                rospy.loginfo(
                    f"[GE_HOVER] Mode changed to {self.fcu_state.mode}, exiting")
                return

            self._pub(x, y, z)
            self._keep_offboard()
            res = self._infer()
            rospy.loginfo_throttle(
                2.0,
                f"[GE_HOVER] z/R={res['z_r_pred']:.1f}  "
                f"α={res['alpha']:.2f}  {res['classification']}({res['class_confidence']:.2f})  "
                f"— waiting for manual land ..."
            )
            self._log("GE_HOVER", x, res)
            self.rate.sleep()

    # ── 主流程 ────────────────────────────────────────────────────────────────
    def run(self):
        if not self._wait_connect():
            return
        x0, y0, _ = self._xyz()
        if not self._arm_offboard(x0, y0, TAKEOFF_ALT):
            return

        rospy.loginfo("=== Scenario 2: Horizontal GE Hover ===")
        rospy.loginfo(
            f"Model: {self.model_type} | Alt={TAKEOFF_ALT}m | "
            f"V_fwd={V_FORWARD}m/s | Cube=1.0m → z/R≈3.1 over cube")

        if not self._phase_takeoff():
            return
        if not self._phase_hover_stable():
            return

        ok, gx, gy, gz = self._phase_forward()

        if not ok:
            rospy.logwarn("GE not detected — switching to AUTO.LOITER for safety")
            self.svc_mode(SetModeRequest(custom_mode="AUTO.LOITER"))
            self._save_log()
            return

        try:
            self._phase_wait_manual_land(gx, gy, gz)
        except KeyboardInterrupt:
            rospy.logwarn("[GE_HOVER] Interrupted — switching to AUTO.LOITER")
            self.svc_mode(SetModeRequest(custom_mode="AUTO.LOITER"))

        self._save_log()
        rospy.loginfo("=== Scenario 2 complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scenario 2: PINN horizontal GE hover (manual land)")
    parser.add_argument("--model", choices=["pinn_a", "pinn_b"], default="pinn_a",
                        help="PINN model variant to use")
    parser.add_argument("--sim", action="store_true",
                        help="仿真模式: 根据位置/高度自动注入 GE 效应 (Gazebo 用)")
    parser.add_argument("--cube-x", type=float, default=3.0,
                        help="仿真模式: 正方体中心 X 坐标 [m] (需与 spawn_cube.py 一致, 默认 3.0)")
    parser.add_argument("--cube-y", type=float, default=0.0,
                        help="仿真模式: 正方体中心 Y 坐标 [m] (默认 0.0)")
    args = parser.parse_args()

    try:
        HorizontalGEHover(
            model_type=args.model,
            sim_mode=args.sim,
            cube_x=args.cube_x,
            cube_y=args.cube_y,
        ).run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.logwarn("Interrupted by user")
