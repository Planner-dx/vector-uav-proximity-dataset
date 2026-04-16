#!/usr/bin/env python3
"""
Scenario 1: PINN 引导的垂直自主降落

流程: 起飞至 1.5m → 悬停 10s → α 缩放匀速下降 (max 0.2m/s)
     → GE 确认悬停 → 悬停 10s → AUTO.LAND

运行方式:
  python3 scenario1_vertical_landing.py --model pinn_a

前置条件 (实机):
  roslaunch mavros px4.launch

前置条件 (Gazebo):
  1) PX4 SITL + Gazebo
  2) roslaunch mavros px4.launch
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
TAKEOFF_ALT      = 1.5    # 起飞目标高度 [m]
HOVER_DURATION   = 10.0   # 起飞后稳定悬停时间 [s]
V_MAX            = 0.2    # 最大下降速度 [m/s]
GE_CONFIRM_ALPHA = 0.15   # alpha 低于此值持续 GE_CONFIRM_TIME s → GE 确认
GE_CONFIRM_TIME  = 1.5    # GE 确认所需持续时间 [s]
HOVER_WAIT       = 10.0   # GE 确认后悬停时间 [s]

# ── 系统参数 ──────────────────────────────────────────────────────────────────
REACH_TOL        = 0.10   # 到达目标高度容忍误差 [m]
CONNECT_TIMEOUT  = 15.0   # FCU 连接超时 [s]
ARM_TIMEOUT      = 15.0   # 解锁超时 [s]
LAND_TIMEOUT     = 30.0   # AUTO.LAND 等待落地超时 [s]
LAND_ALT_THRESH  = 0.15   # 落地判定高度 [m]
WINDOW_SIZE      = 50     # 特征窗口大小 (0.5s @ 100Hz)
DSHOT_SCALE      = 2047.0
SETPOINT_RATE    = 20     # 位置指令发布频率 [Hz]
DT               = 1.0 / SETPOINT_RATE
DEFAULT_HOVER_THRUST = 0.5

# ── 仿真 GE 注入参数 (--sim 模式专用，实机不使用) ─────────────────────────────
SIM_GE_SURFACE_Z  = 0.0   # 地面高度 [m]（Gazebo 世界坐标）
SIM_GE_RANGE      = 10.0  # GE 影响范围 = SIM_GE_RANGE × R [m]


class VerticalLanding:
    def __init__(self, model_type: str = "pinn_a", sim_mode: bool = False):
        self.model_type   = model_type
        self.sim_mode     = sim_mode
        self.fcu_state    = State()
        self.pose         = PoseStamped()
        self.lock         = Lock()
        self.imu_buf      = deque(maxlen=WINDOW_SIZE)
        self.act_buf      = deque(maxlen=WINDOW_SIZE)
        self.hover_thrust = DEFAULT_HOVER_THRUST
        self.log_data     = []

        rospy.init_node("scenario1_vertical_landing", anonymous=True)
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
        Gazebo 仿真专用: 根据当前高度模拟 GE 对传感器特征的影响。
        实机不调用此方法，真实传感器会感受到实际 GE。

        物理依据 (与 gazebo_demo_landing.py 一致):
          effect_strength = max(0, 1 - z_above / (SIM_GE_RANGE × R))
          motor 信号下降, acc/gyro std 上升, hover_thrust 下降
        """
        _, _, z_abs = self._xyz()
        z_above = max(z_abs - SIM_GE_SURFACE_Z, 0.01)
        effect  = max(0.0, 1.0 - z_above / (SIM_GE_RANGE * R))
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

    def _log(self, stage: str, z_cmd: float, res: dict):
        _, _, az = self._xyz()
        self.log_data.append({
            "stage":   stage,
            "alt":     round(az, 4),
            "z_cmd":   round(z_cmd, 4),
            "z_r":     round(res["z_r_pred"], 2),
            "alpha":   round(res["alpha"], 4),
            "cls":     res["classification"],
            "conf":    round(res["class_confidence"], 3),
            "override": res["override_active"],
        })

    def _save_log(self):
        if not self.log_data:
            return
        path = SCRIPT_DIR / "06_simulation" / "scenario1_log.csv"
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
        """预发 100 帧 setpoint，然后尝试切 OFFBOARD + 解锁"""
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
        """爬升至 TAKEOFF_ALT"""
        x, y, _ = self._xyz()
        rospy.loginfo(f"[TAKEOFF] climbing to {TAKEOFF_ALT:.1f}m ...")
        while not rospy.is_shutdown():
            self._pub(x, y, TAKEOFF_ALT)
            self._keep_offboard()
            if abs(self._alt() - TAKEOFF_ALT) < REACH_TOL:
                rospy.loginfo(f"[TAKEOFF] reached {self._alt():.2f}m")
                return True
            self.rate.sleep()
        return False

    def _phase_hover(self, duration: float, label: str = "HOVER") -> bool:
        """原地悬停 duration 秒"""
        x, y, z = self._xyz()
        rospy.loginfo(f"[{label}] holding for {duration:.0f}s at z={z:.2f}m")
        t0 = time.time()
        while not rospy.is_shutdown():
            self._pub(x, y, z)
            self._keep_offboard()
            elapsed = time.time() - t0
            rospy.loginfo_throttle(
                2.0, f"[{label}] {elapsed:.1f}/{duration:.0f}s  alt={self._alt():.2f}m")
            if elapsed >= duration:
                return True
            self.rate.sleep()
        return False

    def _phase_descend(self):
        """
        α 缩放匀速下降:
            v_descent = alpha × V_MAX   (0 ~ 0.2 m/s)
            z_cmd    -= v_descent × DT

        当 alpha 连续 GE_CONFIRM_TIME 秒 < GE_CONFIRM_ALPHA → 返回 GE 确认位置
        """
        x, y, z = self._xyz()
        z_cmd    = z
        ge_timer = 0.0
        rospy.loginfo(
            f"[DESCEND] start z={z:.2f}m  V_MAX={V_MAX}m/s  "
            f"GE_threshold=α<{GE_CONFIRM_ALPHA}")

        while not rospy.is_shutdown():
            res   = self._infer()
            alpha = res["alpha"]

            # 下降速度随 alpha 线性缩放
            v_descent = alpha * V_MAX
            z_cmd    -= v_descent * DT
            z_cmd     = max(z_cmd, 0.05)   # 防止指令穿地

            self._pub(x, y, z_cmd)
            self._keep_offboard()

            alt = self._alt()
            rospy.loginfo(
                f"[DESCEND] alt={alt:.3f}m  z_cmd={z_cmd:.3f}m  "
                f"v={v_descent:.3f}m/s  z/R={res['z_r_pred']:.1f}  "
                f"{res['classification']}({res['class_confidence']:.2f})  "
                f"α={alpha:.2f} {self._bar(alpha)}"
            )
            self._log("DESCEND", z_cmd, res)

            # GE 确认计时
            if alpha < GE_CONFIRM_ALPHA:
                ge_timer += DT
            else:
                ge_timer = 0.0

            if ge_timer >= GE_CONFIRM_TIME:
                rospy.logwarn(
                    f"[DESCEND] *** GE CONFIRMED ***  "
                    f"alt={alt:.3f}m  z/R={res['z_r_pred']:.1f}")
                return True, x, y, alt

            self.rate.sleep()
        return False, x, y, 0.0

    def _phase_ge_hover(self, x: float, y: float, z: float) -> bool:
        """GE 确认后原地悬停 HOVER_WAIT 秒"""
        rospy.logwarn(
            f"[GE_HOVER] hovering at z={z:.3f}m for {HOVER_WAIT:.0f}s ...")
        t0 = time.time()
        while not rospy.is_shutdown():
            self._pub(x, y, z)
            self._keep_offboard()
            elapsed = time.time() - t0
            rospy.loginfo_throttle(
                2.0, f"[GE_HOVER] {elapsed:.1f}/{HOVER_WAIT:.0f}s")
            if elapsed >= HOVER_WAIT:
                return True
            self.rate.sleep()
        return False

    def _phase_land(self):
        """切换 AUTO.LAND，等待落地确认"""
        rospy.logwarn("[LAND] switching to AUTO.LAND ...")
        self.svc_mode(SetModeRequest(custom_mode="AUTO.LAND"))
        deadline = time.time() + LAND_TIMEOUT
        while not rospy.is_shutdown() and time.time() < deadline:
            if self._alt() < LAND_ALT_THRESH:
                rospy.loginfo("[LAND] touchdown confirmed")
                return
            self.rate.sleep()
        rospy.logwarn(f"[LAND] timeout after {LAND_TIMEOUT:.0f}s, force disarm")
        self.svc_arm(CommandBoolRequest(value=False))

    # ── 主流程 ────────────────────────────────────────────────────────────────
    def run(self):
        if not self._wait_connect():
            return
        x0, y0, _ = self._xyz()
        if not self._arm_offboard(x0, y0, TAKEOFF_ALT):
            return

        rospy.loginfo("=== Scenario 1: Vertical GE Landing ===")
        rospy.loginfo(
            f"Model: {self.model_type} | V_MAX={V_MAX}m/s | "
            f"GE_threshold=α<{GE_CONFIRM_ALPHA} | "
            f"sim={'ON (GE injected)' if self.sim_mode else 'OFF (real sensors)'}"
        )

        if not self._phase_takeoff():
            return
        if not self._phase_hover(HOVER_DURATION, label="HOVER_STABLE"):
            return

        ok, gx, gy, gz = self._phase_descend()
        if not ok:
            return
        if not self._phase_ge_hover(gx, gy, gz):
            return

        self._phase_land()
        self._save_log()
        rospy.loginfo("=== Scenario 1 complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scenario 1: PINN vertical GE landing")
    parser.add_argument("--model", choices=["pinn_a", "pinn_b"], default="pinn_a",
                        help="PINN model variant to use")
    parser.add_argument("--sim", action="store_true",
                        help="仿真模式: 根据高度自动注入 GE 效应 (Gazebo 用, 实机不加此参数)")
    args = parser.parse_args()

    try:
        VerticalLanding(model_type=args.model, sim_mode=args.sim).run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.logwarn("Interrupted by user")
