#!/usr/bin/env python3
"""
Gazebo demo: simple waypoint-based platform landing with PINN monitoring.
Only position setpoints are used.
"""
import sys
import time
import argparse
import csv
from collections import deque
from pathlib import Path
from threading import Lock

import numpy as np

import rospy
from mavros_msgs.msg import State, ActuatorControl
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from mavros_msgs.srv import SetMode, SetModeRequest
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from control_head import ControlHead

R = 0.08
WINDOW_SIZE = 50
DSHOT_SCALE = 2047.0
DEFAULT_HOVER_THRUST = 0.5
SETPOINT_RATE = 20
DT = 1.0 / SETPOINT_RATE
CONNECT_TIMEOUT = 10.0
WAYPOINT_TIMEOUT = 20.0
TOTAL_TIMEOUT = 90.0
REACH_TOL = 0.3
GE_CONFIRM_ALPHA = 0.10
GE_CONFIRM_TIME = 1.5
ALPHA_LP_NEW = 0.15
ALPHA_LP_PREV = 0.85
GE_LAND_Z = 1.85
MIN_ALT_DISARM = 1.90
GE_DISARM_ALT = 2.25
GE_LAND_TIMEOUT = 15.0

PAD_X = 3.0
PAD_Y = 0.0
PAD_Z = 2.0
PAD_X_MIN = 2.0
PAD_X_MAX = 4.0

WAYPOINTS = [
    (0.0, 0.0, 2.5, 4.0, "TAKEOFF"),
    (1.5, 0.0, 2.5, 2.0, "CRUISE"),
    (3.0, 0.0, 2.5, 3.0, "ABOVE_PAD"),
    (3.0, 0.0, 2.3, 3.0, "DESCEND_1"),
    (3.0, 0.0, 2.15, 8.0, "GND_EFFECT"),
    (3.0, 0.0, 2.05, 15.0, "FINAL"),
]


class GazeboDemoLanding:
    def __init__(self, model_type="pinn_a"):
        self.model_type = model_type
        self.current_state = State()
        self.current_pose = PoseStamped()
        self.data_lock = Lock()
        self.imu_buffer = deque(maxlen=WINDOW_SIZE)
        self.actuator_buffer = deque(maxlen=WINDOW_SIZE)
        self.hover_thrust = DEFAULT_HOVER_THRUST
        self.log_data = []
        self.alpha_prev = None

        rospy.init_node("gazebo_demo_landing", anonymous=True)
        self.rate = rospy.Rate(SETPOINT_RATE)

        rospy.Subscriber("/mavros/state", State, self._state_cb, queue_size=10)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb, queue_size=10)
        rospy.Subscriber("/mavros/imu/data", Imu, self._imu_cb, queue_size=50)
        rospy.Subscriber("/mavros/target_actuator_control", ActuatorControl, self._actuator_cb, queue_size=50)
        rospy.Subscriber("/mavros/actuator_control", ActuatorControl, self._actuator_cb, queue_size=50)
        rospy.Subscriber("/mavros/hover_thrust_estimate", Float64, self._hover_thrust_cb, queue_size=10)

        self.pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)

        rospy.wait_for_service("/mavros/cmd/arming", timeout=CONNECT_TIMEOUT)
        rospy.wait_for_service("/mavros/set_mode", timeout=CONNECT_TIMEOUT)
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        model_path = SCRIPT_DIR / "04_pinn" / f"{model_type}_model.pth"
        scaler_path = SCRIPT_DIR / "03_mlp" / "scaler.pkl"
        self.ctrl = ControlHead(str(model_path), str(scaler_path), model_type=model_type, device="cpu")
        self.ctrl.reset()

    def _state_cb(self, msg):
        self.current_state = msg

    def _pose_cb(self, msg):
        self.current_pose = msg

    def _imu_cb(self, msg):
        a = msg.linear_acceleration
        g = msg.angular_velocity
        t = msg.header.stamp.to_sec()
        with self.data_lock:
            self.imu_buffer.append(([a.x, a.y, a.z, g.x, g.y, g.z], t))

    def _actuator_cb(self, msg):
        ctrls = list(msg.controls)
        motors = [ctrls[i] * DSHOT_SCALE for i in range(min(6, len(ctrls)))]
        while len(motors) < 6:
            motors.append(0.0)
        t = msg.header.stamp.to_sec()
        with self.data_lock:
            self.actuator_buffer.append((motors, t))

    def _hover_thrust_cb(self, msg):
        self.hover_thrust = msg.data

    def _pub_pos(self, x, y, z):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        self.pos_pub.publish(msg)

    def _keep_offboard(self):
        if self.current_state.mode != "OFFBOARD":
            self.set_mode_client(SetModeRequest(custom_mode="OFFBOARD"))

    def _wait_connection(self):
        start = time.time()
        while not rospy.is_shutdown():
            if self.current_state.connected:
                return True
            if time.time() - start > CONNECT_TIMEOUT:
                rospy.logerr("FCU connection timeout")
                return False
            self.rate.sleep()
        return False

    def _offboard_arm(self, first_wp):
        # Phase 1: 预发100个setpoint（PX4要求OFFBOARD前先收到setpoint流）
        for _ in range(100):
            if rospy.is_shutdown():
                return False
            self._pub_pos(first_wp[0], first_wp[1], first_wp[2])
            self.rate.sleep()

        # Phase 2: 循环尝试切OFFBOARD + arm，持续发setpoint，最多10秒
        deadline = time.time() + 10.0
        last_req = 0.0
        while not rospy.is_shutdown() and time.time() < deadline:
            self._pub_pos(first_wp[0], first_wp[1], first_wp[2])
            now = time.time()
            if now - last_req > 1.0:
                if self.current_state.mode != "OFFBOARD":
                    self.set_mode_client(SetModeRequest(custom_mode="OFFBOARD"))
                elif not self.current_state.armed:
                    self.arming_client(CommandBoolRequest(value=True))
                last_req = now
            if self.current_state.mode == "OFFBOARD" and self.current_state.armed:
                rospy.loginfo("OFFBOARD + Armed confirmed")
                return True
            self.rate.sleep()

        rospy.logerr(f"Arm failed: mode={self.current_state.mode}, armed={self.current_state.armed}")
        return False

    def _disarm(self):
        self.arming_client(CommandBoolRequest(value=False))

    def _pos(self):
        p = self.current_pose.pose.position
        return p.x, p.y, p.z

    def _dist_to(self, target):
        x, y, z = self._pos()
        delta = np.array([x - target[0], y - target[1], z - target[2]], dtype=np.float32)
        return float(np.linalg.norm(delta))

    def build_features(self):
        with self.data_lock:
            imu_list = list(self.imu_buffer)
            act_list = list(self.actuator_buffer)
        if len(imu_list) < 10 or len(act_list) < 5:
            return None

        imu_arr = np.array([row[0] for row in imu_list], dtype=np.float32)
        acc = imu_arr[:, :3]
        gyro = imu_arr[:, 3:]
        motor_arr = np.array([row[0] for row in act_list], dtype=np.float32)

        return np.concatenate([
            acc.mean(axis=0), acc.std(axis=0),
            gyro.mean(axis=0), gyro.std(axis=0),
            motor_arr.mean(axis=0),
            np.array([self.hover_thrust], dtype=np.float32),
        ]).astype(np.float32)

    def inject_ground_effect(self, features_19d, z_abs, x_pos):
        z_relative = z_abs - 2.0
        if not (PAD_X_MIN <= x_pos <= PAD_X_MAX):
            return features_19d, max(z_relative, 0.0), 0.0

        effect_strength = max(0.0, 1.0 - z_relative / (10.0 * 0.0889))
        modified = features_19d.copy()
        for i in range(12, 18):
            modified[i] *= (1.0 - 0.85 * effect_strength)
        for i in range(3, 6):
            modified[i] *= (1.0 + 40.0 * effect_strength)
        for i in range(9, 12):
            modified[i] *= (1.0 + 15.0 * effect_strength)
        modified[18] *= (1.0 - 0.70 * effect_strength)
        return modified, max(z_relative, 0.0), effect_strength

    @staticmethod
    def _bar(alpha):
        width = 20
        filled = int(round(np.clip(alpha, 0.0, 1.0) * width))
        return "#" * filled + "." * (width - filled)

    def _print_status(self, elapsed, wp_name, alt, alt_pad, z_r, cls_name, cls_conf, alpha):
        cls_disp = cls_name.upper() if cls_name == "danger" else cls_name
        print(
            f"[{elapsed:6.1f}s] WP:{wp_name:<12s} ALT={alt:4.2f}m PAD={alt_pad:4.2f}m "
            f"z/R={z_r:4.1f} | {cls_disp:<6s}({cls_conf:0.2f}) | alpha={alpha:0.2f} {self._bar(alpha)}"
        )

    def _record(self, elapsed, wp_name, wp_target, dist, alt, alt_pad, z_r, effect_strength, result):
        x, y, _ = self._pos()
        self.log_data.append({
            "time": round(elapsed, 2),
            "waypoint": wp_name,
            "target_x": wp_target[0],
            "target_y": wp_target[1],
            "target_z": wp_target[2],
            "x": round(x, 3),
            "y": round(y, 3),
            "altitude": round(alt, 3),
            "alt_above_pad": round(alt_pad, 3),
            "distance": round(dist, 3),
            "z_r": round(z_r, 2),
            "effect_strength": round(effect_strength, 3),
            "alpha": round(result["alpha"], 4),
            "alpha_raw": round(result["alpha_raw"], 4),
            "classification": result["classification"],
            "confidence": round(result["class_confidence"], 3),
            "override": result["override_active"],
        })

    def _infer_once(self, wp_name, wp_target, elapsed):
        x, _, alt = self._pos()
        features = self.build_features()
        result = {
            "alpha": 1.0,
            "alpha_raw": 1.0,
            "z_r_pred": 99.0,
            "classification": "safe",
            "class_confidence": 0.0,
            "override_active": False,
        }
        alt_pad = max(alt - PAD_Z, 0.0)
        z_r = alt_pad / R
        effect_strength = 0.0
        if features is not None:
            injected, alt_pad, effect_strength = self.inject_ground_effect(features, alt, x)
            result = self.ctrl.compute_alpha(injected)
            z_r = alt_pad / R

        alpha_new = float(result["alpha"])
        if self.alpha_prev is None:
            alpha_filtered = alpha_new
        else:
            alpha_filtered = ALPHA_LP_NEW * alpha_new + ALPHA_LP_PREV * self.alpha_prev
        self.alpha_prev = alpha_filtered
        result["alpha"] = float(np.clip(alpha_filtered, 0.0, 1.0))

        dist = self._dist_to(wp_target)
        self._print_status(elapsed, wp_name, alt, alt_pad, z_r, result["classification"], result["class_confidence"], result["alpha"])
        self._record(elapsed, wp_name, wp_target, dist, alt, alt_pad, z_r, effect_strength, result)
        return result

    def _auto_land_after_ge(self, t_start):
        print("GE CONFIRMED - AUTO LANDING on platform")
        print(f"Landing target setpoint: (x={PAD_X:.2f}, y={PAD_Y:.2f}, z={GE_LAND_Z:.2f})")
        start = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            elapsed = now - t_start
            self._pub_pos(PAD_X, PAD_Y, GE_LAND_Z)
            self._keep_offboard()
            self._infer_once("AUTO_LAND", (PAD_X, PAD_Y, GE_LAND_Z), elapsed)

            _, _, alt = self._pos()
            if alt <= GE_DISARM_ALT:
                print(f"Touchdown detected (alt={alt:.2f}m <= {GE_DISARM_ALT:.2f}m), disarm")
                self._disarm()
                return False

            if now - start > GE_LAND_TIMEOUT:
                print(f"Auto landing timeout ({GE_LAND_TIMEOUT:.1f}s), disarm for safety")
                self._disarm()
                return False

            if elapsed > TOTAL_TIMEOUT:
                print(f"Total timeout: {TOTAL_TIMEOUT}s")
                self._disarm()
                return False

            self.rate.sleep()
        return False

    def _run_waypoint(self, waypoint, t_start):
        target = waypoint[:3]
        hold_seconds = waypoint[3]
        name = waypoint[4]
        ge_low_timer = 0.0
        leg_start = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            elapsed = now - t_start
            self._pub_pos(*target)
            self._keep_offboard()
            result = self._infer_once(name, target, elapsed)

            _, _, cur_alt = self._pos()
            if cur_alt > 1.5 and result["alpha"] < GE_CONFIRM_ALPHA:
                ge_low_timer += DT
            else:
                ge_low_timer = 0.0
            if ge_low_timer >= GE_CONFIRM_TIME:
                return self._auto_land_after_ge(t_start)

            if self._dist_to(target) < REACH_TOL:
                break
            if now - leg_start > WAYPOINT_TIMEOUT:
                print(f"Waypoint timeout: {name}")
                return True
            if elapsed > TOTAL_TIMEOUT:
                print(f"Total timeout: {TOTAL_TIMEOUT}s")
                self._disarm()
                return False
            self.rate.sleep()

        hold_start = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            elapsed = now - t_start
            if now - hold_start >= hold_seconds:
                return True
            self._pub_pos(*target)
            self._keep_offboard()
            result = self._infer_once(name, target, elapsed)

            _, _, cur_alt = self._pos()
            if cur_alt > 1.5 and result["alpha"] < GE_CONFIRM_ALPHA:
                ge_low_timer += DT
            else:
                ge_low_timer = 0.0
            if ge_low_timer >= GE_CONFIRM_TIME:
                return self._auto_land_after_ge(t_start)
            if elapsed > TOTAL_TIMEOUT:
                print(f"Total timeout: {TOTAL_TIMEOUT}s")
                self._disarm()
                return False
            self.rate.sleep()
        return False

    def run(self):
        if not self._wait_connection():
            return False
        if not self._offboard_arm(WAYPOINTS[0]):
            return False

        print("Simple waypoint demo started")
        print(f"Model: {self.model_type} | Position setpoints only")
        self.ctrl.reset()
        self.alpha_prev = None
        t_start = rospy.Time.now().to_sec()

        ok = True
        for waypoint in WAYPOINTS:
            if not self._run_waypoint(waypoint, t_start):
                ok = False
                break

        if ok:
            print("Waypoints complete, disarming")
            self._disarm()

        self._save_log()
        self._plot()
        return ok

    def _save_log(self):
        if not self.log_data:
            return
        csv_path = SCRIPT_DIR / "06_simulation" / "demo_landing_log.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.log_data[0].keys()))
            writer.writeheader()
            writer.writerows(self.log_data)
        print(f"Saved: {csv_path}")

    def _plot(self):
        if len(self.log_data) < 2:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        t = np.array([row["time"] for row in self.log_data], dtype=np.float32)
        alt = np.array([row["altitude"] for row in self.log_data], dtype=np.float32)
        alt_pad = np.array([row["alt_above_pad"] for row in self.log_data], dtype=np.float32)
        alpha = np.array([row["alpha"] for row in self.log_data], dtype=np.float32)

        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(t, alt, color="#1565C0", linewidth=2, label="Altitude")
        ax1.plot(t, alt_pad, color="#42A5F5", linestyle="--", linewidth=1.5, label="Above pad")
        ax1.axhline(PAD_Z, color="#6D4C41", linestyle=":", linewidth=1.5, label="Pad top")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Altitude (m)")
        ax1.grid(True, alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(t, alpha, color="#E65100", linewidth=2, label="alpha")
        ax2.set_ylabel("alpha")
        ax2.set_ylim(-0.05, 1.05)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
        ax1.set_title(f"Demo Landing ({self.model_type})")
        fig.tight_layout()

        plot_path = SCRIPT_DIR / "06_simulation" / "demo_landing_plot.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"Saved: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple waypoint demo landing")
    parser.add_argument("--model", choices=["pinn_a", "pinn_b"], default="pinn_a")
    args = parser.parse_args()

    try:
        GazeboDemoLanding(model_type=args.model).run()
    except rospy.ROSInterruptException:
        print("ROS interrupted")
    except KeyboardInterrupt:
        print("User interrupted")
