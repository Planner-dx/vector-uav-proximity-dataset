#!/usr/bin/env python3
"""
在 Gazebo 中生成/删除 1m 正方体障碍物

用法:
  # 在 x=3.0m 处生成（默认）
  python3 spawn_cube.py

  # 指定位置
  python3 spawn_cube.py --x 3.0 --y 0.0

  # 删除
  python3 spawn_cube.py --delete
"""
import argparse
import rospy
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion

CUBE_NAME   = "landing_platform"
CUBE_SIZE_X = 1.0   # m
CUBE_SIZE_Y = 1.0   # m
CUBE_SIZE_Z = 1.0   # m  ← 正方体高度

# SDF 模型 (静态, 橙色)
CUBE_SDF = """\
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="col">
        <geometry>
          <box><size>{sx} {sy} {sz}</size></box>
        </geometry>
      </collision>
      <visual name="vis">
        <geometry>
          <box><size>{sx} {sy} {sz}</size></box>
        </geometry>
        <material>
          <ambient>0.85 0.35 0.05 1</ambient>
          <diffuse>0.85 0.35 0.05 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def spawn(x: float, y: float):
    rospy.init_node("spawn_cube", anonymous=True)
    rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=10.0)
    srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

    sdf = CUBE_SDF.format(
        name=CUBE_NAME,
        sx=CUBE_SIZE_X, sy=CUBE_SIZE_Y, sz=CUBE_SIZE_Z,
    )

    # 模型 pose: 中心在 (x, y, CUBE_SIZE_Z/2)，底面贴地
    pose = Pose(
        position=Point(x=x, y=y, z=CUBE_SIZE_Z / 2.0),
        orientation=Quaternion(x=0, y=0, z=0, w=1),
    )

    resp = srv(
        model_name=CUBE_NAME,
        model_xml=sdf,
        robot_namespace="",
        initial_pose=pose,
        reference_frame="world",
    )
    if resp.success:
        rospy.loginfo(
            f"Spawned '{CUBE_NAME}'  center=({x:.2f}, {y:.2f}, {CUBE_SIZE_Z/2:.2f})  "
            f"top_z={CUBE_SIZE_Z:.2f}m  size={CUBE_SIZE_X}×{CUBE_SIZE_Y}×{CUBE_SIZE_Z}m"
        )
    else:
        rospy.logerr(f"Spawn failed: {resp.status_message}")


def delete():
    rospy.init_node("delete_cube", anonymous=True)
    rospy.wait_for_service("/gazebo/delete_model", timeout=10.0)
    srv = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    resp = srv(model_name=CUBE_NAME)
    if resp.success:
        rospy.loginfo(f"Deleted '{CUBE_NAME}'")
    else:
        rospy.logerr(f"Delete failed: {resp.status_message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spawn/delete 1m cube in Gazebo")
    parser.add_argument("--x",      type=float, default=3.0,  help="Cube center X [m] (default: 3.0)")
    parser.add_argument("--y",      type=float, default=0.0,  help="Cube center Y [m] (default: 0.0)")
    parser.add_argument("--delete", action="store_true",      help="Delete the cube instead of spawning")
    args = parser.parse_args()

    try:
        if args.delete:
            delete()
        else:
            spawn(args.x, args.y)
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
