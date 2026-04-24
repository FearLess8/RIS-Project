#!/usr/bin/env python3
"""
interceptor.py

Standalone interceptor that:
  - Subscribes to /ball_position
  - Computes velocity + prediction internally
  - Transforms predicted point into MoveIt's planning frame
  - Moves the robot arm via MoveIt
"""

import sys
import time

import rospy
import moveit_commander
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs


class BallInterceptorTest:
    def __init__(self):
        rospy.init_node("ball_interceptor_test")

        moveit_commander.roscpp_initialize(sys.argv)
        self.group = moveit_commander.MoveGroupCommander("arm")

        # TF listener for frame transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- Trajectory state ---
        self.history = []       # [(x, y, t)]
        self.max_history = 10
        self.predict_dt = 0.5   # seconds ahead to predict

        # --- Robot movement throttle ---
        self.z_height = 0.1       # meters above table / base frame target z
        self.last_move_time = 0.0
        self.move_interval = 0.5  # seconds between MoveIt calls

        # --- Subscribe to ball position ---
        self.sub = rospy.Subscriber(
            "/ball_position",
            PointStamped,
            self.callback
        )

        rospy.loginfo("BallInterceptorTest ready — listening on /ball_position")
        rospy.loginfo(f"MoveIt planning frame: {self.group.get_planning_frame()}")

    def callback(self, msg):
        x = msg.point.x
        y = msg.point.y
        now = time.time()

        # Build history
        self.history.append((x, y, now))
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Need at least 2 points for velocity
        if len(self.history) < 2:
            return

        x1, y1, t1 = self.history[-2]
        x2, y2, t2 = self.history[-1]

        dt = t2 - t1
        if dt <= 0:
            return

        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt

        pred_x = x2 + vx * self.predict_dt
        pred_y = y2 + vy * self.predict_dt

        rospy.loginfo(
            f"pos=({x2:.3f}, {y2:.3f})  "
            f"vel=({vx:.3f}, {vy:.3f})  "
            f"pred=({pred_x:.3f}, {pred_y:.3f})"
        )

        if now - self.last_move_time < self.move_interval:
            return

        self._move_to(pred_x, pred_y)
        self.last_move_time = now

    def _move_to(self, x, y):
        # Point expressed in the table frame
        point_in_table = PointStamped()
        point_in_table.header.stamp = rospy.Time(0)
        point_in_table.header.frame_id = "table"
        point_in_table.point.x = x
        point_in_table.point.y = y
        point_in_table.point.z = 0.0

        planning_frame = self.group.get_planning_frame()

        try:
            point_in_base = self.tf_buffer.transform(
                point_in_table,
                planning_frame,
                rospy.Duration(1.0)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return

        tx = point_in_base.point.x
        ty = point_in_base.point.y
        tz = self.z_height

        rospy.loginfo(
            f"table({x:.3f}, {y:.3f}) -> {planning_frame}({tx:.3f}, {ty:.3f}, {tz:.3f})"
        )

        self.group.set_position_target([tx, ty, tz])

        plan_result = self.group.plan()

        if isinstance(plan_result, tuple):
            success, plan, planning_time, error_code = plan_result
        else:
            plan = plan_result
            success = hasattr(plan, "joint_trajectory") and len(plan.joint_trajectory.points) > 0

        if not success:
            rospy.logwarn(
                f"Planning failed for ({tx:.3f}, {ty:.3f}, {tz:.3f}) in {planning_frame}"
            )
            self.group.clear_pose_targets()
            return

        exec_success = self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

        if exec_success:
            rospy.loginfo(f"Moved to ({tx:.3f}, {ty:.3f}, {tz:.3f})")
        else:
            rospy.logwarn(f"Execution failed for ({tx:.3f}, {ty:.3f}, {tz:.3f})")


if __name__ == "__main__":
    try:
        node = BallInterceptorTest()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass