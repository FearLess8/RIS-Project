#!/usr/bin/env python3
import math
from collections import deque

import rospy
import numpy as np

from geometry_msgs.msg import Point, PointStamped, Vector3Stamped
from visualization_msgs.msg import Marker, MarkerArray
from ur5e_scene.msg import BallTrajectory   # change package name if needed


class TrajectoryPredictor:
    def __init__(self):
        rospy.init_node("ball_trajectory_estimator")

        self.input_topic = rospy.get_param("~input_topic", "/ball_position")
        self.history_len = rospy.get_param("~history_len", 10)
        self.prediction_dt = rospy.get_param("~prediction_dt", 0.4)
        self.min_samples = rospy.get_param("~min_samples", 3)
        self.min_speed = rospy.get_param("~min_speed", 0.01)   # m/s
        self.stale_timeout = rospy.get_param("~stale_timeout", 0.5)
        self.marker_z = rospy.get_param("~marker_z", 0.02)
        self.arrow_scale = rospy.get_param("~arrow_scale", 1.0)

        self.history = deque(maxlen=self.history_len)
        self.last_msg_time = None
        self.frame_id = "table_top"

        self.pos_sub = rospy.Subscriber(
            self.input_topic,
            PointStamped,
            self.position_callback,
            queue_size=1
        )

        self.vel_pub = rospy.Publisher("/ball_velocity", Vector3Stamped, queue_size=1)
        self.pred_pub = rospy.Publisher("/ball_prediction", PointStamped, queue_size=1)
        self.traj_pub = rospy.Publisher("/ball_trajectory", BallTrajectory, queue_size=1)
        self.marker_pub = rospy.Publisher("/ball_trajectory_markers", MarkerArray, queue_size=1)

        rospy.loginfo("BallTrajectoryEstimator started")
        rospy.loginfo("Listening on %s", self.input_topic)

    def position_callback(self, msg):
        now = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time() else rospy.Time.now().to_sec()
        self.last_msg_time = rospy.Time.now()
        self.frame_id = msg.header.frame_id if msg.header.frame_id else "table_top"

        self.history.append((msg.point.x, msg.point.y, now))

        if len(self.history) < self.min_samples:
            self.publish_basic_markers_only(msg.point.x, msg.point.y)
            return

        result = self.estimate_velocity()
        if result is None:
            return

        x, y, vx, vy = result
        speed = math.hypot(vx, vy)

        if speed < self.min_speed:
            vx = 0.0
            vy = 0.0
            speed = 0.0

        pred_x = x + vx * self.prediction_dt
        pred_y = y + vy * self.prediction_dt

        self.publish_velocity(vx, vy)
        self.publish_prediction(pred_x, pred_y)
        self.publish_trajectory_msg(x, y, vx, vy, speed, pred_x, pred_y)
        self.publish_markers(x, y, vx, vy, pred_x, pred_y)

    def estimate_velocity(self):
        if len(self.history) < self.min_samples:
            return None

        arr = np.array(self.history, dtype=np.float64)
        xs = arr[:, 0]
        ys = arr[:, 1]
        ts = arr[:, 2]

        # normalize time for numerical stability
        t0 = ts[0]
        ts = ts - t0

        if ts[-1] <= 0.0:
            return None

        # linear fit: x(t) = ax*t + bx, y(t) = ay*t + by
        try:
            ax, bx = np.polyfit(ts, xs, 1)
            ay, by = np.polyfit(ts, ys, 1)
        except Exception as e:
            rospy.logwarn("Velocity fit failed: %s", e)
            return None

        # current estimated position at latest time
        t_now = ts[-1]
        x_now = ax * t_now + bx
        y_now = ay * t_now + by

        vx = ax
        vy = ay

        return x_now, y_now, vx, vy

    def publish_velocity(self, vx, vy):
        msg = Vector3Stamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.vector.x = vx
        msg.vector.y = vy
        msg.vector.z = 0.0
        self.vel_pub.publish(msg)

    def publish_prediction(self, x, y):
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.point.x = x
        msg.point.y = y
        msg.point.z = 0.0
        self.pred_pub.publish(msg)

    def publish_trajectory_msg(self, x, y, vx, vy, speed, pred_x, pred_y):
        msg = BallTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id

        msg.position.x = x
        msg.position.y = y
        msg.position.z = 0.0

        msg.velocity.x = vx
        msg.velocity.y = vy
        msg.velocity.z = 0.0

        msg.speed = speed

        msg.predicted_position.x = pred_x
        msg.predicted_position.y = pred_y
        msg.predicted_position.z = 0.0

        msg.prediction_dt = self.prediction_dt

        history_points = []
        for hx, hy, _ in self.history:
            p = Point()
            p.x = hx
            p.y = hy
            p.z = 0.0
            history_points.append(p)

        msg.history = history_points
        self.traj_pub.publish(msg)

    def publish_basic_markers_only(self, x, y):
        markers = MarkerArray()

        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = self.frame_id
        m.ns = "ball_trajectory"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = self.marker_z
        m.pose.orientation.w = 1.0
        m.scale.x = 0.04
        m.scale.y = 0.04
        m.scale.z = 0.04
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0
        m.lifetime = rospy.Duration(0.2)
        markers.markers.append(m)

        self.marker_pub.publish(markers)

    def publish_markers(self, x, y, vx, vy, pred_x, pred_y):
        markers = MarkerArray()
        stamp = rospy.Time.now()

        # 1) history path
        path_marker = Marker()
        path_marker.header.stamp = stamp
        path_marker.header.frame_id = self.frame_id
        path_marker.ns = "ball_trajectory"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.pose.orientation.w = 1.0
        path_marker.scale.x = 0.01
        path_marker.color.r = 0.0
        path_marker.color.g = 0.5
        path_marker.color.b = 1.0
        path_marker.color.a = 1.0
        path_marker.lifetime = rospy.Duration(0.3)

        for hx, hy, _ in self.history:
            p = Point()
            p.x = hx
            p.y = hy
            p.z = self.marker_z
            path_marker.points.append(p)

        markers.markers.append(path_marker)

        # 2) velocity arrow
        arrow_marker = Marker()
        arrow_marker.header.stamp = stamp
        arrow_marker.header.frame_id = self.frame_id
        arrow_marker.ns = "ball_trajectory"
        arrow_marker.id = 1
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        arrow_marker.pose.orientation.w = 1.0
        arrow_marker.scale.x = 0.015   # shaft diameter
        arrow_marker.scale.y = 0.03    # head diameter
        arrow_marker.scale.z = 0.04    # head length
        arrow_marker.color.r = 1.0
        arrow_marker.color.g = 0.2
        arrow_marker.color.b = 0.2
        arrow_marker.color.a = 1.0
        arrow_marker.lifetime = rospy.Duration(0.3)

        start = Point()
        start.x = x
        start.y = y
        start.z = self.marker_z

        end = Point()
        end.x = x + vx * self.prediction_dt * self.arrow_scale
        end.y = y + vy * self.prediction_dt * self.arrow_scale
        end.z = self.marker_z

        arrow_marker.points = [start, end]
        markers.markers.append(arrow_marker)

        # 3) predicted point
        pred_marker = Marker()
        pred_marker.header.stamp = stamp
        pred_marker.header.frame_id = self.frame_id
        pred_marker.ns = "ball_trajectory"
        pred_marker.id = 2
        pred_marker.type = Marker.SPHERE
        pred_marker.action = Marker.ADD
        pred_marker.pose.position.x = pred_x
        pred_marker.pose.position.y = pred_y
        pred_marker.pose.position.z = self.marker_z
        pred_marker.pose.orientation.w = 1.0
        pred_marker.scale.x = 0.03
        pred_marker.scale.y = 0.03
        pred_marker.scale.z = 0.03
        pred_marker.color.r = 1.0
        pred_marker.color.g = 1.0
        pred_marker.color.b = 0.0
        pred_marker.color.a = 1.0
        pred_marker.lifetime = rospy.Duration(0.3)
        markers.markers.append(pred_marker)

        self.marker_pub.publish(markers)

    def run(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if self.last_msg_time is not None:
                age = (rospy.Time.now() - self.last_msg_time).to_sec()
                if age > self.stale_timeout:
                    self.history.clear()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = TrajectoryPredictor()
        node.run()
    except rospy.ROSInterruptException:
        pass