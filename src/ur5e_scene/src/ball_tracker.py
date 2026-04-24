#!/usr/bin/env python3
import os
import json
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge


class TapedSquareBallTracker:
    def __init__(self):
        rospy.init_node("taped_square_ball_tracker")

        self.bridge = CvBridge()

        # -----------------------------
        # Params
        # -----------------------------
        self.image_topic = rospy.get_param("~image_topic", "/cam/color/image_raw")
        self.square_size = rospy.get_param("~square_size", 0.80)   # meters
        self.origin_corner = rospy.get_param("~origin_corner", "top_left")
        self.world_frame = rospy.get_param("~world_frame", "table")
        self.ball_diameter = rospy.get_param("~ball_diameter", 0.04)
        self.x_offset = rospy.get_param("~x_offset", 0.0)
        self.y_offset = rospy.get_param("~y_offset", 0.0)
        self.topview_ppm = rospy.get_param("~topview_ppm", 500)

        default_calib = os.path.expanduser("~/.ros/taped_square_calibration.json")
        self.calibration_file = rospy.get_param("~calibration_file", default_calib)

        # -----------------------------
        # Green ball detection (HSV)
        # -----------------------------
        self.green_lower = np.array([35, 60, 40], dtype=np.uint8)
        self.green_upper = np.array([95, 255, 255], dtype=np.uint8)

        self.min_ball_area = 100
        self.max_ball_area = 30000
        self.min_circularity = 0.55

        # -----------------------------
        # Runtime state
        # -----------------------------
        self.latest_frame = None
        self.ball_mask_view = None
        self.topview = None

        self.clicked_points = []       # raw clicked points
        self.square_pts_tl = None      # [tl, tr, br, bl]
        self.square_pts_img = None     # [origin, +x, opposite, +y]
        self.H = None
        self.H_top = None
        self.calibrated = False
        self.last_status = "Waiting for image..."

        # -----------------------------
        # ROS I/O
        # -----------------------------
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1
        )

        self.position_pub = rospy.Publisher("/ball_position", PointStamped, queue_size=1)
        self.marker_pub = rospy.Publisher("/ball_marker", Marker, queue_size=1)

        # -----------------------------
        # OpenCV
        # -----------------------------
        cv2.namedWindow("Ball Tracker", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Ball Tracker", self.mouse_callback)

        # Try loading saved calibration
        self.load_calibration()

        rospy.loginfo("TapedSquareBallTracker started")
        rospy.loginfo("Calibration file: %s", self.calibration_file)

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # ---------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------
    def order_corners_tl_tr_br_bl(self, pts):
        pts = np.array(pts, dtype=np.float32)

        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).reshape(-1)

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def rotate_for_origin(self, ordered_pts):
        tl, tr, br, bl = ordered_pts

        if self.origin_corner == "top_left":
            return np.array([tl, tr, br, bl], dtype=np.float32)
        elif self.origin_corner == "top_right":
            return np.array([tr, br, bl, tl], dtype=np.float32)
        elif self.origin_corner == "bottom_right":
            return np.array([br, bl, tl, tr], dtype=np.float32)
        elif self.origin_corner == "bottom_left":
            return np.array([bl, tl, tr, br], dtype=np.float32)
        else:
            rospy.logwarn("Unknown origin_corner '%s', using top_left", self.origin_corner)
            return np.array([tl, tr, br, bl], dtype=np.float32)

    # ---------------------------------------------------
    # Manual calibration
    # ---------------------------------------------------
    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.calibrated:
            return

        if len(self.clicked_points) < 4:
            self.clicked_points.append([x, y])
            rospy.loginfo("Clicked point %d: (%d, %d)", len(self.clicked_points), x, y)

            if len(self.clicked_points) == 4:
                self.finish_calibration_from_clicks()

    def finish_calibration_from_clicks(self):
        raw_pts = np.array(self.clicked_points, dtype=np.float32)
        self.square_pts_tl = self.order_corners_tl_tr_br_bl(raw_pts)
        self.square_pts_img = self.rotate_for_origin(self.square_pts_tl)
        self.compute_homographies()
        self.calibrated = True
        self.last_status = "Calibration complete and saved"
        self.save_calibration()

    def compute_homographies(self):
        S = float(self.square_size)

        square_pts_world = np.array([
            [0.0, 0.0],
            [S,   0.0],
            [S,   S  ],
            [0.0, S  ]
        ], dtype=np.float32)

        self.H = cv2.getPerspectiveTransform(self.square_pts_img, square_pts_world)

        ppm = int(self.topview_ppm)
        top_size = max(1, int(S * ppm))

        top_pts = np.array([
            [0,          0],
            [top_size-1, 0],
            [top_size-1, top_size-1],
            [0,          top_size-1]
        ], dtype=np.float32)

        self.H_top = cv2.getPerspectiveTransform(self.square_pts_img, top_pts)

    def save_calibration(self):
        if self.square_pts_tl is None:
            return

        data = {
            "square_size": self.square_size,
            "origin_corner": self.origin_corner,
            "points_tl_tr_br_bl": self.square_pts_tl.tolist()
        }

        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        with open(self.calibration_file, "w") as f:
            json.dump(data, f, indent=2)

        rospy.loginfo("Calibration saved to %s", self.calibration_file)

    def load_calibration(self):
        if not os.path.exists(self.calibration_file):
            self.last_status = "No saved calibration. Click 4 corners."
            return False

        try:
            with open(self.calibration_file, "r") as f:
                data = json.load(f)

            pts = np.array(data["points_tl_tr_br_bl"], dtype=np.float32)
            self.square_pts_tl = self.order_corners_tl_tr_br_bl(pts)
            self.square_pts_img = self.rotate_for_origin(self.square_pts_tl)
            self.compute_homographies()
            self.calibrated = True
            self.clicked_points = []
            self.last_status = "Loaded saved calibration"
            rospy.loginfo("Loaded calibration from %s", self.calibration_file)
            return True

        except Exception as e:
            rospy.logwarn("Failed to load calibration: %s", e)
            self.last_status = "Failed to load calibration. Click 4 corners."
            return False

    def reset_calibration(self):
        self.clicked_points = []
        self.square_pts_tl = None
        self.square_pts_img = None
        self.H = None
        self.H_top = None
        self.calibrated = False
        self.topview = None
        self.last_status = "Calibration reset. Click 4 corners."

    def clear_saved_calibration(self):
        if os.path.exists(self.calibration_file):
            os.remove(self.calibration_file)
            rospy.loginfo("Deleted calibration file: %s", self.calibration_file)
        self.reset_calibration()
        self.last_status = "Saved calibration deleted. Click 4 corners."

    # ---------------------------------------------------
    # Ball detection
    # ---------------------------------------------------
    def build_square_fill_mask(self, shape_hw):
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)

        if self.square_pts_img is not None:
            polygon = self.square_pts_img.astype(np.int32)
            cv2.fillConvexPoly(mask, polygon, 255)

        return mask

    def detect_ball(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ball_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

        if self.calibrated and self.square_pts_img is not None:
            square_fill = self.build_square_fill_mask(ball_mask.shape)
            ball_mask = cv2.bitwise_and(ball_mask, square_fill)

        kernel = np.ones((5, 5), np.uint8)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)
        ball_mask = cv2.GaussianBlur(ball_mask, (5, 5), 0)

        self.ball_mask_view = ball_mask.copy()

        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_ball_area or area > self.max_ball_area:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue

            circularity = 4.0 * np.pi * area / (peri * peri)
            if circularity < self.min_circularity:
                continue

            (x, y), r = cv2.minEnclosingCircle(cnt)
            u = int(x)
            v = int(y)
            r = int(r)

            if self.calibrated and self.square_pts_img is not None:
                inside = cv2.pointPolygonTest(
                    self.square_pts_img.astype(np.float32),
                    (float(u), float(v)),
                    False
                )
                if inside < 0:
                    continue

            if area > best_area:
                best_area = area
                best = (u, v, r)

        return best

    # ---------------------------------------------------
    # Coordinate conversion / publishing
    # ---------------------------------------------------
    def image_to_square(self, u, v):
        pt = np.array([[[u, v]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H)

        x = float(mapped[0][0][0]) + self.x_offset
        y = float(mapped[0][0][1]) + self.y_offset

        x = max(0.0, min(self.square_size, x))
        y = max(0.0, min(self.square_size, y))

        return x, y

    def publish_ball_position(self, x, y):
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.world_frame
        msg.point.x = x
        msg.point.y = y
        msg.point.z = 0.0
        self.position_pub.publish(msg)

    def publish_ball_marker(self, x, y):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.world_frame

        marker.ns = "ball"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = self.ball_diameter / 2.0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = self.ball_diameter
        marker.scale.y = self.ball_diameter
        marker.scale.z = self.ball_diameter

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration(0.2)
        self.marker_pub.publish(marker)

    def delete_ball_marker(self):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.world_frame
        marker.ns = "ball"
        marker.id = 0
        marker.action = Marker.DELETE
        self.marker_pub.publish(marker)

    # ---------------------------------------------------
    # Debug displays
    # ---------------------------------------------------
    def make_topview(self, frame, detection):
        if not self.calibrated or self.H_top is None:
            self.topview = None
            return

        ppm = int(self.topview_ppm)
        out_size = max(1, int(self.square_size * ppm))

        warped = cv2.warpPerspective(frame, self.H_top, (out_size, out_size))

        cv2.arrowedLine(warped, (20, 20), (120, 20), (0, 0, 255), 3, tipLength=0.15)
        cv2.putText(warped, "+x", (130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.arrowedLine(warped, (20, 20), (20, 120), (255, 0, 0), 3, tipLength=0.15)
        cv2.putText(warped, "+y", (28, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if detection is not None:
            u, v, _ = detection
            pt = np.array([[[u, v]]], dtype=np.float32)
            top_pt = cv2.perspectiveTransform(pt, self.H_top)
            tx = int(top_pt[0][0][0])
            ty = int(top_pt[0][0][1])

            r_disp = max(6, int(self.ball_diameter * ppm / 2.0))
            cv2.circle(warped, (tx, ty), r_disp, (0, 255, 0), 3)
            cv2.circle(warped, (tx, ty), 4, (0, 0, 255), -1)

        self.topview = warped

    def draw_ui(self, frame, detection):
        display = frame.copy()

        if self.square_pts_tl is not None:
            poly = self.square_pts_tl.astype(np.int32)
            cv2.polylines(display, [poly], True, (255, 0, 0), 2)

            for i, pt in enumerate(self.square_pts_tl):
                p = (int(pt[0]), int(pt[1]))
                cv2.circle(display, p, 5, (255, 0, 255), -1)
                cv2.putText(display, str(i + 1), (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        for i, pt in enumerate(self.clicked_points):
            p = (int(pt[0]), int(pt[1]))
            cv2.circle(display, p, 6, (0, 165, 255), -1)
            cv2.putText(display, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if not self.calibrated:
            cv2.putText(display, "Click 4 square corners", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "Saved after 4 clicks", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "r=reset  c=clear saved  q=quit", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display, "Tracking active", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "r=recalibrate  c=clear saved  q=quit", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(display, self.last_status, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if detection is not None and self.calibrated:
            u, v, r = detection
            x, y = self.image_to_square(u, v)

            cv2.circle(display, (u, v), r, (0, 255, 0), 3)
            cv2.circle(display, (u, v), 5, (0, 0, 255), -1)

            cv2.putText(
                display,
                f"x={100*x:.1f} cm, y={100*y:.1f} cm",
                (max(10, u - 120), max(25, v - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            self.publish_ball_position(x, y)
            self.publish_ball_marker(x, y)
        else:
            self.delete_ball_marker()

        return display

    # ---------------------------------------------------
    # Main loop
    # ---------------------------------------------------
    def run(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if self.latest_frame is None:
                cv2.waitKey(1)
                rate.sleep()
                continue

            frame = self.latest_frame.copy()
            detection = self.detect_ball(frame) if self.calibrated else None

            display = self.draw_ui(frame, detection)
            self.make_topview(frame, detection)

            cv2.imshow("Ball Tracker", display)

            if self.ball_mask_view is not None:
                cv2.imshow("Ball Mask", self.ball_mask_view)

            if self.topview is not None:
                cv2.imshow("Top View", self.topview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                self.reset_calibration()
            elif key == ord('c'):
                self.clear_saved_calibration()
            elif key == ord('q'):
                rospy.signal_shutdown("Quit")

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        node = TapedSquareBallTracker()
        node.run()
    except rospy.ROSInterruptException:
        pass