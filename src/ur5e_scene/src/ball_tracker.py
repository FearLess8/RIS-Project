#!/usr/bin/env python3
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge


class BallTrackerTopViewNode:
    def __init__(self):
        rospy.init_node("ball_tracker_topview_node")

        self.bridge = CvBridge()

        # -----------------------------
        # ROS params
        # -----------------------------
        self.image_topic   = rospy.get_param("~image_topic", "/cam/color/image_raw")
        self.world_frame   = rospy.get_param("~world_frame", "map")

        # Real table dimensions in meters
        self.table_length  = rospy.get_param("~table_length", 1.60)
        self.table_width   = rospy.get_param("~table_width", 0.80)

        # Which detected table corner should be the origin
        # one of: top_left, top_right, bottom_right, bottom_left
        self.origin_corner = rospy.get_param("~origin_corner", "top_left")

        # Ball size in RViz
        self.ball_diameter = rospy.get_param("~ball_diameter", 0.04)

        # Constant correction offsets in meters
        self.x_offset      = rospy.get_param("~x_offset", 0.0)
        self.y_offset      = rospy.get_param("~y_offset", 0.0)

        # Debug top-view resolution
        self.topview_ppm   = rospy.get_param("~topview_ppm", 500)

        # -----------------------------
        # Ball color thresholds (LAB)
        # -----------------------------
        self.lower = np.array([85, 82, 132], dtype=np.uint8)
        self.upper = np.array([130, 100, 150], dtype=np.uint8)

        # Ball detection filters
        self.min_area = 150
        self.max_area = 20000
        self.min_circularity = 0.60

        # -----------------------------
        # Runtime state
        # -----------------------------
        self.latest_frame = None
        self.mask_view = None
        self.topview = None
        self.detect_debug = None

        self.img_pts = None       # 4 table corners in image, in mapping order
        self.world_pts = None     # matching real-world rectangle points
        self.H = None             # image -> world homography
        self.H_top = None         # image -> top-view homography
        self.calibrated = False
        self.auto_detect_attempted = False
        self.last_detect_status = "Waiting for image..."

        # -----------------------------
        # ROS I/O
        # -----------------------------
        self.position_pub = rospy.Publisher("/ball_position", PointStamped, queue_size=1)
        self.marker_pub = rospy.Publisher("/ball_marker", Marker, queue_size=1)

        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1
        )

        # -----------------------------
        # OpenCV windows
        # -----------------------------
        cv2.namedWindow("Ball Tracker", cv2.WINDOW_NORMAL)

        rospy.loginfo("BallTrackerTopViewNode started")
        rospy.loginfo("Table size: %.2f m x %.2f m", self.table_length, self.table_width)
        rospy.loginfo("Origin corner: %s", self.origin_corner)
        rospy.loginfo("Offsets: x=%.3f m, y=%.3f m", self.x_offset, self.y_offset)

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # ---------------------------------------------------
    # Corner ordering / origin assignment
    # ---------------------------------------------------
    def order_corners_tl_tr_br_bl(self, pts):
        pts = np.array(pts, dtype=np.float32)

        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).reshape(-1)

        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(d)]
        bottom_left = pts[np.argmax(d)]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    def rotate_corners_for_origin(self, ordered_pts):
        # ordered_pts assumed in [top_left, top_right, bottom_right, bottom_left]
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
            rospy.logwarn("Unknown origin_corner '%s', falling back to top_left", self.origin_corner)
            return np.array([tl, tr, br, bl], dtype=np.float32)

    # ---------------------------------------------------
    # Automatic table detection
    # ---------------------------------------------------
    def detect_table_corners(self, frame):
        h0, w0 = frame.shape[:2]

        # downscale for faster / more stable contour search
        target_w = min(1280, w0)
        scale = float(target_w) / float(w0)

        if scale < 1.0:
            small = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
        else:
            small = frame.copy()
            scale = 1.0

        h, w = small.shape[:2]
        img_area = float(h * w)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        edges = cv2.Canny(blur, 40, 120)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.dilate(edges, kernel, iterations=1)

        self.detect_debug = edges.copy()

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        best_quad = None
        best_score = -1e9

        expected_ratio = self.table_length / max(self.table_width, 1e-6)

        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)
            if cnt_area < 0.12 * img_area:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue

            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(-1, 2).astype(np.float32)
            else:
                rect = cv2.minAreaRect(cnt)
                quad = cv2.boxPoints(rect).astype(np.float32)

            quad = self.order_corners_tl_tr_br_bl(quad)

            quad_area = cv2.contourArea(quad)
            if quad_area < 0.12 * img_area:
                continue
            if quad_area > 0.97 * img_area:
                continue

            # side lengths
            w1 = np.linalg.norm(quad[1] - quad[0])
            w2 = np.linalg.norm(quad[2] - quad[3])
            h1 = np.linalg.norm(quad[3] - quad[0])
            h2 = np.linalg.norm(quad[2] - quad[1])

            avg_w = 0.5 * (w1 + w2)
            avg_h = 0.5 * (h1 + h2)

            if avg_w < 30 or avg_h < 30:
                continue

            aspect = avg_w / max(avg_h, 1e-6)

            # soft penalties
            aspect_penalty = abs(np.log(max(aspect, 1e-6) / expected_ratio))

            margin = 10
            touches_border = (
                (quad[:, 0] < margin) |
                (quad[:, 0] > (w - 1 - margin)) |
                (quad[:, 1] < margin) |
                (quad[:, 1] > (h - 1 - margin))
            )
            touch_fraction = np.mean(touches_border.astype(np.float32))

            # score: prefer large quads, prefer not touching image border,
            # weakly prefer expected aspect ratio
            score = (
                2.0 * (quad_area / img_area)
                - 0.35 * touch_fraction
                - 0.08 * aspect_penalty
            )

            if score > best_score:
                best_score = score
                best_quad = quad.copy()

        if best_quad is None:
            return None

        if scale < 1.0:
            best_quad[:, 0] /= scale
            best_quad[:, 1] /= scale

        return best_quad

    def auto_calibrate(self, frame):
        detected = self.detect_table_corners(frame)

        if detected is None:
            self.calibrated = False
            self.last_detect_status = "Auto table detection failed"
            return False

        ordered = self.order_corners_tl_tr_br_bl(detected)
        self.img_pts = self.rotate_corners_for_origin(ordered)

        L = float(self.table_length)
        W = float(self.table_width)

        self.world_pts = np.array([
            [0.0, 0.0],
            [L,   0.0],
            [L,   W  ],
            [0.0, W  ]
        ], dtype=np.float32)

        self.H = cv2.getPerspectiveTransform(self.img_pts, self.world_pts)

        ppm = int(self.topview_ppm)
        top_w = max(1, int(L * ppm))
        top_h = max(1, int(W * ppm))

        topview_pts = np.array([
            [0,       0],
            [top_w-1, 0],
            [top_w-1, top_h-1],
            [0,       top_h-1]
        ], dtype=np.float32)

        self.H_top = cv2.getPerspectiveTransform(self.img_pts, topview_pts)

        self.calibrated = True
        self.last_detect_status = "Auto table detection successful"
        rospy.loginfo(self.last_detect_status)
        return True

    def reset_calibration(self):
        self.img_pts = None
        self.world_pts = None
        self.H = None
        self.H_top = None
        self.calibrated = False
        self.topview = None
        self.last_detect_status = "Calibration reset"
        self.delete_ball_marker()

    # ---------------------------------------------------
    # Mapping / publishing
    # ---------------------------------------------------
    def image_to_world(self, u, v):
        pt = np.array([[[u, v]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H)

        x = float(mapped[0][0][0]) + self.x_offset
        y = float(mapped[0][0][1]) + self.y_offset

        x = max(0.0, min(self.table_length, x))
        y = max(0.0, min(self.table_width, y))

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
        marker.pose.position.z = 0.0

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
    # Ball detection
    # ---------------------------------------------------
    def build_table_mask(self, shape_hw):
        h, w = shape_hw
        table_mask = np.zeros((h, w), dtype=np.uint8)

        if self.img_pts is not None:
            polygon = self.img_pts.astype(np.int32)
            cv2.fillConvexPoly(table_mask, polygon, 255)

        return table_mask

    def detect_ball(self, frame):
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

        mask = cv2.inRange(lab, self.lower, self.upper)

        if self.calibrated and self.img_pts is not None:
            table_mask = self.build_table_mask(mask.shape)
            mask = cv2.bitwise_and(mask, table_mask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        self.mask_view = mask.copy()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            (x2d, y2d), radius = cv2.minEnclosingCircle(cnt)
            u = int(x2d)
            v = int(y2d)
            radius = int(radius)

            if self.calibrated and self.img_pts is not None:
                inside = cv2.pointPolygonTest(
                    self.img_pts.astype(np.float32),
                    (float(u), float(v)),
                    False
                )
                if inside < 0:
                    continue

            if area > best_area:
                best_area = area
                best = (u, v, radius)

        return best

    # ---------------------------------------------------
    # Debug views
    # ---------------------------------------------------
    def make_topview_debug(self, frame, detection):
        if not self.calibrated or self.H_top is None:
            self.topview = None
            return

        ppm = int(self.topview_ppm)
        out_w = max(1, int(self.table_length * ppm))
        out_h = max(1, int(self.table_width * ppm))

        warped = cv2.warpPerspective(frame, self.H_top, (out_w, out_h))

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

    def draw_wait_ui(self, frame):
        display = frame.copy()

        cv2.putText(display, "Auto-detecting table...", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(display, self.last_detect_status, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, "r = retry detection | q = quit", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return display

    def draw_tracking_ui(self, frame, detection):
        display = frame.copy()

        if self.img_pts is not None:
            polygon = self.img_pts.astype(np.int32)
            cv2.polylines(display, [polygon], True, (255, 0, 0), 2)

            for i, pt in enumerate(self.img_pts):
                p = (int(pt[0]), int(pt[1]))
                cv2.circle(display, p, 5, (255, 0, 255), -1)
                cv2.putText(display, str(i + 1), (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.putText(display, "Tracking active | r = re-detect table | q = quit",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        if detection is not None:
            u, v, radius = detection
            x_map, y_map = self.image_to_world(u, v)

            cv2.circle(display, (u, v), radius, (0, 255, 0), 3)
            cv2.circle(display, (u, v), 5, (0, 0, 255), -1)

            cv2.putText(
                display,
                f"u={u}, v={v} | x={x_map:.3f} m, y={y_map:.3f} m",
                (max(10, u - 190), max(25, v - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            self.publish_ball_position(x_map, y_map)
            self.publish_ball_marker(x_map, y_map)
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

            if not self.calibrated:
                self.auto_calibrate(frame)
                display = self.draw_wait_ui(frame)
                self.topview = None
            else:
                detection = self.detect_ball(frame)
                display = self.draw_tracking_ui(frame, detection)
                self.make_topview_debug(frame, detection)

            cv2.imshow("Ball Tracker", display)

            if self.mask_view is not None:
                cv2.imshow("Mask", self.mask_view)

            if self.topview is not None:
                cv2.imshow("Top View", self.topview)

            if self.detect_debug is not None and not self.calibrated:
                cv2.imshow("Table Detect Edges", self.detect_debug)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                self.reset_calibration()
            elif key == ord('q'):
                rospy.signal_shutdown("User quit")

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        node = BallTrackerTopViewNode()
        node.run()
    except rospy.ROSInterruptException:
        pass