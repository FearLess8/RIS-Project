#!/usr/bin/env python3
import os
import json
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from cv_bridge import CvBridge


class DebugVisualizer:
    """Handles optional debug visualization windows."""
    
    def __init__(self, enabled=False):
        self.enabled = enabled
    
    def set_enabled(self, enabled):
        self.enabled = enabled
        if not enabled:
            self.close_all()
    
    def show_goal_mask(self, mask):
        """Display the goal detection mask."""
        if self.enabled and mask is not None:
            cv2.imshow("Goal Mask", mask)
    
    def show_detection(self, frame):
        """Display the detection visualization."""
        if self.enabled and frame is not None:
            cv2.imshow("Goal Detection", frame)
    
    def close_all(self):
        """Close all debug windows."""
        try:
            cv2.destroyWindow("Goal Mask")
            cv2.destroyWindow("Goal Detection")
        except:
            pass


class GoalTracker:
    def __init__(self):
        rospy.init_node("goal_tracker")

        self.bridge = CvBridge()

        # -----------------------------
        # Params
        # -----------------------------
        self.image_topic = rospy.get_param("~image_topic", "/cam/color/image_raw")
        self.world_frame = rospy.get_param("~world_frame", "table_top")
        self.debug = rospy.get_param("~debug", False)

        default_calib = os.path.expanduser("~/.ros/taped_square_calibration.json")
        self.calibration_file = rospy.get_param("~calibration_file", default_calib)

        # Square size from calibration (for coordinate conversion)
        self.square_size = 0.80  # Will be overridden by loaded calibration

        # -----------------------------
        # Red goalpost detection (HSV)
        # Red wraps around in HSV, so we need two ranges
        # -----------------------------
        self.red_lower1 = np.array([0, 80, 80], dtype=np.uint8)
        self.red_upper1 = np.array([15, 255, 255], dtype=np.uint8)
        self.red_lower2 = np.array([165, 80, 80], dtype=np.uint8)
        self.red_upper2 = np.array([180, 255, 255], dtype=np.uint8)

        self.min_goal_area = 200
        self.max_goal_area = 100000

        # Line marker parameters
        self.line_length = 0.20  # 20cm line

        # Tracking robustness parameters
        self.position_alpha = 0.3  # Exponential smoothing factor (0-1, lower = more smoothing)
        self.direction_alpha = 0.3
        self.loss_timeout = 0.5  # Seconds to remember last position if tracking is lost

        # -----------------------------
        # Runtime state
        # -----------------------------
        self.latest_frame = None
        self.goal_mask_view = None
        self.detection_view = None

        self.square_pts_img = None
        self.H = None
        self.calibrated = False
        self.last_status = "Initializing..."

        # Temporal tracking state
        self.last_detection_time = None
        self.last_valid_position = None
        self.last_valid_direction = None
        self.smoothed_position = None
        self.smoothed_direction = None

        # -----------------------------
        # ROS I/O
        # -----------------------------
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1
        )

        self.marker_pub = rospy.Publisher("/goal_marker", Marker, queue_size=1)

        # -----------------------------
        # Debug visualizer
        # -----------------------------
        self.debugger = DebugVisualizer(enabled=self.debug)

        # Try loading saved calibration
        self.load_calibration()

        rospy.loginfo("GoalTracker started")
        rospy.loginfo("Calibration file: %s", self.calibration_file)
        rospy.loginfo("Debug mode: %s", "enabled" if self.debug else "disabled")

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # ---------------------------------------------------
    # Calibration loading
    # ---------------------------------------------------
    def load_calibration(self):
        """Load the calibration from the saved file (created by ball_tracker)."""
        if not os.path.exists(self.calibration_file):
            self.last_status = "No calibration file found"
            rospy.logwarn("Calibration file not found: %s", self.calibration_file)
            return False

        try:
            with open(self.calibration_file, "r") as f:
                data = json.load(f)

            self.square_size = float(data.get("square_size", 0.80))
            origin_corner = data.get("origin_corner", "top_left")
            pts = np.array(data["points_tl_tr_br_bl"], dtype=np.float32)

            # Order the points
            self.square_pts_img = self.rotate_for_origin(pts, origin_corner)

            # Compute homography
            self.compute_homography()
            self.calibrated = True
            self.last_status = "Calibration loaded"
            rospy.loginfo("Loaded calibration from %s", self.calibration_file)
            return True

        except Exception as e:
            self.last_status = "Failed to load calibration"
            rospy.logwarn("Failed to load calibration: %s", e)
            return False

    def rotate_for_origin(self, pts, origin_corner):
        """Rotate corner points based on origin corner."""
        pts = np.array(pts, dtype=np.float32)
        tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]

        if origin_corner == "top_left":
            return np.array([tl, tr, br, bl], dtype=np.float32)
        elif origin_corner == "top_right":
            return np.array([tr, br, bl, tl], dtype=np.float32)
        elif origin_corner == "bottom_right":
            return np.array([br, bl, tl, tr], dtype=np.float32)
        elif origin_corner == "bottom_left":
            return np.array([bl, tl, tr, br], dtype=np.float32)
        else:
            return np.array([tl, tr, br, bl], dtype=np.float32)

    def compute_homography(self):
        """Compute the perspective transform matrix."""
        S = float(self.square_size)
        square_pts_world = np.array([
            [0.0, 0.0],
            [S,   0.0],
            [S,   S  ],
            [0.0, S  ]
        ], dtype=np.float32)

        self.H = cv2.getPerspectiveTransform(self.square_pts_img, square_pts_world)

    # ---------------------------------------------------
    # Goal detection
    # ---------------------------------------------------
    def detect_goal(self, frame):
        """
        Detect the red goalpost and return its center and direction.
        Returns: {'center': (x, y), 'direction': (dx, dy), 'contour': contour}
        or None if not detected.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red color detection (two ranges due to HSV wrapping)
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        goal_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        goal_mask = cv2.morphologyEx(goal_mask, cv2.MORPH_OPEN, kernel)
        goal_mask = cv2.morphologyEx(goal_mask, cv2.MORPH_CLOSE, kernel)

        self.goal_mask_view = goal_mask.copy()

        # Find contours
        contours, _ = cv2.findContours(goal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < self.min_goal_area or area > self.max_goal_area:
            return None

        # Get center
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Get orientation using fitLine (fits a line to the contour)
        if len(largest_contour) < 5:
            return None

        try:
            [vx, vy, x0, y0] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
            direction = np.array([vx, vy], dtype=np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-6)
        except:
            return None

        return {
            'center': (cx, cy),
            'direction': direction,
            'contour': largest_contour
        }

    # ---------------------------------------------------
    # Coordinate conversion
    # ---------------------------------------------------
    def image_to_world(self, u, v):
        """Convert image coordinates to world coordinates using the loaded calibration."""
        if self.H is None:
            return None

        pt = np.array([[[u, v]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H)

        # Coordinates in the calibration-square system
        x_local = float(mapped[0][0][0])
        y_local = float(mapped[0][0][1])

        # Convert to table_top frame (same as ball_tracker)
        x_table = 0.40 - y_local
        y_table = 0.80 - x_local

        # Clamp to table bounds
        x_table = max(-0.40, min(0.40, x_table))

        return x_table, y_table

    def image_direction_to_world(self, dir_img):
        """
        Convert image-space direction vector to world-space direction vector.
        This accounts for the perspective transform rotation.
        """
        if self.H is None:
            return None

        # Transform direction by the linear part of the homography
        H_linear = self.H[:2, :2]
        dir_world = H_linear @ dir_img
        dir_world = dir_world / (np.linalg.norm(dir_world) + 1e-6)

        # Swap and negate to match table_top frame conversion
        dir_converted = np.array([-dir_world[1], -dir_world[0]], dtype=np.float32)
        dir_converted = dir_converted / (np.linalg.norm(dir_converted) + 1e-6)

        return dir_converted

    # ---------------------------------------------------
    # Publishing
    # ---------------------------------------------------
    def publish_goal_marker(self, center_x, center_y, direction_x, direction_y):
        """Publish a line marker for the goal."""
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.world_frame

        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # Create line endpoints (half_length in each direction)
        half_length = self.line_length / 2.0

        p1 = Point()
        p1.x = center_x - direction_x * half_length
        p1.y = center_y - direction_y * half_length
        p1.z = 0.0

        p2 = Point()
        p2.x = center_x + direction_x * half_length
        p2.y = center_y + direction_y * half_length
        p2.z = 0.0

        marker.points = [p1, p2]

        # Line properties
        marker.scale.x = 0.02  # Line width

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration(0.2)
        self.marker_pub.publish(marker)

    def delete_goal_marker(self):
        """Delete the goal marker."""
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.world_frame
        marker.ns = "goal"
        marker.id = 0
        marker.action = Marker.DELETE
        self.marker_pub.publish(marker)

    # ---------------------------------------------------
    # Debug visualization
    # ---------------------------------------------------
    def draw_detection(self, frame, detection):
        """Draw detection visualization on frame."""
        display = frame.copy()

        if detection is not None:
            cx, cy = detection['center']
            direction = detection['direction']
            contour = detection['contour']

            # Draw contour
            cv2.drawContours(display, [contour], 0, (0, 255, 0), 2)

            # Draw center
            cv2.circle(display, (cx, cy), 8, (0, 0, 255), -1)

            # Draw direction line (in image space)
            line_length_img = 60
            x2 = int(cx + direction[0] * line_length_img)
            y2 = int(cy + direction[1] * line_length_img)
            cv2.arrowedLine(display, (cx, cy), (x2, y2), (255, 0, 0), 2, tipLength=0.2)

        cv2.putText(display, self.last_status, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return display

    # ---------------------------------------------------
    # Main loop
    # ---------------------------------------------------
    def run(self):
        rate = rospy.Rate(30)

        if not self.calibrated:
            rospy.logwarn("GoalTracker: Waiting for calibration...")

        while not rospy.is_shutdown():
            if self.latest_frame is None:
                cv2.waitKey(1)
                rate.sleep()
                continue

            frame = self.latest_frame.copy()

            if not self.calibrated:
                self.last_status = "Waiting for calibration..."
                cv2.imshow("Goal Tracker", frame)
            else:
                detection = self.detect_goal(frame)

                if detection is not None:
                    cx_img, cy_img = detection['center']
                    dir_img = detection['direction']

                    # Convert to world coordinates
                    world_pos = self.image_to_world(cx_img, cy_img)
                    world_dir = self.image_direction_to_world(dir_img)

                    if world_pos is not None and world_dir is not None:
                        cx_world, cy_world = world_pos
                        dx_world, dy_world = world_dir

                        # Apply exponential smoothing to reduce jitter
                        current_pos = np.array([cx_world, cy_world], dtype=np.float32)
                        current_dir = np.array([dx_world, dy_world], dtype=np.float32)

                        if self.smoothed_position is None:
                            self.smoothed_position = current_pos
                            self.smoothed_direction = current_dir
                        else:
                            self.smoothed_position = (
                                self.position_alpha * current_pos +
                                (1 - self.position_alpha) * self.smoothed_position
                            )
                            self.smoothed_direction = (
                                self.direction_alpha * current_dir +
                                (1 - self.direction_alpha) * self.smoothed_direction
                            )

                        # Normalize direction
                        self.smoothed_direction = self.smoothed_direction / (
                            np.linalg.norm(self.smoothed_direction) + 1e-6
                        )

                        # Update last valid detection
                        self.last_detection_time = rospy.Time.now()
                        self.last_valid_position = self.smoothed_position.copy()
                        self.last_valid_direction = self.smoothed_direction.copy()

                        self.last_status = f"Goal detected: ({100*self.smoothed_position[0]:.1f}cm, {100*self.smoothed_position[1]:.1f}cm)"
                        self.publish_goal_marker(
                            self.smoothed_position[0],
                            self.smoothed_position[1],
                            self.smoothed_direction[0],
                            self.smoothed_direction[1]
                        )
                    else:
                        self.last_status = "Coordinate conversion failed"
                        self.delete_goal_marker()
                else:
                    # Detection failed - check if we can use the last known position
                    if self.last_detection_time is not None:
                        time_since_detection = (rospy.Time.now() - self.last_detection_time).to_sec()

                        if time_since_detection < self.loss_timeout:
                            # Still within timeout, use last known position
                            self.last_status = f"Tracking (occluded {time_since_detection:.2f}s) Goal at ({100*self.last_valid_position[0]:.1f}cm, {100*self.last_valid_position[1]:.1f}cm)"
                            self.publish_goal_marker(
                                self.last_valid_position[0],
                                self.last_valid_position[1],
                                self.last_valid_direction[0],
                                self.last_valid_direction[1]
                            )
                        else:
                            # Timeout exceeded
                            self.last_status = "Lost track of goal (timeout)"
                            self.delete_goal_marker()
                    else:
                        self.last_status = "Goal not detected"
                        self.delete_goal_marker()

                # Debug visualization
                self.detection_view = self.draw_detection(frame, detection)
                cv2.imshow("Goal Tracker", self.detection_view)

            self.debugger.show_goal_mask(self.goal_mask_view)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('d'):
                self.debug = not self.debug
                self.debugger.set_enabled(self.debug)
                status = "enabled" if self.debug else "disabled"
                rospy.loginfo("Debug mode %s", status)
            elif key == ord('q'):
                rospy.signal_shutdown("Quit")

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        node = GoalTracker()
        node.run()
    except rospy.ROSInterruptException:
        pass