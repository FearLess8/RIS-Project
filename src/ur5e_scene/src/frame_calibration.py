#!/usr/bin/env python3
"""
calibrate_homography.py - ROS camera version
"""
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

WORLD_POINTS = np.array([
    [0.40, -0.20],
    [0.40,  0.20],
    [0.70,  0.20],
    [0.70, -0.20],
], dtype=np.float32)

pixel_points = []
frame_display = None
latest_frame = None
bridge = CvBridge()

def image_callback(msg):
    global latest_frame
    latest_frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def mouse_callback(event, x, y, flags, param):
    global frame_display
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_points.append([x, y])
        cv2.circle(frame_display, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(frame_display, str(len(pixel_points)),
                    (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"  Point {len(pixel_points)}: pixel ({x}, {y})")
        cv2.imshow("Calibration", frame_display)

def main():
    global frame_display

    rospy.init_node("homography_calibration")
    rospy.Subscriber("/cam/color/image_raw", Image, image_callback)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    n_points = len(WORLD_POINTS)
    print(f"Click {n_points} points in this order:")
    for i, (wx, wy) in enumerate(WORLD_POINTS):
        print(f"  {i+1}: robot frame ({wx:.3f}, {wy:.3f})")
    print("Press 'q' to quit without saving.")

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        if latest_frame is None:
            rate.sleep()
            continue

        frame_display = latest_frame.copy()

        # Redraw clicked points
        for i, (px, py) in enumerate(pixel_points):
            cv2.circle(frame_display, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(frame_display, str(i + 1),
                        (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(pixel_points) < n_points:
            cv2.putText(frame_display,
                        f"Click point {len(pixel_points)+1}/{n_points}: "
                        f"robot({WORLD_POINTS[len(pixel_points)][0]:.2f}, "
                        f"{WORLD_POINTS[len(pixel_points)][1]:.2f})",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame_display, "All points collected — press Enter to compute",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Calibration", frame_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quit without saving.")
            break
        if key == 13 and len(pixel_points) >= 4:  # Enter
            break

        rate.sleep()

    cv2.destroyAllWindows()

    if len(pixel_points) < 4:
        print("Not enough points collected.")
        return

    src = np.array(pixel_points[:n_points], dtype=np.float32)
    dst = WORLD_POINTS[:n_points]
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC)

    print("\n--- Paste this into ball_tracker.py ---")
    print("self.H = np.array([")
    for row in H:
        print(f"    [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}],")
    print("], dtype=np.float32)")

    print("\nReprojection check:")
    for i, (px, py) in enumerate(pixel_points[:n_points]):
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)
        mx, my = mapped[0][0]
        wx, wy = WORLD_POINTS[i]
        print(f"  pixel({px},{py}) → ({mx:.4f},{my:.4f})  "
              f"target({wx:.4f},{wy:.4f})  "
              f"err={np.hypot(mx-wx, my-wy)*1000:.1f}mm")

if __name__ == "__main__":
    main()