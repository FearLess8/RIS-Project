#!/usr/bin/env python3
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class BallDetectorNode:
    def __init__(self):
        rospy.init_node("ball_detector_node")

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(
            "/cam/color/image_raw",
            Image,
            self.image_callback
        )

        # Calibrated from your image
        self.lower = np.array([85, 82, 132], dtype=np.uint8)
        self.upper = np.array([130, 100, 150], dtype=np.uint8)

        self.min_area = 300
        self.max_area = 10000

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

        # Threshold only the ball color
        mask = cv2.inRange(lab, self.lower, self.upper)

        # Cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = False

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < self.min_area or area > self.max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.7:
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)

            cv2.circle(frame, center, radius, (0, 255, 0), 3)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                "BALL DETECTED",
                (center[0] - 50, center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            print("BALL DETECTED at:", center)
            detected = True
            break

        if not detected:
            print("No ball detected")

        cv2.imshow("Ball Detection", frame)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)


if __name__ == "__main__":
    try:
        BallDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass