#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

class PreciseTableTracker:
    def __init__(self):
        rospy.init_node('table_tracker', anonymous=True)
        self.bridge = CvBridge()
        self.pub = rospy.Publisher('/ball_table_meters', Point, queue_size=1)
        
        # Dimensions for the 180cm x 80cm surface
        self.WIDTH_M = 1.8
        self.HEIGHT_M = 0.8
        
        # Calibration State
        self.clicked_pts = []
        self.is_calibrated = False
        self.M = None
        
        self.instructions = [
            "1. Click UPPER LEFT (Origin 0,0)",
            "2. Click UPPER RIGHT (Width 1.8,0)",
            "3. Click LOWER LEFT (Height 0,0.8)",
            "4. Click LOWER RIGHT (Finish 1.8,0.8)"
        ]

        # Ball Color Ranges (#782128 Burgundy)
        self.lower_red1 = np.array([0, 150, 40])
        self.upper_red1 = np.array([10, 255, 200])
        self.lower_red2 = np.array([170, 150, 40])
        self.upper_red2 = np.array([180, 255, 200])

        # Start subscriber
        self.sub = rospy.Subscriber('/cam/image_raw', Image, self.callback)
        rospy.loginfo("Node started. Waiting for camera frames...")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.clicked_pts) < 4:
                self.clicked_pts.append([x, y])
                print(f"Captured: {x}, {y}")

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr(e)
            return

        if not self.is_calibrated:
            # --- CALIBRATION MODE ---
            cv2.namedWindow("Calibration")
            cv2.setMouseCallback("Calibration", self.mouse_callback)
            
            display_img = frame.copy()
            curr = len(self.clicked_pts)
            
            if curr < 4:
                # UX: On-screen instructions
                cv2.putText(display_img, self.instructions[curr], (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_img, "Press 'r' to reset", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                # Perform the Homography math
                src_pts = np.float32(self.clicked_pts)
                dst_pts = np.float32([[0, 0], [1800, 0], [0, 800], [1800, 800]])
                self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                self.is_calibrated = True
                cv2.destroyWindow("Calibration")
                rospy.loginfo("Calibration Complete!")
                return

            # Draw points as they are clicked
            for i, pt in enumerate(self.clicked_pts):
                cv2.circle(display_img, (pt[0], pt[1]), 5, (0, 255, 0), -1)
                cv2.putText(display_img, str(i+1), (pt[0]+10, pt[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Calibration", display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'): self.clicked_pts = []

        else:
            # --- TRACKING MODE ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                
                if radius > 8:
                    # Map pixel to meter space using Homography
                    px_point = np.array([[[x, y]]], dtype=np.float32)
                    transformed = cv2.perspectiveTransform(px_point, self.M)[0][0]
                    
                    # Convert 1800px scale to 1.8m scale
                    m_x = transformed[0] / 1000.0
                    m_y = transformed[1] / 1000.0
                    
                    # Publish for the UR5e
                    self.pub.publish(Point(m_x, m_y, radius))

                    # UX: Visual tracking
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    cv2.putText(frame, f"X:{m_x:.2f}m Y:{m_y:.2f}m", (int(x), int(y)-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Ball Tracker (X=Width, Y=Height)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown("User quit")

if __name__ == '__main__':
    try:
        PreciseTableTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()