#!/usr/bin/env python3
"""
HSV Color Range Tuner
Adjust sliders to find the perfect color range for your ball detection.
"""
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class HSVTuner:
    def __init__(self):
        rospy.init_node("hsv_tuner")
        self.bridge = CvBridge()
        
        self.image_topic = rospy.get_param("~image_topic", "/cam/color/image_raw")
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1
        )
        
        self.latest_frame = None
        self.setup_windows()
        
    def setup_windows(self):
        cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        
        # Create trackbars for HSV range
        cv2.createTrackbar("H_min", "HSV Tuner", 35, 180, self.on_trackbar)
        cv2.createTrackbar("H_max", "HSV Tuner", 95, 180, self.on_trackbar)
        cv2.createTrackbar("S_min", "HSV Tuner", 60, 255, self.on_trackbar)
        cv2.createTrackbar("S_max", "HSV Tuner", 255, 255, self.on_trackbar)
        cv2.createTrackbar("V_min", "HSV Tuner", 40, 255, self.on_trackbar)
        cv2.createTrackbar("V_max", "HSV Tuner", 255, 255, self.on_trackbar)
        
        rospy.loginfo("Adjust sliders to find your color range. Press 'q' to quit.")
        
    def on_trackbar(self, x):
        pass
    
    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    
    def run(self):
        rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            if self.latest_frame is None:
                cv2.waitKey(1)
                rate.sleep()
                continue
            
            frame = self.latest_frame.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Get current trackbar values
            h_min = cv2.getTrackbarPos("H_min", "HSV Tuner")
            h_max = cv2.getTrackbarPos("H_max", "HSV Tuner")
            s_min = cv2.getTrackbarPos("S_min", "HSV Tuner")
            s_max = cv2.getTrackbarPos("S_max", "HSV Tuner")
            v_min = cv2.getTrackbarPos("V_min", "HSV Tuner")
            v_max = cv2.getTrackbarPos("V_max", "HSV Tuner")
            
            # Create mask
            lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Morphology
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Display info
            display = frame.copy()
            info = f"H: {h_min}-{h_max}  S: {s_min}-{s_max}  V: {v_min}-{v_max}"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Find contours and show largest
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                cv2.drawContours(display, [largest], 0, (0, 255, 0), 2)
                cv2.putText(display, f"Area: {area}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("HSV Tuner", display)
            cv2.imshow("Mask", mask)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\nFinal values:")
                print(f"self.green_lower = np.array([{h_min}, {s_min}, {v_min}], dtype=np.uint8)")
                print(f"self.green_upper = np.array([{h_max}, {s_max}, {v_max}], dtype=np.uint8)")
                rospy.signal_shutdown("Quit")
            
            rate.sleep()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        node = HSVTuner()
        node.run()
    except rospy.ROSInterruptException:
        pass