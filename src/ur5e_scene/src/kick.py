#!/usr/bin/env python3

import rospy
import moveit_commander
from geometry_msgs.msg import Pose

class SimpleKick:
    def __init__(self):
        rospy.init_node("simple_kick_node")

        moveit_commander.roscpp_initialize([])
        self.group = moveit_commander.MoveGroupCommander("manipulator")

        # Optional: faster movement
        self.group.set_max_velocity_scaling_factor(0.5)
        self.group.set_max_acceleration_scaling_factor(0.5)

    def go_to_pose(self, pose):
        self.group.set_pose_target(pose)
        self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

    def perform_kick(self):
        rospy.loginfo("Starting kick motion...")

        # --- Pre-kick pose ---
        pre_kick = Pose()
        pre_kick.position.x = 0.4
        pre_kick.position.y = 0.0
        pre_kick.position.z = 0.2

        pre_kick.orientation.x = 0
        pre_kick.orientation.y = 1
        pre_kick.orientation.z = 0
        pre_kick.orientation.w = 0

        # --- Kick pose (forward push) ---
        kick_pose = Pose()
        kick_pose.position.x = 0.6   # move forward
        kick_pose.position.y = 0.0
        kick_pose.position.z = 0.2

        kick_pose.orientation = pre_kick.orientation

        # --- Retreat pose ---
        retreat_pose = Pose()
        retreat_pose.position.x = 0.3  # move back
        retreat_pose.position.y = 0.0
        retreat_pose.position.z = 0.2

        retreat_pose.orientation = pre_kick.orientation

        # Execute sequence
        self.go_to_pose(pre_kick)
        rospy.sleep(1)

        self.go_to_pose(kick_pose)
        rospy.sleep(0.5)

        self.go_to_pose(retreat_pose)
        rospy.sleep(1)

        rospy.loginfo("Kick complete!")

if __name__ == "__main__":
    sk = SimpleKick()
    rospy.sleep(2)
    sk.perform_kick()