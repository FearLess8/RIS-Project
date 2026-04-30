#!/usr/bin/env python3

import sys
import rospy
import moveit_commander

from std_msgs.msg import String


class MoveArmOnCommand:
    def __init__(self):
        rospy.init_node("move_arm_on_command", anonymous=True)

        moveit_commander.roscpp_initialize(sys.argv)

        # Change this if your MoveIt group has another name, e.g. "manipulator"
        self.group_name = rospy.get_param("~move_group", "arm")

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(self.group_name)

        # -----------------------------
        # MoveIt planning settings
        # -----------------------------
        self.group.set_planning_time(10.0)
        self.group.set_num_planning_attempts(10)
        self.group.allow_replanning(True)

        # Optional but usually good
        self.group.set_max_velocity_scaling_factor(0.25)
        self.group.set_max_acceleration_scaling_factor(0.25)

        # Joint target from your screenshot, in radians
        self.target_joint_values = [
            -0.196,   # arm_shoulder_pan_joint
            -0.697,   # arm_shoulder_lift_joint
             1.356,   # arm_elbow_joint
             2.832,   # arm_wrist_1_joint
            -1.674,   # arm_wrist_2_joint
             2.675    # arm_wrist_3_joint
        ]

        # Optional: add a simple table/floor collision object
        self.add_basic_collision_objects()

        rospy.Subscriber("/arm_command", String, self.command_callback)

        rospy.loginfo("Ready. Send '1' on /arm_command to move the robot.")

    def add_basic_collision_objects(self):
        """
        MoveIt avoids collisions only with objects it knows about.
        This adds a simple floor/table collision box.
        Adjust position and size for your real setup.
        """

        rospy.sleep(1.0)

        frame = self.robot.get_planning_frame()

        from geometry_msgs.msg import PoseStamped

        table_pose = PoseStamped()
        table_pose.header.frame_id = frame

        # Adjust these values for your table/workspace
        table_pose.pose.position.x = 0.0
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = -0.05

        table_pose.pose.orientation.w = 1.0

        # size = x, y, z in meters
        self.scene.add_box(
            name="table",
            pose=table_pose,
            size=(1.5, 1.5, 0.10)
        )

        rospy.loginfo("Added table collision object to planning scene.")

    def command_callback(self, msg):
        command = msg.data.strip()

        if command == "1":
            rospy.loginfo("Received command 1. Planning motion...")

            success = self.plan_and_execute(self.target_joint_values)

            if success:
                rospy.loginfo("Motion completed successfully.")
            else:
                rospy.logwarn("Motion failed or no valid collision-free plan found.")

    def plan_and_execute(self, joint_goal):
        self.group.stop()
        self.group.clear_pose_targets()
        self.group.set_start_state_to_current_state()

        current_joints = self.group.get_current_joint_values()

        rospy.loginfo("Current joints:")
        rospy.loginfo(current_joints)

        rospy.loginfo("Target joints:")
        rospy.loginfo(joint_goal)

        self.group.set_joint_value_target(joint_goal)

        plan_result = self.group.plan()

        # MoveIt Python API differs slightly between versions
        if isinstance(plan_result, tuple):
            success = plan_result[0]
            plan = plan_result[1]
        else:
            plan = plan_result
            success = len(plan.joint_trajectory.points) > 0

        if not success:
            rospy.logwarn("Planning failed. Robot will not move.")
            return False

        rospy.loginfo("Plan found. Executing...")

        execute_success = self.group.execute(plan, wait=True)

        self.group.stop()
        self.group.clear_pose_targets()

        return execute_success


if __name__ == "__main__":
    try:
        node = MoveArmOnCommand()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

    finally:
        moveit_commander.roscpp_shutdown()