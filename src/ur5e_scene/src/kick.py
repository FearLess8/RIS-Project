#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState

READY_JOINTS = {
    "arm_elbow_joint":         -2.166,
    "arm_shoulder_lift_joint": -2.140,
    "arm_shoulder_pan_joint":  -3.145,
    "arm_wrist_1_joint":       -0.403,
    "arm_wrist_2_joint":        1.567,
    "arm_wrist_3_joint":        0.0,
}
JOINT_TOL = 0.05

# ── Tune these ────────────────────────────────────────────────────────────────
BACK_SPEED     = 0.2   # m/s  +Y (windup)
BACK_DURATION  = 0.3   # s

FORWARD_SPEED  = 0.5   # m/s  -Y (kick)
FORWARD_DURATION = 0.3 # s
# ─────────────────────────────────────────────────────────────────────────────

RATE_HZ = 50
DT = 1.0 / RATE_HZ


def switch_controller(start, stop):
    rospy.wait_for_service("/controller_manager/switch_controller")
    svc = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
    svc(start_controllers=start, stop_controllers=stop, strictness=2)


def is_at_ready():
    try:
        js = rospy.wait_for_message("/joint_states", JointState, timeout=3.0)
        for name, target in READY_JOINTS.items():
            if name in js.name:
                if abs(js.position[js.name.index(name)] - target) > JOINT_TOL:
                    return False
        return True
    except Exception:
        return False


def go_to_ready(arm):
    switch_controller(["scaled_pos_joint_traj_controller"], ["twist_controller"])
    arm.set_joint_value_target(READY_JOINTS)
    arm.set_max_velocity_scaling_factor(0.7)
    arm.set_max_acceleration_scaling_factor(0.7)
    success = arm.go(wait=True)
    arm.stop()
    if not success:
        rospy.logerr("[KICK] Failed to reach ready position!")
        sys.exit(1)
    switch_controller(["twist_controller"], ["scaled_pos_joint_traj_controller"])


def twist_y(pub, vy):
    msg = TwistStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "table_top"
    msg.twist.linear.y = vy
    pub.publish(msg)


# ── Init ──────────────────────────────────────────────────────────────────────
rospy.init_node("kick")
moveit_commander.roscpp_initialize(sys.argv)

arm = moveit_commander.MoveGroupCommander("arm")
pub = rospy.Publisher("twist_controller/zoned_command", TwistStamped, queue_size=1)
rate = rospy.Rate(RATE_HZ)

if not is_at_ready():
    rospy.loginfo("[KICK] Moving to ready position...")
    go_to_ready(arm)
else:
    rospy.loginfo("[KICK] Already at ready position.")

rospy.sleep(0.5)

# ── Main loop ─────────────────────────────────────────────────────────────────
while not rospy.is_shutdown():
    print("\n[KICK] Place the ball then press ENTER to kick (Ctrl+C to quit)...")
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        break

    # Back (+Y)
    rospy.loginfo("[KICK] Back...")
    elapsed = 0.0
    while elapsed < BACK_DURATION and not rospy.is_shutdown():
        twist_y(pub, BACK_SPEED)
        rate.sleep()
        elapsed += DT

    # Forward (-Y)
    rospy.loginfo("[KICK] Forward!")
    elapsed = 0.0
    while elapsed < FORWARD_DURATION and not rospy.is_shutdown():
        twist_y(pub, -FORWARD_SPEED)
        rate.sleep()
        elapsed += DT

    twist_y(pub, 0.0)
    rospy.loginfo("[KICK] Done. Returning to ready...")
    go_to_ready(arm)
    rospy.loginfo("[KICK] Ready.")
