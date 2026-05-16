#!/usr/bin/env python3
import sys
import math
import rospy
import moveit_commander
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import TwistStamped, PointStamped
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import JointState
from ur5e_scene.msg import BallArrival

# --- Robot config ---
READY_JOINTS = {
    "arm_elbow_joint":         -2.166,
    "arm_shoulder_lift_joint": -2.140,
    "arm_shoulder_pan_joint":  -3.145,
    "arm_wrist_1_joint":       -0.403,
    "arm_wrist_2_joint":        1.567,
    "arm_wrist_3_joint":        0.0,
}
JOINT_TOL = 0.05  # rad

# --- Motion tuning ---
SPEED_MAX  = 0.8  # m/s
ACCEL      = 0.8   # m/s²  — 0→max in ~0.56s
DECEL_DIST = 0.05  # m — only brake in the last 5cm so robot doesn't slow early
POS_TOL    = 0.02  # m — "close enough" to target X

# --- Catch detection ---
CATCH_RADIUS = 0.08  # m — max |tool_x - ball_x| to count as caught
CATCH_LINE_Y = 0.55  # m — ball must be at least this far in Y to trigger catch check

DT = 1.0 / 20

# --- States ---
WAITING   = "WAITING"
CATCHING  = "CATCHING"
CAUGHT    = "CAUGHT"
RETURNING = "RETURNING"


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


def switch_controller(start, stop):
    rospy.wait_for_service("/controller_manager/switch_controller")
    svc = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
    svc(start_controllers=start, stop_controllers=stop, strictness=2)


def go_to_ready():
    rospy.loginfo("[MOVER] Switching to trajectory controller...")
    switch_controller(["scaled_pos_joint_traj_controller"], ["twist_controller"])
    arm = moveit_commander.MoveGroupCommander("arm")
    arm.set_joint_value_target(READY_JOINTS)
    rospy.loginfo("[MOVER] Moving to ready pose...")
    success = arm.go(wait=True)
    arm.stop()
    if not success:
        rospy.logerr("[MOVER] Failed to reach ready pose!")
        sys.exit(1)
    rospy.loginfo("[MOVER] Switching back to twist controller...")
    switch_controller(["twist_controller"], ["scaled_pos_joint_traj_controller"])


# --- Node init ---
rospy.init_node("table_mover")
moveit_commander.roscpp_initialize(sys.argv)

# --- Shared data from callbacks ---
latest_arrival = None
latest_ball_pos = None

def arrival_cb(msg):
    global latest_arrival
    if msg.valid:
        latest_arrival = msg

def ball_cb(msg):
    global latest_ball_pos
    latest_ball_pos = msg

rospy.Subscriber("/ball_arrival", BallArrival, arrival_cb)
rospy.Subscriber("/ball_position", PointStamped, ball_cb)

# --- Go to ready on startup ---
if not is_at_ready():
    go_to_ready()
else:
    rospy.loginfo("[MOVER] Already at ready position.")

tf_buffer = Buffer()
TransformListener(tf_buffer)
pub = rospy.Publisher("twist_controller/zoned_command", TwistStamped, queue_size=1)

rospy.sleep(1.0)
rospy.loginfo("[MOVER] === Ready to catch the ball! Roll the ball when ready. ===")

# --- Main loop ---
rate = rospy.Rate(20)
state = WAITING
target_x = None
current_speed = 0.0
prev_direction = 0.0
tool_x = 0.0

while not rospy.is_shutdown():

    # Update tool position
    try:
        tf = tf_buffer.lookup_transform("table_top", "arm_tool0", rospy.Time(0))
        tool_x = tf.transform.translation.x
    except Exception:
        pass

    twist_x = 0.0

    # ── WAITING ──────────────────────────────────────────────────────────────
    if state == WAITING:
        if latest_arrival is not None:
            target_x = latest_arrival.arrival_position.x
            t = latest_arrival.time_to_arrival
            rospy.loginfo(f"[MOVER] Ball incoming! Predicted X={target_x:.3f}m in {t:.2f}s — intercepting...")
            latest_arrival = None
            current_speed = 0.0
            prev_direction = 0.0
            state = CATCHING

    # ── CATCHING ─────────────────────────────────────────────────────────────
    elif state == CATCHING:
        # Update target from fresh predictions while still moving
        if latest_arrival is not None:
            new_x = latest_arrival.arrival_position.x
            if abs(new_x - target_x) > 0.01:
                rospy.loginfo(f"[MOVER] Prediction updated: X={new_x:.3f}m")
                target_x = new_x
            latest_arrival = None

        error = target_x - tool_x
        dist  = abs(error)
        direction = math.copysign(1.0, error) if dist > POS_TOL else 0.0

        # Reset speed on direction reversal
        if direction != 0.0 and prev_direction != 0.0 and direction != prev_direction:
            current_speed = 0.0
        prev_direction = direction

        if direction == 0.0:
            # Arrived — hold position
            current_speed = 0.0
            twist_x = 0.0
            rospy.loginfo_throttle(1.0, f"[MOVER] Holding at x={tool_x:.3f}m, waiting for ball...")
        else:
            target_speed = SPEED_MAX * min(1.0, dist / DECEL_DIST)
            if current_speed < target_speed:
                current_speed = min(target_speed, current_speed + ACCEL * DT)
            else:
                current_speed = max(target_speed, current_speed - ACCEL * DT)
            twist_x = current_speed * direction

        # Collision check
        if latest_ball_pos is not None:
            bx = latest_ball_pos.point.x
            by = latest_ball_pos.point.y
            if by >= CATCH_LINE_Y and abs(tool_x - bx) < CATCH_RADIUS:
                rospy.loginfo(f"[MOVER] Ball caught! tool_x={tool_x:.3f}m  ball_x={bx:.3f}m  ball_y={by:.3f}m")
                state = CAUGHT
                twist_x = 0.0
                current_speed = 0.0

    # ── CAUGHT ───────────────────────────────────────────────────────────────
    elif state == CAUGHT:
        twist_x = 0.0
        # Publish a zero twist before sleeping so the robot stops cleanly
        stop = TwistStamped()
        stop.header.stamp = rospy.Time.now()
        stop.header.frame_id = "table_top"
        pub.publish(stop)

        rospy.loginfo("[MOVER] Holding for 2 seconds...")
        rospy.sleep(2.0)
        rospy.loginfo("[MOVER] Returning to ready position...")
        latest_arrival = None
        latest_ball_pos = None
        state = RETURNING

    # ── RETURNING ────────────────────────────────────────────────────────────
    elif state == RETURNING:
        go_to_ready()
        current_speed = 0.0
        prev_direction = 0.0
        state = WAITING
        latest_arrival = None
        rospy.loginfo("[MOVER] === Ready to catch the ball! Roll the ball when ready. ===")

    # Publish twist
    msg = TwistStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "table_top"
    msg.twist.linear.x = twist_x
    pub.publish(msg)

    rate.sleep()
