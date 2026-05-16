#!/usr/bin/env python3
import sys
import math
import subprocess
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

# --- Catching motion tuning ---
SPEED_MAX  = 0.8   # m/s
ACCEL      = 0.8   # m/s²
DECEL_DIST = 0.05  # m
POS_TOL    = 0.02  # m

# --- Catch detection ---
CATCH_RADIUS = 0.08  # m
CATCH_LINE_Y = 0.55  # m
MISS_LINE_Y  = 0.70  # m — ball past this without catch/kick = miss

# --- Kick tuning ---
KICK_TRIGGER_T   = 0.5    # s before ball arrival to fire the kick

BACK_SPEED       = 0.3    # m/s  pull-back speed (+Y)
BACK_ACCEL_T     = 0.15   # s
BACK_HOLD_T      = 0.1    # s
BACK_DECEL_T     = 0.15   # s

FORWARD_SPEED    = 0.7    # m/s  kick speed (-Y)
FORWARD_ACCEL_T  = 0.05   # s
FORWARD_HOLD_T   = 0.15   # s
FORWARD_DECEL_T  = 0.17   # s

KICK_RATE_HZ = 50
KICK_DT      = 1.0 / KICK_RATE_HZ

KP_Z = 3.0   # proportional gain for Z-hold during kick (m/s per m of error)

DT = 1.0 / 20

# --- States ---
WAITING   = "WAITING"
CATCHING  = "CATCHING"
CAUGHT    = "CAUGHT"
RETURNING = "RETURNING"


def say(text):
    """Non-blocking text-to-speech (fails silently if espeak is missing)."""
    try:
        subprocess.Popen(
            ["espeak", "-s", "140", "-a", "200", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


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

latest_arrival  = None
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

if not is_at_ready():
    go_to_ready()
else:
    rospy.loginfo("[MOVER] Already at ready position.")

tf_buffer = Buffer()
TransformListener(tf_buffer)
pub = rospy.Publisher("twist_controller/zoned_command", TwistStamped, queue_size=1)

rospy.sleep(1.0)
rospy.loginfo("[MOVER] === Ready! Roll the ball when ready. ===")
say("Ready. Roll the ball.")


def _publish_kick(vy, z_target):
    """Publish Y velocity with proportional Z correction to hold height."""
    vz = 0.0
    try:
        tf = tf_buffer.lookup_transform("table_top", "arm_tool0", rospy.Time(0))
        z_now = tf.transform.translation.z
        vz = KP_Z * (z_target - z_now)
        vz = max(-0.10, min(0.10, vz))   # clamp correction to ±100 mm/s
    except Exception:
        pass
    msg = TwistStamped()
    msg.header.stamp    = rospy.Time.now()
    msg.header.frame_id = "table_top"
    msg.twist.linear.y  = vy
    msg.twist.linear.z  = vz
    pub.publish(msg)


def ramp_y(v_start, v_end, duration, z_target):
    steps     = max(1, int(duration * KICK_RATE_HZ))
    increment = (v_end - v_start) / steps
    v         = v_start
    for _ in range(steps):
        if rospy.is_shutdown():
            break
        _publish_kick(v, z_target)
        v += increment
        rospy.sleep(KICK_DT)


def hold_y(v, duration, z_target):
    steps = max(1, int(duration * KICK_RATE_HZ))
    for _ in range(steps):
        if rospy.is_shutdown():
            break
        _publish_kick(v, z_target)
        rospy.sleep(KICK_DT)


def execute_kick():
    # Capture Z height at kick start — maintain it throughout the kick
    z_target = 0.1
    try:
        tf = tf_buffer.lookup_transform("table_top", "arm_tool0", rospy.Time(0))
        z_target = tf.transform.translation.z
        rospy.loginfo(f"[MOVER] Kick Z target: {z_target:.3f}m")
    except Exception:
        rospy.logwarn("[MOVER] Could not read tool Z — using fallback 0.1 m")

    rospy.loginfo("[MOVER] Kick: pulling back...")
    ramp_y(0.0,          +BACK_SPEED,    BACK_ACCEL_T,    z_target)
    hold_y(+BACK_SPEED,                  BACK_HOLD_T,     z_target)
    ramp_y(+BACK_SPEED,   0.0,           BACK_DECEL_T,    z_target)

    rospy.loginfo("[MOVER] Kick: striking forward!")
    ramp_y(0.0,           -FORWARD_SPEED, FORWARD_ACCEL_T, z_target)
    hold_y(-FORWARD_SPEED,                FORWARD_HOLD_T,  z_target)
    ramp_y(-FORWARD_SPEED, 0.0,           FORWARD_DECEL_T, z_target)

    _publish_kick(0.0, z_target)


rate          = rospy.Rate(20)
state         = WAITING
target_x      = None
current_speed = 0.0
prev_direction = 0.0
tool_x        = 0.0
kick_ref_msg  = None   # last BallArrival message, kept for time-remaining calculation

while not rospy.is_shutdown():

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
            kick_ref_msg   = latest_arrival
            latest_arrival = None
            current_speed  = 0.0
            prev_direction = 0.0
            state = CATCHING

    # ── CATCHING ─────────────────────────────────────────────────────────────
    elif state == CATCHING:
        if latest_arrival is not None:
            kick_ref_msg = latest_arrival          # update timing reference
            new_x = latest_arrival.arrival_position.x
            if abs(new_x - target_x) > 0.01:
                rospy.loginfo(f"[MOVER] Prediction updated: X={new_x:.3f}m")
                target_x = new_x
            latest_arrival = None

        # Check if it's time to kick
        if kick_ref_msg is not None:
            elapsed        = (rospy.Time.now() - kick_ref_msg.header.stamp).to_sec()
            time_remaining = kick_ref_msg.time_to_arrival - elapsed
            if time_remaining <= KICK_TRIGGER_T:
                rospy.loginfo(f"[MOVER] Kick trigger! {time_remaining:.3f}s to arrival — firing kick!")
                say("Kicking")
                _publish_kick(0.0, 0.0)   # stop x motion cleanly
                execute_kick()
                kick_ref_msg    = None
                latest_arrival  = None
                latest_ball_pos = None
                state = RETURNING
                continue          # skip bottom publish; execute_kick handled it

        # X-axis positioning
        error     = target_x - tool_x
        dist      = abs(error)
        direction = math.copysign(1.0, error) if dist > POS_TOL else 0.0

        if direction != 0.0 and prev_direction != 0.0 and direction != prev_direction:
            current_speed = 0.0
        prev_direction = direction

        if direction == 0.0:
            current_speed = 0.0
            rospy.loginfo_throttle(1.0, f"[MOVER] Holding at x={tool_x:.3f}m, waiting for ball...")
        else:
            target_speed  = SPEED_MAX * min(1.0, dist / DECEL_DIST)
            if current_speed < target_speed:
                current_speed = min(target_speed, current_speed + ACCEL * DT)
            else:
                current_speed = max(target_speed, current_speed - ACCEL * DT)
            twist_x = current_speed * direction

        # Fallback: passive catch if kick timing was missed
        if latest_ball_pos is not None:
            bx = latest_ball_pos.point.x
            by = latest_ball_pos.point.y
            if by >= CATCH_LINE_Y and abs(tool_x - bx) < CATCH_RADIUS:
                rospy.loginfo(f"[MOVER] Ball caught (fallback)! tool_x={tool_x:.3f}m  ball_x={bx:.3f}m")
                state         = CAUGHT
                twist_x       = 0.0
                current_speed = 0.0
            elif by >= MISS_LINE_Y:
                rospy.loginfo(f"[MOVER] Ball missed! ball_y={by:.3f}m — resetting.")
                say("Uh-oh, missed.")
                kick_ref_msg    = None
                latest_arrival  = None
                latest_ball_pos = None
                twist_x         = 0.0
                current_speed   = 0.0
                state           = RETURNING

    # ── CAUGHT (fallback — kick timing was missed) ────────────────────────────
    elif state == CAUGHT:
        stop = TwistStamped()
        stop.header.stamp    = rospy.Time.now()
        stop.header.frame_id = "table_top"
        pub.publish(stop)
        say("Caught. Resetting.")
        rospy.loginfo("[MOVER] Holding for 1 second then returning...")
        rospy.sleep(1.0)
        latest_arrival  = None
        latest_ball_pos = None
        kick_ref_msg    = None
        state = RETURNING

    # ── RETURNING ────────────────────────────────────────────────────────────
    elif state == RETURNING:
        say("Resetting")
        go_to_ready()
        current_speed  = 0.0
        prev_direction = 0.0
        kick_ref_msg   = None
        latest_arrival = None
        state = WAITING
        rospy.loginfo("[MOVER] === Ready! Roll the ball when ready. ===")
        say("Ready. Roll the ball.")

    msg = TwistStamped()
    msg.header.stamp    = rospy.Time.now()
    msg.header.frame_id = "table_top"
    msg.twist.linear.x  = twist_x
    pub.publish(msg)

    rate.sleep()
