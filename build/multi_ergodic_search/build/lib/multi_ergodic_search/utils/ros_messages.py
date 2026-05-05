"""ROS2 message contract used by experiment outputs.

Control commands are represented as trajectory_msgs/MultiDOFJointTrajectoryPoint:
- transforms[0].translation.x/y store planar position.
- velocities[0].linear.x/y store planar velocity.
- accelerations[0].linear.x/y store planar acceleration command.
- time_from_start is filled when the command belongs to a recorded sequence.
"""

from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import Transform, Twist, Vector3
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint


CONTROL_COMMAND_MSG = MultiDOFJointTrajectoryPoint


def control_topic(topic_pattern, robot_id):
    return topic_pattern.format(robot_id=robot_id)


def seconds_to_nanoseconds(seconds):
    return int(round(float(seconds) * 1_000_000_000))


def seconds_to_time_msg(seconds):
    seconds = float(seconds)
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1_000_000_000))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


def seconds_to_duration_msg(seconds):
    time_msg = seconds_to_time_msg(seconds)
    return Duration(sec=time_msg.sec, nanosec=time_msg.nanosec)


def build_control_command(x_data, u_data, time_from_start=None):
    command = CONTROL_COMMAND_MSG()
    transform = Transform()
    transform.translation = Vector3(
        x=float(x_data[0]),
        y=float(x_data[1]),
        z=0.0,
    )
    transform.rotation.w = 1.0

    velocity = Twist()
    velocity.linear.x = float(x_data[2])
    velocity.linear.y = float(x_data[3])

    acceleration = Twist()
    acceleration.linear.x = float(u_data[0])
    acceleration.linear.y = float(u_data[1])

    command.transforms = [transform]
    command.velocities = [velocity]
    command.accelerations = [acceleration]
    if time_from_start is not None:
        command.time_from_start = seconds_to_duration_msg(time_from_start)
    return command

