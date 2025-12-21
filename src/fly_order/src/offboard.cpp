#include "nav_msgs/Odometry.h"
#include "quadrotor_msgs/PositionCommand.h"
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <Eigen/Dense>

ros::Publisher pos_cmd_pub;
Eigen::Vector3d pos(Eigen::Vector3d::Zero()), vel(Eigen::Vector3d::Zero()), acc(Eigen::Vector3d::Zero());
quadrotor_msgs::PositionCommand cmd;
double pos_gain[3] = {0, 0, 0};
double vel_gain[3] = {0, 0, 0};
double last_yaw_, last_yaw_dot_;
int drone_id;
bool control_start = false;
void posCmdCallback(const quadrotor_msgs::PositionCommand::ConstPtr& msg) {
  control_start = true;
  pos(0) = msg->position.x;
  pos(1) = msg->position.y;
  pos(2) = msg->position.z;
  vel(0) = msg->velocity.x;
  vel(1) = msg->velocity.y;
  vel(2) = msg->velocity.z;
  acc(0) = msg->acceleration.x;
  acc(1) = msg->acceleration.y;
  acc(2) = msg->acceleration.z;
}
void cmdCallback(const ros::TimerEvent &e)
{
  // cmd.header.stamp = time_now;
  if (control_start)
  { 
    cmd.header.frame_id = "world";
    cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
    cmd.trajectory_id = drone_id;
    
    cmd.position.x = pos(0);
    cmd.position.y = pos(1);
    cmd.position.z = 1.0;

    cmd.velocity.x = vel(0);
    cmd.velocity.y = vel(1);
    cmd.velocity.z = vel(2);

    cmd.acceleration.x = acc(0);
    cmd.acceleration.y = acc(1);
    cmd.acceleration.z = acc(2);
    
    cmd.yaw = 0;
    cmd.yaw_dot = 0;

    pos_cmd_pub.publish(cmd); 
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "fly_order");
  ros::NodeHandle nh("~");

  pos_cmd_pub = nh.advertise<quadrotor_msgs::PositionCommand>("/position_cmd", 50);
  ros::Subscriber pos_sub = nh.subscribe<quadrotor_msgs::PositionCommand>("/trajectory/control_sequence", 50, posCmdCallback);


  ros::Timer cmd_timer = nh.createTimer(ros::Duration(0.01), cmdCallback);
  nh.param("drone_id", drone_id, 0);
  /* control parameter */
  cmd.kx[0] = pos_gain[0];
  cmd.kx[1] = pos_gain[1];
  cmd.kx[2] = pos_gain[2];

  cmd.kv[0] = vel_gain[0];
  cmd.kv[1] = vel_gain[1];
  cmd.kv[2] = vel_gain[2];

  // std::cin >> dummy; 
  ros::Duration(4.0).sleep();
  ros::Rate rate(30.0); // 设置循环频率为20Hz
  ROS_WARN("[Order server]: ready.");
  while (ros::ok()){
    ros::spinOnce();
    rate.sleep(); 
  }
  
  // ros::spin();
  return 0;
}



