#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <string>

int main(int argc, char** argv) {
    ros::init(argc, argv, "replay_probability_cloud");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    // 参数：bag 文件路径 和 是否使用 bag 时间戳
    std::string bag_path = "/home/cyc/Decay_Ergodic_Search/src/ergodic_search/datas/bags/prob_connect/mapinfo.bag";  // ←←← 改成你的实际路径
    bool use_bag_time = true;  // 若为 false，则按当前时间发布
    double rate = 1.0;         // 回放速度倍率（仅当 use_bag_time=true 时有效）
    private_nh.param<bool>("use_bag_time", use_bag_time, true);
    private_nh.param<double>("rate", rate, 1.0);

    // 打开 bag
    rosbag::Bag bag;
    try {
        bag.open(bag_path, rosbag::bagmode::Read);
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to open bag file: %s", e.what());
        return 1;
    }

    // 指定要读取的话题
    std::vector<std::string> topics;
    topics.push_back("/distribution/probability_cloud");

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    // 创建发布器
    ros::Publisher pub = nh.advertise<sensor_msgs::Image>("/probability_map", 10);

    ROS_INFO("Replaying PointCloud2 from bag: %s", bag_path.c_str());
    ROS_INFO("Publishing on topic: /distribution/probability_cloud");

    ros::Time start_time;
    bool first_message = true;

    for (const rosbag::MessageInstance& m : view) {
        ROS_ERROR("111");   
        if (!ros::ok()) break;

        sensor_msgs::PointCloud2::ConstPtr msg = m.instantiate<sensor_msgs::PointCloud2>();
        if (msg == nullptr) continue;

        sensor_msgs::PointCloud2 out_msg = *msg; // 拷贝

        start_time = ros::Time::now();
        
        while (ros::ok()) {
            out_msg.header.stamp = ros::Time::now(); // 或保留原 stamp，取决于 RViz 需求
            pub.publish(out_msg);
            ros::Duration(0.001).sleep(); // 1ms sleep
        }
        
    }
    ROS_INFO("over");
    bag.close();
    return 0;
}