#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <std_msgs/String.h>
#include <quadrotor_msgs/PositionCommand.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <filesystem>

namespace fs = std::filesystem;

class Bag2Publisher
{
private:
    ros::NodeHandle nh_;
    ros::Publisher control_pub_;
    ros::Publisher path_pub_;
    
    std::string bag_path_;
    std::string robot_id_;
    double publish_rate_;
    std::string frame_id_;
    
    std::vector<quadrotor_msgs::PositionCommand> control_data_;
    std::vector<ros::Time> timestamps_;

public:
    Bag2Publisher(const std::string& bag_path, 
                        const std::string& robot_id,
                        double publish_rate = 10.0,
                        const std::string& frame_id = "map")
        : bag_path_(bag_path)
        , robot_id_(robot_id)
        , publish_rate_(publish_rate)
        , frame_id_(frame_id){
        
        // 加载bag文件
        if (!loadFromBag()) {
            ROS_ERROR("loading bag files failed!");
            return;
        }
        // 创建发布器
        createPublishers();
    }
    ~Bag2Publisher() = default;
    
private:
    bool loadFromBag()
    {
        // 检查文件是否存在
        if (!fs::exists(bag_path_)) {
            return false;
        }
        try {
            rosbag::Bag bag;
            bag.open(bag_path_, rosbag::bagmode::Read);
            // 获取bag文件信息
            rosbag::View view(bag);
            // 清空数据
            control_data_.clear();
            timestamps_.clear();
            // 读取所有消息
            std::vector<std::tuple<std::string, quadrotor_msgs::PositionCommand, ros::Time>> messages;
            
            for (const rosbag::MessageInstance& m : view) {
                if (m.getTopic() == "/trajectory/sequence") {
                    auto msg = m.instantiate<quadrotor_msgs::PositionCommand>();
                    if (msg != nullptr) {
                        messages.emplace_back(m.getTopic(), *msg, m.getTime());
                    }
                }
            }
            // 按时间排序
            std::sort(messages.begin(), messages.end(),
                [](const auto& a, const auto& b) {
                    return std::get<2>(a) < std::get<2>(b);
                });
            
            // 提取数据
            for (const auto& [topic, msg, timestamp] : messages) {
                control_data_.push_back(msg);
                timestamps_.push_back(timestamp);
            }
            bag.close();
            return true;
            
        } catch (const rosbag::BagException& e) {
            ROS_ERROR_STREAM("加载失败: " << e.what());
            return false;
        } catch (const std::exception& e) {
            ROS_ERROR_STREAM("加载失败: " << e.what());
            return false;
        }
    }
    
    void createPublishers()
    {
        std::string control_topic = "/trajectory/" + robot_id_ + "/control_sequence";
        control_pub_ = nh_.advertise<quadrotor_msgs::PositionCommand>(control_topic, 10);
        
    }
    
    void publishPoint(size_t idx){
        quadrotor_msgs::PositionCommand pose_msg = control_data_[idx];
        // 更新时间戳
        pose_msg.header.stamp = ros::Time::now();
        pose_msg.header.frame_id = frame_id_;
        control_pub_.publish(pose_msg);
    }
    
    void publishSequence(){
        if (control_data_.empty()) {
            ROS_WARN("没有数据可发布");
            return;
        }
        ros::Rate rate(publish_rate_);
        size_t idx = 0;
        while (ros::ok() && idx < control_data_.size()) {
            publishPoint(idx);
            idx++;
            rate.sleep();
        }
    }
    
public:
    void run()
    {
        publishSequence();
    }
    
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "bag_to_topic_publisher");
    ros::NodeHandle nh("~");
    double publish_rate;
    std::string bag_path, frame_id, robot_id;
    nh.param<std::string>("robot_id", robot_id, "robot_0");
    nh.param<double>("publish_rate", publish_rate, 20.0);
    nh.param<std::string>("frame", frame_id, "map1");
    nh.param<std::string>("bag_path", bag_path, std::string(""));
    // 创建并运行发布器
    Bag2Publisher publisher(
        bag_path,
        robot_id,
        publish_rate,
        frame_id
    );
    publisher.run();
    
    return 0;
}