/* 随机地图生成器节点 - 用于生成无人机/机器人仿真环境的随机障碍物地图
 * 主要功能：
 *  1. 生成随机圆柱体障碍物和环状障碍物
 *  2. 处理里程计数据
 *  3. 发布全局和局部点云地图
 *  4. 支持点击添加障碍物
 *
 * 注意：此代码实现了2种障碍物生成算法：
 *  - RandomMapGenerateCylinder(): 改进的圆柱体障碍物生成
 */

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// #include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>  // 使用FLANN实现的KD树进行最近邻搜索
#include <pcl_conversions/pcl_conversions.h>  // PCL与ROS消息转换
#include <iostream>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3.h>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>  // 点云ROS消息
#include <Eigen/Eigen>  // 线性代数库
#include <random>  // 随机数生成

using namespace std;
std::vector<double> cylinder_x, cylinder_y, cylinder_w, cylinder_h;
std::vector<double> circle_x, circle_y, circle_z, circle_theta, circle_radius1, circle_radius2;
// KD树用于快速点云搜索
//pcl::search::KdTree<pcl::PointXYZ> kdtreeLocalMap;
pcl::KdTreeFLANN<pcl::PointXYZ> kdtreeLocalMap;
vector<int> pointIdxRadiusSearch;       // 搜索结果索引
vector<float> pointRadiusSquaredDistance; // 搜索结果距离平方

// ROS发布器
ros::Publisher _local_map_pub;    // 局部地图
ros::Publisher _all_map_pub;      // 全局地图
ros::Publisher click_map_pub_;    // 点击添加的地图
ros::Subscriber _odom_sub;        // 里程计订阅

// 系统状态 [x, y, z, vx, vy, vz, ...]
vector<double> _state;

// 地图参数
int cylinder_num_;        // 障碍物数量
double _x_size, _y_size, _z_size; // 地图尺寸
double _x_l, _x_h, _y_l, _y_h;   // XY范围边界
double _w_l, _w_h, _h_l, _h_h;   // 障碍物尺寸范围
double _z_limit, _sensing_range, _resolution, _sense_rate, _init_x, _init_y;
double _min_dist;    // 最小障碍物间距
double _inf;         // 膨胀系数

// 系统状态标志
bool _map_ok = false;  // 地图是否准备好
bool _has_odom = false; // 是否收到里程计

// 环形障碍物参数
int circle_num_;     // 环形障碍物数量
double radius_l_, radius_h_, z_l_, z_h_; // 环半径和高低范围
// 点云存储
sensor_msgs::PointCloud2 globalMap_pcd; // 全局点云(ROS消息)
pcl::PointCloud<pcl::PointXYZ> cloudMap; // 全局点云(PCL格式)

sensor_msgs::PointCloud2 localMap_pcd; // 局部点云(ROS消息)
pcl::PointCloud<pcl::PointXYZ> clicked_cloud_; // 点击添加的点云

// 改进的圆柱体障碍物生成函数
void CustomMapGenerateCylinder() {
  pcl::PointXYZ pt_random;
  vector<Eigen::Vector2d> obs_position; // 存储已生成障碍物位置

  // 生成圆柱体障碍物
  for (int i = 0; i < cylinder_num_ && ros::ok(); i++) {
    double x, y, w, h, inf;
    x = cylinder_x[i];
    y = cylinder_y[i];
    w = cylinder_w[i];
    h = cylinder_h[i];
    inf = _inf; // 膨胀系数

    // 位置检查（避免靠近起始点和目标点）
    if (sqrt(pow(x - _init_x, 2) + pow(y - _init_y, 2)) < 2.0) {
      i--;
      continue;
    }
    if (sqrt(pow(x - 19.0, 2) + pow(y - 0.0, 2)) < 2.0) {
      i--;
      continue;
    }
    
    // 新增：检查障碍物间距
    bool flag_continue = false;
    for (auto p : obs_position) {
      if ((Eigen::Vector2d(x,y) - p).norm() < _min_dist) {
        i--; // 间距过小，重新生成
        flag_continue = true;
        break;
      }
    }
    if (flag_continue) continue;

    // 记录已生成位置
    obs_position.push_back(Eigen::Vector2d(x,y));
    
    // 对齐网格
    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;

    // 计算膨胀后半径和网格数
    double radius = inf + (w / 2); // 应用膨胀后的半径
    int widNum = ceil((w + 2 * inf) / _resolution);
    int heiNum = ceil(h / _resolution);
    // 生成圆柱体点云
    for (int r = -widNum / 2.0; r < widNum / 2.0; r++) {
      for (int s = -widNum / 2.0; s < widNum / 2.0; s++) {
        for (int t = -30; t < heiNum; t++) {
          double temp_x = x + (r + 0.5) * _resolution + 1e-2;
          double temp_y = y + (s + 0.5) * _resolution + 1e-2;
          double temp_z = (t + 0.5) * _resolution + 1e-2;
          
          // 检查点是否在圆内（形成圆柱体）
          if ((Eigen::Vector2d(temp_x,temp_y) - Eigen::Vector2d(x,y)).norm() <= radius) {
            pt_random.x = temp_x;
            pt_random.y = temp_y;
            pt_random.z = temp_z;
            cloudMap.points.push_back(pt_random);
          }
        }
      }
    }
  }

  // 生成环形障碍物（与RandomMapGenerate()相同）
  for (int i = 0; i < circle_num_; ++i) {
    double x, y, z;
    x = circle_x[i];
    y = circle_y[i];
    z = circle_z[i];

    // 位置检查
    if (sqrt(pow(x - _init_x, 2) + pow(y - _init_y, 2)) < 2.0) {
      i--;
      continue;
    }
    if (sqrt(pow(x - 19.0, 2) + pow(y - 0.0, 2)) < 2.0) {
      i--;
      continue;
    }

    // 对齐网格
    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;
    z = floor(z / _resolution) * _resolution + _resolution / 2.0;

    Eigen::Vector3d translate(x, y, z);
    double theta = circle_theta[i];
    Eigen::Matrix3d rotate;
    rotate << cos(theta), -sin(theta), 0.0, 
             sin(theta), cos(theta), 0.0, 
             0, 0, 1;

    double radius1 = circle_radius1[i];
    double radius2 = circle_radius2[i];

    // 沿椭圆生成点
    Eigen::Vector3d cpt;
    for (double angle = 0.0; angle < 6.282; angle += _resolution / 2) {
      cpt(0) = 0.0;
      cpt(1) = radius1 * cos(angle);
      cpt(2) = radius2 * sin(angle);

      // 点膨胀（实际未使用）
      Eigen::Vector3d cpt_if;
      for (int ifx = -0; ifx <= 2; ++ifx)
        for (int ify = -0; ify <= 2; ++ify)
          for (int ifz = -0; ifz <= 2; ++ifz) {
            cpt_if = cpt + Eigen::Vector3d(ifx * _resolution, ify * _resolution,
                                           ifz * _resolution);
            cpt_if = rotate * cpt_if + translate;
            pt_random.x = cpt_if(0);
            pt_random.y = cpt_if(1);
            pt_random.z = cpt_if(2);
            cloudMap.push_back(pt_random);
          }
    }
  }

  // 添加地板（被注释掉）
  /*
  pcl::PointXYZ pt;
  pt.z = 0.1;
  for (pt.x = _x_l; pt.x <= _x_h; pt.x += _resolution)
    for (pt.y = _y_l; pt.y <= _y_h; pt.y += _resolution) {
      cloudMap.push_back(pt);
    }
  */

  // 设置点云属性
  cloudMap.width = cloudMap.points.size();
  cloudMap.height = 1;
  cloudMap.is_dense = true;

  ROS_WARN("Finished generate custom map ");

  // 构建KD树
  kdtreeLocalMap.setInputCloud(cloudMap.makeShared());

  _map_ok = true;
}

// 里程计回调函数
void rcvOdometryCallbck(const nav_msgs::Odometry odom) {
  // 跳过特定类型的odom消息
  if (odom.child_frame_id == "X" || odom.child_frame_id == "O") return;
  
  _has_odom = true; // 标记已收到里程计

  // 提取位置和速度信息
  _state = {
    odom.pose.pose.position.x,
    odom.pose.pose.position.y,
    odom.pose.pose.position.z,
    odom.twist.twist.linear.x,
    odom.twist.twist.linear.y,
    odom.twist.twist.linear.z,
    0.0,  // 预留
    0.0,  // 预留
    0.0   // 预留
  };
}

// 发布感知到的点云
int i = 0;
void pubSensedPoints() {
  // 早期版本只发布前10次（已废弃）
  // if (i < 10) {
  pcl::toROSMsg(cloudMap, globalMap_pcd); // PCL转ROS消息
  globalMap_pcd.header.frame_id = "world";
  _all_map_pub.publish(globalMap_pcd); // 发布全局地图
  // }
  i++;
  
  return; // 下面的局部地图发布已禁用

  /* ---------- 仅发布当前位置周围的点（已禁用） ---------- */
  if (!_map_ok || !_has_odom) return;

  pcl::PointCloud<pcl::PointXYZ> localMap;
  pcl::PointXYZ searchPoint(_state[0], _state[1], _state[2]); // 当前位姿
  
  // 重置搜索变量
  pointIdxRadiusSearch.clear();
  pointRadiusSquaredDistance.clear();

  pcl::PointXYZ pt;

  // 检查搜索点是否有效
  if (isnan(searchPoint.x) || isnan(searchPoint.y) || isnan(searchPoint.z))
    return;

  // 半径搜索
  if (kdtreeLocalMap.radiusSearch(searchPoint, _sensing_range,
                                  pointIdxRadiusSearch,
                                  pointRadiusSquaredDistance) > 0) {
    // 收集搜索到的点
    for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
      pt = cloudMap.points[pointIdxRadiusSearch[i]];
      localMap.points.push_back(pt);
    }
  } else {
    ROS_ERROR("[Map server] No obstacles.");
    return;
  }

  // 设置局部点云属性
  localMap.width = localMap.points.size();
  localMap.height = 1;
  localMap.is_dense = true;

  // 发布局部地图
  pcl::toROSMsg(localMap, localMap_pcd);
  localMap_pcd.header.frame_id = "world";
  _local_map_pub.publish(localMap_pcd);
}

// 主函数
int main(int argc, char** argv) {
  ros::init(argc, argv, "custom_map_sensing");
  ros::NodeHandle n("~"); // 私有命名空间

  // 创建发布器和订阅器
  _local_map_pub = n.advertise<sensor_msgs::PointCloud2>("/map_generator/local_cloud", 1);
  _all_map_pub = n.advertise<sensor_msgs::PointCloud2>("/map_generator/global_cloud", 1);
  _odom_sub = n.subscribe("odometry", 50, rcvOdometryCallbck);
  click_map_pub_ = n.advertise<sensor_msgs::PointCloud2>("/pcl_render_node/local_map", 1);
  
  // 从参数服务器获取参数
  n.param("init_state_x", _init_x, 0.0); // 初始X位置，默认0
  n.param("init_state_y", _init_y, 0.0); // 初始Y位置，默认0
  
  n.param("map/x_size", _x_size, 50.0); // 地图X尺寸，默认50m
  n.param("map/y_size", _y_size, 50.0); // 地图Y尺寸，默认50m
  n.param("map/z_size", _z_size, 5.0);  // 地图Z尺寸，默认5m
  n.param("map/cylinder_num", cylinder_num_, 0); // 障碍物数量，默认30
  n.param("map/resolution", _resolution, 0.1); // 地图分辨率，默认0.1m
  n.param("map/circle_num", circle_num_, 0);  // 环形障碍物数量，默认30
  n.param("map/inf", _inf, 0.0); // 膨胀系数，默认1.0（不膨胀）

  cylinder_x.resize(cylinder_num_);
  cylinder_y.resize(cylinder_num_);
  cylinder_w.resize(cylinder_num_);
  cylinder_h.resize(cylinder_num_);
  for (int i = 0; i < cylinder_num_; i++){
    n.param("cylinder_obs/x" + std::to_string(i), cylinder_x[i], 0.0); 
    n.param("cylinder_obs/y" + std::to_string(i), cylinder_y[i], 0.0);
    n.param("cylinder_obs/w" + std::to_string(i), cylinder_w[i], 0.0);
    n.param("cylinder_obs/h" + std::to_string(i), cylinder_h[i], 0.0);
  }
  circle_x.resize(circle_num_);
  circle_y.resize(circle_num_);
  circle_z.resize(circle_num_);
  circle_theta.resize(circle_num_);
  circle_radius1.resize(circle_num_);
  circle_radius2.resize(circle_num_);
  for (int i = 0; i < circle_num_; i++){
    n.param("circle_obs/x" + std::to_string(i), circle_x[i], 0.0); 
    n.param("circle_obs/y" + std::to_string(i), circle_y[i], 0.0);
    n.param("circle_obs/z" + std::to_string(i), circle_z[i], 0.0);
    n.param("circle_obs/theta" + std::to_string(i), circle_theta[i], 0.0);
    n.param("circle_obs/radius1_" + std::to_string(i), circle_radius1[i], 0.0); 
    n.param("circle_obs/radius2_" + std::to_string(i), circle_radius2[i], 0.0);
  }
  // 感知参数
  n.param("sensing/radius", _sensing_range, 10.0); // 感知半径
  n.param("sensing/radius", _sense_rate, 10.0);    // 感知频率

  n.param("min_distance", _min_dist, 1.0); // 最小障碍物间距

  // 计算地图边界
  _x_l = -_x_size / 2.0;
  _x_h = +_x_size / 2.0;
  _y_l = -_y_size / 2.0;
  _y_h = +_y_size / 2.0;

  // 限制障碍物数量
  cylinder_num_ = min(cylinder_num_, (int)_x_size * 10);
  _z_limit = _z_size; // Z限制

  // 短暂延迟
  ros::Duration(0.5).sleep();

  CustomMapGenerateCylinder(); // 使用改进的圆柱体生成方法

  // 设置循环频率
  ros::Rate loop_rate(_sense_rate);

  // 主循环
  while (ros::ok()) {
    pubSensedPoints(); // 发布点云
    ros::spinOnce();   // 处理回调
    loop_rate.sleep(); // 按频率休眠
  }
}