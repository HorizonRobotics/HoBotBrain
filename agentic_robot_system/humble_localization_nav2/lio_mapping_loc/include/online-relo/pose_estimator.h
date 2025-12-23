#pragma once

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <deque>
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <thread>

#include "../FRICP-toolkit/registeration.h"
#include "../common_lib.h"
#include "../multi-session/Incremental_mapping.hpp"
#include "../tool_color_printf.h"
#include "../utils/tictoc.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#define MAX_TIME_DIFF 0.05  // seconds, max time difference
class pose_estimator {
 public:
  pose_estimator(rclcpp::Node::SharedPtr& node);
  ~pose_estimator() {}

  // Parameters
  std::string priorDir;
  std::string cloudTopic;
  std::string poseTopic;
  std::string cloudTopic_repub;
  std::string poseTopic_repub;
  float searchDis;
  int searchNum;
  float trustDis;
  int regMode;
  rclcpp::Node::SharedPtr node;
  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subCloud;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subPose;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      subExternalPose;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloud;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubPose;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPriorMap;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPriorPath;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubInitCloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubReloWorldCloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRelocBodyCloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubNearCloud;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      pubMeasurementEdge;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;

  // Point clouds
  pcl::PointCloud<PointTypeXYZI>::Ptr priorMap;
  pcl::PointCloud<PointTypeXYZI>::Ptr priorPath;
  pcl::PointCloud<PointTypeXYZI>::Ptr reloCloudInMap;
  pcl::PointCloud<PointTypeXYZI>::Ptr cloudInBody;
  pcl::PointCloud<PointTypeXYZI>::Ptr initCloud;
  pcl::PointCloud<PointTypeXYZI>::Ptr initCloudInOdom;
  pcl::PointCloud<PointTypeXYZI>::Ptr nearCloud;
  pcl::PointCloud<PointTypeXYZI>::Ptr localMapDS;

  // Pose info
  PointTypePose externalPose;
  PointTypePose initPose;
  PointTypePose pose_zero;
  PointTypePose pose_ext;

  std::vector<double> extrinT_;
  std::vector<double> extrinR_;
  Eigen::Vector3d extrinT;
  Eigen::Matrix3d extrinR;
  int relo_interval;

  // KD-tree
  std::vector<int> idxVec;
  std::vector<float> disVec;
  pcl::KdTreeFLANN<PointTypeXYZI>::Ptr kdtreeGlobalMapPoses;

  std::vector<int> idxVec_copy;
  std::vector<float> disVec_copy;
  pcl::KdTreeFLANN<PointTypeXYZI>::Ptr kdtreeGlobalMapPoses_copy;

  pcl::VoxelGrid<PointTypeXYZI> downSizeFilterPub, downSizeFilterLocalMap,
      downSizeFilterCurrentCloud;
  pcl::VoxelGrid<PointTypeXYZI> downSizeInitCloud, downSizeFilterNearCloud;
  pcl::NormalDistributionsTransform<PointTypeXYZI, PointTypeXYZI> NDT;
  int idx = 1;
  std::deque<pcl::PointCloud<PointTypeXYZI>::Ptr> cloudBuffer;
  std::deque<double> cloudtimeBuffer;
  std::deque<Eigen::Affine3f> poseBuffer_6D;
  std::deque<double> posetimeBuffer;
  std::deque<PointTypeXYZI> poseBuffer_3D;
  std::ofstream fout_relo;
  std::mutex cloudBufferMutex;
  std::mutex poseBufferMutex;
  std::condition_variable sig_buffer;
  // PointTypePose currentPoseInOdom;
  Eigen::Affine3f currentPoseInOdom;
  Eigen::Affine3f lastPoseInOdom = Eigen::Affine3f::Identity();
  Eigen::Affine3f deltaPose;
  PointTypeXYZI currentPose3d;
  Eigen::Affine3f currentPoseInMap;
  Eigen::Affine3f lastPoseInMap = Eigen::Affine3f::Identity();

  pcl::PointCloud<PointTypeXYZI>::Ptr currentCloud,currentCloudDs;
  pcl::PointCloud<PointTypeXYZI>::Ptr currentCloudInMap;
  pcl::PointCloud<PointTypeXYZI>::Ptr currentCloudDsInMap;
  pcl::PointCloud<PointTypeXYZI>::Ptr currentCloudDsInOdom;
  double currentCloudTime;

  // Path and odometry
  double ld_time;
  nav_msgs::msg::Path path;
  nav_msgs::msg::Odometry odomAftMapped;
  geometry_msgs::msg::PoseStamped msg_body_pose;
  std::deque<PointTypePose> reloPoseBuffer;

  // Sessions and registration
  std::vector<MultiSession::Session> sessions;
  std::vector<Registeration> reg;
  std::pair<int, float> detectResult;
  std::vector<int> invalid_idx;

  // Flags
  bool buffer_flg = true;
  bool global_flg = false;
  bool external_flg = false;
  bool receive_ext_flg = false;
  bool sc_init_enable = false;
  bool sc_flg = false;

  float height;
  int cout_count = 0;
  int cout_count_ = 0;
  int sc_new = 1;
  int sc_old = -1;
  std::thread thread_loc_, thread_pub_;

  // Methods
  void allocateMemory();

  void cloudCBK(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void poseCBK(const nav_msgs::msg::Odometry::SharedPtr msg);
  void externalCBK(
      const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);

  void run(rclcpp::Node::SharedPtr& node);
  void publish_cloud(rclcpp::Node::SharedPtr& node);
  void start_localization();
  bool easyToRelo(const PointTypeXYZI& pose3d);
  bool globalRelo();
  bool relocalization();
  void lio_incremental();
  void publish_odometry(const Eigen::Affine3f& trans_aft);
  void publish_odometry(
      const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr& pub);
  void publish_path(
      const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr& pub);
};
