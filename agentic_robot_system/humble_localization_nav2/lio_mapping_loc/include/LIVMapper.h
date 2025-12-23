/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIV_MAPPER_H
#define LIV_MAPPER_H

#include <utils/types.h>

#include <atomic>

#include "IMU_Processing.h"
#include "common_lib.h"
#include "preprocess.h"
#include "sc-relo/Scancontext.h"
#include "vio.h"
#ifdef PRE_ROS_IRON
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.hpp>
#endif
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <vikit/camera_loader.h>

#include <condition_variable>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <image_transport/image_transport.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>

#include "fast_livo/srv/save_map.hpp"
#include "ground_detection.h"
#include "wheel_odometry.h"
#include "zupt.h"
using namespace ScanContext;
class LIVMapper {
 public:
  LIVMapper(rclcpp::Node::SharedPtr &node, std::string node_name,
            const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~LIVMapper();
  void initializeSubscribersAndPublishers(rclcpp::Node::SharedPtr &nh,
                                          image_transport::ImageTransport &it_);
  void initializeComponents(rclcpp::Node::SharedPtr &node);
  void initializeFiles();
  void run(rclcpp::Node::SharedPtr &node);
  void gravityAlignment();
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleVIO();
  bool handleLIO();
  void savePCD();
  void processImu();
  void processRobotOdometry(LidarMeasureGroup &meas);
  void extractWheelVel(LidarMeasureGroup &meas);
  void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odometryMsg);
  void saveKeyFramesAndFactor();
  void correctPoses();
  bool detectLoopClosureDistance(int *latestID, int *closestID);
  void loopFindNearKeyframes(pcl::PointCloud<PointTypeXYZI>::Ptr &nearKeyframes,
                             const int &key, const int &searchNum);
  void performLoopClosure();
  void loopClosureThread();
  void visualizeLoopClosure();
  void addLoopFactor();
  void addOdomFactor();
  bool saveFrame();
  void saveKeyFrame();
  void saveOptimizedVerticesKITTIformat(gtsam::Values _estimates,
                                        std::string _filename);
  void updatePath(const PointTypePose &pose_in);
  void getCurPose(StatesGroup cur_state);
  void publishCloud(
      rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub,
      pcl::PointCloud<PointTypeXYZI>::Ptr thisCloud, rclcpp::Time thisStamp,
      std::string thisFrame);
  bool sync_packages(LidarMeasureGroup &meas);
  void prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr,
                     V3D angvel_avr);
  void imu_prop_callback();
  void transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t,
                      const PointCloudXYZI::Ptr &input_cloud,
                      PointCloudXYZI::Ptr &trans_cloud);
  void transform2Camera(const Eigen::Matrix3d &rot, const Eigen::Vector3d &t,
                        const PointCloudXYZRGB::Ptr &input_cloud,
                        PointCloudXYZRGB::Ptr &trans_cloud);
  void pointBodyToWorld(const PointType &pi, PointType &po);
  nav_msgs::msg::Odometry odomToLidarConverter(
      const nav_msgs::msg::Odometry &odom_in);
  void RGBpointBodyToWorld(PointType const *const pi, PointType *const po);
  void standard_pcl_cbk(
      const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void livox_pcl_cbk(
      const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg_in);
  void imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr &msg_in);
  void img_cbk(const sensor_msgs::msg::Image::ConstSharedPtr &msg_in);
  void publish_img_rgb(const image_transport::Publisher &pubImage,
                       VIOManagerPtr vio_manager);

  void publish_frame_world(
      const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
          &pubLaserCloudFullRes,
      VIOManagerPtr vio_manager);
  void publish_frame_lidar(
      const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
          &pubLaserCloudFullRes);
  void publish_submap_world(const PointCloudXYZRGB::Ptr &rgb_cloud);
  void publish_visual_sub_map(
      const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
          &pubSubVisualMap);
  void publish_effect_world(
      const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
          &pubLaserCloudEffect,
      const std::vector<PointToPlane> &ptpl_list);
  void publish_odometry(
      const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr
          &pmavros_pose_publisherubOdomAftMapped);
  void publish_mavros(
      const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
          &mavros_pose_publisher);
  void publish_path(
      const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pubPath);
  void readParameters(rclcpp::Node::SharedPtr &node);
  template <typename T>
  void set_posestamp(T &out);
  template <typename T>
  void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi,
                        Eigen::Matrix<T, 3, 1> &po);
  template <typename T>
  Eigen::Matrix<T, 3, 1> pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi);
  cv::Mat getImageFromMsg(
      const sensor_msgs::msg::Image::ConstSharedPtr &img_msg);

  std::mutex mtx_buffer, mtx_buffer_imu_prop;
  std::condition_variable sig_buffer;
  std::mutex mtx;
  std::mutex mtxLoopInfo;
  SLAM_MODE slam_mode_;
  //   std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;

  string root_dir, map_save_path;
  string lid_topic, imu_topic, seq_name, img_topic;

  bool enable_zupt = true;
  bool enable_wheel_odom = true;
  bool est_wheel_extrinsic = false;
  double zupt_noise = 0.1;
  int zupt_interval = 10;
  V3D extT;
  M3D extR;

  int feats_down_size = 0, max_iterations = 0;

  double res_mean_last = 0.05;
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;
  double blind_rgb_points = 0.0;
  double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0,
         last_timestamp_img = -1.0;
  double filter_size_surf_min = 0;
  double filter_size_pcd = 0;
  double _first_lidar_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;

  bool lidar_map_inited = false, pcd_save_en = false,
       pub_effect_point_en = false, pose_output_en = false,
       ros_driver_fix_en = false;
  int pcd_save_interval = -1, pcd_index = 0;
  int rgb_cloud_interval = 1;
  int pub_scan_num = 1;
  bool img_filter_en = false;
  int img_filter_fre = 1;
  StatesGroup imu_propagate, latest_ekf_state;

  bool new_imu = false, state_update_flg = false, imu_prop_enable = true,
       ekf_finish_once = false;
  deque<sensor_msgs::msg::Imu> prop_imu_buffer;
  sensor_msgs::msg::Imu newest_imu;
  double latest_ekf_time;
  nav_msgs::msg::Odometry imu_prop_odom;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuPropOdom;
  double imu_time_offset = 0.0;

  bool gravity_align_en = false, gravity_align_finished = false;

  bool sync_jump_flag = false;

  bool lidar_pushed = false, imu_en, gravity_est_en, flg_reset = false,
       ba_bg_est_en = true;
  bool dense_map_en = false;
  int img_en = 1, imu_int_frame = 3;
  bool normal_en = true;
  bool pub_rgb_cloud_en = false;
  bool depth_en = false;
  bool exposure_estimate_en = false;
  double exposure_time_init = 0.0;
  bool inverse_composition_en = false;
  bool raycast_en = false;
  int lidar_en = 1;
  bool is_first_frame = false;
  int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;
  int outlier_threshold;
  double plot_time;
  int frame_cnt;
  double img_time_offset = 0.0;
  deque<PointCloudXYZI::Ptr> lid_raw_data_buffer;
  deque<double> lid_header_time_buffer;
  deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;
  deque<cv::Mat> img_buffer;
  deque<double> img_time_buffer;
  vector<pointWithVar> _pv_list;
  vector<double> extrinT;
  vector<double> extrinR;
  vector<double> cameraextrinT;
  vector<double> cameraextrinR;
  int IMG_POINT_COV;
  // wheel odometry
  //   Eigen::Vector3d p_wheel_to_imu;
  //   Eigen::Matrix3d R_wheel_to_imu;
  vector<double> p_wheel_to_imu;
  vector<double> R_wheel_to_imu;
  vector<double> wheel_ext_cov_;
  vector<double> wheel_mea_cov_;
  Eigen::Vector3d wheel_linear_velocity_;
  Eigen::Vector3d imu_angular_velocity_;
  bool wheel_odom_updated_ = true;
  PointCloudXYZI::Ptr visual_sub_map;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_body_ground;
  PointCloudXYZI::Ptr ds_ground_pc;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_ng_w_wait_pub;
  PointCloudXYZI::Ptr pcl_g_w_wait_pub;
  PointCloudXYZI::Ptr pcl_body_wait_pub;

  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZRGB::Ptr pcl_wait_save, pcl_wait_save_global_rgb;
  PointCloudXYZRGB::Ptr local_rgb_cloud;
  deque<PointCloudXYZRGB::Ptr> local_rgb_cloud_buffer;

  PointCloudXYZI::Ptr pcl_wait_save_intensity, pcl_wait_save_intensity_cp,
      pcl_wait_save_global;

  // PGO
  vector<pcl::PointCloud<PointTypeXYZI>::Ptr>
      cornerCloudKeyFrames;  // 历史所有关键帧的角点集合(降采样)
  vector<pcl::PointCloud<PointTypeXYZI>::Ptr>
      surfCloudKeyFrames;  // 历史所有关键帧的平面点集合(降采样)
  pcl::PointCloud<PointTypeXYZI>::Ptr cloudKeyPoses3D =
      pcl::PointCloud<PointTypeXYZI>::Ptr(new pcl::PointCloud<PointTypeXYZI>());
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D =
      pcl::PointCloud<PointTypePose>::Ptr(new pcl::PointCloud<PointTypePose>());
  pcl::PointCloud<PointTypeXYZI>::Ptr copy_cloudKeyPoses3D =
      pcl::PointCloud<PointTypeXYZI>::Ptr(new pcl::PointCloud<PointTypeXYZI>());
  pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D =
      pcl::PointCloud<PointTypePose>::Ptr(new pcl::PointCloud<PointTypePose>());
  float transformTobeMapped[6];
  pcl::KdTreeFLANN<PointTypeXYZI>::Ptr kdtreeHistoryKeyPoses =
      pcl::KdTreeFLANN<PointTypeXYZI>::Ptr(
          new pcl::KdTreeFLANN<PointTypeXYZI>());
  bool aLoopIsClosed = false;
  map<int, int> loopIndexContainer;       // from new to old
  vector<pair<int, int>> loopIndexQueue;  // 回环索引队列
  vector<gtsam::Pose3> loopPoseQueue;     // 回环位姿队列
  vector<gtsam::noiseModel::Diagonal::shared_ptr>
      loopNoiseQueue;  // 回环噪声队列
  // deque<std_msgs::Float64MultiArray> loopInfoVec;
  double lidar_end_time = 0.0;
  float keyframeAddingDistThreshold;   // 判断是否为关键帧的距离阈值,yaml
  float keyframeAddingAngleThreshold;  // 判断是否为关键帧的角度阈值,yaml
  float surroundingKeyframeDensity;
  // gtsam
  gtsam::NonlinearFactorGraph gtSAMgraph;  // 实例化一个空的因子图
  gtsam::Values initialEstimate;
  gtsam::Values optimizedEstimate;
  gtsam::ISAM2 *isam;
  gtsam::Values isamCurrentEstimate;
  Eigen::MatrixXd poseCovariance;
  // pose_graph  saver
  std::fstream pgSaveStream;  // pg: pose-graph
  std::vector<std::string> edges_str;
  std::vector<std::string> vertices_str;
  // loop
  bool startFlag = true;
  bool loopClosureEnableFlag;
  float loopClosureFrequency;           // 回环检测频率
  float historyKeyframeSearchRadius;    // 回环检测radius kdtree搜索半径
  float historyKeyframeSearchTimeDiff;  // 帧间时间阈值
  int historyKeyframeSearchNum;         // 回环时多少个keyframe拼成submap
  float historyKeyframeFitnessScore;    // icp匹配阈值
  bool potentialLoopFlag = false;
  bool enable_gtsam = true;
  nav_msgs::msg::Path globalPath;
  ScanContext::SCManager scLoop;  // sc 类
  // giseop，Scan Context的输入格式
  enum class SCInputType { SINGLE_SCAN_FULL, MULTI_SCAN_FEAT };
  std::thread loopthread;

  ofstream fout_pre, fout_out, fout_pcd_pos, fout_points;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterGround;

  V3D euler_cur;

  LidarMeasureGroup LidarMeasures;
  StatesGroup _state;
  StatesGroup state_propagat;
  StatesGroup state_last;
  std::deque<nav_msgs::msg::Odometry> odomQueue;

  nav_msgs::msg::Path path;
  nav_msgs::msg::Odometry odomAftMapped;
  geometry_msgs::msg::Quaternion geoQuat;
  geometry_msgs::msg::PoseStamped msg_body_pose;

  PreprocessPtr p_pre;
  ImuProcessPtr p_imu;
  VoxelMapManagerPtr voxelmap_manager;
  std::shared_ptr<ZUPT> zupt;
  std::shared_ptr<WheelOdometryConstraint> wheel_odometry_;
  VIOManagerPtr vio_manager;
  rclcpp::CallbackGroup::SharedPtr callbackGroupLidar, callbackGroupImu,
      callbackGroupImg, callbackGroupOdom;

  rmw_qos_profile_t qos_profile_imu{RMW_QOS_POLICY_HISTORY_KEEP_LAST,
                                    2000,
                                    RMW_QOS_POLICY_RELIABILITY_RELIABLE,
                                    RMW_QOS_POLICY_DURABILITY_VOLATILE,
                                    RMW_QOS_DEADLINE_DEFAULT,
                                    RMW_QOS_LIFESPAN_DEFAULT,
                                    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
                                    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
                                    false};

  rclcpp::QoS qos_imu = rclcpp::QoS(
      rclcpp::QoSInitialization(qos_profile_imu.history, qos_profile_imu.depth),
      qos_profile_imu);

  rmw_qos_profile_t qos_profile_lidar{RMW_QOS_POLICY_HISTORY_KEEP_LAST,
                                      100,
                                      RMW_QOS_POLICY_RELIABILITY_RELIABLE,
                                      RMW_QOS_POLICY_DURABILITY_VOLATILE,
                                      RMW_QOS_DEADLINE_DEFAULT,
                                      RMW_QOS_LIFESPAN_DEFAULT,
                                      RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
                                      RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
                                      false};

  rclcpp::QoS qos_lidar =
      rclcpp::QoS(rclcpp::QoSInitialization(qos_profile_lidar.history,
                                            qos_profile_lidar.depth),
                  qos_profile_lidar);
  rmw_qos_profile_t qos_profile_img{RMW_QOS_POLICY_HISTORY_KEEP_LAST,
                                    100,
                                    RMW_QOS_POLICY_RELIABILITY_RELIABLE,
                                    RMW_QOS_POLICY_DURABILITY_VOLATILE,
                                    RMW_QOS_DEADLINE_DEFAULT,
                                    RMW_QOS_LIFESPAN_DEFAULT,
                                    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
                                    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
                                    false};
  rclcpp::QoS qos_img = rclcpp::QoS(
      rclcpp::QoSInitialization(qos_profile_img.history, qos_profile_img.depth),
      qos_profile_img);
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_pub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxel_pub;
  std::shared_ptr<rclcpp::SubscriptionBase> sub_pcl;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubRobotOdom;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pubLaserCloudFullRes,pubLaserUndistortCloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pubLaserLocalCloudFullRes;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubNormal;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSubVisualMap;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pubLaserCloudEffect;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudDyn;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pubLaserCloudDynRmed;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pubLaserCloudDynDbg;
  image_transport::Publisher pubImage;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
      mavros_pose_publisher;
  image_transport::Publisher pubDepth;
  image_transport::Publisher pubOverlay;

  rclcpp::TimerBase::SharedPtr imu_prop_timer;
  rclcpp::Node::SharedPtr node;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pubHistoryKeyFrames;  // 发布loop history keyframe submap
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubIcpKeyFrames;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pubRecentKeyFrames;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrame;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      pubLoopConstraintEdge;
  rclcpp::Service<fast_livo::srv::SaveMap>::SharedPtr srvSaveMap;
  // 新增发布器
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pubCamPose;

  int frame_num = 0;
  double aver_time_consu = 0;
  double aver_time_icp = 0;
  double aver_time_map_inre = 0;
  bool colmap_output_en = false;
  double lidar_time_offset = 0.0;
  std::mutex odoLock;
  std::shared_ptr<ERASOR> dect_ground;
  // 快照结构（发布时用，避免长时间持锁）
  struct VizSnapshot {
    // 只放发布需要的最小数据；可按需再加
    // pcl::PointCloud<PointTypeRGB> local_rgb_cloud_copy;
    PointCloudXYZRGB::Ptr local_rgb_cloud_copy;
    cv::Mat img_rgb;
    cv::Mat img_cp;
    double last_lidar_ts = 0.0;
    double image_time = 0.0;
    bool has_img = false;
    bool has_cloud = false;
    bool has_local_rgb = false;
    // 相机内参与尺寸（新增）
    double fx = 0, fy = 0, cx = 0, cy = 0;
    int width = 0, height = 0;

    VIOManagerPtr vio_mgr;  // 只读指针（假设生命周期贯穿程序）
    // 新增：相机位姿（T_f_w_）
    Eigen::Vector3d cam_t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond cam_q = Eigen::Quaterniond::Identity();
    bool has_cam_pose = false;
  };
  void publish_img_depth(const VizSnapshot &snap);

 private:
  // ----- 可视化快照队列（不丢帧） -----
  std::deque<VizSnapshot> viz_queue_;
  std::mutex viz_queue_mtx_;
  std::condition_variable viz_queue_cv_;
  // 最大容量：达到后生产者阻塞（不丢帧）
  size_t viz_queue_cap_{0};  // 0 表示不限制，可在参数里设
  bool viz_shutdown_{false};
  int viz_timer_period_ms_{50};  // 可在参数里设，单位毫秒
  // 定时器回调
  void vizTimerCb();
  rclcpp::TimerBase::SharedPtr viz_timer_;
  // 生成快照
  void enqueueSnapshot();          // 生产
  bool dequeueOne(VizSnapshot &);  // 消费单帧
  void processSnapshot(const VizSnapshot &snap);
  void makeSnapshot(VizSnapshot &snap);
};
#endif
