/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper.h"

#include <vikit/camera_loader.h>
#define USE_CIRCLE_DRAW 1
using namespace Sophus;
LIVMapper::LIVMapper(rclcpp::Node::SharedPtr &node, std::string node_name,
                     const rclcpp::NodeOptions &options)
    : node(std::make_shared<rclcpp::Node>(node_name, options)),
      extT(0, 0, 0),
      extR(M3D::Identity()) {
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);

  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  readParameters(this->node);
  VoxelMapConfig voxel_config;
  loadVoxelConfig(this->node, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  ds_ground_pc.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_body_ground.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_ng_w_wait_pub.reset(new PointCloudXYZI());
  pcl_g_w_wait_pub.reset(new PointCloudXYZI());
  pcl_body_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_global_rgb.reset(new PointCloudXYZRGB());
  local_rgb_cloud.reset(new PointCloudXYZRGB());
  pcl_wait_save_global.reset(new PointCloudXYZI());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  pcl_wait_save_intensity_cp.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config));
  local_rgb_cloud_buffer.clear();
  zupt.reset(new ZUPT());
  wheel_odometry_.reset(new WheelOdometryConstraint());
  // 初始化轮速外参
  // wheel_extrinsic_ekf_->initialize(
  //     M3D::Identity(), V3D(0.0, 0.0, 0.0),
  //     0.01 * Eigen::Matrix<double, 6, 6>::Identity(), 0.01 * M3D::Identity(),
  //     0.01 * Eigen::Matrix<double, 6, 6>::Identity());
  vio_manager.reset(new VIOManager());
  dect_ground.reset(new ERASOR());
  root_dir = ROOT_DIR;
  initializeFiles();
  initializeComponents(this->node);  // initialize components errors
  path.header.stamp = this->node->now();
  path.header.frame_id = "camera_init";
  // 回环检测线程
  // std::thread loopthread(&loopClosureThread);
  // loopthread = std::thread(&LIVMapper::loopClosureThread, this);
  // 例如 20Hz 发布（可调）
  // 构造函数里：
  if (depth_en) {
    viz_timer_ = this->node->create_wall_timer(
        std::chrono::milliseconds(viz_timer_period_ms_),
        std::bind(&LIVMapper::vizTimerCb, this));
  }

  auto saveMapService =
      [this](const std::shared_ptr<rmw_request_id_t> request_header,
             const std::shared_ptr<fast_livo::srv::SaveMap::Request> req,
             std::shared_ptr<fast_livo::srv::SaveMap::Response> res) -> void {
    (void)request_header;
    saveKeyFrame();
  };

  srvSaveMap = this->node->create_service<fast_livo::srv::SaveMap>(
      "fast_livo/save_map", saveMapService);
}

LIVMapper::~LIVMapper() {
  {
    std::lock_guard<std::mutex> lk(viz_queue_mtx_);
    viz_shutdown_ = true;
  }
  viz_queue_cv_.notify_all();
  if (loopthread.joinable()) loopthread.join();
}

void LIVMapper::readParameters(rclcpp::Node::SharedPtr &node) {
  // declare parameters
  auto try_declare = [node]<typename ParameterT>(
                         const std::string &name,
                         const ParameterT &default_value) {
    if (!node->has_parameter(name)) {
      return node->declare_parameter<ParameterT>(name, default_value);
    } else {
      return node->get_parameter(name).get_value<ParameterT>();
    }
  };

  // declare parameter
  try_declare.template operator()<std::string>("common.lid_topic",
                                               "/livox/lidar");
  try_declare.template operator()<std::string>("common.imu_topic",
                                               "/livox/imu");
  try_declare.template operator()<bool>("common.ros_driver_bug_fix", false);
  try_declare.template operator()<int>("common.img_en", 1);
  try_declare.template operator()<int>("common.lidar_en", 1);
  try_declare.template operator()<std::string>("common.img_topic",
                                               "/left_camera/image");
  try_declare.template operator()<std::string>("common.map_save_path", "/map");

  // zupt
  try_declare.template operator()<bool>("common.enable_zupt", true);
  //   // zupt_noise
  try_declare.template operator()<double>("common.zupt_noise", 0.1);
  // zupt_interval
  try_declare.template operator()<int>("common.zupt_interval", 10);
  // enable_wheel_odom
  try_declare.template operator()<bool>("wheel.enable_wheel_odom", false);
  // est_wheel_extrinsic
  try_declare.template operator()<bool>("wheel.est_wheel_extrinsic", false);

  try_declare.template operator()<bool>("vio.normal_en", true);
  try_declare.template operator()<bool>("vio.inverse_composition_en", false);
  try_declare.template operator()<int>("vio.max_iterations", 5);
  try_declare.template operator()<int>("vio.img_point_cov", 100);
  try_declare.template operator()<bool>("vio.raycast_en", false);
  try_declare.template operator()<bool>("vio.exposure_estimate_en", true);
  try_declare.template operator()<double>("vio.inv_expo_cov", 0.1);
  try_declare.template operator()<int>("vio.grid_size", 5);
  try_declare.template operator()<int>("vio.grid_n_height", 17);
  try_declare.template operator()<int>("vio.patch_pyrimid_level", 4);
  try_declare.template operator()<int>("vio.patch_size", 8);
  try_declare.template operator()<int>("vio.outlier_threshold", 100);
  try_declare.template operator()<double>("time_offset.exposure_time_init",
                                          0.0);
  try_declare.template operator()<double>("time_offset.img_time_offset", 0.0);
  try_declare.template operator()<bool>("uav.imu_rate_odom", false);
  try_declare.template operator()<bool>("uav.gravity_align_en", false);
  try_declare.template operator()<double>("time_offset.lidar_time_offset", 0.0);
  try_declare.template operator()<std::string>("evo.seq_name", "01");
  try_declare.template operator()<bool>("evo.pose_output_en", false);
  try_declare.template operator()<double>("imu.gyr_cov", 1.0);
  try_declare.template operator()<double>("imu.acc_cov", 1.0);
  try_declare.template operator()<int>("imu.imu_int_frame", 30);
  try_declare.template operator()<bool>("imu.imu_en", true);
  try_declare.template operator()<bool>("imu.gravity_est_en", true);
  try_declare.template operator()<bool>("imu.ba_bg_est_en", true);

  try_declare.template operator()<double>("preprocess.blind", 0.01);
  // max_range
  try_declare.template operator()<double>("preprocess.max_range", 100);
  try_declare.template operator()<double>("preprocess.filter_size_surf", 0.5);
  try_declare.template operator()<int>("preprocess.lidar_type", AVIA);
  try_declare.template operator()<int>("preprocess.scan_line", 6);
  try_declare.template operator()<int>("preprocess.point_filter_num", 3);
  try_declare.template operator()<int>("preprocess.scan_rate", 10);
  try_declare.template operator()<bool>("preprocess.feature_extract_enabled",
                                        false);
  try_declare.template operator()<bool>("preprocess.img_filter_en", false);
  // img_filter_num
  try_declare.template operator()<int>("preprocess.img_filter_fre", 1.0);

  try_declare.template operator()<int>("pcd_save.interval", -1);
  try_declare.template operator()<bool>("pcd_save.pcd_save_en", false);
  try_declare.template operator()<bool>("pcd_save.colmap_output_en", false);
  try_declare.template operator()<double>("pcd_save.filter_size_pcd", 0.10);
  try_declare.template operator()<vector<double>>("extrin_calib.extrinsic_T",
                                                  vector<double>{});
  try_declare.template operator()<vector<double>>("extrin_calib.extrinsic_R",
                                                  vector<double>{});
  try_declare.template operator()<vector<double>>("extrin_calib.Pcl",
                                                  vector<double>{});
  try_declare.template operator()<vector<double>>("extrin_calib.Rcl",
                                                  vector<double>{});
  // p_wheel_to_imu
  try_declare.template operator()<vector<double>>(
      "wheel.p_wheel_to_imu", vector<double>{0.0, 0.0, 0.0});
  try_declare.template operator()<vector<double>>(
      "wheel.R_wheel_to_imu",
      vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
  // wheel_ext_cov_
  try_declare.template operator()<vector<double>>(
      "wheel.wheel_ext_cov",
      vector<double>{0.01, 0.01, 0.01, 0.01, 0.01, 0.01});
  // wheel_mea_cov_
  try_declare.template operator()<vector<double>>(
      "wheel.wheel_mea_cov", vector<double>{0.01, 0.01, 0.01});
  try_declare.template operator()<double>("debug.plot_time", -10);
  try_declare.template operator()<int>("debug.frame_cnt", 6);

  try_declare.template operator()<double>("publish.blind_rgb_points", 0.01);
  try_declare.template operator()<int>("publish.pub_scan_num", 1);
  try_declare.template operator()<bool>("publish.pub_effect_point_en", false);
  try_declare.template operator()<bool>("publish.dense_map_en", false);
  try_declare.template operator()<bool>("publish.pub_rgb_cloud_en", true);
  try_declare.template operator()<bool>("publish.depth_en", true);
  try_declare.template operator()<int>("publish.viz_timer_ms", 50);
  // loop clousre
  try_declare.template operator()<bool>("loop.loop_closure_enable_flag", true);
  try_declare.template operator()<float>("loop.loop_closure_frequency", 1.0);
  try_declare.template operator()<float>("loop.history_keyframe_search_radius",
                                         10.0);
  try_declare.template operator()<float>(
      "loop.history_keyframe_search_time_diff", 30.0);
  try_declare.template operator()<int>("loop.history_keyframe_search_num", 10);
  try_declare.template operator()<float>("loop.history_keyframe_fitness_score",
                                         0.3);
  try_declare.template operator()<float>("mapping.keyframeAddingDistThreshold",
                                         1.0);
  try_declare.template operator()<float>("mapping.keyframeAddingAngleThreshold",
                                         0.2);
  // enable_gtsam
  try_declare.template operator()<bool>("mapping.enable_gtsam", true);
  // get parameter
  this->node->get_parameter("common.lid_topic", lid_topic);
  this->node->get_parameter("common.imu_topic", imu_topic);
  this->node->get_parameter("common.ros_driver_bug_fix", ros_driver_fix_en);
  this->node->get_parameter("common.img_en", img_en);
  this->node->get_parameter("common.lidar_en", lidar_en);
  this->node->get_parameter("common.img_topic", img_topic);
  this->node->get_parameter("common.map_save_path", map_save_path);
  this->node->get_parameter("common.enable_zupt", enable_zupt);
  // zupt_noise
  this->node->get_parameter("common.zupt_noise", zupt_noise);
  // zupt_interval
  this->node->get_parameter("common.zupt_interval", zupt_interval);
  // enable_wheel_odom
  this->node->get_parameter("wheel.enable_wheel_odom", enable_wheel_odom);
  // est_wheel_extrinsic
  this->node->get_parameter("wheel.est_wheel_extrinsic", est_wheel_extrinsic);
  this->node->get_parameter("vio.normal_en", normal_en);
  this->node->get_parameter("vio.inverse_composition_en",
                            inverse_composition_en);
  this->node->get_parameter("vio.max_iterations", max_iterations);
  this->node->get_parameter("vio.img_point_cov", IMG_POINT_COV);
  this->node->get_parameter("vio.raycast_en", raycast_en);
  this->node->get_parameter("vio.exposure_estimate_en", exposure_estimate_en);
  this->node->get_parameter("vio.inv_expo_cov", inv_expo_cov);
  this->node->get_parameter("vio.grid_size", grid_size);
  this->node->get_parameter("vio.grid_n_height", grid_n_height);
  this->node->get_parameter("vio.patch_pyrimid_level", patch_pyrimid_level);
  this->node->get_parameter("vio.patch_size", patch_size);
  this->node->get_parameter("vio.outlier_threshold", outlier_threshold);
  this->node->get_parameter("time_offset.exposure_time_init",
                            exposure_time_init);
  this->node->get_parameter("time_offset.img_time_offset", img_time_offset);
  this->node->get_parameter("time_offset.lidar_time_offset", lidar_time_offset);
  this->node->get_parameter("uav.imu_rate_odom", imu_prop_enable);
  this->node->get_parameter("uav.gravity_align_en", gravity_align_en);

  this->node->get_parameter("evo.seq_name", seq_name);
  this->node->get_parameter("evo.pose_output_en", pose_output_en);
  this->node->get_parameter("imu.gyr_cov", gyr_cov);
  this->node->get_parameter("imu.acc_cov", acc_cov);
  this->node->get_parameter("imu.imu_int_frame", imu_int_frame);
  this->node->get_parameter("imu.imu_en", imu_en);
  this->node->get_parameter("imu.gravity_est_en", gravity_est_en);
  this->node->get_parameter("imu.ba_bg_est_en", ba_bg_est_en);

  this->node->get_parameter("preprocess.blind", p_pre->blind);
  this->node->get_parameter("preprocess.max_range", p_pre->max_range);
  this->node->get_parameter("preprocess.filter_size_surf",
                            filter_size_surf_min);
  this->node->get_parameter("preprocess.lidar_type", p_pre->lidar_type);
  this->node->get_parameter("preprocess.scan_line", p_pre->N_SCANS);
  this->node->get_parameter("preprocess.scan_rate", p_pre->SCAN_RATE);
  this->node->get_parameter("preprocess.point_filter_num",
                            p_pre->point_filter_num);
  this->node->get_parameter("preprocess.feature_extract_enabled",
                            p_pre->feature_enabled);
  // img_filter_en
  this->node->get_parameter("preprocess.img_filter_en", img_filter_en);
  this->node->get_parameter("preprocess.img_filter_fre", img_filter_fre);
  this->node->get_parameter("pcd_save.interval", pcd_save_interval);
  this->node->get_parameter("pcd_save.pcd_save_en", pcd_save_en);
  this->node->get_parameter("pcd_save.colmap_output_en", colmap_output_en);
  this->node->get_parameter("pcd_save.filter_size_pcd", filter_size_pcd);
  this->node->get_parameter("extrin_calib.extrinsic_T", extrinT);
  this->node->get_parameter("extrin_calib.extrinsic_R", extrinR);
  this->node->get_parameter("extrin_calib.Pcl", cameraextrinT);
  this->node->get_parameter("extrin_calib.Rcl", cameraextrinR);
  // p_wheel_to_imu
  this->node->get_parameter("wheel.p_wheel_to_imu", p_wheel_to_imu);
  this->node->get_parameter("wheel.R_wheel_to_imu", R_wheel_to_imu);
  // wheel_ext_cov_
  this->node->get_parameter("wheel.wheel_ext_cov", wheel_ext_cov_);
  // wheel_mea_cov_
  this->node->get_parameter("wheel.wheel_mea_cov", wheel_mea_cov_);

  this->node->get_parameter("debug.plot_time", plot_time);
  this->node->get_parameter("debug.frame_cnt", frame_cnt);

  this->node->get_parameter("publish.blind_rgb_points", blind_rgb_points);
  this->node->get_parameter("publish.pub_scan_num", pub_scan_num);
  this->node->get_parameter("publish.pub_effect_point_en", pub_effect_point_en);
  this->node->get_parameter("publish.dense_map_en", dense_map_en);
  this->node->get_parameter("publish.pub_rgb_cloud_en", pub_rgb_cloud_en);
  this->node->get_parameter("publish.depth_en", depth_en);
  this->node->get_parameter("publish.rgb_cloud_interval", rgb_cloud_interval);
  this->node->get_parameter("publish.viz_timer_ms", viz_timer_period_ms_);

  // loop clousre
  this->node->get_parameter("loop.loop_closure_enable_flag",
                            loopClosureEnableFlag);
  this->node->get_parameter("loop.loop_closure_frequency",
                            loopClosureFrequency);
  this->node->get_parameter("loop.history_keyframe_search_radius",
                            historyKeyframeSearchRadius);
  this->node->get_parameter("loop.history_keyframe_search_time_diff",
                            historyKeyframeSearchTimeDiff);
  this->node->get_parameter("loop.history_keyframe_search_num",
                            historyKeyframeSearchNum);
  this->node->get_parameter("loop.history_keyframe_fitness_score",
                            historyKeyframeFitnessScore);
  this->node->get_parameter("mapping.keyframeAddingDistThreshold",
                            keyframeAddingDistThreshold);
  this->node->get_parameter("mapping.keyframeAddingAngleThreshold",
                            keyframeAddingAngleThreshold);
  // enable_gtsam
  this->node->get_parameter("mapping.enable_gtsam", enable_gtsam);
  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

void LIVMapper::initializeComponents(rclcpp::Node::SharedPtr &node) {
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min,
                                 filter_size_surf_min);
  downSizeFilterGround.setLeafSize(filter_size_surf_min * 0.5,
                                   filter_size_surf_min * 0.5,
                                   filter_size_surf_min * 0.5);
  // extrinT.assign({0.04165, 0.02326, -0.0284});
  // extrinR.assign({1, 0, 0, 0, 1, 0, 0, 0, 1});
  // cameraextrinT.assign({0.0194384, 0.104689,-0.0251952});
  // cameraextrinR.assign({0.00610193,-0.999863,-0.0154172,-0.00615449,0.0153796,-0.999863,0.999962,0.00619598,-0.0060598});

  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);

  if (!vk::camera_loader::loadFromRosNs(this->node, "camera", vio_manager->cam))
    throw std::runtime_error("Camera model not correctly specified.");

  vio_manager->grid_size = grid_size;
  vio_manager->patch_size = patch_size;
  vio_manager->outlier_threshold = outlier_threshold;
  vio_manager->setImuToLidarExtrinsic(extT, extR);
  vio_manager->setLidarToCameraExtrinsic(cameraextrinR, cameraextrinT);
  vio_manager->state = &_state;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->exposure_estimate_en = exposure_estimate_en;
  vio_manager->colmap_output_en = colmap_output_en;
  vio_manager->max_voxel_num_ =
      voxelmap_manager->config_setting_.MAX_VOXEL_NUM * 0.25;
  vio_manager->initializeVIO();

  // wheel_odometry_
  wheel_odometry_->initialize(wheel_mea_cov_, R_wheel_to_imu, p_wheel_to_imu);
  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_imu_init_frame_num(imu_int_frame);
  // ISAM2参数
  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  isam = new gtsam::ISAM2(parameters);
  if (!imu_en) p_imu->disable_imu();
  if (!gravity_est_en) p_imu->disable_gravity_est();
  if (!ba_bg_est_en) p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();

  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;
}

void LIVMapper::initializeFiles() {
  // if (pcd_save_en && colmap_output_en)
  // {
  //     const std::string folderPath = std::string(ROOT_DIR) +
  //     "scripts/colmap_output.sh";

  //     std::string chmodCommand = "chmod +x " + folderPath;

  //     int chmodRet = system(chmodCommand.c_str());
  //     if (chmodRet != 0) {
  //         std::cerr << "Failed to set execute permissions for the script." <<
  //         std::endl; return;
  //     }

  //     int executionRet = system(folderPath.c_str());
  //     if (executionRet != 0) {
  //         std::cerr << "Failed to execute the script." << std::endl;
  //         return;
  //     }
  // }
  if (colmap_output_en)
    fout_points.open(std::string(ROOT_DIR) + "Log/Colmap/sparse/0/points3D.txt",
                     std::ios::out);
  if (pcd_save_interval > 0) {
    fout_pcd_pos.open(std::string(ROOT_DIR) + "Log/PCD/scans_pos.json",
                      std::ios::out);
  }
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
}

void LIVMapper::initializeSubscribersAndPublishers(
    rclcpp::Node::SharedPtr &node, image_transport::ImageTransport &it_) {
  image_transport::ImageTransport it(this->node);
  callbackGroupLidar = this->node->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
  callbackGroupImu = this->node->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
  callbackGroupOdom = this->node->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);

  auto lidarOpt = rclcpp::SubscriptionOptions();
  lidarOpt.callback_group = callbackGroupLidar;
  auto imuOpt = rclcpp::SubscriptionOptions();
  imuOpt.callback_group = callbackGroupImu;
  auto imgOpt = rclcpp::SubscriptionOptions();
  imgOpt.callback_group = callbackGroupImg;
  auto odomOpt = rclcpp::SubscriptionOptions();
  imuOpt.callback_group = callbackGroupOdom;

  if (p_pre->lidar_type == AVIA) {
    sub_pcl =
        this->node->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            lid_topic, qos_lidar,
            std::bind(&LIVMapper::livox_pcl_cbk, this, std::placeholders::_1),
            lidarOpt);
  } else {
    sub_pcl = this->node->create_subscription<sensor_msgs::msg::PointCloud2>(
        lid_topic, 200000,
        std::bind(&LIVMapper::standard_pcl_cbk, this, std::placeholders::_1));
  }
  sub_imu = this->node->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, qos_imu,
      std::bind(&LIVMapper::imu_cbk, this, std::placeholders::_1), imuOpt);
  sub_img = this->node->create_subscription<sensor_msgs::msg::Image>(
      img_topic, qos_img,
      std::bind(&LIVMapper::img_cbk, this, std::placeholders::_1), imgOpt);
  subOdom = this->node->create_subscription<nav_msgs::msg::Odometry>(
      "/robot_odom", qos_imu,
      std::bind(&LIVMapper::odometryHandler, this, std::placeholders::_1));
  pubRobotOdom = this->node->create_publisher<nav_msgs::msg::Odometry>(
      "/robot_odom_convert", 20000);

  pubLaserCloudFullRes =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/cloud_registered", 100);
  pubLaserUndistortCloud =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/undistort_cloud", 100);
  pubLaserLocalCloudFullRes =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/local_cloud_registered", 100);
  pubNormal =
      this->node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "/visualization_marker", 100);
  pubSubVisualMap = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/cloud_visual_sub_map_before", 100);
  pubLaserCloudEffect =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/cloud_effected", 100);
  pubLaserCloudMap =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map",
                                                                  100);
  pubOdomAftMapped = this->node->create_publisher<nav_msgs::msg::Odometry>(
      "/aft_mapped_to_init", 10);
  pubPath = this->node->create_publisher<nav_msgs::msg::Path>("/path", 10);
  plane_pub = this->node->create_publisher<visualization_msgs::msg::Marker>(
      "/planner_normal", 1);
  voxel_pub =
      this->node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "/voxels", 1);
  pubLaserCloudDyn =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>("/dyn_obj",
                                                                  100);
  pubLaserCloudDynRmed =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/dyn_obj_removed", 100);
  pubLaserCloudDynDbg =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/dyn_obj_dbg_hist", 100);
  mavros_pose_publisher =
      this->node->create_publisher<geometry_msgs::msg::PoseStamped>(
          "/mavros/vision_pose/pose", 10);

  // 发布回环匹配关键帧局部map
  pubHistoryKeyFrames =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/icp_loop_closure_history_cloud", 1);
  // 发布当前关键帧经过回环优化后的位姿变换之后的特征点云
  pubIcpKeyFrames = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/icp_loop_closure_corrected_cloud", 1);
  // 发布回环边,rviz中表现为回环帧之间的连线
  pubLoopConstraintEdge =
      this->node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "/loop_closure_constraints", 1);

  pubImage = it.advertise("/rgb_img", 10);
  pubDepth = it.advertise("/depth_img", 10);
  pubOverlay = it.advertise("/overlay_img", 10);
  pubImuPropOdom = this->node->create_publisher<nav_msgs::msg::Odometry>(
      "/LIVO2/imu_propagate", 10000);
  imu_prop_timer = this->node->create_wall_timer(
      0.004s, std::bind(&LIVMapper::imu_prop_callback, this));
  voxelmap_manager->voxel_map_pub_ =
      this->node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "/planes", 10000);
  pubCamPose = this->node->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/camera_pose", 10);
}

void LIVMapper::handleFirstFrame() {
  if (!is_first_frame) {
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    p_imu->first_lidar_time = _first_lidar_time;  // Only for IMU data log
    is_first_frame = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment() {
  if (!p_imu->imu_need_init && !gravity_align_finished) {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(_state.gravity);
    Eigen::Quaterniond G_q_I0 = Eigen::Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

nav_msgs::msg::Odometry LIVMapper::odomToLidarConverter(
    const nav_msgs::msg::Odometry &odom_in) {
  static bool first_odom = true;
  static Eigen::Vector3d first_pos;
  static Eigen::Quaterniond first_quat;

  // 右手坐标系 (x前, y左, z上) -> 激光雷达坐标系 (x前, y右, z下)
  Eigen::Vector3d pos_odom(odom_in.pose.pose.position.x,
                           odom_in.pose.pose.position.y,
                           odom_in.pose.pose.position.z);
  Eigen::Vector3d pos_lidar(pos_odom.x(), -pos_odom.y(), -pos_odom.z());

  Eigen::Quaterniond q_odom(
      odom_in.pose.pose.orientation.w, odom_in.pose.pose.orientation.x,
      odom_in.pose.pose.orientation.y, odom_in.pose.pose.orientation.z);
  Eigen::Quaterniond q_lidar(q_odom.w(), q_odom.x(), -q_odom.y(), -q_odom.z());

  if (first_odom) {
    first_odom = false;
    first_pos = pos_lidar;
    first_quat = q_lidar;
    std::cout << "第一帧odom (转换后): " << first_pos.x() << " "
              << first_pos.y() << " " << first_pos.z() << std::endl;
  }

  nav_msgs::msg::Odometry odom_out = odom_in;

  // 位置变换
  Eigen::Vector3d relative_pos = first_quat.inverse() * (pos_lidar - first_pos);
  odom_out.pose.pose.position.x = relative_pos.x();
  odom_out.pose.pose.position.y = relative_pos.y();
  odom_out.pose.pose.position.z = relative_pos.z();

  // 姿态变换
  Eigen::Quaterniond relative_quat = first_quat.inverse() * q_lidar;
  odom_out.pose.pose.orientation.x = relative_quat.x();
  odom_out.pose.pose.orientation.y = relative_quat.y();
  odom_out.pose.pose.orientation.z = relative_quat.z();
  odom_out.pose.pose.orientation.w = relative_quat.w();

  return odom_out;
}

void LIVMapper::odometryHandler(
    const nav_msgs::msg::Odometry::SharedPtr odometryMsg) {
  std::lock_guard<std::mutex> lock(odoLock);
  nav_msgs::msg::Odometry thisOdom = *odometryMsg;
  odomQueue.emplace_back(thisOdom);
  // thisOdom.header.stamp = odometryMsg->header.stamp;
  // thisOdom.header.frame_id = "camera_init";
  // thisOdom.child_frame_id = "aft_mapped";
  // pubRobotOdom->publish(thisOdom);
}

void LIVMapper::processImu() {
  // double t0 = omp_get_wtime();
  state_last = _state;  // 保存上一次的状态
  p_imu->Process2(LidarMeasures, _state, feats_undistort);

  if (gravity_align_en) gravityAlignment();

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;
  zupt->setState(_state);

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() <<
  // std::endl; std::cout << "[ Mapping ] predict cov: " <<
  // _state.cov.diagonal().transpose() << std::endl; std::cout << "[ Mapping ]
  // predict sta: " << state_propagat.pos_end.transpose() <<
  // state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping() {
  switch (LidarMeasures.lio_vio_flg) {
    case VIO: {
      handleVIO();
      if (enable_gtsam) {
        getCurPose(_state);  // 更新transformTobeMapped
        saveKeyFramesAndFactor();
        correctPoses();
      }
    
      break;
    }
    case LIO:
    case LO: {
      handleLIO();
      break;
    }
  }
}

void LIVMapper::handleVIO() {
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << std::setw(20)
           << LidarMeasures.last_lio_update_time - _first_lidar_time << " "
           << euler_cur.transpose() * 57.3 << " " << _state.pos_end.transpose()
           << " " << _state.vel_end.transpose() << " "
           << _state.bias_g.transpose() << " " << _state.bias_a.transpose()
           << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;

  if (pcl_w_wait_pub->empty() || (pcl_w_wait_pub == nullptr)) {
    std::cout << "[ VIO ] No point!!!" << std::endl;
    return;
  }

  std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub->points.size()
            << std::endl;

  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) -
           plot_time) < (frame_cnt / 2 * 0.1)) {
    vio_manager->plot_flag = true;
  } else {
    vio_manager->plot_flag = false;
  }

  vio_manager->processFrame(LidarMeasures.measures.back().img, _pv_list,
                            voxelmap_manager->voxel_map_,
                            LidarMeasures.measures.back().vio_time);
  if (imu_prop_enable) {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  Eigen::Quaterniond quat(_state.rot_end);
  geoQuat.w = quat.w();
  geoQuat.x = quat.x();
  geoQuat.y = quat.y();
  geoQuat.z = quat.z();
  publish_frame_world(pubLaserCloudFullRes, vio_manager);
  publish_odometry(pubOdomAftMapped);

  // publish rgb, depth, and camera pose
  if (depth_en)
    enqueueSnapshot();
  else
    publish_img_rgb(pubImage, vio_manager);

  // publish_img_rgb(pubImage, vio_manager);
}
// TODO: rviz展示回环边, can be used for relocalization
void LIVMapper::visualizeLoopClosure() {
  // 使用 ROS 2 的时间戳
  rclcpp::Time timeLaserInfoStamp = this->node->now();
  std::string odometryFrame = "camera_init";

  if (loopIndexContainer.empty()) return;

  visualization_msgs::msg::MarkerArray markerArray;

  // 回环顶点
  visualization_msgs::msg::Marker markerNode;
  markerNode.header.frame_id = odometryFrame;
  markerNode.header.stamp = timeLaserInfoStamp;
  markerNode.action = visualization_msgs::msg::Marker::ADD;
  markerNode.type = visualization_msgs::msg::Marker::SPHERE_LIST;
  markerNode.ns = "loop_nodes";
  markerNode.id = 0;
  markerNode.pose.orientation.w = 1.0;
  markerNode.scale.x = 0.3;
  markerNode.scale.y = 0.3;
  markerNode.scale.z = 0.3;
  markerNode.color.r = 0.0;
  markerNode.color.g = 0.8;
  markerNode.color.b = 1.0;
  markerNode.color.a = 1.0;

  // 回环边
  visualization_msgs::msg::Marker markerEdge;
  markerEdge.header.frame_id = odometryFrame;
  markerEdge.header.stamp = timeLaserInfoStamp;
  markerEdge.action = visualization_msgs::msg::Marker::ADD;
  markerEdge.type = visualization_msgs::msg::Marker::LINE_LIST;
  markerEdge.ns = "loop_edges";
  markerEdge.id = 1;
  markerEdge.pose.orientation.w = 1.0;
  markerEdge.scale.x = 0.1;
  markerEdge.color.r = 0.9;
  markerEdge.color.g = 0.9;
  markerEdge.color.b = 0.0;
  markerEdge.color.a = 1.0;

  // 遍历回环
  for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end();
       ++it) {
    int key_cur = it->first;
    int key_pre = it->second;

    geometry_msgs::msg::Point p;
    p.x = copy_cloudKeyPoses6D->points[key_cur].x;
    p.y = copy_cloudKeyPoses6D->points[key_cur].y;
    p.z = copy_cloudKeyPoses6D->points[key_cur].z;
    markerNode.points.emplace_back(p);
    markerEdge.points.emplace_back(p);

    p.x = copy_cloudKeyPoses6D->points[key_pre].x;
    p.y = copy_cloudKeyPoses6D->points[key_pre].y;
    p.z = copy_cloudKeyPoses6D->points[key_pre].z;
    markerNode.points.emplace_back(p);
    markerEdge.points.emplace_back(p);
  }

  markerArray.markers.emplace_back(markerNode);
  markerArray.markers.emplace_back(markerEdge);

  // 发布回环约束
  pubLoopConstraintEdge->publish(markerArray);
}

// Get the Cur Pose
// object将更新的pose赋值到transformTobeMapped中来表示位姿（因子图要用）
void LIVMapper::getCurPose(StatesGroup cur_state) {
  // 欧拉角是没有群的性质,所以从SO3还是一般的rotation matrix,转换过来的结果一样
  // Eigen::Vector3d eulerAngle = cur_state.rot.matrix().eulerAngles(2, 1, 0);
  // // yaw-pitch-roll,单位:弧度
  // std::cout << "getCurPose: " << cur_state.rot_end << std::endl;
  Eigen::Vector3d eulerAngle =
      cur_state.rot_end.eulerAngles(2, 1, 0);  // yaw-pitch-roll,单位:弧度

  transformTobeMapped[0] = eulerAngle(2);         // roll
  transformTobeMapped[1] = eulerAngle(1);         // pitch
  transformTobeMapped[2] = eulerAngle(0);         // yaw
  transformTobeMapped[3] = cur_state.pos_end(0);  // x
  transformTobeMapped[4] = cur_state.pos_end(1);  // y
  transformTobeMapped[5] = cur_state.pos_end(2);  // z

  // if(tollerance_en){  // TODO: human constraint in z and roll oitch
  //     transformTobeMapped[0] =
  //     constraintTransformation(transformTobeMapped[0], rotation_tollerance);
  //     // roll transformTobeMapped[1] =
  //     constraintTransformation(transformTobeMapped[1], rotation_tollerance);
  //     // pitch transformTobeMapped[5] =
  //     constraintTransformation(transformTobeMapped[5], z_tollerance);
  // }
}
// 计算当前帧与前一帧位姿变换,如果变化太小,不设为关键帧,反之设为关键帧
bool LIVMapper::saveFrame() {
  if (pcl_body_wait_pub->points.size() == 0) return false;
  if (cloudKeyPoses3D->points.empty()) return true;

  // 前一帧位姿,注:最开始没有的时候,在函数extractCloud里面有
  Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
  // 当前帧位姿
  Eigen::Affine3f transFinal = trans2Affine3f(transformTobeMapped);
  // 位姿变换增量
  Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
  float x, y, z, roll, pitch, yaw;
  // pcl::getTranslationAndEulerAngles是根据仿射矩阵计算x,y,z,roll,pitch,yaw
  pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch,
                                    yaw);  // 获取上一帧相对当前帧的位姿

  // 旋转和平移量都较小,当前帧不设为关键帧
  if (abs(roll) < keyframeAddingAngleThreshold &&
      abs(pitch) < keyframeAddingAngleThreshold &&
      abs(yaw) < keyframeAddingAngleThreshold &&
      sqrt(x * x + y * y + z * z) < keyframeAddingDistThreshold)
    return false;
  std::cout << "Now is keyframe" << endl;
  return true;
}

// 添加激光里程计因子
void LIVMapper::addOdomFactor() {
  // 如果是第一帧
  if (cloudKeyPoses3D->points.empty()) {
    std::cout << "First frame addOdomFactor" << std::endl;
    // 给出一个噪声模型,也就是协方差矩阵
    gtsam::noiseModel::Diagonal::shared_ptr priorNoise =
        gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12)
                .finished());
    // 加入先验因子PriorFactor,固定这个顶点,对第0个节点增加约束
    gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(
        0, trans2gtsamPose(transformTobeMapped), priorNoise));
    // 节点设置初始值,将这个顶点的值加入初始值中
    initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));

    // 变量节点设置初始值
    writeVertex(0, trans2gtsamPose(transformTobeMapped), vertices_str);
    std::cout << "First frame addOdomFactor done" << endl;
  }
  // 不是第一帧,增加帧间约束
  else {
    // 添加激光里程计因子
    std::cout << "addOdomFactor" << std::endl;
    gtsam::noiseModel::Diagonal::shared_ptr odometryNoise =
        gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4)
                .finished());
    gtsam::Pose3 poseFrom =
        pclPointTogtsamPose3(cloudKeyPoses6D->points.back());    // 上一个位姿
    gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);  // 当前位姿
    gtsam::Pose3 relPose = poseFrom.between(poseTo);
    // 参数:前一帧id;当前帧id;前一帧与当前帧的位姿变换poseFrom.between(poseTo)
    // = poseFrom.inverse()*poseTo;噪声协方差;
    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(
        cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(),
        poseFrom.between(poseTo), odometryNoise));
    // 变量节点设置初始值,将这个顶点的值加入初始值中
    initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);

    writeVertex(cloudKeyPoses3D->size(), poseTo, vertices_str);
    writeEdge({cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size()}, relPose,
              edges_str);
  }
}

// 添加回环因子
void LIVMapper::addLoopFactor() {
  if (loopIndexQueue.empty()) {
    std::cout << "no loop" << endl;
    return;
  }

  // 把队列里面所有的回环约束添加进行
  std::cout << "addLoopFactor" << endl;
  for (int i = 0; i < (int)loopIndexQueue.size(); ++i) {
    int indexFrom = loopIndexQueue[i].first;  // 回环帧索引
    int indexTo = loopIndexQueue[i].second;   // 当前帧索引
    // 两帧的位姿变换（帧间约束）
    gtsam::Pose3 poseBetween = loopPoseQueue[i];
    // 回环的置信度就是icp的得分？
    gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
    // 加入约束
    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(
        indexFrom, indexTo, poseBetween, noiseBetween));

    writeEdge({indexFrom, indexTo}, poseBetween, edges_str);
  }
  // 清空回环相关队列
  loopIndexQueue.clear();  // it's very necessary
  loopPoseQueue.clear();
  loopNoiseQueue.clear();

  aLoopIsClosed = true;
}

/**
 * @brief
 * 设置当前帧为关键帧并执行因子图优化
 * 1、计算当前帧与前一帧位姿变换,如果变化太小,不设为关键帧,反之设为关键帧
 * 2、添加激光里程计因子、回环因子
 * 3、执行因子图优化
 * 4、得到当前帧优化后位姿,位姿协方差
 * 5、添加cloudKeyPoses3D,cloudKeyPoses6D,更新transformTobeMapped,添加当前关键帧的角点、平面点集合
 */
void LIVMapper::saveKeyFramesAndFactor() {
  // 计算当前帧与前一帧位姿变换,如果变化太小,不设为关键帧,反之设为关键帧
  if (saveFrame() == false) return;

  // 激光里程计因子(from fast-lio)
  addOdomFactor();

  // 回环因子
  addLoopFactor();

  // 执行优化,更新图模型
  isam->update(gtSAMgraph, initialEstimate);
  isam->update();

  // TODO: 如果加入了回环约束,isam需要进行更多次的优化
  if (aLoopIsClosed == true) {
    cout << "pose is upated by isam " << endl;
    isam->update();
    isam->update();
    isam->update();
    isam->update();
  }

  // update之后要清空一下保存的因子图,注:清空不会影响优化,ISAM保存起来了
  gtSAMgraph.resize(0);
  initialEstimate.clear();

  PointTypeXYZI thisPose3D;
  PointTypePose thisPose6D;
  gtsam::Pose3 latestEstimate;

  // 通过接口获得所以变量的优化结果
  isamCurrentEstimate = isam->calculateBestEstimate();
  // 取出优化后的当前帧位姿结果
  latestEstimate =
      isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size() - 1);
  // 位移信息取出来保存进clouKeyPoses3D这个结构中
  thisPose3D.x = latestEstimate.translation().x();
  thisPose3D.y = latestEstimate.translation().y();
  thisPose3D.z = latestEstimate.translation().z();
  std::cout << "get isam result" << endl;
  // 其中索引作为intensity
  thisPose3D.intensity =
      cloudKeyPoses3D->size();  // 使用intensity作为该帧点云的index
  cloudKeyPoses3D->emplace_back(thisPose3D);  // 新关键帧帧放入队列中
  // 同样6D的位姿也保存下来
  thisPose6D.x = thisPose3D.x;
  thisPose6D.y = thisPose3D.y;
  thisPose6D.z = thisPose3D.z;

  // TODO:
  thisPose3D.z = 0.0;  // FIXME: right?

  thisPose6D.intensity = thisPose3D.intensity;
  thisPose6D.roll = latestEstimate.rotation().roll();
  thisPose6D.pitch = latestEstimate.rotation().pitch();
  thisPose6D.yaw = latestEstimate.rotation().yaw();
  // thisPose6D.time = lidar_end_time;
  thisPose6D.time = LidarMeasures.measures.back().vio_time;
  cloudKeyPoses6D->emplace_back(thisPose6D);
  // 保存当前位姿的位姿协方差（置信度）
  poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

  // // ESKF状态和方差更新
  StatesGroup state_updated = _state;  // 获取cur_pose(还没修正)
  Eigen::Vector3d pos(latestEstimate.translation().x(),
                      latestEstimate.translation().y(),
                      latestEstimate.translation().z());

  Eigen::Matrix3d rotationMatrix = latestEstimate.rotation().matrix();

  // 更新状态量
  state_updated.pos_end = pos;
  state_updated.rot_end = rotationMatrix;
  // state_point = state_updated;  //
  // 对state_point进行更新,state_point可视化用到

  if (aLoopIsClosed == true)
    _state = state_updated;  // 更新状态量,如果没有回环,不更新

  pcl::PointCloud<PointTypeXYZI>::Ptr thisSurfKeyFrame(
      new pcl::PointCloud<PointTypeXYZI>());
  pcl::PointCloud<PointTypeXYZI>::Ptr thisSurfKeyFrameDS(
      new pcl::PointCloud<PointTypeXYZI>());  // 降采样后的
  pcl::copyPointCloud(*pcl_body_wait_pub,
                      *thisSurfKeyFrame);  // 存储关键帧,没有降采样的点云
  // 降采样thisSurfKeyFrame
  // pcl::VoxelGrid<PointTypeXYZI> downSizeFilter;
  // downSizeFilter.setLeafSize(0.1, 0.1, 0.1);  // 设置降采样体素大小
  // downSizeFilter.setInputCloud(thisSurfKeyFrame);
  // downSizeFilter.filter(*thisSurfKeyFrameDS);  // 执行降采样
  // std::cout << RED << "pcl_body_wait_pub size: " << pcl_body_wait_pub->size()
  //           << endl;
  surfCloudKeyFrames.emplace_back(thisSurfKeyFrame);

  // updatePath(thisPose6D);  // 可视化update后的最新位姿
}
// 更新里程计轨迹
void LIVMapper::updatePath(const PointTypePose &pose_in) {
  std::string odometryFrame = "camera_init";
  geometry_msgs::msg::PoseStamped pose_stamped;
  pose_stamped.header.stamp =
      rclcpp::Time(pose_in.time * 1e9);  // Convert seconds to nanoseconds

  pose_stamped.header.frame_id = odometryFrame;
  pose_stamped.pose.position.x = pose_in.x;
  pose_stamped.pose.position.y = pose_in.y;
  pose_stamped.pose.position.z = pose_in.z;
  tf2::Quaternion q;
  q.setRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
  pose_stamped.pose.orientation.x = q.x();
  pose_stamped.pose.orientation.y = q.y();
  pose_stamped.pose.orientation.z = q.z();
  pose_stamped.pose.orientation.w = q.w();

  // Eigen::Matrix3d R =
  //     Exp((double)pose_in.roll, (double)pose_in.pitch,
  //     (double)pose_in.yaw);
  // Eigen::Vector3d t((double)pose_in.x, (double)pose_in.y,
  // (double)pose_in.z); pose update_pose; update_pose.R = R; update_pose.t =
  // t; update_nokf_poses.emplace_back(update_pose);

  // fout_update_pose << std::fixed << R(0, 0) << " " << R(0, 1) << " " <<
  // R(0, 2) << " " << t[0] << " "
  //     << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << " " << t[1] << " "
  //     << R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << " " << t[2] <<
  //     std::endl;

  globalPath.poses.emplace_back(pose_stamped);
}
// 更新因子图中所有变量节点的位姿,也就是所有历史关键帧的位姿,调整全局轨迹,重构ikdtree
void LIVMapper::correctPoses() {
  // std::cout<<"start correctPoses"<<endl;
  if (cloudKeyPoses3D->points.empty()) return;
  // 只有回环以及才会触发全局调整
  if (aLoopIsClosed == true) {
    // 清空里程计轨迹
    globalPath.poses.clear();
    // 更新因子图中所有变量节点的位姿,也就是所有历史关键帧的位姿
    int numPoses = isamCurrentEstimate.size();
    for (int i = 0; i < numPoses; ++i) {
      cloudKeyPoses3D->points[i].x =
          isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
      cloudKeyPoses3D->points[i].y =
          isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
      cloudKeyPoses3D->points[i].z =
          isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

      cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
      cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
      cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
      cloudKeyPoses6D->points[i].roll =
          isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
      cloudKeyPoses6D->points[i].pitch =
          isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
      cloudKeyPoses6D->points[i].yaw =
          isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

      // 更新里程计轨迹
      updatePath(cloudKeyPoses6D->points[i]);
    }

    // // 清空局部map, reconstruct  ikdtree submap
    // if (recontructKdTree){
    //     recontructIKdTree();
    // }

    RCLCPP_INFO(this->node->get_logger(), "ISMA2 Update");
    aLoopIsClosed = false;
  }
}

/**
 * @brief
 * 检测最新帧是否和其它帧形成回环
 * 回环检测三大要素
 * 1.设置最小时间差,太近没必要
 * 2.控制回环的频率,避免频繁检测,每检测一次,就做一次等待
 * 3.根据当前最小距离重新计算等待时间
 */
bool LIVMapper::detectLoopClosureDistance(int *latestID, int *closestID) {
  int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;  // 当前关键帧索引
  int loopKeyPre = -1;

  // 检查一下当前帧是否和别的形成了回环,如果已经有回环就不再继续
  auto it = loopIndexContainer.find(loopKeyCur);
  if (it != loopIndexContainer.end()) return false;
  // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
  std::vector<int> pointSearchIndLoop;      // 候选关键帧索引
  std::vector<float> pointSearchSqDisLoop;  // 候选关键帧距离
  kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
  kdtreeHistoryKeyPoses->radiusSearch(
      copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius,
      pointSearchIndLoop, pointSearchSqDisLoop, 0);
  for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i) {
    int id = pointSearchIndLoop[i];
    // 历史帧必须比当前帧间隔historyKeyframeSearchTimeDiff以上,必须满足时间阈值,才是一个有效回环,一次找一个回环帧就行了
    if (abs(copy_cloudKeyPoses6D->points[id].time - lidar_end_time) >
        historyKeyframeSearchTimeDiff) {
      loopKeyPre = id;
      break;
    }
  }
  // 如果没有找到回环或者回环找到自己身上去了,就认为是本次回环寻找失败
  if (loopKeyPre == -1 || loopKeyCur == loopKeyPre) return false;
  // 赋值当前帧和历史回环帧的id
  *latestID = loopKeyCur;
  *closestID = loopKeyPre;

  // ROS_INFO("Find loop clousre frame ");
  return true;
}

/**
 * @brief 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合,降采样
 *
 */
void LIVMapper::loopFindNearKeyframes(
    pcl::PointCloud<PointTypeXYZI>::Ptr &nearKeyframes, const int &key,
    const int &searchNum) {
  nearKeyframes->clear();
  int cloudSize = copy_cloudKeyPoses6D->size();
  // searchNum是搜索范围,遍历帧的范围
  for (int i = -searchNum; i <= searchNum; ++i) {
    int keyNear = key + i;
    // 超出范围,退出
    if (keyNear < 0 || keyNear >= cloudSize) continue;
    // 注意:cloudKeyPoses6D存储的是T_w_b,而点云是lidar系下的,构建submap时,要把点云转到cur点云系下
    if (i == 0) {
      *nearKeyframes += *surfCloudKeyFrames[keyNear];  // cur点云本身保持不变
    } else {
      Eigen::Affine3f keyTrans =
          pcl::getTransformation(copy_cloudKeyPoses6D->points[key].x,
                                 copy_cloudKeyPoses6D->points[key].y,
                                 copy_cloudKeyPoses6D->points[key].z,
                                 copy_cloudKeyPoses6D->points[key].roll,
                                 copy_cloudKeyPoses6D->points[key].pitch,
                                 copy_cloudKeyPoses6D->points[key].yaw);
      Eigen::Affine3f keyNearTrans =
          pcl::getTransformation(copy_cloudKeyPoses6D->points[keyNear].x,
                                 copy_cloudKeyPoses6D->points[keyNear].y,
                                 copy_cloudKeyPoses6D->points[keyNear].z,
                                 copy_cloudKeyPoses6D->points[keyNear].roll,
                                 copy_cloudKeyPoses6D->points[keyNear].pitch,
                                 copy_cloudKeyPoses6D->points[keyNear].yaw);
      Eigen::Affine3f finalTrans = keyTrans.inverse() * keyNearTrans;
      pcl::PointCloud<PointTypeXYZI>::Ptr tmp(
          new pcl::PointCloud<PointTypeXYZI>());
      transformPointCloud(surfCloudKeyFrames[keyNear], finalTrans, tmp);
      *nearKeyframes += *tmp;
      // *nearKeyframes += *getBodyCloud(surfCloudKeyFrames[keyNear],
      // copy_cloudKeyPoses6D->points[key],
      // copy_cloudKeyPoses6D->points[keyNear]); // TODO:
      // fast-lio没有进行特征提取,默认点云就是surf
    }
  }
  if (nearKeyframes->empty()) return;
}

void LIVMapper::publishCloud(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub,
    pcl::PointCloud<PointTypeXYZI>::Ptr thisCloud, rclcpp::Time thisStamp,
    std::string thisFrame) {
  sensor_msgs::msg::PointCloud2 tempCloud;
  pcl::toROSMsg(*thisCloud, tempCloud);
  tempCloud.header.stamp = thisStamp;
  tempCloud.header.frame_id = thisFrame;
  if (thisPub->get_subscription_count() != 0) thisPub->publish(tempCloud);
  // return tempCloud;
}

/**
 * @brief 回环检测函数 //
 * TODO：没有用SC全局回环，因为很容易出现错误匹配，与其这样不如相信odometry
 *    先最近距离再sc，我们可以相信odometry 是比较准的
 *
 */
void LIVMapper::performLoopClosure() {
  // 使用 ROS 2 的时间戳
  rclcpp::Time timeLaserInfoStamp = this->node->now();
  std::string odometryFrame = "camera_init";
  // 没有关键帧,没法进行回环检测
  if (cloudKeyPoses3D->points.empty() == true) {
    return;
  }
  // 把存储关键帧位姿的点云copy出来,避免线程冲突,cloudKeyPoses3D就是关键帧的位置,cloudKeyPoses6D就是关键帧的位姿
  mtx.lock();
  *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
  *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
  mtx.unlock();

  int loopKeyCur;  // 当前关键帧索引
  int loopKeyPre;  // 候选回环匹配帧索引

  // 根据里程计的距离来检测回环,如果还没有回环则直接返回
  if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false) {
    return;
  }
  cout << "[Nearest Pose found] " << " curKeyFrame: " << loopKeyCur
       << " loopKeyFrame: " << loopKeyPre << endl;

  // 提取scan2map
  pcl::PointCloud<PointTypeXYZI>::Ptr cureKeyframeCloud(
      new pcl::PointCloud<PointTypeXYZI>());  // 当前关键帧的点云
  pcl::PointCloud<PointTypeXYZI>::Ptr prevKeyframeCloud(
      new pcl::PointCloud<
          PointTypeXYZI>());  // 历史回环帧周围的点云（局部地图）
  {
    // 提取当前关键帧特征点集合,降采样
    loopFindNearKeyframes(
        cureKeyframeCloud, loopKeyCur,
        historyKeyframeSearchNum);  // 将cureKeyframeCloud保持在cureKeyframeCloud系下
    // 提取历史回环帧周围的点云特征点集合,降采样
    loopFindNearKeyframes(
        prevKeyframeCloud, loopKeyPre,
        historyKeyframeSearchNum);  // 选取historyKeyframeSearchNum个keyframe拼成submap，并转换到cureKeyframeCloud系下
    // 发布回环局部地图submap
    if (pubHistoryKeyFrames->get_subscription_count() > 0) {
      pcl::PointCloud<PointTypeXYZI>::Ptr pubKrevKeyframeCloud(
          new pcl::PointCloud<PointTypeXYZI>());
      *pubKrevKeyframeCloud += *transformPointCloud(
          prevKeyframeCloud,
          &copy_cloudKeyPoses6D
               ->points[loopKeyCur]);  // 将submap转换到world系再发布
      publishCloud(pubHistoryKeyFrames, pubKrevKeyframeCloud,
                   timeLaserInfoStamp, odometryFrame);
    }
  }

  // TODO: 生成sc，进行匹配，再次声明一次，这里没有使用sc的全局回环
  Eigen::MatrixXd cureKeyframeSC = scLoop.makeScancontext(*cureKeyframeCloud);
  Eigen::MatrixXd prevKeyframeSC = scLoop.makeScancontext(*prevKeyframeCloud);
  std::pair<double, int> simScore =
      scLoop.distanceBtnScanContext(cureKeyframeSC, prevKeyframeSC);
  double dist = simScore.first;
  int align = simScore.second;
  if (dist > scLoop.SC_DIST_THRES) {
    cout << "but they can not be detected by SC." << endl;
    return;
  }
  std::cout.precision(3);  // TODO: 如果使用sc全局定位，必须将保存的scd精度为3
  cout << "[SC Loop found]" << " curKeyFrame: " << loopKeyCur
       << " loopKeyFrame: " << loopKeyPre << " distance: " << dist
       << " nn_align: " << align * scLoop.PC_UNIT_SECTORANGLE << " deg."
       << endl;

  // ICP设置
  pcl::IterativeClosestPoint<PointTypeXYZI, PointTypeXYZI> icp;
  icp.setMaxCorrespondenceDistance(200);
  icp.setMaximumIterations(100);  // 迭代停止条件一:设置最大的迭代次数
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);
  icp.setRANSACIterations(0);  // 设置RANSAC运行次数

  float com_yaw = align * scLoop.PC_UNIT_SECTORANGLE;
  PointTypePose com;
  com.x = 0.0;
  com.y = 0.0;
  com.z = 0.0;
  com.yaw = -com_yaw;
  com.pitch = 0.0;
  com.roll = 0.0;
  cureKeyframeCloud = transformPointCloud(cureKeyframeCloud, &com);

  // map-to-map,调用icp匹配
  icp.setInputSource(cureKeyframeCloud);  // 设置原始点云
  icp.setInputTarget(prevKeyframeCloud);  // 设置目标点云
  pcl::PointCloud<PointTypeXYZI>::Ptr unused_result(
      new pcl::PointCloud<PointTypeXYZI>());
  icp.align(*unused_result);  // 进行ICP配准,输出变换后点云

  // 检测icp是否收敛以及得分是否满足要求
  if (icp.hasConverged() == false ||
      icp.getFitnessScore() > historyKeyframeFitnessScore) {
    cout << "but they can not be registered by ICP."
         << " icpFitnessScore: " << icp.getFitnessScore() << endl;
    return;
  }
  std::cout.precision(3);
  cout << "[ICP Regiteration success ] " << " curKeyFrame: " << loopKeyCur
       << " loopKeyFrame: " << loopKeyPre
       << " icpFitnessScore: " << icp.getFitnessScore() << endl;

  // 发布当前关键帧经过回环优化后的位姿变换之后的特征点云供可视化使用
  if (pubIcpKeyFrames->get_subscription_count() > 0) {
    // TODO: icp.getFinalTransformation()可以用来得到精准位姿
    pcl::PointCloud<PointTypeXYZI>::Ptr closed_cloud(
        new pcl::PointCloud<PointTypeXYZI>());
    pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud,
                             icp.getFinalTransformation());
    closed_cloud = transformPointCloud(
        closed_cloud, &copy_cloudKeyPoses6D->points[loopKeyCur]);
    publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp,
                 odometryFrame);
  }

  // 回环优化得到的当前关键帧与回环关键帧之间的位姿变换
  float x, y, z, roll, pitch, yaw;
  Eigen::Affine3f correctionLidarFrame;
  correctionLidarFrame =
      icp.getFinalTransformation();  // 获得两个点云的变换矩阵结果
  // 回环优化前的当前帧位姿
  Eigen::Affine3f tWrong =
      pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
  // 回环优化后的当前帧位姿:将icp结果补偿过去,就是当前帧的更为准确的位姿结果
  Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;

  // 将回环优化后的当前帧位姿换成平移和旋转
  pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
  // 将回环优化后的当前帧位姿换成gtsam的形式,From和To相当于帧间约束的因子,To是历史回环帧的位姿
  gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw),
                                       gtsam::Point3(x, y, z));
  gtsam::Pose3 poseTo =
      pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
  // 使用icp的得分作为他们的约束噪声项
  gtsam::Vector Vector6(6);
  float noiseScore = icp.getFitnessScore();  // 获得icp的得分
  Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore,
      noiseScore;
  gtsam::noiseModel::Diagonal::shared_ptr constraintNoise =
      gtsam::noiseModel::Diagonal::Variances(Vector6);
  std::cout << "loopNoiseQueue   =   " << noiseScore << std::endl;

  // 添加回环因子需要的数据:将两帧索引,两帧相对位姿和噪声作为回环约束送入对列
  mtx.lock();
  loopIndexQueue.emplace_back(make_pair(loopKeyCur, loopKeyPre));
  loopPoseQueue.emplace_back(poseFrom.between(poseTo));
  loopNoiseQueue.emplace_back(constraintNoise);
  mtx.unlock();

  loopIndexContainer[loopKeyCur] = loopKeyPre;  // 使用hash map存储回环对
}

// 回环检测线程
void LIVMapper::loopClosureThread() {
  // 不进行回环检测
  if (loopClosureEnableFlag == false) {
    std::cout << "loopClosureEnableFlag   ==  false " << std::endl;
    return;
  }
  std::cout << "loopClosureThread" << endl;
  rclcpp::Rate rate(
      loopClosureFrequency);  // 设置回环检测的频率loopClosureFrequency默认为1hz
  while (rclcpp::ok() && startFlag) {
    // 执行完一次就必须sleep一段时间,否则该线程的cpu占用会非常高
    rate.sleep();
    performLoopClosure();    // 回环检测函数
    visualizeLoopClosure();  // rviz展示回环边
  }
}

bool LIVMapper::handleLIO() {
  euler_cur = RotMtoEuler(_state.rot_end);
  // fout_pre << setw(20) << LidarMeasures.last_lio_update_time -
  // _first_lidar_time
  //          << " " << euler_cur.transpose() * 57.3 << " "
  //          << _state.pos_end.transpose() << " " << _state.vel_end.transpose()
  //          << " " << _state.bias_g.transpose() << " "
  //          << _state.bias_a.transpose() << " "
  //          << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;

  if (feats_undistort->empty() || (feats_undistort == nullptr)) {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return false;
  }
  double g0 = omp_get_wtime();
  // dect_ground->extract_ground(*feats_undistort);

  double t0 = omp_get_wtime();
  if (feats_undistort->points.size() < 500) {
    std::cout << "[ LIO ]: Not enough points!!!" << std::endl;
    pcl::copyPointCloud(*feats_undistort, *feats_down_body);
  } else {
    downSizeFilterSurf.setInputCloud(feats_undistort);
    downSizeFilterSurf.filter(*feats_down_body);
    // dect_ground->extract_ground(*feats_undistort);
    // pcl::PointCloud<PointType>::Ptr non_ground_pc(
    //     new pcl::PointCloud<PointType>(dect_ground->non_ground_pc_));
    // pcl::PointCloud<PointType>::Ptr ds_non_ground_pc(
    //     new pcl::PointCloud<PointType>());
    // pcl::PointCloud<PointType>::Ptr ground_pc(
    //     new pcl::PointCloud<PointType>(dect_ground->ground_pc_));
    // ds_ground_pc->reserve(ground_pc->size());
    // ds_non_ground_pc->reserve(non_ground_pc->size() + ground_pc->size());

    // // 对非地面点进行降采样
    // downSizeFilterSurf.setInputCloud(non_ground_pc);
    // downSizeFilterSurf.filter(*ds_non_ground_pc);
    // downSizeFilterGround.setInputCloud(ground_pc);
    // downSizeFilterGround.filter(*ds_ground_pc);
    // *ds_non_ground_pc += *ds_ground_pc;
    // std::cout << "ds ground points size: " << ds_ground_pc->points.size()
    //           << std::endl;
    // std::cout << "ds all points size: " << ds_non_ground_pc->points.size()
    //           << std::endl;
    // pcl::copyPointCloud(*ds_non_ground_pc, *feats_down_body);
    // std::cout << "[ LIO ] downsampled from " <<
    // feats_undistort->points.size()
    //           << " to " << feats_down_body->points.size() << std::endl;

    // ds_ground_pc->clear();

    // dect_ground->reset();
  }
  double g1 = omp_get_wtime();

  double t_down = omp_get_wtime();

  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body;
  // transformLidar(_state.rot_end, _state.pos_end, feats_down_body,
  //                feats_down_world);
  // voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;

  if (!lidar_map_inited) {
    lidar_map_inited = true;
    transformLidar(_state.rot_end, _state.pos_end, feats_down_body,
                   feats_down_world);
    // voxelmap_manager->feats_down_world_ = feats_down_world;
    voxelmap_manager->BuildVoxelMapLRU(feats_down_world);
    // voxelmap_manager->BuildVoxelMap();
  }

  double t1 = omp_get_wtime();
  // before state estimation
  // cout << "[ LIO ] Start State Estimation" << endl;
  // cout << "Before pos: " << _state.pos_end.transpose() << endl;
  // cout << "Before vel: " << _state.vel_end.transpose() << endl;
  // cout << "Before rpy: " << RotMtoEuler(_state.rot_end).transpose() << endl;

  voxelmap_manager->StateEstimation(state_propagat);
  // voxelmap_manager->StateEstimation2(state_propagat);
  _state = voxelmap_manager->state_;
  // cout << "After pos: " << _state.pos_end.transpose() << endl;
  // cout << "After vel: " << _state.vel_end.transpose() << endl;
  // cout << "After rpy: " << RotMtoEuler(_state.rot_end).transpose() << endl;

  if (enable_zupt && frame_num % zupt_interval == 0) {
    zupt->setState(_state);
    zupt->setMeasurement(zupt_noise * zupt_noise, 0.0);
    zupt->applyZConstraint();
    _state = zupt->state_;
  }
  if (enable_wheel_odom && wheel_odom_updated_) {
    // wheel_odometry_->update_vel(_state, wheel_linear_velocity_);
    if (est_wheel_extrinsic) {
      wheel_odometry_->update_state_joint(_state, wheel_linear_velocity_);
    } else {
      wheel_odometry_->update_state(_state, wheel_linear_velocity_);
    }
  }
  // wheel_extrinsic_ekf_->update(_state, wheel_linear_velocity_);
  // _pv_list = voxelmap_manager->pv_list_;

  double t2 = omp_get_wtime();

  if (imu_prop_enable) {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  if (pose_output_en) {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    if (!pos_opend) {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt",
                   std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open())
        RCLCPP_ERROR(this->node->get_logger(), "open fail\n");
    } else {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt",
                   std::ios::app);
      if (!evoFile.is_open())
        RCLCPP_ERROR(this->node->get_logger(), "open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0]
            << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
            << std::endl;
  }

  double t3 = omp_get_wtime();

  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);
  for (size_t i = 0; i < world_lidar->points.size(); i++) {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x,
        world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
    M3D var = voxelmap_manager->body_cov_list_[i];
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) *
              (-point_crossmat).transpose() +
          _state.cov.block<3, 3>(3, 3);
    voxelmap_manager->pv_list_[i].var = var;
  }
  voxelmap_manager->UpdateVoxelMapLRU(voxelmap_manager->pv_list_);
  // std::cout << "[ LIO ] Update Voxel Map" << std::endl;
  _pv_list = voxelmap_manager->pv_list_;

  double t4 = omp_get_wtime();

  // if (voxelmap_manager->config_setting_.map_sliding_en) {
  //   voxelmap_manager->mapSliding();
  // }

  // laserCloudFullRes默认使用包含地面点信息的feats_undistort
  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort
                                                     : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  // double g0 = omp_get_wtime();
  // dect_ground->extract_ground(*laserCloudFullRes);
  // double g1 = omp_get_wtime();
  // dect_ground->reset();

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                        &laserCloudWorld->points[i]);
  }
  {
    *pcl_w_wait_pub = *laserCloudWorld;
    *pcl_body_wait_pub = *laserCloudFullRes;
  }
  // pcl_body_wait_pub
  if (!img_en) publish_frame_lidar(pubLaserUndistortCloud);
  if (!img_en) publish_frame_world(pubLaserCloudFullRes, vio_manager);
  if (pub_effect_point_en)
    publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_)
    voxelmap_manager->pubVoxelMapLRU();
  publish_path(pubPath);
  publish_mavros(mavros_pose_publisher);
  if (!img_en) {
    Eigen::Quaterniond quat(_state.rot_end);
    geoQuat.w = quat.w();
    geoQuat.x = quat.x();
    geoQuat.y = quat.y();
    geoQuat.z = quat.z();
    publish_odometry(pubOdomAftMapped);
  }

  frame_num++;
  aver_time_consu =
      aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

  // aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) /
  // frame_num; aver_time_map_inre = aver_time_map_inre * (frame_num - 1) /
  // frame_num + (t4 - t3) / frame_num; aver_time_solve = aver_time_solve *
  // (frame_num - 1) / frame_num + (solve_time) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) /
  // frame_num + solve_const_H_time / frame_num; printf("[ mapping time ]: per
  // scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP:
  // %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total:
  //         %0.6f \n", t_prop - t0, t1 - t_prop, match_time, solve_time, t3 -
  //         t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time,
  //         aver_time_consu);

  // printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs,
  // map incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
  //         "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map
  //         incre: %0.6f secs, total: %0.6f secs.\033[0m\n", t2 - t1, t4 -
  //         t3, t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_consu);
  printf(
      "\033[1;34m+-----------------------------------------------------------"
      "--"
      "+\033[0m\n");
  printf(
      "\033[1;34m|                         LIO Mapping Time                  "
      "  "
      "|\033[0m\n");
  printf(
      "\033[1;34m+-----------------------------------------------------------"
      "--"
      "+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage",
         "Time (secs)");
  printf(
      "\033[1;34m+-----------------------------------------------------------"
      "--"
      "+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
  printf(
      "\033[1;34m+-----------------------------------------------------------"
      "--"
      "+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Ground Detection", g1 - g0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  // // save t4-t0 to /Log/time.txt
  // std::ofstream fout_time;
  // fout_time.open(std::string(ROOT_DIR) + "Log/result/lio_time.txt",
  //                std::ios::app);
  // if (!fout_time.is_open())
  //   RCLCPP_ERROR(this->node->get_logger(), "open fail\n");
  // fout_time << std::fixed << std::setprecision(6) << t4 - t0 << std::endl;

  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time",
         aver_time_consu);
  printf(
      "\033[1;34m+-----------------------------------------------------------"
      "--"
      "+\033[0m\n");

  euler_cur = RotMtoEuler(_state.rot_end);
  // fout_out << std::setw(20)
  //          << LidarMeasures.last_lio_update_time - _first_lidar_time << " "
  //          << euler_cur.transpose() * 57.3 << " " << _state.pos_end.transpose()
  //          << " " << _state.vel_end.transpose() << " "
  //          << _state.bias_g.transpose() << " " << _state.bias_a.transpose()
  //          << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " "
  //          << feats_undistort->points.size() << std::endl;
  return true;
}

void LIVMapper::savePCD() {
  std::cout << "**************** data saver runs when programe is closing "
               "****************"
            << std::endl;
  std::cout << "pcl_wait_save size: " << pcl_wait_save->points.size()
            << std::endl;
  std::cout << "pcl_wait_save_intensity size: "
            << pcl_wait_save_intensity_cp->points.size() << std::endl;
  if (pcd_save_en && (pcl_wait_save_global_rgb->points.size() > 0 ||
                      pcl_wait_save_intensity->points.size() > 0 ||
                      pcl_wait_save_intensity_cp->points.size() > 0)) {
    std::string raw_points_dir =
        std::string(ROOT_DIR) + "Log/PCD/all_raw_points.pcd";
    std::string downsampled_points_dir =
        std::string(ROOT_DIR) + "Log/PCD/all_downsampled_points.pcd";

    pcl::PCDWriter pcd_writer;

    if (img_en) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(pcl_wait_save_global_rgb);
      voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd,
                               filter_size_pcd);
      voxel_filter.filter(*downsampled_cloud);
      pcl::VoxelGrid<PointType> voxel_filter_raw;
      pcl::PointCloud<PointType>::Ptr raw_downsampled_cloud(
          new pcl::PointCloud<PointType>);
      voxel_filter_raw.setInputCloud(pcl_wait_save_global);
      voxel_filter_raw.setLeafSize(filter_size_pcd, filter_size_pcd,
                                   filter_size_pcd);
      voxel_filter_raw.filter(*raw_downsampled_cloud);
      pcd_writer.writeBinary(
          raw_points_dir,
          *raw_downsampled_cloud);  // Save the raw point cloud data
      //  Downsampled point cloud data
      std::cout << GREEN
                << "Raw lidar point cloud data saved to: " << raw_points_dir
                << " with point count: " << pcl_wait_save_global->points.size()
                << RESET << std::endl;

      pcd_writer.writeBinary(
          downsampled_points_dir,
          *downsampled_cloud);  // Save the downsampled point cloud data
      std::cout << GREEN
                << "Downsampled point cloud data in camera fov saved to: "
                << downsampled_points_dir
                << " with point count after filtering: "
                << downsampled_cloud->points.size() << RESET << std::endl;

      if (colmap_output_en) {
        fout_points << "# 3D point list with one line of data per point\n";
        fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
        for (size_t i = 0; i < downsampled_cloud->size(); ++i) {
          const auto &point = downsampled_cloud->points[i];
          fout_points << i << " " << std::fixed << std::setprecision(6)
                      << point.x << " " << point.y << " " << point.z << " "
                      << static_cast<int>(point.r) << " "
                      << static_cast<int>(point.g) << " "
                      << static_cast<int>(point.b) << " " << 0 << std::endl;
          // std::cout << GREEN << "the colmap map has been saved" << endl;
        }
      }
      // std::cout << GREEN << "the pcd map has been saved" << endl;
    } else {
      // pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity);
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity_cp);
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir
                << " with point count: "
                << pcl_wait_save_intensity_cp->points.size() << RESET
                << std::endl;
    }
  }
  std::cout << GREEN << "the pcd map has been saved" << endl;
}

void LIVMapper::saveOptimizedVerticesKITTIformat(gtsam::Values _estimates,
                                                 std::string _filename) {
  using namespace gtsam;

  // ref from gtsam's original code "dataset.cpp"
  std::fstream stream(_filename.c_str(), fstream::out);

  for (const auto &key_value : _estimates) {
    auto p = dynamic_cast<const GenericValue<Pose3> *>(&key_value.value);
    if (!p) continue;

    const Pose3 &pose = p->value();

    Point3 t = pose.translation();
    Rot3 R = pose.rotation();
    auto col1 = R.column(1);  // Point3
    auto col2 = R.column(2);  // Point3
    auto col3 = R.column(3);  // Point3

    stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x()
           << " " << col1.y() << " " << col2.y() << " " << col3.y() << " "
           << t.y() << " " << col1.z() << " " << col2.z() << " " << col3.z()
           << " " << t.z() << std::endl;
  }
}
void LIVMapper::saveKeyFrame() {
  /**************** data saver runs when programe is closing ****************/
  std::cout << "**************** data saver runs when programe is closing "
               "****************"
            << std::endl;

  if (!((surfCloudKeyFrames.size() == cloudKeyPoses3D->points.size()) &&
        (cloudKeyPoses3D->points.size() == cloudKeyPoses6D->points.size()))) {
    std::cout << surfCloudKeyFrames.size() << " "
              << cloudKeyPoses3D->points.size() << " "
              << cloudKeyPoses6D->points.size() << std::endl;
    std::cout << " the condition --surfCloudKeyFrames.size() == "
                 "cloudKeyPoses3D->points.size() == "
                 "cloudKeyPoses6D->points.size()-- is not satisfied"
              << std::endl;
    return;
  } else {
    std::cout << "the num of total keyframe is " << surfCloudKeyFrames.size()
              << std::endl;
  }
  isam->update();
  isam->update();
  isamCurrentEstimate = isam->calculateBestEstimate();

  // save key frame poses
  // string save_dir(std::string(ROOT_DIR) + "map/");
  string save_dir = map_save_path;
  fsmkdir(save_dir);

  // save pose graph
  cout << "****************************************************" << endl;
  cout << "save map to " << save_dir << endl;
  cout << "Saving  posegraph" << endl;
  pgSaveStream =
      std::fstream(save_dir + "singlesession_posegraph.g2o", std::fstream::out);

  for (auto &_line : vertices_str) pgSaveStream << _line << std::endl;
  for (auto &_line : edges_str) pgSaveStream << _line << std::endl;
  pgSaveStream.close();
  std::cout << "pose graph saved to: "
            << save_dir + "singlesession_posegraph.g2o" << std::endl;

  // save key frame poses in KITTI format
  std::cout << "Saving key frame poses in special format" << std::endl;
  const std::string kitti_format_pg_filename{save_dir + "optimized_poses.txt"};
  saveOptimizedVerticesKITTIformat(isamCurrentEstimate,
                                   kitti_format_pg_filename);
  std::string mapping_traj_path = save_dir + "mapping.txt";
  std::ofstream foutC2(mapping_traj_path, std::ios::app);
  for (int i = 0; i < cloudKeyPoses6D->size(); i++) {
    cloudKeyPoses6D->points[i].x =
        isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
    cloudKeyPoses6D->points[i].y =
        isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
    cloudKeyPoses6D->points[i].z =
        isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();
    cloudKeyPoses6D->points[i].roll =
        isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
    cloudKeyPoses6D->points[i].pitch =
        isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
    cloudKeyPoses6D->points[i].yaw =
        isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

    tf2::Quaternion quat;
    quat.setRPY(cloudKeyPoses6D->points[i].roll,
                cloudKeyPoses6D->points[i].pitch,
                cloudKeyPoses6D->points[i].yaw);

    foutC2 << i << " " << std::fixed << std::setprecision(4)
           << cloudKeyPoses6D->points[i].x << " "
           << cloudKeyPoses6D->points[i].y << " "
           << cloudKeyPoses6D->points[i].z << " " << quat.x() << " " << quat.y()
           << " " << quat.z() << " " << quat.w() << std::endl;
  }
  foutC2.close();

  string traj_dir(save_dir + "trajectory.pcd");
  pcl::io::savePCDFileASCII(traj_dir, *cloudKeyPoses3D);
  string trans_dir(save_dir + "transformations.pcd");
  pcl::io::savePCDFileASCII(trans_dir, *cloudKeyPoses6D);
  std::cout << "Completed saving key frame pose" << std::endl;
  // save sc and keyframe cloud
  const SCInputType sc_input_type =
      SCInputType::SINGLE_SCAN_FULL;  // TODO: change this in ymal
  bool soMany = false;
  std::cout << "save sc and keyframe" << std::endl;
  std::string scd_path = save_dir + "keyframe_scancontext/";
  fsmkdir(scd_path);
  string pcd_path = save_dir + "keyframe_cloud/";
  fsmkdir(pcd_path);
  for (size_t i = 0; i < cloudKeyPoses6D->size(); i++) {
    pcl::PointCloud<PointTypeXYZI>::Ptr save_cloud(
        new pcl::PointCloud<PointTypeXYZI>());
    if (sc_input_type == SCInputType::SINGLE_SCAN_FULL) {
      pcl::copyPointCloud(*surfCloudKeyFrames[i], *save_cloud);
      scLoop.makeAndSaveScancontextAndKeys(*save_cloud);
    } else if (sc_input_type == SCInputType::MULTI_SCAN_FEAT) {
      pcl::PointCloud<PointTypeXYZI>::Ptr multiKeyFrameFeatureCloud(
          new pcl::PointCloud<PointTypeXYZI>());
      loopFindNearKeyframes(multiKeyFrameFeatureCloud, i,
                            historyKeyframeSearchNum);
      if (soMany) {
        pcl::copyPointCloud(*multiKeyFrameFeatureCloud, *save_cloud);
      } else {
        pcl::copyPointCloud(*surfCloudKeyFrames[i], *save_cloud);
      }
      scLoop.makeAndSaveScancontextAndKeys(*save_cloud);
    }

    // save sc data
    const auto &curr_scd = scLoop.getConstRefRecentSCD();
    std::string curr_scd_node_idx = padZeros(scLoop.polarcontexts_.size() - 1);
    writeSCD(scd_path + curr_scd_node_idx + ".scd", curr_scd);

    string all_points_dir(pcd_path + string(curr_scd_node_idx) + ".pcd");
    pcl::io::savePCDFileASCII(all_points_dir, *save_cloud);
  }

  pcl::PointCloud<PointTypeXYZI>::Ptr globalCornerCloud(
      new pcl::PointCloud<PointTypeXYZI>());
  pcl::PointCloud<PointTypeXYZI>::Ptr globalSurfCloud(
      new pcl::PointCloud<PointTypeXYZI>());
  pcl::PointCloud<PointTypeXYZI>::Ptr globalMapCloud(
      new pcl::PointCloud<PointTypeXYZI>());
  pcl::PointCloud<PointTypeXYZI>::Ptr globalMapCloudDS(
      new pcl::PointCloud<PointTypeXYZI>());
  for (int i = 0; i < (int)cloudKeyPoses6D->size(); i++) {
    *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i],
                                             &cloudKeyPoses6D->points[i]);
  }
  *globalMapCloud += *globalSurfCloud;
  pcl::VoxelGrid<PointTypeXYZI>
      downSizeFilterGlobalMapKeyFrames;  // for global map
  downSizeFilterGlobalMapKeyFrames.setLeafSize(
      1.0, 1.0, 1.0);  // for global map visualization
  downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapCloud);
  downSizeFilterGlobalMapKeyFrames.filter(*globalMapCloudDS);
  pcl::io::savePLYFileASCII(save_dir + "cloudGlobal.ply", *globalMapCloudDS);
  pcl::io::savePCDFileASCII(save_dir + "cloudGlobal.pcd", *globalMapCloudDS);
  cout << "*************************Saving map to pcd files "
          "completed***************************"
       << endl;
  savePCD();
}
void LIVMapper::run(rclcpp::Node::SharedPtr &node) {
  rclcpp::Rate rate(5000);
  while (rclcpp::ok()) {
    rclcpp::spin_some(this->node);
    if (!sync_packages(LidarMeasures)) {
      rate.sleep();
      continue;
    }
    handleFirstFrame();

    processImu();
    // processRobotOdometry(LidarMeasures);
    extractWheelVel(LidarMeasures);

    // if (!p_imu->imu_time_init) continue;

    stateEstimationAndMapping();
  }

  startFlag = false;
}

void LIVMapper::extractWheelVel(LidarMeasureGroup &meas) {
  std::lock_guard<std::mutex> lock2(odoLock);
  // newest_imu.header.stamp
  // double start_time = meas.lidar_frame_beg_time;
  double start_time = meas.last_lio_update_time;
  static double first_time = start_time;
  // std::cout << "processRobotOdometry: lidar_end_time: " << start_time
  //           << std::endl;
  std::cout << std::fixed << std::setprecision(19)
            << "extractWheelVel: lidar_start_time: " << start_time << std::endl;
  while (!odomQueue.empty()) {
    if (stamp2Sec(odomQueue.front().header.stamp) < start_time - 0.01)
      odomQueue.pop_front();
    else
      break;
  }
  // copy odomQueue to localOdomQueue for processing
  std::deque<nav_msgs::msg::Odometry> localOdomQueue = odomQueue;
  odoLock.unlock();
  if (localOdomQueue.empty()) {
    std::cout << "extractWheelVel: odomQueue is empty" << std::endl;
    wheel_odom_updated_ = false;
    return;
  }

  // get start odometry at the beinning of the scan
  nav_msgs::msg::Odometry startOdomMsg;
  for (size_t i = 0; i < (size_t)localOdomQueue.size(); ++i) {
    startOdomMsg = localOdomQueue[i];
    if (stamp2Sec(startOdomMsg.header.stamp) < start_time)
      continue;
    else
      break;
  }

  wheel_linear_velocity_[0] = startOdomMsg.twist.twist.linear.x;
  wheel_linear_velocity_[1] = startOdomMsg.twist.twist.linear.y;
  wheel_linear_velocity_[2] = startOdomMsg.twist.twist.linear.z;
  if(abs(wheel_linear_velocity_[2])<0.200) wheel_linear_velocity_[2] = 0.0;
  wheel_odom_updated_ = true;
  std::cout << "current wheel vel time: "
            << stamp2Sec(startOdomMsg.header.stamp) << std::endl;
  std::cout << "wheel linear velocity: " << wheel_linear_velocity_.transpose()
            << std::endl;
}

void LIVMapper::processRobotOdometry(LidarMeasureGroup &meas) {
  std::lock_guard<std::mutex> lock2(odoLock);
  // newest_imu.header.stamp
  // double start_time = meas.lidar_frame_beg_time;
  double start_time = meas.last_lio_update_time;
  static double first_time = start_time;
  // std::cout << "processRobotOdometry: lidar_end_time: " << start_time
  //           << std::endl;
  std::cout << std::fixed << std::setprecision(19)
            << "processRobotOdometry: lidar_start_time: " << start_time
            << std::endl;
  double end_time = img_time_buffer.front();
  std::cout << std::fixed << std::setprecision(19)
            << "processRobotOdometry: lidar_end_time: " << end_time
            << std::endl;
  while (!odomQueue.empty()) {
    if (stamp2Sec(odomQueue.front().header.stamp) < start_time - 0.002)
      odomQueue.pop_front();
    else
      break;
  }
  // copy odomQueue to localOdomQueue for processing
  std::deque<nav_msgs::msg::Odometry> localOdomQueue = odomQueue;
  odoLock.unlock();
  if (localOdomQueue.empty()) {
    std::cout << "processRobotOdometry: odomQueue is empty" << std::endl;
    return;
  }

  if (stamp2Sec(localOdomQueue.front().header.stamp) > start_time) return;

  // get start odometry at the beinning of the scan
  nav_msgs::msg::Odometry startOdomMsg;
  for (size_t i = 0; i < (size_t)localOdomQueue.size(); ++i) {
    startOdomMsg = localOdomQueue[i];
    if (stamp2Sec(startOdomMsg.header.stamp) < start_time)
      continue;
    else
      break;
  }

  tf2::Quaternion orientation;
  tf2::fromMsg(startOdomMsg.pose.pose.orientation, orientation);
  double roll, pitch, yaw;
  tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  if (stamp2Sec(localOdomQueue.back().header.stamp) < end_time) return;

  nav_msgs::msg::Odometry endOdomMsg;
  for (int i = 0; i < (int)localOdomQueue.size(); ++i) {
    endOdomMsg = localOdomQueue[i];
    if (stamp2Sec(endOdomMsg.header.stamp) < end_time)
      continue;
    else
      break;
  }

  Eigen::Affine3f transBegin = pcl::getTransformation(
      startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y,
      startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

  tf2::fromMsg(endOdomMsg.pose.pose.orientation, orientation);
  tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
  Eigen::Affine3f transEnd = pcl::getTransformation(
      endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y,
      endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

  Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

  std::cout << "processRobotOdometry: transBt: " << transBt.translation().x()
            << ", " << transBt.translation().y() << ", "
            << transBt.translation().z() << std::endl;
  std::cout << "processRobotOdometry: transBt rotation: "
            << transBt.rotation().eulerAngles(0, 1, 2).transpose() << std::endl;
  StatesGroup state_last = latest_ekf_state;
  // update state_last with the odometry increment
  state_last.pos_end[0] += transBt.translation().x();
  state_last.pos_end[1] += transBt.translation().y();
  state_last.pos_end[2] += transBt.translation().z();
  state_last.rot_end = state_last.rot_end * transBt.rotation().cast<double>();
  // if (transBt.translation().norm() > 0.0 &&
  //     abs(start_time - first_time) > 2.0) {
  //   // 更新_state
  //   // _state.pos_end = state_last.pos_end;
  //   cout << "last state rot: " << _state.rot_end.transpose() << endl;
  //   _state.rot_end = state_last.rot_end;
  //   cout << "current state rot: " << _state.rot_end.transpose() << endl;
  // }
}
void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt,
                              V3D acc_avr, V3D angvel_avr) {
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr +
                V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1],
                    imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end +
                           imu_prop_state.vel_end * dt +
                           0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback() {
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) {
    return;
  }
  mtx_buffer_imu_prop.lock();
  new_imu = false;  // 控制 propagate 频率和 IMU 频率一致
  if (imu_prop_enable && !prop_imu_buffer.empty()) {
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg) {
      imu_propagate = latest_ekf_state;
      // drop all useless imu pkg
      while (
          (!prop_imu_buffer.empty() &&
           stamp2Sec(prop_imu_buffer.front().header.stamp) < latest_ekf_time)) {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer.size(); i++) {
        double t_from_lidar_end_time =
            stamp2Sec(prop_imu_buffer[i].header.stamp) - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " <<
        // last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x,
                    prop_imu_buffer[i].linear_acceleration.y,
                    prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x,
                    prop_imu_buffer[i].angular_velocity.y,
                    prop_imu_buffer[i].angular_velocity.z);
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    } else {
      V3D acc_imu(newest_imu.linear_acceleration.x,
                  newest_imu.linear_acceleration.y,
                  newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y,
                  newest_imu.angular_velocity.z);
      double t_from_lidar_end_time =
          stamp2Sec(newest_imu.header.stamp) - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom->publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot,
                               const Eigen::Vector3d t,
                               const PointCloudXYZI::Ptr &input_cloud,
                               PointCloudXYZI::Ptr &trans_cloud) {
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  // #pragma omp parallel for
  for (size_t i = 0; i < input_cloud->size(); i++) {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.emplace_back(pi);
  }
}

void LIVMapper::transform2Camera(const Eigen::Matrix3d &rot,
                                 const Eigen::Vector3d &t,
                                 const PointCloudXYZRGB::Ptr &input_cloud,
                                 PointCloudXYZRGB::Ptr &trans_cloud) {
  trans_cloud->clear();
  trans_cloud->resize(input_cloud->size());
#pragma omp parallel for
  for (size_t i = 0; i < input_cloud->size(); i++) {
    const auto &p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = rot * p + t;
    PointTypeRGB pi;
    pi.x = p.x();
    pi.y = p.y();
    pi.z = p.z();
    pi.r = p_c.r;
    pi.g = p_c.g;
    pi.b = p_c.b;
    trans_cloud->points[i] = pi;
  }
}
void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po) {
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T>
void LIVMapper::pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi,
                                 Eigen::Matrix<T, 3, 1> &po) {
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T>
Eigen::Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(
    const Eigen::Matrix<T, 3, 1> &pi) {
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Eigen::Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi,
                                    PointType *const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
  if (!lidar_en) return;
  mtx_buffer.lock();
  // cout<<"got feature"<<endl;
  if (stamp2Sec(msg->header.stamp) < last_timestamp_lidar) {
    RCLCPP_ERROR(this->node->get_logger(), "lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", stamp2Sec(msg->header.stamp));
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lid_raw_data_buffer.emplace_back(ptr);
  lid_header_time_buffer.emplace_back(stamp2Sec(msg->header.stamp));
  last_timestamp_lidar = stamp2Sec(msg->header.stamp);

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::livox_pcl_cbk(
    const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg_in) {
  if (!lidar_en) return;
  mtx_buffer.lock();
  livox_ros_driver2::msg::CustomMsg::SharedPtr msg(
      new livox_ros_driver2::msg::CustomMsg(*msg_in));
  // if ((abs(stamp2Sec(msg->header.stamp) - last_timestamp_lidar) > 0.2 &&
  // last_timestamp_lidar > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("lidar jumps %.3f\n", stamp2Sec(msg->header.stamp) -
  //   last_timestamp_lidar); sync_jump_flag = true; msg->header.stamp =
  //   rclcpp::Time().fromSec(last_timestamp_lidar + 0.1);
  // }
  if (abs(last_timestamp_imu - stamp2Sec(msg->header.stamp)) > 1.0 &&
      !imu_buffer.empty()) {
    double timediff_imu_wrt_lidar =
        last_timestamp_imu - stamp2Sec(msg->header.stamp);
    RCLCPP_INFO(
        this->node->get_logger(),
        "\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m",
        timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;
  }

  // double cur_head_time = stamp2Sec(msg->header.stamp);
  double cur_head_time = stamp2Sec(msg->header.stamp) + lidar_time_offset;
  // RCLCPP_INFO(this->node->get_logger(), "Get LiDAR, its header time: %.6f",
  //             cur_head_time);
  if (cur_head_time < last_timestamp_lidar) {
    RCLCPP_ERROR(this->node->get_logger(), "lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // // time diff
  // double time_diff = cur_head_time - last_timestamp_lidar;
  // // write to file
  // std::ofstream ofs;
  // ofs.open(
  //     "/home/users/tingyang.xiao/3D_Recon/datasets/G1/D-robotics/3f-0829/"
  //     "rosbag2_2025_08_29-15_13_22/lidar_time_diff.txt",
  //     std::ios::app);
  // ofs << std::fixed << std::setprecision(10) << cur_head_time << " "
  //     << time_diff << std::endl;
  // ofs.close();
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);

  if (!ptr || ptr->empty()) {
    RCLCPP_ERROR(this->node->get_logger(), "Received an empty point cloud");
    mtx_buffer.unlock();
    return;
  }

  lid_raw_data_buffer.emplace_back(ptr);
  lid_header_time_buffer.emplace_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr &msg_in) {
  if (!imu_en) return;

  if (last_timestamp_lidar < 0.0) return;
  sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
  msg->header.stamp = sec2Stamp(stamp2Sec(msg->header.stamp) - imu_time_offset);
  double timestamp = stamp2Sec(msg->header.stamp);

  if (ros_driver_fix_en)
    timestamp += std::round(last_timestamp_lidar - timestamp);
  msg->header.stamp = sec2Stamp(timestamp);

  mtx_buffer.lock();

  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu) {
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    RCLCPP_ERROR(this->node->get_logger(), "imu loop back, offset: %lf \n",
                 last_timestamp_imu - timestamp);
    return;
  }

  last_timestamp_imu = timestamp;

  imu_buffer.emplace_back(msg);
  mtx_buffer.unlock();

  double front_imu_time = stamp2Sec(imu_buffer.front()->header.stamp);
  if (fabs(last_timestamp_lidar - front_imu_time) > 0.5 &&
      (!ros_driver_fix_en)) {
    RCLCPP_WARN(this->node->get_logger(),
                "IMU and LiDAR not synced! delta time: %lf .\n",
                last_timestamp_lidar - timestamp);
  }

  if (imu_prop_enable) {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init) {
      prop_imu_buffer.emplace_back(*msg);
    }
    newest_imu = *msg;
    new_imu = true;
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(
    const sensor_msgs::msg::Image::ConstSharedPtr &img_msg) {
  cv::Mat img;
  img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
  return img;
}

// static int i = 0;
void LIVMapper::img_cbk(const sensor_msgs::msg::Image::ConstSharedPtr &msg_in) {
  if (!img_en) return;
  if (img_filter_en) {
    static int frame_counter = 0;
    if (++frame_counter % img_filter_fre != 0) return;
  }
  sensor_msgs::msg::Image::SharedPtr msg(new sensor_msgs::msg::Image(*msg_in));
  // if ((abs(stamp2Sec(msg->header.stamp) - last_timestamp_img) > 0.2 &&
  // last_timestamp_img > 0) || sync_jump_flag)
  // {
  //   RCLCPP_WARN(this->node->get_logger(), "img jumps %.3f\n",
  //   stamp2Sec(msg->header.stamp) - last_timestamp_img); sync_jump_flag =
  //   true; msg->header.stamp = rclcpp::Time().fromSec(last_timestamp_img +
  //   0.1);
  // }

  // double msg_header_time =  stamp2Sec(msg->header.stamp);
  double msg_header_time = stamp2Sec(msg->header.stamp) + img_time_offset;

  if (abs(msg_header_time - last_timestamp_img) < 0.001) return;
  // RCLCPP_INFO(this->node->get_logger(), "Get image, its header time: %.6f",
  //             msg_header_time);
  if (last_timestamp_lidar < 0) return;

  if (msg_header_time < last_timestamp_img) {
    RCLCPP_ERROR(this->node->get_logger(), "image loop back. \n");
    return;
  }

  mtx_buffer.lock();

  double img_time_correct = msg_header_time;  // last_timestamp_lidar + 0.105;

  if (img_time_correct - last_timestamp_img < 0.02) {
    RCLCPP_WARN(this->node->get_logger(), "Image need Jumps: %.6f",
                img_time_correct);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);
  img_buffer.emplace_back(img_cur);
  img_time_buffer.emplace_back(img_time_correct);

  // ROS_INFO("Correct Image time: %.6f", img_time_correct);

  last_timestamp_img = img_time_correct;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas) {
  if (lid_raw_data_buffer.empty() && lidar_en) return false;
  if (img_buffer.empty() && img_en) return false;
  if (imu_buffer.empty() && imu_en) return false;

  switch (slam_mode_) {
    case ONLY_LIO: {
      if (meas.last_lio_update_time < 0.0)
        meas.last_lio_update_time = lid_header_time_buffer.front();
      if (!lidar_pushed) {
        // If not push the lidar into measurement data buffer
        meas.lidar = lid_raw_data_buffer.front();  // push the first lidar topic
        if (meas.lidar->points.size() <= 1) return false;

        meas.lidar_frame_beg_time =
            lid_header_time_buffer.front();  // generate lidar_frame_beg_time
        meas.lidar_frame_end_time =
            meas.lidar_frame_beg_time +
            meas.lidar->points.back().curvature /
                double(1000);  // calc lidar scan end time
        meas.pcl_proc_cur = meas.lidar;
        lidar_pushed = true;  // flag
      }

      if (imu_en &&
          last_timestamp_imu <
              meas.lidar_frame_end_time) {  // waiting imu message needs to be
        // larger than _lidar_frame_end_time,
        // make sure complete propagate.
        // ROS_ERROR("out sync");
        return false;
      }

      struct MeasureGroup m;  // standard method to keep imu message.

      m.imu.clear();
      m.lio_time = meas.lidar_frame_end_time;
      lidar_end_time = meas.lidar_frame_end_time;
      mtx_buffer.lock();
      while (!imu_buffer.empty()) {
        if (stamp2Sec(imu_buffer.front()->header.stamp) >
            meas.lidar_frame_end_time)
          break;
        m.imu.emplace_back(imu_buffer.front());
        imu_buffer.pop_front();
      }
      lid_raw_data_buffer.pop_front();
      lid_header_time_buffer.pop_front();
      mtx_buffer.unlock();
      sig_buffer.notify_all();

      meas.lio_vio_flg =
          LIO;  // process lidar topic, so timestamp should be lidar scan end.
      meas.measures.emplace_back(m);
      // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
      lidar_pushed = false;  // sync one whole lidar scan.
      return true;

      break;
    }

    case LIVO: {
      /*** For LIVO mode, the time of LIO update is set to be the same as VIO,
       * LIO first than VIO imediatly ***/
      EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
      // double t0 = omp_get_wtime();
      switch (last_lio_vio_flg) {
        // double img_capture_time = meas.lidar_frame_beg_time +
        // exposure_time_init;
        case WAIT:
        case VIO: {
          // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
          double img_capture_time =
              img_time_buffer.front() + exposure_time_init;
          /*** has img topic, but img topic timestamp larger than lidar end
           * time, process lidar topic. After LIO update, the
           * meas.lidar_frame_end_time will be refresh. ***/
          if (meas.last_lio_update_time < 0.0)
            meas.last_lio_update_time = lid_header_time_buffer.front();
          // printf("[ Data Cut ] wait \n");
          // printf("[ Data Cut ] last_lio_update_time: %lf \n",
          // meas.last_lio_update_time);

          double lid_newest_time =
              lid_header_time_buffer.back() +
              lid_raw_data_buffer.back()->points.back().curvature /
                  double(1000);
          double imu_newest_time = stamp2Sec(imu_buffer.back()->header.stamp);

          if (img_capture_time < meas.last_lio_update_time + 0.00001) {
            img_buffer.pop_front();
            img_time_buffer.pop_front();
            RCLCPP_ERROR(this->node->get_logger(),
                         "[ Data Cut ] Throw one image frame! \n");
            return false;
          }

          if (img_capture_time > lid_newest_time ||
              img_capture_time > imu_newest_time) {
            // RCLCPP_ERROR(this->node->get_logger(), "lost first camera
            // frame"); printf("img_capture_time, lid_newest_time,
            // imu_newest_time: %lf , %lf , %lf \n", img_capture_time,
            // lid_newest_time, imu_newest_time);
            return false;
          }

          struct MeasureGroup m;

          // printf("[ Data Cut ] LIO \n");
          // printf("[ Data Cut ] img_capture_time: %lf \n",
          // img_capture_time);
          m.imu.clear();
          m.lio_time = img_capture_time;
          mtx_buffer.lock();
          while (!imu_buffer.empty()) {
            if (stamp2Sec(imu_buffer.front()->header.stamp) > m.lio_time) break;

            if (stamp2Sec(imu_buffer.front()->header.stamp) >
                meas.last_lio_update_time)
              m.imu.emplace_back(imu_buffer.front());

            imu_buffer.pop_front();
            // printf("[ Data Cut ] imu time: %lf \n",
            // stamp2Sec(imu_buffer.front()->header.stamp));
          }
          mtx_buffer.unlock();
          sig_buffer.notify_all();

          *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
          PointCloudXYZI().swap(*meas.pcl_proc_next);

          int lid_frame_num = lid_raw_data_buffer.size();
          int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
          meas.pcl_proc_cur->reserve(max_size);
          meas.pcl_proc_next->reserve(max_size);
          // deque<PointCloudXYZI::Ptr> lidar_buffer_tmp;

          while (!lid_raw_data_buffer.empty()) {
            if (lid_header_time_buffer.front() > img_capture_time) break;
            auto pcl(lid_raw_data_buffer.front()->points);
            double frame_header_time(lid_header_time_buffer.front());
            float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

            for (int i = 0; i < pcl.size(); i++) {
              auto pt = pcl[i];
              if (pcl[i].curvature < max_offs_time_ms) {
                pt.curvature +=
                    (frame_header_time - meas.last_lio_update_time) * 1000.0f;
                meas.pcl_proc_cur->points.emplace_back(pt);
              } else {
                pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
                meas.pcl_proc_next->points.emplace_back(pt);
              }
            }
            lid_raw_data_buffer.pop_front();
            lid_header_time_buffer.pop_front();
          }

          meas.measures.emplace_back(m);
          meas.lio_vio_flg = LIO;
          // meas.last_lio_update_time = m.lio_time;
          // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
          // printf("[ Data Cut ] pcl_proc_cur number: %d \n",
          // meas.pcl_proc_cur
          // ->points.size()); printf("[ Data Cut ] LIO process time: %lf \n",
          // omp_get_wtime() - t0);
          return true;
        }

        case LIO: {
          double img_capture_time =
              img_time_buffer.front() + exposure_time_init;
          meas.lio_vio_flg = VIO;
          // printf("[ Data Cut ] VIO \n");
          meas.measures.clear();
          double imu_time = stamp2Sec(imu_buffer.front()->header.stamp);

          struct MeasureGroup m;
          m.vio_time = img_capture_time;
          m.lio_time = meas.last_lio_update_time;
          m.img = img_buffer.front();
          mtx_buffer.lock();
          // while ((!imu_buffer.empty() && (imu_time < img_capture_time)))
          // {
          //   imu_time = stamp2Sec(imu_buffer.front()->header.stamp);
          //   if (imu_time > img_capture_time) break;
          //   m.imu.emplace_back(imu_buffer.front());
          //   imu_buffer.pop_front();
          //   printf("[ Data Cut ] imu time: %lf \n",
          //   stamp2Sec(imu_buffer.front()->header.stamp));
          // }
          img_buffer.pop_front();
          img_time_buffer.pop_front();
          mtx_buffer.unlock();
          sig_buffer.notify_all();
          meas.measures.emplace_back(m);
          lidar_pushed = false;  // after VIO update, the
                                 // _lidar_frame_end_time will be refresh.
          // printf("[ Data Cut ] VIO process time: %lf \n", omp_get_wtime() -
          // t0);
          return true;
        }

          lidar_end_time = meas.last_lio_update_time;

        default: {
          // printf("!! WRONG EKF STATE !!");
          return false;
        }
          // return false;
      }
      break;
    }

    case ONLY_LO: {
      if (!lidar_pushed) {
        // If not in lidar scan, need to generate new meas
        if (lid_raw_data_buffer.empty()) return false;
        meas.lidar = lid_raw_data_buffer.front();  // push the first lidar topic
        meas.lidar_frame_beg_time =
            lid_header_time_buffer.front();  // generate lidar_beg_time
        meas.lidar_frame_end_time =
            meas.lidar_frame_beg_time +
            meas.lidar->points.back().curvature /
                double(1000);  // calc lidar scan end time
        lidar_pushed = true;
      }
      struct MeasureGroup m;  // standard method to keep imu message.
      m.lio_time = meas.lidar_frame_end_time;
      lidar_end_time = meas.lidar_frame_end_time;
      mtx_buffer.lock();
      lid_raw_data_buffer.pop_front();
      lid_header_time_buffer.pop_front();
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      lidar_pushed = false;  // sync one whole lidar scan.
      meas.lio_vio_flg =
          LO;  // process lidar topic, so timestamp should be lidar scan end.
      meas.measures.emplace_back(m);
      return true;
      break;
    }

    default: {
      printf("!! WRONG SLAM TYPE !!");
      return false;
    }
  }
  RCLCPP_ERROR(this->node->get_logger(), "out sync");
}

void LIVMapper::makeSnapshot(VizSnapshot &snap) {
  if (!local_rgb_cloud->empty()) {
    // snap.local_rgb_cloud_copy = *local_rgb_cloud;
    snap.local_rgb_cloud_copy = local_rgb_cloud;
    snap.has_local_rgb = true;
  }
  snap.last_lidar_ts = last_timestamp_lidar;
  if (vio_manager) {
    snap.vio_mgr = vio_manager;
    vio_manager->pinhole_cam->undistortImage(vio_manager->img_rgb,
                                             snap.img_rgb);
    snap.img_cp = vio_manager->img_cp.clone();
    snap.has_img = !snap.img_rgb.empty();
    snap.fx = vio_manager->fx;
    snap.fy = vio_manager->fy;
    snap.cx = vio_manager->cx;
    snap.cy = vio_manager->cy;
    snap.width = vio_manager->width;
    snap.height = vio_manager->height;
    if (vio_manager->new_frame_) {
      snap.cam_t = vio_manager->new_frame_->T_f_w_.translation();
      snap.cam_q =
          Eigen::Quaterniond(vio_manager->new_frame_->T_f_w_.rotationMatrix());
      snap.image_time = vio_manager->image_time;
      snap.has_cam_pose = true;
    }
  }
}
void LIVMapper::enqueueSnapshot() {
  VizSnapshot snap;
  makeSnapshot(snap);
  // 若没有任何要发布的内容则直接返回
  if (!snap.has_img && !snap.has_cam_pose && !snap.has_local_rgb) return;
  std::unique_lock<std::mutex> lk(viz_queue_mtx_);
  viz_queue_.emplace_back(std::move(snap));
  lk.unlock();
  viz_queue_cv_.notify_one();
}

bool LIVMapper::dequeueOne(VizSnapshot &out) {
  std::lock_guard<std::mutex> lk(viz_queue_mtx_);
  if (viz_queue_.empty()) return false;
  out = std::move(viz_queue_.front());
  viz_queue_.pop_front();
  viz_queue_cv_.notify_one();  // 让可能阻塞的生产者继续
  return true;
}
// 单帧发布（替代原 publish_image_depth_pose）
void LIVMapper::processSnapshot(const VizSnapshot &snap) {
  // Pose
  if (snap.has_cam_pose) {
    geometry_msgs::msg::PoseStamped cam_pose;
    cam_pose.header.stamp = rclcpp::Time(snap.image_time * 1e9);

    cam_pose.header.frame_id = "camera_init";
    Eigen::Matrix3d R_w2c = snap.cam_q.toRotationMatrix();
    Eigen::Vector3d t_w2c = snap.cam_t;
    Eigen::Affine3d T_w2c = Eigen::Affine3d::Identity();
    T_w2c.linear() = R_w2c;
    T_w2c.translation() = t_w2c;
    Eigen::Affine3d T_c2w = T_w2c.inverse();
    Eigen::Quaterniond q_c2w(T_c2w.linear());
    cam_pose.pose.position.x = T_c2w.translation().x();
    cam_pose.pose.position.y = T_c2w.translation().y();
    cam_pose.pose.position.z = T_c2w.translation().z();
    cam_pose.pose.orientation.w = q_c2w.w();
    cam_pose.pose.orientation.x = q_c2w.x();
    cam_pose.pose.orientation.y = q_c2w.y();
    cam_pose.pose.orientation.z = q_c2w.z();

    pubCamPose->publish(cam_pose);
  }

  // RGB
  if (snap.has_img) {
    cv_bridge::CvImage img_msg;
    // 使用图像采集时间或 last_lidar_ts 二选一，这里用 image_time 优先（无则
    // fallback）
    double stamp_sec = snap.image_time;
    img_msg.header.stamp = rclcpp::Time(stamp_sec * 1e9);
    img_msg.header.frame_id = "camera_init";
    img_msg.encoding = sensor_msgs::image_encodings::BGR8;
    img_msg.image = snap.img_rgb;
    pubImage.publish(img_msg.toImageMsg());
  }
  // publish_submap_world
  if (snap.has_local_rgb) {
    publish_submap_world(snap.local_rgb_cloud_copy);
  }
  // Depth
  // if (snap.has_local_rgb && snap.vio_mgr) {
  //   auto cloud_ptr =
  //       std::make_shared<PointCloudXYZRGB>(snap.local_rgb_cloud_copy);
  //   publish_img_depth(pubDepth, cloud_ptr, snap.vio_mgr);
  // }
  // Depth
  if (snap.has_local_rgb && snap.has_img && snap.has_cam_pose &&
      snap.width > 0 && snap.height > 0) {
    publish_img_depth(snap);
  }
}
void LIVMapper::vizTimerCb() {
  VizSnapshot snap;
  bool processed = false;
  while (dequeueOne(snap)) {
    processed = true;
    auto start_time = omp_get_wtime();
    processSnapshot(snap);
    auto end_time = omp_get_wtime();
    RCLCPP_INFO(this->node->get_logger(),
                "Processed publish snapshot in %.3f seconds",
                end_time - start_time);
  }
}

void LIVMapper::publish_img_rgb(const image_transport::Publisher &pubImage,
                                VIOManagerPtr vio_manager) {
  cv::Mat img_rgb = vio_manager->img_cp;
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = this->node->get_clock()->now();
  // out_msg.header.frame_id = "camera_init";
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_rgb;
  pubImage.publish(out_msg.toImageMsg());
}
// ...existing code...
// 优化版：基于快照
// ...existing code...
void LIVMapper::publish_img_depth(const VizSnapshot &snap) {
  if (!img_en) return;
  if (!snap.has_local_rgb || !snap.has_cam_pose || !snap.has_img) return;
  if (snap.width <= 0 || snap.height <= 0) return;

  const int width = snap.width;
  const int height = snap.height;
  const float fx = snap.fx, fy = snap.fy, cx = snap.cx, cy = snap.cy;
  const Eigen::Matrix3d R_w2c = snap.cam_q.toRotationMatrix();
  const Eigen::Vector3d t_w2c = snap.cam_t;

  constexpr float min_z = 0.1f;
  constexpr float max_range = 80.f;
  const size_t N = snap.local_rgb_cloud_copy->points.size();
  const int pixels = width * height;

  // 全局深度数组（最终）
  std::vector<float> depth_global(pixels, max_range);

#pragma omp parallel
  {
    std::vector<float> depth_local(pixels, max_range);
#pragma omp for nowait schedule(static)
    for (long i = 0; i < (long)N; ++i) {
      const auto &pt = snap.local_rgb_cloud_copy->points[i];
      Eigen::Vector3d pw(pt.x, pt.y, pt.z);
      Eigen::Vector3d pc = R_w2c * pw + t_w2c;
      float Z = (float)pc.z();
      if (Z <= min_z || Z > max_range) continue;
      float invZ = 1.0f / Z;
      int u = (int)std::lround(fx * (float)pc.x() * invZ + cx);
      int v = (int)std::lround(fy * (float)pc.y() * invZ + cy);
      if ((unsigned)u >= (unsigned)width || (unsigned)v >= (unsigned)height)
        continue;
      int idx = v * width + u;
      float &cur = depth_local[idx];
      if (Z < cur) cur = Z;
    }
    //
#pragma omp critical
    {
      for (int i = 0; i < pixels; ++i) {
        float dl = depth_local[i];
        if (dl < depth_global[i]) depth_global[i] = dl;
      }
    }
  }

  // 转成 Mat
  cv::Mat depth_img(height, width, CV_32FC1);
  float *depth_ptr = (float *)depth_img.data;
  std::memcpy(depth_ptr, depth_global.data(), pixels * sizeof(float));

  // 统计有效深度、min/max
  float dmin = max_range, dmax = 0.f;
  int valid_cnt = 0;
  for (int i = 0; i < pixels; ++i) {
    float d = depth_ptr[i];
    if (d >= max_range - 1e-6f) {
      depth_ptr[i] = 0.f;
    } else {
      if (d < dmin) dmin = d;
      if (d > dmax) dmax = d;
      valid_cnt++;
    }
  }
  if (valid_cnt == 0) return;
  if (dmax - dmin < 1e-6f) dmax = dmin + 1.f;

  const cv::Mat &base_img = snap.img_cp.empty() ? snap.img_rgb : snap.img_cp;
  if (base_img.empty() || base_img.size() != cv::Size(width, height)) return;
  cv::Mat overlay = base_img.clone();

  static cv::Mat lut;
  if (lut.empty()) {
    cv::Mat grad(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) grad.data[i] = (uchar)i;
    cv::applyColorMap(grad, lut, cv::COLORMAP_JET);
  }

  const float invSpan = 1.f / (dmax - dmin);
  for (int v = 0; v < height; ++v) {
    float *row_d = depth_img.ptr<float>(v);
    cv::Vec3b *row_pix = overlay.ptr<cv::Vec3b>(v);
    for (int u = 0; u < width; ++u) {
      float d = row_d[u];
      if (d <= 0.f) continue;
      float norm = (d - dmin) * invSpan;
      if (norm < 0)
        norm = 0;
      else if (norm > 1)
        norm = 1;
      int idx = (int)(norm * 255.f + 0.5f);
      const cv::Vec3b &c = lut.at<cv::Vec3b>(0, idx);
#ifdef USE_CIRCLE_DRAW
      cv::circle(overlay, {u, v}, 1, cv::Scalar(c[0], c[1], c[2]), cv::FILLED,
                 cv::LINE_8);
#else
      row_pix[u] = c;
#endif
    }
  }

  cv::Mat depth_mm;
  depth_img.convertTo(depth_mm, CV_16UC1, 1000.0);

  cv_bridge::CvImage out_depth, out_overlay;
  double stamp_sec = snap.image_time;
  out_depth.header.stamp = rclcpp::Time(stamp_sec * 1e9);
  out_depth.header.frame_id = "camera";
  out_overlay.header.stamp = out_depth.header.stamp;
  out_overlay.header.frame_id = out_depth.header.frame_id;
  // 发布伪彩覆盖
  out_overlay.encoding = sensor_msgs::image_encodings::BGR8;
  out_depth.encoding = sensor_msgs::image_encodings::MONO16;
  out_overlay.image = overlay;
  out_depth.image = depth_mm;
  pubOverlay.publish(out_overlay.toImageMsg());
  pubDepth.publish(out_depth.toImageMsg());

  RCLCPP_DEBUG(this->node->get_logger(),
               "Depth publish pts=%zu valid=%d range=[%.2f,%.2f] (parallel)", N,
               valid_cnt, dmin, dmax);
}
void LIVMapper::publish_frame_lidar(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        &pubLaserUndistortCloud){
  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*pcl_body_wait_pub, laserCloudmsg);
  PointCloudXYZI().swap(*pcl_body_wait_pub);
  laserCloudmsg.header.stamp = rclcpp::Time(last_timestamp_lidar * 1e9);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserUndistortCloud->publish(laserCloudmsg);
}
void LIVMapper::publish_frame_world(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        &pubLaserCloudFullRes,
    VIOManagerPtr vio_manager) {
  if (pcl_w_wait_pub->empty()) return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  PointCloudXYZRGB::Ptr laserNoGroundCloudWorldRGB(new PointCloudXYZRGB());
  // PointCloudXYZI::Ptr laserCloudWorldXYZ(new PointCloudXYZI());
  if (img_en && pub_rgb_cloud_en) {
    static int pub_num = 1;
    *pcl_wait_pub += *pcl_w_wait_pub;
    if (pub_num == pub_scan_num) {
      pub_num = 1;
      size_t size = pcl_wait_pub->points.size();
      laserCloudWorldRGB->reserve(size);
      laserNoGroundCloudWorldRGB->reserve(size);
      // double inv_expo = _state.inv_expo_time;

      // TODO：publish img_rgb with pcl_w_wait_pub
      //
      cv::Mat img_rgb = vio_manager->img_rgb;
      size_t ground_num = 0;
      for (size_t i = 0; i < size; i++) {
        V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y,
                pcl_wait_pub->points[i].z);
        V3D pf(vio_manager->new_frame_->w2f(p_w));
        if (pf[2] < 0) continue;
        PointTypeRGB pointRGB;
        PointType pointXYZI;
        pointXYZI.x = pcl_wait_pub->points[i].x;
        pointXYZI.y = pcl_wait_pub->points[i].y;
        pointXYZI.z = pcl_wait_pub->points[i].z;
        pointXYZI.intensity = pcl_wait_pub->points[i].intensity;

        pointRGB.x = pcl_wait_pub->points[i].x;
        pointRGB.y = pcl_wait_pub->points[i].y;
        pointRGB.z = pcl_wait_pub->points[i].z;
        V2D pc(vio_manager->new_frame_->w2c(p_w));

        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(),
                                                     3))  // 100
        {
          V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);
          pointRGB.r = pixel[2];
          pointRGB.g = pixel[1];
          pointRGB.b = pixel[0];
          if (pf.norm() > blind_rgb_points) {
            laserCloudWorldRGB->emplace_back(pointRGB);
          }

          // // 选取非地面点云
          if (pf.norm() > blind_rgb_points && pointXYZI.intensity == 0) {
            laserNoGroundCloudWorldRGB->emplace_back(pointRGB);
            ground_num++;
          }
        }
      }
    } else {
      pub_num++;
    }
  }

  // std::cout << "ground_num: " << laserNoGroundCloudWorldRGB->size()
  //           << std::endl;
  // std::cout << "all_num: " << pcl_wait_pub->points.size() << std::endl;
  /*** Publish Frame ***/
  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  // For relocalization, we need to publish the pcl_w_wait_pub
  if (img_en && pub_rgb_cloud_en) {
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  } else {
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg);
  }

  // from last_timestamp_lidar
  laserCloudmsg.header.stamp = rclcpp::Time(last_timestamp_lidar * 1e9);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes->publish(laserCloudmsg);

  // local map
  if (depth_en) {
    static int local_cloud_num = 0;
    local_cloud_num++;
    if (laserCloudWorldRGB->size() > 0 &&
        local_cloud_num % rgb_cloud_interval == 0 && img_en) {
      local_rgb_cloud_buffer.emplace_back(laserCloudWorldRGB);
    }
    // 维护一个局部点云buffer
    if (local_rgb_cloud_buffer.size() > 20) {
      local_rgb_cloud->clear();
      for (auto &cloud : local_rgb_cloud_buffer) {
        *local_rgb_cloud += *cloud;
      }
      // 保持buffer大小不超过10
      while (local_rgb_cloud_buffer.size() > 20) {
        local_rgb_cloud_buffer.pop_front();
      }
    }
  }
  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  if (pcd_save_en) {
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    static int scan_wait_num = 0;
    static int scan_num = 0;
    if (img_en) {
      // 所有上色后的点云
      *pcl_wait_save += *laserCloudWorldRGB;
      // 所有非地面点云
      *pcl_wait_save_global_rgb += *laserNoGroundCloudWorldRGB;
      // LiDAR FOV下的原始点云
      *pcl_wait_save_global += *pcl_w_wait_pub;
    } else {
      *pcl_wait_save_intensity += *pcl_w_wait_pub;
      *pcl_wait_save_intensity_cp += *pcl_w_wait_pub;
    }
    scan_num++;
    scan_wait_num++;

    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) &&
        pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval) {
      pcd_index++;
      string all_points_dir(string(string(ROOT_DIR) + "Log/PCD/") +
                            to_string(pcd_index) + string(".pcd"));
      pcl::PCDWriter pcd_writer;
      if (pcd_save_en) {
        if (img_en) {
          pcd_writer.writeBinary(
              all_points_dir,
              *pcl_wait_save);  // pcl::io::savePCDFileASCII(all_points_dir,
                                // *pcl_wait_save);
          // 每次清空
          // PointCloudXYZRGB().swap(*pcl_wait_save);
        } else {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
          // PointCloudXYZI().swap(*pcl_wait_save_intensity);
        }
        Eigen::Quaterniond q(_state.rot_end);
        fout_pcd_pos << _state.pos_end[0] << " " << _state.pos_end[1] << " "
                     << _state.pos_end[2] << " " << q.w() << " " << q.x() << " "
                     << q.y() << " " << q.z() << " " << endl;
        scan_wait_num = 0;
      }
    }
  }

  PointCloudXYZI().swap(*pcl_wait_pub);
  PointCloudXYZI().swap(*pcl_w_wait_pub);
  PointCloudXYZRGB().swap(*pcl_wait_save);
  PointCloudXYZI().swap(*pcl_wait_save_intensity);
}

void LIVMapper::publish_submap_world(const PointCloudXYZRGB::Ptr &rgb_cloud) {
  sensor_msgs::msg::PointCloud2 laserLocalCloudmsg;
  pcl::toROSMsg(*rgb_cloud, laserLocalCloudmsg);
  // from last_timestamp_lidar
  laserLocalCloudmsg.header.stamp = rclcpp::Time(last_timestamp_lidar * 1e9);
  laserLocalCloudmsg.header.frame_id = "camera_init";
  pubLaserLocalCloudFullRes->publish(laserLocalCloudmsg);
}

void LIVMapper::publish_visual_sub_map(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        &pubSubVisualMap) {
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);
  int size = laserCloudFullRes->points.size();
  if (size == 0) return;
  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1) {
    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = this->node->get_clock()->now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualMap->publish(laserCloudmsg);
  }
}

void LIVMapper::publish_effect_world(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        &pubLaserCloudEffect,
    const std::vector<PointToPlane> &ptpl_list) {
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++) {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = this->node->get_clock()->now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect->publish(laserCloudFullRes3);
}

template <typename T>
void LIVMapper::set_posestamp(T &out) {
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

void LIVMapper::publish_odometry(
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr
        &pubOdomAftMapped) {
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";
  // odomAftMapped.header.stamp =
  //     this->node->get_clock()
  //         ->now();  //.ros::Time()fromSec(last_timestamp_lidar);
  odomAftMapped.header.stamp = rclcpp::Time(
      last_timestamp_lidar * 1e9);  // Convert seconds to nanoseconds
  set_posestamp(odomAftMapped.pose.pose);

  static std::shared_ptr<tf2_ros::TransformBroadcaster> br;
  br = std::make_shared<tf2_ros::TransformBroadcaster>(this->node);

  tf2::Transform transform;
  tf2::Quaternion q;
  transform.setOrigin(
      tf2::Vector3(_state.pos_end(0), _state.pos_end(1), _state.pos_end(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  br->sendTransform(geometry_msgs::msg::TransformStamped(createTransformStamped(
      transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped")));
  pubOdomAftMapped->publish(odomAftMapped);
}

void LIVMapper::publish_mavros(
    const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
        &mavros_pose_publisher) {
  msg_body_pose.header.stamp = this->node->get_clock()->now();
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher->publish(msg_body_pose);
}

void LIVMapper::publish_path(
    const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pubPath) {
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = this->node->get_clock()->now();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.emplace_back(msg_body_pose);
  pubPath->publish(path);
}
