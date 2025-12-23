// ROS2 Humble version of pose_estimator.cpp (partial conversion -
// initialization only)
#include "pose_estimator.h"

#include "../FRICP-toolkit/registeration.h"

pose_estimator::pose_estimator(rclcpp::Node::SharedPtr &node_) : node(node_) {
  allocateMemory();

  this->node->declare_parameter("relo.priorDir", std::string(" "));
  this->node->declare_parameter("relo.cloudTopic",
                                std::string("/cloud_registered"));
  this->node->declare_parameter("relo.poseTopic", std::string("/Odometry"));
  this->node->declare_parameter("relo.searchDis", 10.0);
  this->node->declare_parameter("relo.searchNum", 3);
  this->node->declare_parameter("relo.trustDis", 5.0);
  this->node->declare_parameter("relo.regMode", 5);
  this->node->declare_parameter("relo.extrinsic_T", std::vector<double>());
  this->node->declare_parameter("relo.extrinsic_R", std::vector<double>());
  this->node->declare_parameter("relo.relo_interval", 10);
  // external_flg
  this->node->declare_parameter("relo.sc_init_enable", false);
  this->node->declare_parameter("relo.external_flg", false);

  this->node->get_parameter("relo.priorDir", priorDir);
  this->node->get_parameter("relo.cloudTopic", cloudTopic);
  this->node->get_parameter("relo.poseTopic", poseTopic);
  this->node->get_parameter("relo.searchDis", searchDis);
  this->node->get_parameter("relo.searchNum", searchNum);
  this->node->get_parameter("relo.trustDis", trustDis);
  this->node->get_parameter("relo.regMode", regMode);
  this->node->get_parameter("relo.extrinsic_T", extrinT_);
  this->node->get_parameter("relo.extrinsic_R", extrinR_);
  // relo_interval
  this->node->get_parameter("relo.relo_interval", relo_interval);
  // external_flg
  this->node->get_parameter("relo.external_flg", external_flg);
  // sc_init_enable
  this->node->get_parameter("relo.sc_init_enable", sc_init_enable);

  extrinT << VEC_FROM_ARRAY(extrinT_);
  extrinR << MAT_FROM_ARRAY(extrinR_);
  // std::cout << "extrinT: " << extrinT << "\n" << "extrinR: " << extrinR
  //           << std::endl;

  // Eigen::Matrix<double, 3, 1> euler_ext = RotMtoEuler(extrinR);
  Eigen::Matrix<double, 3, 1> euler_ext =
      extrinR.eulerAngles(0, 1, 2);  // roll, pitch, yaw
  pose_ext.x = extrinT(0);
  pose_ext.y = extrinT(1);
  pose_ext.z = extrinT(2);
  pose_ext.roll = euler_ext(0, 0);
  pose_ext.pitch = euler_ext(1, 0);
  pose_ext.yaw = euler_ext(2, 0);

  pose_zero.x = 0.0;
  pose_zero.y = 0.0;
  pose_zero.z = 0.0;
  pose_zero.roll = 0.0;
  pose_zero.pitch = 0.0;
  pose_zero.yaw = 0.0;
  currentCloudTime = 0.0;
  subCloud = this->node->create_subscription<sensor_msgs::msg::PointCloud2>(
      cloudTopic, 100,
      std::bind(&pose_estimator::cloudCBK, this, std::placeholders::_1));

  subPose = this->node->create_subscription<nav_msgs::msg::Odometry>(
      poseTopic, 500,
      std::bind(&pose_estimator::poseCBK, this, std::placeholders::_1));

  pubCloud =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud", 10);
  pubPose = this->node->create_publisher<nav_msgs::msg::Odometry>("/pose", 10);

  fout_relo.open(priorDir + "relo_pose.txt", std::ios::out);

  subExternalPose =
      this->node
          ->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
              "/initialpose", 10,
              std::bind(&pose_estimator::externalCBK, this,
                        std::placeholders::_1));

  pubPriorMap = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/prior_map", 10);
  pubPriorPath = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/prior_path", 10);
  pubReloWorldCloud =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/relo_world_cloud", 10);
  pubRelocBodyCloud =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/reloc_body_cloud", 10);

  pubInitCloud = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/init_cloud", 10);
  pubNearCloud = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/near_cloud", 10);
  pubMeasurementEdge =
      this->node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "measurement", 10);
  // pubPath = this->node->create_publisher<nav_msgs::msg::Path>("/path_loc",
  // 10);
  RCLCPP_INFO(this->node->get_logger(), "rostopic is ok");

  sessions.push_back(MultiSession::Session(1, "priorMap", priorDir, true));
  *priorMap += *sessions[0].globalMap;
  *priorPath += *sessions[0].cloudKeyPoses3D;
  publishCloud(pubPriorMap, priorMap, this->node->now(), "world");

  downSizeFilterPub.setLeafSize(3.0, 3.0, 3.0);
  downSizeFilterLocalMap.setLeafSize(0.2, 0.2, 0.2);
  downSizeFilterCurrentCloud.setLeafSize(0.2, 0.2, 0.2);

  height = priorPath->points[0].z;

  kdtreeGlobalMapPoses->setInputCloud(priorPath);
  kdtreeGlobalMapPoses_copy->setInputCloud(priorPath);
  RCLCPP_INFO(this->node->get_logger(), "load prior knowledge");

  // reg.push_back(Registeration(regMode));
  invalid_idx.emplace_back(-1);
}

void pose_estimator::allocateMemory() {
  priorMap.reset(new pcl::PointCloud<PointTypeXYZI>());
  priorPath.reset(new pcl::PointCloud<PointTypeXYZI>());
  reloCloudInMap.reset(new pcl::PointCloud<PointTypeXYZI>());
  cloudInBody.reset(new pcl::PointCloud<PointTypeXYZI>());
  initCloud.reset(new pcl::PointCloud<PointTypeXYZI>());
  initCloudInOdom.reset(new pcl::PointCloud<PointTypeXYZI>());
  nearCloud.reset(new pcl::PointCloud<PointTypeXYZI>());
  localMapDS.reset(new pcl::PointCloud<PointTypeXYZI>());
  kdtreeGlobalMapPoses.reset(new pcl::KdTreeFLANN<PointTypeXYZI>());
  kdtreeGlobalMapPoses_copy.reset(new pcl::KdTreeFLANN<PointTypeXYZI>());
  currentCloud.reset(new pcl::PointCloud<PointTypeXYZI>());
  currentCloudDs.reset(new pcl::PointCloud<PointTypeXYZI>());
  currentCloudInMap.reset(new pcl::PointCloud<PointTypeXYZI>());
  currentCloudDsInOdom.reset(new pcl::PointCloud<PointTypeXYZI>());
}

void pose_estimator::cloudCBK(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  pcl::PointCloud<PointTypeXYZI>::Ptr msgCloud(
      new pcl::PointCloud<PointTypeXYZI>());
  pcl::fromROSMsg(*msg, *msgCloud);
  if (msgCloud->empty()) return;
  msgCloud->width = msgCloud->points.size();
  msgCloud->height = 1;
  cloudBufferMutex.lock();
  cloudBuffer.emplace_back(msgCloud);
  cloudtimeBuffer.emplace_back(msg->header.stamp.sec +
                               msg->header.stamp.nanosec * 1e-9);
  cloudBufferMutex.unlock();
  sig_buffer.notify_all();
}

void pose_estimator::poseCBK(const nav_msgs::msg::Odometry::SharedPtr msg) {
  Eigen::Affine3f pose;
  Eigen::Vector4d q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                    msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
  quaternionNormalize(q);
  Eigen::Matrix3d rot = quaternionToRotation(q);
  pose.linear() = rot.cast<float>();
  pose.translation() << msg->pose.pose.position.x, msg->pose.pose.position.y,
      msg->pose.pose.position.z;
  double time_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

  poseBufferMutex.lock();
  poseBuffer_6D.push_back(pose);
  posetimeBuffer.push_back(time_stamp);
  poseBufferMutex.unlock();
  sig_buffer.notify_all();
}

void pose_estimator::externalCBK(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
  // if (external_flg) return;
  RCLCPP_INFO(this->node->get_logger(),
              "please set your external pose now ...");
  externalPose.x = msg->pose.pose.position.x;
  externalPose.y = msg->pose.pose.position.y;
  externalPose.z = 0.0;
  tf2::Quaternion q;
  tf2::fromMsg(msg->pose.pose.orientation, q);
  double roll, pitch, yaw;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  externalPose.roll = 0.0;
  externalPose.pitch = 0.0;
  externalPose.yaw = yaw;
  RCLCPP_INFO(this->node->get_logger(),
              "Get initial pose: %.6f %.6f %.6f %.6f %.6f %.6f", externalPose.x,
              externalPose.y, externalPose.z, externalPose.roll,
              externalPose.pitch, externalPose.yaw);
  receive_ext_flg = true;
}

void pose_estimator::run(rclcpp::Node::SharedPtr &node) {
  rclcpp::Rate rate(50);
  if (!rclcpp::ok()) {
    RCLCPP_ERROR(node->get_logger(), "Node is not ok, exiting run method.");
    return;
  }
  while (rclcpp::ok()) {
    rclcpp::spin_some(this->node);

    // Check if there are enough clouds and poses in the buffers
    cloudBufferMutex.lock();
    poseBufferMutex.lock();
    if (cloudBuffer.empty() || poseBuffer_6D.empty()) {
      cloudBufferMutex.unlock();
      poseBufferMutex.unlock();
      rate.sleep();
      continue;
    }

    // Synchronize poseBuffer_6D and cloudBuffer
    while (!poseBuffer_6D.empty() && !cloudBuffer.empty() &&
           std::abs(posetimeBuffer.front() - cloudtimeBuffer.front()) >
               MAX_TIME_DIFF) {
      if (posetimeBuffer.front() < cloudtimeBuffer.front()) {
        RCLCPP_WARN(
            this->node->get_logger(),
            "Pose timestamp: %.5f is earlier than cloud timestamp: %.5f",
            posetimeBuffer.front(), cloudtimeBuffer.front());
        poseBuffer_6D.pop_front();
        posetimeBuffer.pop_front();
      } else {
        cloudBuffer.pop_front();
        cloudtimeBuffer.pop_front();
      }
    }
    // print time stamp of cloud and pose
    if (!poseBuffer_6D.empty() && !cloudtimeBuffer.empty()) {
      currentCloud = cloudBuffer.front();
      cloudBuffer.pop_front();
      currentCloudTime = cloudtimeBuffer.front();
      cloudtimeBuffer.pop_front();
      cloudBufferMutex.unlock();

      currentPoseInOdom = poseBuffer_6D.front();
      poseBuffer_6D.pop_front();
      posetimeBuffer.pop_front();
      poseBufferMutex.unlock();

      sig_buffer.notify_all();

    } else {
      RCLCPP_WARN(this->node->get_logger(),
                  "Buffers are empty, cannot print timestamps.");
      cloudBufferMutex.unlock();
      poseBufferMutex.unlock();
      sig_buffer.notify_all();
      continue;
    }

    // Initialize the pose if not done yet
    if (!global_flg) {
      if (cout_count_ < 1) {
        std::cout << ANSI_COLOR_RED
                  << "wait for global pose initialization ... "
                  << ANSI_COLOR_RESET << std::endl;
      }
      global_flg = globalRelo();
      cout_count_ = 1;
      lastPoseInMap = currentPoseInMap;
      lastPoseInOdom = currentPoseInOdom;
      continue;
    }

    pcl::PointCloud<PointTypeXYZI>::Ptr relo_pt =
        std::make_shared<pcl::PointCloud<PointTypeXYZI>>();


    // Lk to Lk-1
    deltaPose = lastPoseInOdom.inverse() * currentPoseInOdom;

    // predict current pose in map frame
    currentPoseInMap = lastPoseInMap * deltaPose;
    PointTypeXYZI currentPose3dInMap;
    currentPose3dInMap.x = currentPoseInMap.translation().x();
    currentPose3dInMap.y = currentPoseInMap.translation().y();
    currentPose3dInMap.z = currentPoseInMap.translation().z();
    relo_pt->points.emplace_back(currentPose3dInMap);

    currentCloudDs->clear();
    downSizeFilterCurrentCloud.setInputCloud(currentCloud);
    downSizeFilterCurrentCloud.filter(*currentCloudDs);
    std::cout << "current ds cloud in lidar size: " << currentCloud->points.size()
              << std::endl;

    bool relo_success = true;
    // 初始化稳定后降低重定位频率
    if (!relo_pt->points.empty() && easyToRelo(relo_pt->points[0]) &&
        idx % relo_interval == 0 && currentCloudDs->points.size()>0) {
      relo_success = relocalization();
      relo_pt->clear();
      if (!relo_success) {
        lio_incremental();
      }
    } else {
      lio_incremental();
    }
    idx++;
    lastPoseInMap = currentPoseInMap;
    lastPoseInOdom= currentPoseInOdom;

    rate.sleep();
  }
}
// 直接计算lidar to map的位姿
bool pose_estimator::relocalization() {
  std::cout << ANSI_COLOR_GREEN << "relo mode for frame: " << idx
            << ANSI_COLOR_RESET << std::endl;

  nearCloud->clear();
  localMapDS->clear();
  for (auto &it : idxVec) {
    *nearCloud +=
        *transformPointCloud(sessions[0].cloudKeyFrames[it].all_cloud,
                             &sessions[0].cloudKeyPoses6D->points[it]);
  }

  // downsize
  downSizeFilterLocalMap.setLeafSize(0.20, 0.20, 0.20);
  downSizeFilterLocalMap.setInputCloud(nearCloud);
  downSizeFilterLocalMap.filter(*localMapDS);
  std::cout << "downsized local map size: " << localMapDS->points.size()
            << std::endl;
  if (localMapDS->points.size() < 1000) {
    std::cout << "local map size is too small, skip this relo" << std::endl;
    return false;
  }
  // // 统计耗时
  auto start = std::chrono::high_resolution_clock::now();
  // TODO: overflow when ndt registeration. icp is working well
  pcl::IterativeClosestPoint<PointTypeXYZI, PointTypeXYZI> NDT;
  NDT.setMaxCorrespondenceDistance(0.20);
  NDT.setMaximumIterations(50);
  NDT.setTransformationEpsilon(1e-6);
  NDT.setEuclideanFitnessEpsilon(1e-6);
  NDT.setRANSACIterations(1);
  NDT.setInputSource(currentCloudDs);
  NDT.setInputTarget(localMapDS);
  // // NDT = pcl::NormalDistributionsTransform<PointTypeXYZI,
  // //                                         PointTypeXYZI>();  // 重置NDT
  // NDT.setResolution(0.5f);             // 体素分辨率(米)，可在 0.5~2.0 调整
  // NDT.setStepSize(0.1);                // line search 步长
  // NDT.setTransformationEpsilon(1e-4);  // 收敛阈值
  // NDT.setMaximumIterations(60);        // 迭代次数
  // NDT.setInputSource(currentCloudDsInOdom);
  // NDT.setInputTarget(localMapDS);
  Eigen::Affine3f init_pose = currentPoseInMap;
  pcl::PointCloud<PointTypeXYZI>::Ptr alignedCloud(
      new pcl::PointCloud<PointTypeXYZI>());
  NDT.align(*alignedCloud,init_pose.matrix());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "NDT registration time: " << elapsed.count() << " seconds"
            << std::endl;
  Eigen::Matrix4f transform = NDT.getFinalTransformation();
  Eigen::Matrix3d rot = transform.matrix().block<3, 3>(0, 0).cast<double>();
  Eigen::Vector3d linear = transform.matrix().block<3, 1>(0, 3).cast<double>();
  std::cout << "NDT has converged: " << NDT.hasConverged()
            << " with score: " << NDT.getFitnessScore() << std::endl;
  std::cout << "NDT lidar to map transformation: " << linear.transpose() << " "
            << rot.eulerAngles(0, 1, 2).transpose() << std::endl;
  if (!NDT.hasConverged() || NDT.getFitnessScore() > 1.0) {
    std::cout << "NDT did not converge or fitness score too high, skip this "
                 "relo"
              << std::endl;
    return false;
  }

  // lidar to map
  currentPoseInMap = Eigen::Affine3f(transform);
  // cloud in map frame
  reloCloudInMap->clear();
  transformPointCloud(currentCloudDs, currentPoseInMap, reloCloudInMap);
  publishCloud(pubReloWorldCloud, reloCloudInMap, this->node->now(), "world");

  // body pose in map
  Eigen::Affine3f lidar2body;
  lidar2body.translation() << 0.0, 0.0, 0.0;
  Eigen::Matrix3f extrinR;
  extrinR << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;
  lidar2body.linear() = extrinR.cast<float>();

  // cloud in body frame for local planning 
  cloudInBody->clear();
  transformPointCloud(currentCloud, lidar2body, cloudInBody);
  publishCloud(pubRelocBodyCloud, cloudInBody,
               /*rclcpp::Time(currentCloudTime * 1e9)*/ this->node->now(),
               "base_link");

  std::cout << "lidar to map transformation: " << currentPoseInMap.translation().x()
            << " " << currentPoseInMap.translation().y() << " "
            << currentPoseInMap.translation().z() << " "
            << currentPoseInMap.rotation().eulerAngles(0, 1, 2).transpose()
            << std::endl;
  publish_odometry(currentPoseInMap);

  return true;
}

void pose_estimator::lio_incremental() {
  std::cout << ANSI_COLOR_RED << "livo mode for frame: " << idx
            << ANSI_COLOR_RESET << std::endl;

  Eigen::Affine3f lidar2body;
  lidar2body.translation() << 0.0, 0.0, 0.0;
  Eigen::Matrix3f extrinR;
  extrinR << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;
  lidar2body.linear() = extrinR.cast<float>();

  // cloud in body frame
  cloudInBody->clear();
  transformPointCloud(currentCloud, lidar2body, cloudInBody);
  publishCloud(pubRelocBodyCloud, cloudInBody,
               /*rclcpp::Time(currentCloudTime * 1e9)*/ this->node->now(),
               "base_link");
  
  // cloud in map frame
  reloCloudInMap->clear();
  transformPointCloud(currentCloudDs, currentPoseInMap, reloCloudInMap);
  publishCloud(pubReloWorldCloud, reloCloudInMap, this->node->now(), "world");

  std::cout << "livo transformation in map: " << currentPoseInMap.translation().x()
            << " " << currentPoseInMap.translation().y() << " "
            << currentPoseInMap.translation().z() << " "
            << currentPoseInMap.rotation().eulerAngles(0, 1, 2).transpose()
            << std::endl;
  publish_odometry(currentPoseInMap);
}
void pose_estimator::publish_odometry(const Eigen::Affine3f &trans_aft) {
  Eigen::Matrix<double, 3, 3> ang_rot = trans_aft.rotation().cast<double>();
  Eigen::Quaterniond quaternion(ang_rot);
  odomAftMapped.pose.pose.position.x = trans_aft.translation().x();
  odomAftMapped.pose.pose.position.y = trans_aft.translation().y();
  odomAftMapped.pose.pose.position.z = trans_aft.translation().z();
  odomAftMapped.pose.pose.orientation.x = quaternion.x();
  odomAftMapped.pose.pose.orientation.y = quaternion.y();
  odomAftMapped.pose.pose.orientation.z = quaternion.z();
  odomAftMapped.pose.pose.orientation.w = quaternion.w();
  publish_odometry(pubPose);
}

bool pose_estimator::easyToRelo(const PointTypeXYZI &pose3d) {
  idxVec.clear();
  disVec.clear();
  idxVec_copy.clear();
  disVec_copy.clear();

  // Perform radius search on the primary kdtree
  kdtreeGlobalMapPoses->radiusSearch(pose3d, searchDis, idxVec, disVec);

  // Check if the search results are valid
  if (!disVec.empty() && disVec[0] <= searchDis && idxVec.size() > searchNum) {
    return true;
  }
  // Perform radius search on the secondary kdtree with an extended radius
  kdtreeGlobalMapPoses_copy->radiusSearchT(pose3d, searchDis * 2.0, idxVec_copy,
                                           disVec_copy);

  if (idxVec_copy.size() > 4 && disVec[0] <= searchDis * 2.0) {
    std::cout << ANSI_COLOR_RED << "relo by secondary search with "
              << idxVec_copy.size() << " points" << ANSI_COLOR_RESET
              << std::endl;
    // If the secondary search yields enough results, return true
    return true;
  }
  // Validate results from the secondary search
  // for (const auto &index : idxVec_copy) {
  //   if (priorPath->points.size() >
  //       index + 100) {  // Ensure sufficient prior points
  //     std::cout << ANSI_COLOR_GREEN_BG << "lio -> relo with "
  //               << priorPath->points.size() << " to " << index
  //               << ANSI_COLOR_RESET << std::endl;
  //     idxVec.emplace_back(index);
  //     // return true;
  //   }
  // }

  // for (const auto &index : idxVec_copy) {
  //     idxVec.emplace_back(index);
  // }
  return false;
}

bool pose_estimator::globalRelo() {
  int detectID = -1;
  static int cloud_count = 0;

  if (!sc_flg && sc_init_enable) {
    // 点云过少不进行初始化
    if (currentCloud->points.size() < 5000) {
      std::cout << ANSI_COLOR_RED
                << "current cloud size is too small, wait for next frame ..."
                << ANSI_COLOR_RESET << std::endl;
      return false;
    }
    std::cout << ANSI_COLOR_GREEN << "global relo by sc ... "
              << ANSI_COLOR_RESET << std::endl;

    // Must be body frame when calculating scancontext
    pcl::PointCloud<PointTypeXYZI>::Ptr cloud_lidar = currentCloud;

    Eigen::MatrixXd initSC = sessions[0].scManager.makeScancontext(*cloud_lidar);
    Eigen::MatrixXd ringkey =
        sessions[0].scManager.makeRingkeyFromScancontext(initSC);
    Eigen::MatrixXd sectorkey =
        sessions[0].scManager.makeSectorkeyFromScancontext(initSC);
    std::vector<float> polarcontext_invkey_vec =
        ScanContext::eig2stdvec(ringkey);
    detectResult = sessions[0].scManager.detectClosestKeyframeID(
        0, invalid_idx, polarcontext_invkey_vec, initSC);

    std::cout << " current cloud size: " << cloud_lidar->points.size()
              << std::endl;
    std::cout << ANSI_COLOR_RED << "init relocalization by current SC id: " << 0
              << " in prior map's SC id: " << detectResult.first
              << " yaw offset: " << detectResult.second << ANSI_COLOR_RESET
              << std::endl;
    if (detectResult.first != -1) {
      detectID = detectResult.first;
      PointTypePose pose_com;
      pose_com.x = 0.0;
      pose_com.y = 0.0;
      pose_com.z = 0.0;
      pose_com.roll = 0.0;
      pose_com.pitch = 0.0;
      pose_com.yaw = -detectResult.second;

      initCloud->clear();
      *initCloud += *getAddCloud(cloud_lidar, pose_com, pose_ext);

      nearCloud->clear();
      *nearCloud += *transformPointCloud(
          sessions[0].cloudKeyFrames[detectResult.first].all_cloud,
          &sessions[0].cloudKeyPoses6D->points[detectResult.first]);
    } else {
      initCloud->clear();
      *initCloud += *cloud_lidar;
      std::cout << ANSI_COLOR_RED << "can not relo by SC ... "
                << ANSI_COLOR_RESET << std::endl;
    }
  }

  std::cout << ANSI_COLOR_GREEN << "global relocalization processing ... "
            << ANSI_COLOR_RESET << std::endl;

  if (detectID > -1) {
    std::cout << ANSI_COLOR_GREEN << "init relo by SC-pose ... "
              << ANSI_COLOR_RESET << std::endl;
    std::cout << ANSI_COLOR_GREEN << "use prior frame " << detectID
              << " to relo init cloud ..." << ANSI_COLOR_RESET << std::endl;

    nearCloud->clear();
    PointTypeXYZI tmp;
    PointTypePose poseSC =
        sessions[0].cloudKeyPoses6D->points[detectResult.first];
    tmp.x = poseSC.x;
    tmp.y = poseSC.y;
    tmp.z = poseSC.z;

    idxVec.clear();
    disVec.clear();
    kdtreeGlobalMapPoses->nearestKSearch(tmp, searchNum, idxVec, disVec);
    for (int i = 0; i < idxVec.size(); i++) {
      *nearCloud +=
          *transformPointCloud(sessions[0].cloudKeyFrames[idxVec[i]].all_cloud,
                               &sessions[0].cloudKeyPoses6D->points[idxVec[i]]);
    }

    std::cout << "raw local map point cloud size: " << nearCloud->points.size()
              << std::endl;

    PointTypePose poseOffset;
    poseOffset.x = poseSC.x;
    poseOffset.y = poseSC.y;
    poseOffset.z = poseSC.z;
    poseOffset.roll = poseSC.roll;
    poseOffset.pitch = poseSC.pitch;
    poseOffset.yaw = poseSC.yaw;
    std::cout << "sc prior frame pose: " << poseOffset.x << " " << poseOffset.y
              << " " << poseOffset.z << " " << poseOffset.roll << " "
              << poseOffset.pitch << " " << poseOffset.yaw << std::endl;
    initCloud = transformPointCloud(initCloud, &poseOffset);
    std::cout << "init cloud size: " << initCloud->points.size() << std::endl;

    std::cout << ANSI_COLOR_GREEN << "get precise pose by NDT ... "
              << ANSI_COLOR_RESET << std::endl;
    // Eigen::MatrixXd transform = reg[0].run(initCloud, nearCloud);
    pcl::PointCloud<PointTypeXYZI>::Ptr laserCloudSource(
        new pcl::PointCloud<PointTypeXYZI>());
    pcl::copyPointCloud(*initCloud, *laserCloudSource);
    pcl::PointCloud<PointTypeXYZI>::Ptr laserCloudTarget(
        new pcl::PointCloud<PointTypeXYZI>());

    pcl::copyPointCloud(*nearCloud, *laserCloudTarget);
    std::cout << "downsized initialization local map point cloud size: "
              << laserCloudTarget->points.size() << std::endl;
    pcl::IterativeClosestPoint<PointTypeXYZI, PointTypeXYZI> ndt;
    ndt.setMaxCorrespondenceDistance(0.1);
    ndt.setMaximumIterations(50);
    ndt.setTransformationEpsilon(1e-6);
    ndt.setEuclideanFitnessEpsilon(1e-6);
    ndt.setRANSACIterations(1);
    ndt.setInputSource(laserCloudSource);
    ndt.setInputTarget(laserCloudTarget);
    pcl::PointCloud<PointTypeXYZI>::Ptr alignedCloud(
        new pcl::PointCloud<PointTypeXYZI>());
    Eigen::Matrix4f matricInitGuess = Eigen::Matrix4f::Identity();
    ndt.align(*alignedCloud, matricInitGuess);
    Eigen::Matrix4f transform = ndt.getFinalTransformation();

    // 初始化不准，后续可以纠正
    if (!ndt.hasConverged() || ndt.getFitnessScore() > 0.05) {
      std::cout << ANSI_COLOR_RED
                << "NDT registration failed, fitness score too high: "
                << ndt.getFitnessScore() << ANSI_COLOR_RESET << std::endl;
      invalid_idx.emplace_back(detectID);
      sc_flg = false;
      return false;
    } else {
      std::cout << "NDT has converged: " << ndt.hasConverged()
                << " with score: " << ndt.getFitnessScore() << std::endl;
      sc_flg = true;
    }
    Eigen::Matrix3d rot = transform.matrix().block<3, 3>(0, 0).cast<double>();
    Eigen::Vector3d linear =
        transform.matrix().block<3, 1>(0, 3).cast<double>();
    // Eigen::Matrix<double, 3, 1> euler = RotMtoEuler(rot);
    Eigen::Matrix<double, 3, 1> euler =
        rot.eulerAngles(0, 1, 2);  // roll, pitch, yaw

    PointTypePose poseReg;
    poseReg.x = linear(0, 0);
    poseReg.y = linear(1, 0);
    poseReg.z = linear(2, 0);
    poseReg.roll = euler(0, 0);
    poseReg.pitch = euler(1, 0);
    poseReg.yaw = euler(2, 0);

    Eigen::Affine3f trans_com =
        pcl::getTransformation(0.0, 0.0, 0.0, 0.0, 0.0, -detectResult.second);
    Eigen::Affine3f trans_offset = pcl::getTransformation(
        poseOffset.x, poseOffset.y, poseOffset.z, poseOffset.roll,
        poseOffset.pitch, poseOffset.yaw);
    Eigen::Affine3f trans_reg =
        pcl::getTransformation(poseReg.x, poseReg.y, poseReg.z, poseReg.roll,
                               poseReg.pitch, poseReg.yaw);

    Eigen::Affine3f trans_init =
        trans_com * trans_offset * trans_reg * currentPoseInOdom.inverse();

    float pose_init[6];
    pcl::getTranslationAndEulerAngles(trans_init, pose_init[0], pose_init[1],
                                      pose_init[2], pose_init[3], pose_init[4],
                                      pose_init[5]);
    initPose.x = pose_init[0];
    initPose.y = pose_init[1];
    initPose.z = pose_init[2];
    initPose.roll = pose_init[3];
    initPose.pitch = pose_init[4];
    initPose.yaw = pose_init[5];

    global_flg = true;
    std::cout << ANSI_COLOR_GREEN << "get optimized pose: " << initPose.x << " "
              << initPose.y << " " << initPose.z << " " << initPose.roll << " "
              << initPose.pitch << " " << initPose.yaw << ANSI_COLOR_RESET
              << std::endl;
    std::cout << ANSI_COLOR_GREEN
              << "init relocalization has been finished ... "
              << ANSI_COLOR_RESET << std::endl;
  } else if (external_flg && receive_ext_flg) {
    std::cout << ANSI_COLOR_GREEN << "init relo by external-pose ... "
              << ANSI_COLOR_RESET << std::endl;
    // 初始化时对点云进行累加，至少5帧 currentCloud在lidar系下
    cloud_count++;
    pcl::PointCloud<PointTypeXYZI>::Ptr tmpCloud(new pcl::PointCloud<PointTypeXYZI>());
    transformPointCloud(currentCloud, currentPoseInOdom, tmpCloud);
    *initCloudInOdom +=*tmpCloud;
    if(cloud_count < 5) {
      std::cout << ANSI_COLOR_RED
                << "wait for more cloud frame, current frame count: "
                << cloud_count << ANSI_COLOR_RESET << std::endl;
      return false;
    }
    cloud_count = 0;
    std::cout << "init cloud size: " << initCloudInOdom->points.size() << std::endl;
    if (initCloudInOdom->points.size() < 2000) {
      std::cout << ANSI_COLOR_RED
                << "current cloud size is too small, wait for next frame ..."
                << ANSI_COLOR_RESET << std::endl;
      return false;
    }

    // odom to map
    // odom frame has been aligned to gravity direction
    PointTypePose pose_offset;
    pose_offset.x = externalPose.x;
    pose_offset.y = externalPose.y;
    pose_offset.z = 0.0;
    pose_offset.roll = 0.0;
    pose_offset.pitch = 0.0;
    pose_offset.yaw = externalPose.yaw;
    Eigen::Affine3f init_odom2map = pcl::getTransformation(
        pose_offset.x, pose_offset.y, pose_offset.z, pose_offset.roll,
        pose_offset.pitch, pose_offset.yaw);

    // search nearby keyframes
    idxVec.clear();
    disVec.clear();
    PointTypeXYZI tmp;
    tmp.x = externalPose.x;
    tmp.y = externalPose.y;
    tmp.z = externalPose.z;
    kdtreeGlobalMapPoses->radiusSearchT(tmp, searchDis * 2, idxVec, disVec);
    PointTypePose pose_new = sessions[0].cloudKeyPoses6D->points[idxVec[0]];
    
    // get local map for initialization
    nearCloud->clear();
    for (int i = 0; i < idxVec.size(); i++) {
      *nearCloud +=
          *transformPointCloud(sessions[0].cloudKeyFrames[idxVec[i]].all_cloud,
                               &sessions[0].cloudKeyPoses6D->points[idxVec[i]]);
    }
    std::cout << "local cloud size for initialization: " << nearCloud->points.size() << std::endl;
    std::cout << ANSI_COLOR_GREEN << "get precise pose by NDT ... "
              << ANSI_COLOR_RESET << std::endl;

    currentCloudDsInOdom->clear();
    downSizeInitCloud.setLeafSize(0.20, 0.20, 0.20);
    downSizeInitCloud.setInputCloud(initCloudInOdom);
    downSizeInitCloud.filter(*currentCloudDsInOdom);
    initCloudInOdom->clear();

    localMapDS->clear();
    downSizeFilterNearCloud.setLeafSize(0.20, 0.20, 0.20);
    downSizeFilterNearCloud.setInputCloud(nearCloud);
    downSizeFilterNearCloud.filter(*localMapDS);
    nearCloud->clear();

    // pcl::NormalDistributionsTransform<PointTypeXYZI, PointTypeXYZI> NDT;
    // NDT清理
    // NDT = pcl::NormalDistributionsTransform<PointTypeXYZI,
    //                                         PointTypeXYZI>();  // 重置NDT
    // NDT.setResolution(0.5f);             // 体素分辨率(米)，可在 0.5~2.0 调整
    // NDT.setStepSize(0.1);                // line search 步长
    // NDT.setTransformationEpsilon(1e-4);  // 收敛阈值
    // NDT.setMaximumIterations(60);        // 迭代次数
    // NDT.setInputSource(currentCloudDsInOdom);
    // NDT.setInputTarget(localMapDS);
    pcl::IterativeClosestPoint<PointTypeXYZI, PointTypeXYZI> NDT;
    NDT.setMaxCorrespondenceDistance(0.25);
    NDT.setMaximumIterations(50);
    NDT.setTransformationEpsilon(1e-4);
    NDT.setEuclideanFitnessEpsilon(1e-4);
    NDT.setRANSACIterations(2);
    NDT.setInputSource(currentCloudDsInOdom);
    NDT.setInputTarget(localMapDS);
    pcl::PointCloud<PointTypeXYZI>::Ptr alignedCloud;
    alignedCloud.reset(new pcl::PointCloud<PointTypeXYZI>());
    NDT.align(*alignedCloud,init_odom2map.matrix());

    // transform is from odom to map
    Eigen::Matrix4f transform = NDT.getFinalTransformation();
    Eigen::Affine3f transformAffine;
    transformAffine.matrix() = transform;
    // convert current cloud in odom to map frame for visualization
    reloCloudInMap->clear();
    transformPointCloud(currentCloudDsInOdom, transformAffine, reloCloudInMap);
    publishCloud(pubReloWorldCloud, reloCloudInMap, this->node->now(), "world");
    Eigen::Matrix4f trans_diff =
        init_odom2map.matrix().inverse() * transform;
    float pos_diff = trans_diff.block<3, 1>(0, 3).norm();
    std::cout << "NDT delta pose difference: " << pos_diff << std::endl;

    if (NDT.getFitnessScore() > 0.2) {
      std::cout << ANSI_COLOR_RED
                << "NDT registration failed, fitness score too high: "
                << NDT.getFitnessScore() << ANSI_COLOR_RESET << std::endl;
      std::cout << ANSI_COLOR_RED << "or position diff too large: " << pos_diff
                << ANSI_COLOR_RESET << std::endl;
      std::cout << "aligned cloud size: " << alignedCloud->points.size()
                << std::endl;
      receive_ext_flg = false;
      std::cout << ANSI_COLOR_RED
                << "please update external pose to continue ..."
                << ANSI_COLOR_RESET << std::endl;
      return false;
    }

    std::cout << "NDT has converged: " << NDT.hasConverged()
              << " with score: " << NDT.getFitnessScore() << std::endl;
    std::cout << "NDT aligned init cloud size: " << alignedCloud->points.size()
              << std::endl;
    sc_flg = true;

    // optimazed odom to map transformation
    Eigen::Matrix3d rot = transform.matrix().block<3, 3>(0, 0).cast<double>();
    Eigen::Vector3d linear =
        transform.matrix().block<3, 1>(0, 3).cast<double>();
    currentPoseInMap= transformAffine * currentPoseInOdom;
    global_flg = true;

    std::cout << ANSI_COLOR_GREEN << "init lidar to map pose: " << currentPoseInMap.translation().x() << " "
              << currentPoseInMap.translation().y() << " " << currentPoseInMap.translation().z() << std::endl;
    std::cout << ANSI_COLOR_GREEN
              << "init relocalization by external-pose has been finished ... "
              << ANSI_COLOR_RESET << std::endl;
  } else {
    // std::cout << ANSI_COLOR_RED
    //           << "can not relo by SC and no external pose ... "
    //           << ANSI_COLOR_RESET << std::endl;
    // std::cout << ANSI_COLOR_RED << "please set external pose to continue ..."
    //           << ANSI_COLOR_RESET << std::endl;
    sc_flg = false;
    return false;
  }
  return true;
}

void pose_estimator::publish_odometry(
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pub) {
  odomAftMapped.header.frame_id = "world";
  odomAftMapped.child_frame_id = "base";
  odomAftMapped.header.stamp = this->node->now();

  static std::shared_ptr<tf2_ros::TransformBroadcaster> br;
  br = std::make_shared<tf2_ros::TransformBroadcaster>(this->node);
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = odomAftMapped.header.stamp;
  transform.header.frame_id = "map";
  transform.child_frame_id = "base";
  transform.transform.translation.x = odomAftMapped.pose.pose.position.x;
  transform.transform.translation.y = odomAftMapped.pose.pose.position.y;
  transform.transform.translation.z = odomAftMapped.pose.pose.position.z;
  transform.transform.rotation = odomAftMapped.pose.pose.orientation;
  br->sendTransform(transform);

  static std::shared_ptr<tf2_ros::StaticTransformBroadcaster> br_static;
  br_static = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this->node);
  geometry_msgs::msg::TransformStamped static_transform;
  static_transform.header.stamp = odomAftMapped.header.stamp;
  static_transform.header.frame_id = "base";
  static_transform.child_frame_id = "base_link";
  static_transform.transform.translation.x = 0;
  static_transform.transform.translation.y = 0;
  static_transform.transform.translation.z = 0;
  static_transform.transform.rotation.x = 1;
  static_transform.transform.rotation.y = 0;
  static_transform.transform.rotation.z = 0;
  static_transform.transform.rotation.w = 0;
  br_static->sendTransform(static_transform);
  pub->publish(odomAftMapped);
}
void pose_estimator::publish_path(
    const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pub) {
  msg_body_pose.header.frame_id = "world";
  msg_body_pose.header.stamp = this->node->now();

  path.header.frame_id = "world";
  path.header.stamp = this->node->now();
  path.poses.push_back(msg_body_pose);
  pub->publish(path);
}