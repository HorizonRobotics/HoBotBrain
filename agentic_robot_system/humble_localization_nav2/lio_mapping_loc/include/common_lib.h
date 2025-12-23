/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <tf2_ros/transform_broadcaster.h>
#include <utils/color.h>
#include <utils/so3_math.h>
#include <utils/types.h>
#include <utils/utils.h>

#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sophus/se3.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Transform.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// pcl
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/search/impl/search.hpp>
// eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <experimental/filesystem>  // file gcc>=8
#include <experimental/optional>
#include <filesystem>  // C++17 standard
#include <optional>
#include <unordered_map>
// #include <Python.h>
// gtsam
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <ctime>
#include <deque>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// #include "math_tools.h"
// #include <eigen_conversions/eigen_msg.h>
using namespace std;
// using namespace Eigen;   // avoid cmake error: reference to ‘Matrix’ is
// ambiguous
using namespace Sophus;
namespace fs = std::filesystem;
#define print_line std::cout << __FILE__ << ", " << __LINE__ << std::endl;
#define G_m_s2 (9.81)   // Gravaty const in GuangDong/China
#define DIM_STATE (19)  // Dimension of states (Let Dim(SO(3)) = 3)
#define INIT_COV (0.01)
#define SIZE_LARGE (500)
#define SIZE_SMALL (100)
#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "Log/" + name))
enum LID_TYPE {
  AVIA = 1,
  VELO16 = 2,
  OUST64 = 3,
  L515 = 4,
  XT32 = 5,
  PANDAR128 = 6
};
enum SLAM_MODE { ONLY_LO = 0, ONLY_LIO = 1, LIVO = 2 };
enum EKF_STATE { WAIT = 0, VIO = 1, LIO = 2, LO = 3 };

struct MeasureGroup {
  double vio_time;
  double lio_time;
  deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu;
  cv::Mat img;
  MeasureGroup() {
    vio_time = 0.0;
    lio_time = 0.0;
  };
};

struct LidarMeasureGroup {
  double lidar_frame_beg_time;
  double lidar_frame_end_time;
  double last_lio_update_time;
  PointCloudXYZI::Ptr lidar;
  PointCloudXYZI::Ptr pcl_proc_cur;
  PointCloudXYZI::Ptr pcl_proc_next;
  deque<struct MeasureGroup> measures;
  EKF_STATE lio_vio_flg;
  int lidar_scan_index_now;

  LidarMeasureGroup() {
    lidar_frame_beg_time = -0.0;
    lidar_frame_end_time = 0.0;
    last_lio_update_time = -1.0;
    lio_vio_flg = WAIT;
    this->lidar.reset(new PointCloudXYZI());
    this->pcl_proc_cur.reset(new PointCloudXYZI());
    this->pcl_proc_next.reset(new PointCloudXYZI());
    this->measures.clear();
    lidar_scan_index_now = 0;
    last_lio_update_time = -1.0;
  };
};

typedef struct pointWithVar {
  Eigen::Vector3d point_b;      // point in the lidar body frame
  Eigen::Vector3d point_i;      // point in the imu body frame
  Eigen::Vector3d point_w;      // point in the world frame
  Eigen::Matrix3d var_nostate;  // the var removed the state covarience
  Eigen::Matrix3d body_var;
  Eigen::Matrix3d var;
  Eigen::Matrix3d point_crossmat;
  Eigen::Vector3d normal;
  pointWithVar() {
    var_nostate = Eigen::Matrix3d::Zero();
    var = Eigen::Matrix3d::Zero();
    body_var = Eigen::Matrix3d::Zero();
    point_crossmat = Eigen::Matrix3d::Zero();
    point_b = Eigen::Vector3d::Zero();
    point_i = Eigen::Vector3d::Zero();
    point_w = Eigen::Vector3d::Zero();
    normal = Eigen::Vector3d::Zero();
  };
} pointWithVar;

struct StatesGroup {
  StatesGroup() {
    this->rot_end = M3D::Identity();
    this->pos_end = V3D::Zero();
    this->vel_end = V3D::Zero();
    this->ang_vel_end = V3D::Zero();
    this->bias_g = V3D::Zero();
    this->bias_a = V3D::Zero();
    this->gravity = V3D::Zero();
    this->inv_expo_time = 1.0;
    this->cov = MD(DIM_STATE, DIM_STATE)::Identity() * INIT_COV;
    this->cov(6, 6) = 0.00001;
    this->cov.block<9, 9>(10, 10) = MD(9, 9)::Identity() * 0.00001;
  };

  StatesGroup(const StatesGroup& b) {
    this->rot_end = b.rot_end;
    this->pos_end = b.pos_end;
    this->vel_end = b.vel_end;
    this->ang_vel_end = b.ang_vel_end;
    this->bias_g = b.bias_g;
    this->bias_a = b.bias_a;
    this->gravity = b.gravity;
    this->inv_expo_time = b.inv_expo_time;
    this->cov = b.cov;
  };

  StatesGroup& operator=(const StatesGroup& b) {
    this->rot_end = b.rot_end;
    this->pos_end = b.pos_end;
    this->vel_end = b.vel_end;
    this->ang_vel_end = b.ang_vel_end;
    this->bias_g = b.bias_g;
    this->bias_a = b.bias_a;
    this->gravity = b.gravity;
    this->inv_expo_time = b.inv_expo_time;
    this->cov = b.cov;
    return *this;
  };

  StatesGroup operator+(const Matrix<double, DIM_STATE, 1>& state_add) {
    StatesGroup a;
    a.rot_end =
        this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
    a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
    a.inv_expo_time = this->inv_expo_time + state_add(6, 0);
    a.vel_end = this->vel_end + state_add.block<3, 1>(7, 0);
    a.bias_g = this->bias_g + state_add.block<3, 1>(10, 0);
    a.bias_a = this->bias_a + state_add.block<3, 1>(13, 0);
    a.gravity = this->gravity + state_add.block<3, 1>(16, 0);
    a.ang_vel_end = this->ang_vel_end;
    a.cov = this->cov;
    return a;
  };

  StatesGroup& operator+=(const Matrix<double, DIM_STATE, 1>& state_add) {
    this->rot_end =
        this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
    this->pos_end += state_add.block<3, 1>(3, 0);
    this->inv_expo_time += state_add(6, 0);
    this->vel_end += state_add.block<3, 1>(7, 0);
    this->bias_g += state_add.block<3, 1>(10, 0);
    this->bias_a += state_add.block<3, 1>(13, 0);
    this->gravity += state_add.block<3, 1>(16, 0);
    return *this;
  };

  Matrix<double, DIM_STATE, 1> operator-(const StatesGroup& b) {
    Matrix<double, DIM_STATE, 1> a;
    M3D rotd(b.rot_end.transpose() * this->rot_end);
    a.block<3, 1>(0, 0) = Log(rotd);
    a.block<3, 1>(3, 0) = this->pos_end - b.pos_end;
    a(6, 0) = this->inv_expo_time - b.inv_expo_time;
    a.block<3, 1>(7, 0) = this->vel_end - b.vel_end;
    a.block<3, 1>(10, 0) = this->bias_g - b.bias_g;
    a.block<3, 1>(13, 0) = this->bias_a - b.bias_a;
    a.block<3, 1>(16, 0) = this->gravity - b.gravity;
    return a;
  };

  void resetpose() {
    this->rot_end = M3D::Identity();
    this->pos_end = V3D::Zero();
    this->vel_end = V3D::Zero();
  }

  M3D rot_end;  // the estimated attitude (rotation matrix) at the end lidar
                // point
  V3D pos_end;  // the estimated position at the end lidar point (world frame)
  V3D vel_end;  // the estimated velocity at the end lidar point (world frame)
  V3D ang_vel_end;  // the estimated angular velocity at the end lidar point
                    // (body frame)
  double inv_expo_time;  // the estimated inverse exposure time (no scale)
  V3D bias_g;            // gyroscope bias
  V3D bias_a;            // accelerator bias
  V3D gravity;           // the estimated gravity acceleration
  Matrix<double, DIM_STATE, DIM_STATE> cov;  // states covariance
};

template <typename T>
auto set_pose6d(const double t, const Matrix<T, 3, 1>& a,
                const Matrix<T, 3, 1>& g, const Matrix<T, 3, 1>& v,
                const Matrix<T, 3, 1>& p, const Matrix<T, 3, 3>& R) {
  Pose6D rot_kp;
  rot_kp.offset_time = t;
  for (int i = 0; i < 3; i++) {
    rot_kp.acc[i] = a(i);
    rot_kp.gyr[i] = g(i);
    rot_kp.vel[i] = v(i);
    rot_kp.pos[i] = p(i);
    for (int j = 0; j < 3; j++) rot_kp.rot[i * 3 + j] = R(i, j);
  }
  // Map<M3D>(rot_kp.rot, 3,3) = R;
  return move(rot_kp);
}
// keyframe
struct KeyFrame {
  pcl::PointCloud<PointTypeXYZI>::Ptr all_cloud;  // original pointcloud
  std::vector<std::pair<int, std::vector<int>>>
      object_cloud;        // TODO: segmented object <object_id, ptIdx>
  Eigen::MatrixXd scv_od;  // TODO: T-GRS paper is a variant of scan context
  int reloTargetIdx;
  float reloScore;
};

// edge
struct Edge {
  int from_idx;
  int to_idx;
  gtsam::Pose3 relative;
};

// node
struct Node {
  int idx;
  gtsam::Pose3 initial;
};

using SessionNodes = std::multimap<int, Node>;  // from_idx, Node
using SessionEdges = std::multimap<int, Edge>;  // from_idx, Edge

// g2o
struct G2oLineInfo {
  std::string type;

  int prev_idx = -1;  // for vertex, this member is null
  int curr_idx;

  std::vector<double> trans;
  std::vector<double> quat;

  inline static const std::string kVertexTypeName = "VERTEX_SE3:QUAT";
  inline static const std::string kEdgeTypeName = "EDGE_SE3:QUAT";
};
// structure
// trajector
/* comment
plane equation: Ax + By + Cz + D = 0
convert to: A/D*x + B/D*y + C/D*z = -1
solve: A0*x0 = b0
where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
normvec:  normalized x0
*/

inline float calc_dist(PointTypeXYZI p1, PointTypeXYZI p2) {  // 3d-distance
  float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) +
            (p1.z - p2.z) * (p1.z - p2.z);
  return d;
}

inline float calc_dist_(PointTypeXYZI p1, PointTypeXYZI p2) {  // 2d-distance
  float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
  return d;
}

// padding with zero for invalid parts in scd (SC)
inline std::string padZeros(int val, int num_digits = 6) {
  std::ostringstream out;
  out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
  return out.str();
}

// save scd for each keyframe
inline void writeSCD(std::string fileName, Eigen::MatrixXd matrix,
                     std::string delimiter = " ") {
  // delimiter: ", " or " " etc.

  int precision = 3;  // or Eigen::FullPrecision, but SCD does not require such
                      // accruate precisions so 3 is enough.
  const static Eigen::IOFormat the_format(precision, Eigen::DontAlignCols,
                                          delimiter, "\n");

  std::ofstream file(fileName);
  if (file.is_open()) {
    file << matrix.format(the_format);
    file.close();
  }
}

// TODO: pubilsh any-type pointcloud
inline sensor_msgs::msg::PointCloud2 publishCloud(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub,
    pcl::PointCloud<PointTypeXYZI>::Ptr& thisCloud, rclcpp::Time thisStamp,
    std::string thisFrame) {
  sensor_msgs::msg::PointCloud2 tempCloud;
  pcl::toROSMsg(*thisCloud, tempCloud);
  tempCloud.header.stamp = thisStamp;
  tempCloud.header.frame_id = thisFrame;
  if (thisPub->get_subscription_count() != 0) thisPub->publish(tempCloud);
  return tempCloud;
}

// read bin
inline void readBin(std::string _bin_path,
                    pcl::PointCloud<PointTypeXYZI>::Ptr _pcd_ptr) {
  std::fstream input(_bin_path.c_str(), ios::in | ios::binary);
  if (!input.good()) {
    cerr << "Could not read file: " << _bin_path << endl;
    exit(EXIT_FAILURE);
  }
  input.seekg(0, ios::beg);

  for (int ii = 0; input.good() && !input.eof(); ii++) {
    PointTypeXYZI point;

    input.read((char*)&point.x, sizeof(float));
    input.read((char*)&point.y, sizeof(float));
    input.read((char*)&point.z, sizeof(float));
    input.read((char*)&point.intensity, sizeof(float));

    _pcd_ptr->push_back(point);
  }
  input.close();
}

template <typename T>
std::vector<T> linspace(T a, T b, size_t N) {
  T h = (b - a) / static_cast<T>(N - 1);
  std::vector<T> xs(N);
  typename std::vector<T>::iterator x;
  T val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) *x = val;
  return xs;
}

const static inline int kSessionStartIdxOffset =
    1000000;  // int max 2147483647 so ok.

inline int ungenGlobalNodeIdx(const int& _session_idx,
                              const int& _idx_in_graph) {
  return (_idx_in_graph - 1) / (_session_idx * kSessionStartIdxOffset);
}  // ungenGlobalNodeIdx

inline int genGlobalNodeIdx(const int& _session_idx, const int& _node_offset) {
  return (_session_idx * kSessionStartIdxOffset) + _node_offset + 1;
}  // genGlobalNodeIdx

inline int genAnchorNodeIdx(const int& _session_idx) {
  return (_session_idx * kSessionStartIdxOffset);
}  // genAnchorNodeIdx

inline gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
  return gtsam::Pose3(
      gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch),
                          double(thisPoint.yaw)),
      gtsam::Point3(double(thisPoint.x), double(thisPoint.y),
                    double(thisPoint.z)));
}

inline gtsam::Pose3 trans2gtsamPose(float transformIn[]) {
  return gtsam::Pose3(
      gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
      gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

inline Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint) {
  return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z,
                                thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

inline Eigen::Affine3f trans2Affine3f(float transformIn[]) {
  return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5],
                                transformIn[0], transformIn[1], transformIn[2]);
}

inline PointTypePose trans2PointTypePose(float transformIn[]) {
  PointTypePose thisPose6D;
  thisPose6D.x = transformIn[3];
  thisPose6D.y = transformIn[4];
  thisPose6D.z = transformIn[5];
  thisPose6D.roll = transformIn[0];
  thisPose6D.pitch = transformIn[1];
  thisPose6D.yaw = transformIn[2];
  return thisPose6D;
}

inline float pointDistance(PointTypeXYZI p) {
  return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

inline float pointDistance(PointTypeXYZI p1, PointTypeXYZI p2) {
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) +
              (p1.z - p2.z) * (p1.z - p2.z));
}

inline void fsmkdir(const std::string& _path) {
  std::filesystem::path pathObj(_path);
  if (fs::is_directory(pathObj) && fs::exists(pathObj)) fs::remove_all(pathObj);
  if (!fs::is_directory(pathObj) || !fs::exists(pathObj))
    fs::create_directories(pathObj);  // create src folder
}  // fsmkdir

inline pcl::PointCloud<PointTypeXYZI>::Ptr transformPointCloud(
    pcl::PointCloud<PointTypeXYZI>::Ptr cloudIn, PointTypePose* transformIn) {
  pcl::PointCloud<PointTypeXYZI>::Ptr cloudOut(
      new pcl::PointCloud<PointTypeXYZI>());

  PointTypeXYZI* pointFrom;

  int cloudSize = cloudIn->size();
  cloudOut->resize(cloudSize);

  Eigen::Affine3f transCur = pcl::getTransformation(
      transformIn->x, transformIn->y, transformIn->z, transformIn->roll,
      transformIn->pitch, transformIn->yaw);

  int numberOfCores = 8;  // TODO: move to yaml
#pragma omp parallel for num_threads(numberOfCores)
  for (int i = 0; i < cloudSize; ++i) {
    pointFrom = &cloudIn->points[i];
    cloudOut->points[i].x = transCur(0, 0) * pointFrom->x +
                            transCur(0, 1) * pointFrom->y +
                            transCur(0, 2) * pointFrom->z + transCur(0, 3);
    cloudOut->points[i].y = transCur(1, 0) * pointFrom->x +
                            transCur(1, 1) * pointFrom->y +
                            transCur(1, 2) * pointFrom->z + transCur(1, 3);
    cloudOut->points[i].z = transCur(2, 0) * pointFrom->x +
                            transCur(2, 1) * pointFrom->y +
                            transCur(2, 2) * pointFrom->z + transCur(2, 3);
    cloudOut->points[i].intensity = pointFrom->intensity;
  }
  return cloudOut;
}

template <typename CloudT>
inline void transformPointCloud(const CloudT& cloudIn_,
                                const Eigen::Affine3f& transCur_,
                                CloudT& cloudOut_) {
  int cloudSize = cloudIn_->points.size();
  cloudOut_->points.resize(cloudSize);

  int numberOfCores = 5;  // TODO: move to yaml
#pragma omp parallel for num_threads(numberOfCores)
  for (int i = 0; i < cloudSize; i++) {
    cloudOut_->points[i].x = transCur_(0, 0) * cloudIn_->points[i].x +
                             transCur_(0, 1) * cloudIn_->points[i].y +
                             transCur_(0, 2) * cloudIn_->points[i].z +
                             transCur_(0, 3);
    cloudOut_->points[i].y = transCur_(1, 0) * cloudIn_->points[i].x +
                             transCur_(1, 1) * cloudIn_->points[i].y +
                             transCur_(1, 2) * cloudIn_->points[i].z +
                             transCur_(1, 3);
    cloudOut_->points[i].z = transCur_(2, 0) * cloudIn_->points[i].x +
                             transCur_(2, 1) * cloudIn_->points[i].y +
                             transCur_(2, 2) * cloudIn_->points[i].z +
                             transCur_(2, 3);
    cloudOut_->points[i].intensity = cloudIn_->points[i].intensity;
  }
}

inline pcl::PointCloud<PointTypeXYZI>::Ptr getBodyCloud(
    const pcl::PointCloud<PointTypeXYZI>::Ptr& cloud, const PointTypePose& body,
    const PointTypePose& other) {
  pcl::PointCloud<PointTypeXYZI>::Ptr cloud_trans(
      new pcl::PointCloud<PointTypeXYZI>());
  Eigen::Affine3f trans_body = pcl::getTransformation(
      body.x, body.y, body.z, body.roll, body.pitch, body.yaw);
  Eigen::Affine3f trans_other = pcl::getTransformation(
      other.x, other.y, other.z, other.roll, other.pitch, other.yaw);
  Eigen::Affine3f trans_diff = trans_body.inverse() * trans_other;
  transformPointCloud(cloud, trans_diff, cloud_trans);
  return cloud_trans;
}
inline pcl::PointCloud<PointTypeXYZI>::Ptr getBodyCloud(
    const pcl::PointCloud<PointTypeXYZI>::Ptr& cloud,
    const Eigen::Affine3f& trans_body) {
  pcl::PointCloud<PointTypeXYZI>::Ptr cloud_trans(
      new pcl::PointCloud<PointTypeXYZI>());
  Eigen::Affine3f trans_diff = trans_body.inverse();
  transformPointCloud(cloud, trans_diff, cloud_trans);
  return cloud_trans;
}
inline pcl::PointCloud<PointTypeXYZI>::Ptr getAddCloud(
    const pcl::PointCloud<PointTypeXYZI>::Ptr& cloud, const PointTypePose& body,
    const PointTypePose& other) {
  pcl::PointCloud<PointTypeXYZI>::Ptr cloud_trans(
      new pcl::PointCloud<PointTypeXYZI>());
  Eigen::Affine3f trans_body = pcl::getTransformation(
      body.x, body.y, body.z, body.roll, body.pitch, body.yaw);
  Eigen::Affine3f trans_other = pcl::getTransformation(
      other.x, other.y, other.z, other.roll, other.pitch, other.yaw);
  Eigen::Affine3f trans_diff = trans_body * trans_other;
  transformPointCloud(cloud, trans_diff, cloud_trans);
  return cloud_trans;
}

template <typename T1, typename T2>
inline void floor(T1& a, T2 b) {
  if (a >= b) {
    a = b;
  }
}

// std::vector<std::pair<double, int>> sortVecWithIdx(
//     const std::vector<double>& arr) {
//   std::vector<std::pair<double, int>> vp;
//   for (int i = 0; i < arr.size(); ++i) vp.push_back(std::make_pair(arr[i],
//   i));

//   std::sort(vp.begin(), vp.end(), std::greater<>());
//   return vp;
// }

// std::vector<double> splitPoseLine(std::string _str_line, char _delimiter) {
//   std::vector<double> parsed;
//   std::stringstream ss(_str_line);
//   std::string temp;
//   while (getline(ss, temp, _delimiter)) {
//     parsed.push_back(std::stod(temp));  // convert string to "double"
//   }
//   return parsed;
// }

inline bool isTwoStringSame(std::string _str1, std::string _str2) {
  return !(_str1.compare(_str2));
}

inline void collect_digits(std::vector<int>& digits, int num) {
  if (num > 9) {
    collect_digits(digits, num / 10);
  }
  digits.push_back(num % 10);
}

// read g2o
// example：VERTEX_SE3:QUAT 99 -61.332581 -9.253125 0.131973 -0.004256 -0.005810
// -0.625732 0.780005
inline G2oLineInfo splitG2oFileLine(std::string _str_line) {
  std::stringstream ss(_str_line);

  std::vector<std::string> parsed_elms;
  std::string elm;
  char delimiter = ' ';
  while (getline(ss, elm, delimiter)) {
    parsed_elms.push_back(elm);  // convert string to "double"
  }

  G2oLineInfo parsed;
  // determine whether edge or node
  if (isTwoStringSame(parsed_elms.at(0), G2oLineInfo::kVertexTypeName)) {
    parsed.type = parsed_elms.at(0);
    parsed.curr_idx = std::stoi(parsed_elms.at(1));
    parsed.trans.push_back(std::stod(parsed_elms.at(2)));
    parsed.trans.push_back(std::stod(parsed_elms.at(3)));
    parsed.trans.push_back(std::stod(parsed_elms.at(4)));
    parsed.quat.push_back(std::stod(parsed_elms.at(5)));
    parsed.quat.push_back(std::stod(parsed_elms.at(6)));
    parsed.quat.push_back(std::stod(parsed_elms.at(7)));
    parsed.quat.push_back(std::stod(parsed_elms.at(8)));
  }
  if (isTwoStringSame(parsed_elms.at(0), G2oLineInfo::kEdgeTypeName)) {
    parsed.type = parsed_elms.at(0);
    parsed.prev_idx = std::stoi(parsed_elms.at(1));
    parsed.curr_idx = std::stoi(parsed_elms.at(2));
    parsed.trans.push_back(std::stod(parsed_elms.at(3)));
    parsed.trans.push_back(std::stod(parsed_elms.at(4)));
    parsed.trans.push_back(std::stod(parsed_elms.at(5)));
    parsed.quat.push_back(std::stod(parsed_elms.at(6)));
    parsed.quat.push_back(std::stod(parsed_elms.at(7)));
    parsed.quat.push_back(std::stod(parsed_elms.at(8)));
    parsed.quat.push_back(std::stod(parsed_elms.at(9)));
  }

  return parsed;
}
// record poses
inline void writePose3ToStream(std::fstream& _stream, gtsam::Pose3 _pose) {
  gtsam::Point3 t = _pose.translation();
  gtsam::Rot3 R = _pose.rotation();

  // r1 means column 1 (see https://gtsam.org/doxygen/a02759.html)
  std::string sep = " ";  // separator
  _stream << R.r1().x() << sep << R.r2().x() << sep << R.r3().x() << sep
          << t.x() << sep << R.r1().y() << sep << R.r2().y() << sep
          << R.r3().y() << sep << t.y() << sep << R.r1().z() << sep
          << R.r2().z() << sep << R.r3().z() << sep << t.z() << std::endl;
}

// std::vector<int> linspace(int a, int b, int N) {
//   int h = (b - a) / static_cast<int>(N - 1);
//   std::vector<int> xs(N);
//   typename std::vector<int>::iterator x;
//   int val;
//   for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) *x = val;
//   return xs;
// }

inline float poseDistance(const gtsam::Pose3& p1, const gtsam::Pose3& p2) {
  auto p1x = p1.translation().x();
  auto p1y = p1.translation().y();
  auto p1z = p1.translation().z();
  auto p2x = p2.translation().x();
  auto p2y = p2.translation().y();
  auto p2z = p2.translation().z();

  return sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y) +
              (p1z - p2z) * (p1z - p2z));
}

inline Eigen::Quaterniond EulerToQuat(float roll_, float pitch_, float yaw_) {
  Eigen::Quaterniond q;  // 四元数q和-q是相等的
  Eigen::AngleAxisd roll(double(roll_), Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitch(double(pitch_), Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yaw(double(yaw_), Eigen::Vector3d::UnitZ());
  q = yaw * pitch * roll;
  q.normalize();
  return q;
}

// std::set<int> convertIntVecToSet(const std::vector<int>& v) {
//   std::set<int> s;
//   for (int x : v) {
//     s.insert(x);
//   }
//   return s;
// }
// read scd
inline Eigen::MatrixXd readSCD(std::string fileToOpen) {
  // ref:
  // https://aleksandarhaber.com/eigen-matrix-library-c-tutorial-saving-and-loading-data-in-from-a-csv-file/

  std::vector<double> matrixEntries;
  std::ifstream matrixDataFile(fileToOpen);
  std::string matrixRowString;
  std::string matrixEntry;

  int matrixRowNumber = 0;
  while (getline(matrixDataFile, matrixRowString)) {
    std::stringstream matrixRowStringStream(
        matrixRowString);  // convert matrixRowString that is a string to a
                           // stream variable.
    while (getline(matrixRowStringStream, matrixEntry,
                   ' '))  // here we read pieces of the stream
                          // matrixRowStringStream until every comma, and store
                          // the resulting character into the matrixEntry
      matrixEntries.push_back(stod(
          matrixEntry));  // here we convert the string to double and fill in
                          // the row vector storing all the matrix entries

    matrixRowNumber++;  // update the column numbers
  }
  return Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      matrixEntries.data(), matrixRowNumber,
      matrixEntries.size() / matrixRowNumber);
}
inline void writeVertex(const int _node_idx, const gtsam::Pose3& _initPose,
                        std::vector<std::string>& vertices_str) {
  gtsam::Point3 t = _initPose.translation();
  gtsam::Rot3 R = _initPose.rotation();

  std::string curVertexInfo{
      "VERTEX_SE3:QUAT " + std::to_string(_node_idx) + " " +
      std::to_string(t.x()) + " " + std::to_string(t.y()) + " " +
      std::to_string(t.z()) + " " + std::to_string(R.toQuaternion().x()) + " " +
      std::to_string(R.toQuaternion().y()) + " " +
      std::to_string(R.toQuaternion().z()) + " " +
      std::to_string(R.toQuaternion().w())};

  // pgVertexSaveStream << curVertexInfo << std::endl;
  vertices_str.emplace_back(curVertexInfo);
}

inline void writeEdge(const std::pair<int, int> _node_idx_pair,
                      const gtsam::Pose3& _relPose,
                      std::vector<std::string>& edges_str) {
  gtsam::Point3 t = _relPose.translation();
  gtsam::Rot3 R = _relPose.rotation();

  std::string curEdgeInfo{
      "EDGE_SE3:QUAT " + std::to_string(_node_idx_pair.first) + " " +
      std::to_string(_node_idx_pair.second) + " " + std::to_string(t.x()) +
      " " + std::to_string(t.y()) + " " + std::to_string(t.z()) + " " +
      std::to_string(R.toQuaternion().x()) + " " +
      std::to_string(R.toQuaternion().y()) + " " +
      std::to_string(R.toQuaternion().z()) + " " +
      std::to_string(R.toQuaternion().w())};

  // pgEdgeSaveStream << curEdgeInfo << std::endl;
  edges_str.emplace_back(curEdgeInfo);
}

// functions
template <typename T>
inline T rad2deg(T radians)  // rad2deg
{
  return radians * 180.0 / PI_M;
}

template <typename T>
inline T deg2rad(T degrees)  // deg2rad
{
  return degrees * PI_M / 180.0;
}
template <typename PointT>
inline float pointDistance3d(const PointT& p1, const PointT& p2) {
  return (float)sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                     (p1.y - p2.y) * (p1.y - p2.y) +
                     (p1.z - p2.z) * (p1.z - p2.z));
}

template <typename PointT>
inline float pointDistance2d(const PointT& p1, const PointT& p2) {
  return (float)sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                     (p1.y - p2.y) * (p1.y - p2.y));
}

template <typename PointT>
inline float pointDistance3d(const PointT& p1) {
  return (float)sqrt((p1.x) * (p1.x) + (p1.y) * (p1.y) + (p1.z) * (p1.z));
}

template <typename PointT>
inline float pointDistance2d(const PointT& p1) {
  return (float)sqrt((p1.x) * (p1.x) + (p1.y) * (p1.y));
}

template <typename PointT>
inline float getPolarAngle(const PointT& p) {
  if (p.x == 0 && p.y == 0) {
    return 0.f;
  } else if (p.y >= 0) {
    return (float)rad2deg((float)atan2(p.y, p.x));
  } else if (p.y < 0) {
    return (float)rad2deg((float)atan2(p.y, p.x) + 2 * M_PI);
  }
}

template <typename PointT>
inline float getAzimuth(const PointT& p) {
  return (float)rad2deg((float)atan2(p.z, (float)pointDistance2d(p)));
}

template <typename T>
inline void addVec(std::vector<T>& vec_central_,
                   const std::vector<T>& vec_add_) {
  vec_central_.insert(vec_central_.end(), vec_add_.begin(), vec_add_.end());
}

template <typename T>
inline void reduceVec(std::vector<T>& vec_central_,
                      const std::vector<T>& vec_reduce_) {
  for (auto it = vec_reduce_.begin(); it != vec_reduce_.end(); it++) {
    vec_central_.erase(
        std::remove(vec_central_.begin(), vec_central_.end(), *it),
        vec_central_.end());
  }
}

template <typename T>
inline void sampleVec(std::vector<T>& vec_central_) {
  std::sort(vec_central_.begin(), vec_central_.end());
  vec_central_.erase(std::unique(vec_central_.begin(), vec_central_.end()),
                     vec_central_.end());
}

inline bool findNameInVec(const int& name_, const std::vector<int>& vec_) {
  if (std::count(vec_.begin(), vec_.end(), name_)) {
    return true;
  } else {
    return false;
  }
}

#endif