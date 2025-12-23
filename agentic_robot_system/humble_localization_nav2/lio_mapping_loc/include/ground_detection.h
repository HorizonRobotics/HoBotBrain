#pragma once
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_lib.h"
// using namespace _UTILITY_LIDAR_ODOMETRY_H_;
class ERASOR {
 public:
  ERASOR();
  ~ERASOR();

 public:
  // ------ Ground extraction ------------------------
  double th_dist_;        // params!
  int iter_groundfilter;  // params!
  int num_lprs_;
  double th_seeds_heights_;
  int num_lowest_pts;

  double min_h;

  Eigen::MatrixXf normal_;
  double th_dist_d_, d_;

  pcl::PointCloud<PointType> piecewise_ground_, non_ground_;
  pcl::PointCloud<PointType> ground_pc_, non_ground_pc_;

  void estimate_plane_(const pcl::PointCloud<PointType> &ground);

  void extract_initial_seeds_(const pcl::PointCloud<PointType> &p_sorted,
                              pcl::PointCloud<PointType> &init_seeds);

  void extract_ground(pcl::PointCloud<PointType> &src);
  void reset() {
    piecewise_ground_.clear();
    non_ground_.clear();
    ground_pc_.clear();
    non_ground_pc_.clear();
  }
};
