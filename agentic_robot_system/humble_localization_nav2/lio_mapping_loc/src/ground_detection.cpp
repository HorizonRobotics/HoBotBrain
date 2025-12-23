#include "ground_detection.h"
using namespace std;

ERASOR::ERASOR() {
  th_dist_ = 0.10;
  iter_groundfilter = 3;
  num_lprs_ = 10;
  th_seeds_heights_ = -0.6;

  num_lowest_pts = 5;
  min_h = 1.5;  // 1.5m

  piecewise_ground_.reserve(130000);
  non_ground_.reserve(130000);
  ground_pc_.reserve(130000);
  non_ground_pc_.reserve(130000);
}

ERASOR::~ERASOR() {
  piecewise_ground_.clear();
  non_ground_.clear();
  ground_pc_.clear();
  non_ground_pc_.clear();
  normal_.resize(0, 0);
}

bool point_cmp(PointType a, PointType b) { return a.z > b.z; }
void ERASOR::estimate_plane_(const pcl::PointCloud<PointType> &ground) {
  Eigen::Matrix3f cov;
  Eigen::Vector4f pc_mean;
  pcl::computeMeanAndCovarianceMatrix(ground, cov, pc_mean);
  // Singular Value Decomposition: SVD
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(
      cov, Eigen::DecompositionOptions::ComputeFullU);
  // use the least singular vector as normal
  normal_ = (svd.matrixU().col(2));
  // mean ground seeds value
  Eigen::Vector3f seeds_mean = pc_mean.head<3>();

  // according to normal.T*[x,y,z] = -d
  d_ = -(normal_.transpose() * seeds_mean)(0, 0);
  // set distance threhold to `th_dist - d`
  //   th_dist_d_ = th_dist_ - d_;
}

void ERASOR::extract_initial_seeds_(const pcl::PointCloud<PointType> &p_sorted,
                                    pcl::PointCloud<PointType> &init_seeds) {
  init_seeds.points.clear();
  pcl::PointCloud<PointType> g_seeds_pc;

  // LPR is the mean of low point representative
  double sum = 0;
  int cnt = 0;

  // Calculate the mean height value.

  for (int i = num_lowest_pts; i < p_sorted.points.size() && cnt < num_lprs_;
       i++) {
    sum += p_sorted.points[i].z;
    cnt++;
  }
  double lpr_height = cnt != 0 ? sum / cnt : 0;  // in case divide by 0
  std::cout << "lpr height: " << lpr_height << std::endl;
  g_seeds_pc.clear();
  // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
  for (int i = 0; i < p_sorted.points.size(); i++) {
    if (p_sorted.points[i].z > lpr_height + th_seeds_heights_) {
      g_seeds_pc.points.emplace_back(p_sorted.points[i]);
    }
  }
  //  std::cout<<"hey!! g seeds"<<g_seeds_pc.points.size()<<std::endl;
  // return seeds points
  init_seeds = g_seeds_pc;
}

void ERASOR::extract_ground(pcl::PointCloud<PointType> &src) {
  // if (!dst.empty()) dst.clear();
  // if (!outliers.empty()) outliers.clear();

  auto src_copy = src;
  std::sort(src_copy.points.begin(), src_copy.points.end(), point_cmp);
  // 1. remove_outliers;
  auto it = src_copy.points.begin();
  for (int i = 0; i < src_copy.points.size(); i++) {
    // if(src_copy.points[i].z < -0.5*SENSOR_HEIGHTS){
    // z轴反向
    if (src_copy.points[i].z > min_h) {
      it++;
    } else {
      break;
    }
  }
  src_copy.points.erase(src_copy.points.begin(), it);

  // 2. set seeds!
  if (!ground_pc_.empty()) ground_pc_.clear();
  if (!non_ground_pc_.empty()) non_ground_pc_.clear();

  extract_initial_seeds_(src_copy, ground_pc_);
  //  std::cout<<"\033[1;032m [Ground] num:
  //  "<<ground_pc.points.size()<<std::endl;

  // 3. Extract ground
  for (int i = 0; i < iter_groundfilter; i++) {
    estimate_plane_(ground_pc_);
    ground_pc_.clear();
    // pointcloud to matrix
    Eigen::MatrixXf points(src.points.size(), 3);
    int j = 0;
    for (auto p : src.points) {
      points.row(j++) << p.x, p.y, p.z;
    }
    // ground plane model
    // Eigen::VectorXf result = points * normal_ + d_;
    Eigen::VectorXf result =
        points * normal_ + Eigen::VectorXf::Constant(points.rows(), d_);
    // threshold filter
    for (int r = 0; r < result.rows(); r++) {
      PointType p;
      p.x = src[r].x;
      p.y = src[r].y;
      p.z = src[r].z;
      if (abs(result[r]) < th_dist_) {
        ground_pc_.points.emplace_back(p);
        if (i == (iter_groundfilter - 1))
          src[r].intensity = 1;  // mark as ground
      } else {
        if (i == (iter_groundfilter - 1)) {  // Last iteration
          non_ground_pc_.points.emplace_back(p);
          src[r].intensity = 0;  // mark as non-ground
        }
      }
    }
    std::cout << "normal: " << normal_.transpose() << ", d: " << d_
              << std::endl;
  }
}
