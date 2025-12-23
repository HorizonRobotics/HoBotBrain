/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "voxel_map.h"
using namespace Eigen;
void calcBodyCov(Eigen::Vector3d &pb, const float range_inc,
                 const float degree_inc, Eigen::Matrix3d &cov) {
  if (pb[2] == 0) pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
      pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0,
      -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1,
                               -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
      base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov = direction * range_var * direction.transpose() +
        A * direction_var * A.transpose();
}

void loadVoxelConfig(rclcpp::Node::SharedPtr &node,
                     VoxelMapConfig &voxel_config) {
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
  try_declare.template operator()<bool>("publish.pub_plane_en", false);
  try_declare.template operator()<int>("lio.max_layer", 1);
  try_declare.template operator()<double>("lio.voxel_size", 0.5);
  try_declare.template operator()<double>("lio.min_eigen_value", 0.01);
  try_declare.template operator()<double>("lio.sigma_num", 3);
  try_declare.template operator()<double>("lio.beam_err", 0.02);
  try_declare.template operator()<double>("lio.dept_err", 0.05);

  // Declaration of parameter of type std::vector<int> won't build,
  // https://github.com/ros2/rclcpp/issues/1585
  try_declare.template operator()<vector<int64_t>>(
      "lio.layer_init_num", std::vector<int64_t>{5, 5, 5, 5, 5});
  try_declare.template operator()<int>("lio.max_points_num", 50);
  try_declare.template operator()<int>("lio.min_iterations", 5);
  // MAX_VOXEL_NUM
  try_declare.template operator()<int>("lio.max_voxel_num", 5000);
  try_declare.template operator()<bool>("local_map.map_sliding_en", false);
  try_declare.template operator()<int>("local_map.half_map_size", 100);
  try_declare.template operator()<double>("local_map.sliding_thresh", 8.0);

  // get parameter
  node->get_parameter("publish.pub_plane_en", voxel_config.is_pub_plane_map_);
  node->get_parameter("lio.max_layer", voxel_config.max_layer_);
  node->get_parameter("lio.voxel_size", voxel_config.max_voxel_size_);
  node->get_parameter("lio.min_eigen_value", voxel_config.planner_threshold_);
  node->get_parameter("lio.sigma_num", voxel_config.sigma_num_);
  node->get_parameter("lio.beam_err", voxel_config.beam_err_);
  node->get_parameter("lio.dept_err", voxel_config.dept_err_);
  node->get_parameter("lio.layer_init_num", voxel_config.layer_init_num_);
  node->get_parameter("lio.max_points_num", voxel_config.max_points_num_);
  node->get_parameter("lio.min_iterations", voxel_config.max_iterations_);
  node->get_parameter("lio.max_voxel_num", voxel_config.MAX_VOXEL_NUM);

  node->get_parameter("local_map.map_sliding_en", voxel_config.map_sliding_en);
  node->get_parameter("local_map.half_map_size", voxel_config.half_map_size);
  node->get_parameter("local_map.sliding_thresh", voxel_config.sliding_thresh);
}

void VoxelOctoTree::init_plane(const std::vector<pointWithVar> &points,
                               VoxelPlane *plane) {
  plane->plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
  plane->covariance_ = Eigen::Matrix3d::Zero();
  plane->center_ = Eigen::Vector3d::Zero();
  plane->normal_ = Eigen::Vector3d::Zero();
  plane->points_size_ = points.size();
  plane->radius_ = 0;
  for (auto pv : points) {
    plane->covariance_ += pv.point_w * pv.point_w.transpose();
    plane->center_ += pv.point_w;
  }
  plane->center_ = plane->center_ / plane->points_size_;
  plane->covariance_ = plane->covariance_ / plane->points_size_ -
                       plane->center_ * plane->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3f::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
  Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
  Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
  Eigen::Matrix3d J_Q;
  J_Q << 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_, 0, 0, 0,
      1.0 / plane->points_size_;
  // && evalsReal(evalsMid) > 0.05
  //&& evalsReal(evalsMid) > 0.01
  if (evalsReal(evalsMin) < planer_threshold_) {
    for (int i = 0; i < points.size(); i++) {
      Eigen::Matrix<double, 6, 3> J;
      Eigen::Matrix3d F;
      for (int m = 0; m < 3; m++) {
        if (m != (int)evalsMin) {
          Eigen::Matrix<double, 1, 3> F_m =
              (points[i].point_w - plane->center_).transpose() /
              ((plane->points_size_) * (evalsReal[evalsMin] - evalsReal[m])) *
              (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() +
               evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
          F.row(m) = F_m;
        } else {
          Eigen::Matrix<double, 1, 3> F_m;
          F_m << 0, 0, 0;
          F.row(m) = F_m;
        }
      }
      J.block<3, 3>(0, 0) = evecs.real() * F;
      J.block<3, 3>(3, 0) = J_Q;
      plane->plane_var_ += J * points[i].var * J.transpose();
    }

    plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
        evecs.real()(2, evalsMin);
    plane->y_normal_ << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
        evecs.real()(2, evalsMid);
    plane->x_normal_ << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
        evecs.real()(2, evalsMax);
    plane->min_eigen_value_ = evalsReal(evalsMin);
    plane->mid_eigen_value_ = evalsReal(evalsMid);
    plane->max_eigen_value_ = evalsReal(evalsMax);
    plane->radius_ = sqrt(evalsReal(evalsMax));
    plane->d_ = -(plane->normal_(0) * plane->center_(0) +
                  plane->normal_(1) * plane->center_(1) +
                  plane->normal_(2) * plane->center_(2));
    plane->is_plane_ = true;
    plane->is_update_ = true;
    if (!plane->is_init_) {
      plane->id_ = voxel_plane_id;
      voxel_plane_id++;
      plane->is_init_ = true;
    }
  } else {
    plane->is_update_ = true;
    plane->is_plane_ = false;
  }
}

void VoxelOctoTree::init_octo_tree() {
  if (temp_points_.size() > points_size_threshold_) {
    init_plane(temp_points_, plane_ptr_);
    if (plane_ptr_->is_plane_ == true) {
      octo_state_ = 0;
      // new added
      if (temp_points_.size() > max_points_num_) {
        update_enable_ = false;
        std::vector<pointWithVar>().swap(temp_points_);
        new_points_ = 0;
      }
    } else {
      octo_state_ = 1;
      cut_octo_tree();
    }
    init_octo_ = true;
    new_points_ = 0;
  }
}

void VoxelOctoTree::cut_octo_tree() {
  if (layer_ >= max_layer_) {
    octo_state_ = 0;
    return;
  }
  for (size_t i = 0; i < temp_points_.size(); i++) {
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].point_w[0] > voxel_center_[0]) {
      xyz[0] = 1;
    }
    if (temp_points_[i].point_w[1] > voxel_center_[1]) {
      xyz[1] = 1;
    }
    if (temp_points_[i].point_w[2] > voxel_center_[2]) {
      xyz[2] = 1;
    }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr) {
      leaves_[leafnum] =
          new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1],
                            max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] =
          voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] =
          voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] =
          voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
    }
    leaves_[leafnum]->temp_points_.emplace_back(temp_points_[i]);
    leaves_[leafnum]->new_points_++;
  }
  for (uint i = 0; i < 8; i++) {
    if (leaves_[i] != nullptr) {
      if (leaves_[i]->temp_points_.size() >
          leaves_[i]->points_size_threshold_) {
        init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
        if (leaves_[i]->plane_ptr_->is_plane_) {
          leaves_[i]->octo_state_ = 0;
          // new added
          if (leaves_[i]->temp_points_.size() > leaves_[i]->max_points_num_) {
            leaves_[i]->update_enable_ = false;
            std::vector<pointWithVar>().swap(leaves_[i]->temp_points_);
            new_points_ = 0;
          }
        } else {
          leaves_[i]->octo_state_ = 1;
          leaves_[i]->cut_octo_tree();
        }
        leaves_[i]->init_octo_ = true;
        leaves_[i]->new_points_ = 0;
      }
    }
  }
}

void VoxelOctoTree::UpdateOctoTree(const pointWithVar &pv) {
  if (!init_octo_) {
    new_points_++;
    temp_points_.emplace_back(pv);
    if (temp_points_.size() > points_size_threshold_) {
      init_octo_tree();
    }
  } else {
    if (plane_ptr_->is_plane_) {
      if (update_enable_) {
        new_points_++;
        temp_points_.emplace_back(pv);
        if (new_points_ > update_size_threshold_) {
          init_plane(temp_points_, plane_ptr_);
          new_points_ = 0;
        }
        if (temp_points_.size() >= max_points_num_) {
          update_enable_ = false;
          std::vector<pointWithVar>().swap(temp_points_);
          new_points_ = 0;
        }
      }
    } else {
      if (layer_ < max_layer_) {
        int xyz[3] = {0, 0, 0};
        if (pv.point_w[0] > voxel_center_[0]) {
          xyz[0] = 1;
        }
        if (pv.point_w[1] > voxel_center_[1]) {
          xyz[1] = 1;
        }
        if (pv.point_w[2] > voxel_center_[2]) {
          xyz[2] = 1;
        }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] != nullptr) {
          leaves_[leafnum]->UpdateOctoTree(pv);
        } else {
          leaves_[leafnum] = new VoxelOctoTree(
              max_layer_, layer_ + 1, layer_init_num_[layer_ + 1],
              max_points_num_, planer_threshold_);
          leaves_[leafnum]->layer_init_num_ = layer_init_num_;
          leaves_[leafnum]->voxel_center_[0] =
              voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[1] =
              voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[2] =
              voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
          leaves_[leafnum]->quater_length_ = quater_length_ / 2;
          leaves_[leafnum]->UpdateOctoTree(pv);
        }
      } else {
        if (update_enable_) {
          new_points_++;
          temp_points_.emplace_back(pv);
          if (new_points_ > update_size_threshold_) {
            init_plane(temp_points_, plane_ptr_);
            new_points_ = 0;
          }
          if (temp_points_.size() > max_points_num_) {
            update_enable_ = false;
            std::vector<pointWithVar>().swap(temp_points_);
            new_points_ = 0;
          }
        }
      }
    }
  }
}

VoxelOctoTree *VoxelOctoTree::find_correspond(Eigen::Vector3d pw) {
  if (!init_octo_ || plane_ptr_->is_plane_ || (layer_ >= max_layer_))
    return this;

  int xyz[3] = {0, 0, 0};
  xyz[0] = pw[0] > voxel_center_[0] ? 1 : 0;
  xyz[1] = pw[1] > voxel_center_[1] ? 1 : 0;
  xyz[2] = pw[2] > voxel_center_[2] ? 1 : 0;
  int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

  // printf("leafnum: %d. \n", leafnum);

  return (leaves_[leafnum] != nullptr) ? leaves_[leafnum]->find_correspond(pw)
                                       : this;
}

VoxelOctoTree *VoxelOctoTree::Insert(const pointWithVar &pv) {
  if ((!init_octo_) || (init_octo_ && plane_ptr_->is_plane_) ||
      (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ >= max_layer_))) {
    new_points_++;
    temp_points_.emplace_back(pv);
    return this;
  }

  if (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ < max_layer_)) {
    int xyz[3] = {0, 0, 0};
    xyz[0] = pv.point_w[0] > voxel_center_[0] ? 1 : 0;
    xyz[1] = pv.point_w[1] > voxel_center_[1] ? 1 : 0;
    xyz[2] = pv.point_w[2] > voxel_center_[2] ? 1 : 0;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] != nullptr) {
      return leaves_[leafnum]->Insert(pv);
    } else {
      leaves_[leafnum] =
          new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1],
                            max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] =
          voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] =
          voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] =
          voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      return leaves_[leafnum]->Insert(pv);
    }
  }
  return nullptr;
}

void VoxelMapManager::StateEstimation(StatesGroup &state_propagat) {
  // cross_mat_list_.clear();
  // cross_mat_list_.reserve(feats_down_size_);
  cross_mat_list_.resize(feats_down_size_);
  // body_cov_list_.clear();
  // body_cov_list_.reserve(feats_down_size_);
  body_cov_list_.resize(feats_down_size_);

// build_residual_time = 0.0;
// ekf_time = 0.0;
// double t0 = omp_get_wtime();

// for (size_t i = 0; i < feats_down_body_->size(); i++) {
//   V3D point_this(feats_down_body_->points[i].x,
//   feats_down_body_->points[i].y,
//                  feats_down_body_->points[i].z);
//   if (point_this[2] == 0) {
//     point_this[2] = 0.001;
//   }
//   M3D var;
//   calcBodyCov(point_this, config_setting_.dept_err_,
//               config_setting_.beam_err_, var);
//   body_cov_list_.emplace_back(var);
//   point_this = extR_ * point_this + extT_;
//   M3D point_crossmat;
//   point_crossmat << SKEW_SYM_MATRX(point_this);
//   cross_mat_list_.emplace_back(point_crossmat);
// }
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (size_t i = 0; i < feats_down_body_->size(); i++) {
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y,
                   feats_down_body_->points[i].z);
    if (point_this[2] == 0) {
      point_this[2] = 0.001;
    }
    M3D var;
    // // 增加地面点权重
    double depth_err = config_setting_.dept_err_;
    double beam_err = config_setting_.beam_err_;
    if (abs(point_this[2]) > 1.2) {
      depth_err *= 0.1;
      beam_err *= 0.1;
    }
    calcBodyCov(point_this, depth_err, beam_err, var);
    body_cov_list_[i] = var;
    point_this = extR_ * point_this + extT_;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    cross_mat_list_[i] = point_crossmat;
  }
  vector<pointWithVar>().swap(pv_list_);
  pv_list_.resize(feats_down_size_);

  int rematch_num = 0;
  MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
  G.setZero();
  H_T_H.setZero();
  I_STATE.setIdentity();

  bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0;
  for (int iterCount = 0; iterCount < config_setting_.max_iterations_;
       iterCount++) {
    double total_residual = 0.0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(
        new pcl::PointCloud<pcl::PointXYZI>);
    TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_,
                   world_lidar);
    M3D rot_var = state_.cov.block<3, 3>(0, 0);
    M3D t_var = state_.cov.block<3, 3>(3, 3);
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (size_t i = 0; i < feats_down_body_->size(); i++) {
      pointWithVar &pv = pv_list_[i];
      pv.point_b << feats_down_body_->points[i].x,
          feats_down_body_->points[i].y, feats_down_body_->points[i].z;
      pv.point_w << world_lidar->points[i].x, world_lidar->points[i].y,
          world_lidar->points[i].z;

      M3D cov = body_cov_list_[i];
      M3D point_crossmat = cross_mat_list_[i];
      cov = state_.rot_end * cov * state_.rot_end.transpose() +
            (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;
      pv.var = cov;
      pv.body_var = body_cov_list_[i];
    }
    ptpl_list_.clear();

    // double t1 = omp_get_wtime();

    // BuildResidualListOMP(pv_list_, ptpl_list_);
    BuildResidualListOMPLRU(pv_list_, ptpl_list_);

    // build_residual_time += omp_get_wtime() - t1;

    for (int i = 0; i < ptpl_list_.size(); i++) {
      total_residual += fabs(ptpl_list_[i].dis_to_plane_);
    }
    effct_feat_num_ = ptpl_list_.size();
    cout << "[ LIO ] Raw feature num: " << feats_undistort_->size()
         << ", downsampled feature num:" << feats_down_size_
         << " effective feature num: " << effct_feat_num_
         << " average residual: " << total_residual / effct_feat_num_ << endl;

    /*** Computation of Measuremnt Jacobian matrix H and measurents covarience
     * ***/
    MatrixXd Hsub(effct_feat_num_, 6);
    MatrixXd Hsub_T_R_inv(6, effct_feat_num_);
    VectorXd R_inv(effct_feat_num_);
    VectorXd meas_vec(effct_feat_num_);
    meas_vec.setZero();
    for (int i = 0; i < effct_feat_num_; i++) {
      auto &ptpl = ptpl_list_[i];
      V3D point_this(ptpl.point_b_);
      point_this = extR_ * point_this + extT_;
      V3D point_body(ptpl.point_b_);
      M3D point_crossmat;
      point_crossmat << SKEW_SYM_MATRX(point_this);

      /*** get the normal vector of closest surface/corner ***/

      V3D point_world =
          state_propagat.rot_end * point_this + state_propagat.pos_end;
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = point_world - ptpl_list_[i].center_;
      J_nq.block<1, 3>(0, 3) = -ptpl_list_[i].normal_;

      M3D var;
      // V3D normal_b = state_.rot_end.inverse() * ptpl_list_[i].normal_;
      // V3D point_b = ptpl_list_[i].point_b_;
      // double cos_theta = fabs(normal_b.dot(point_b) / point_b.norm());
      // ptpl_list_[i].body_cov_ = ptpl_list_[i].body_cov_ * (1.0 / cos_theta) *
      // (1.0 / cos_theta);

      // point_w cov
      // var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ *
      // (state_propagat.rot_end * extR_).transpose() +
      //       state_propagat.cov.block<3, 3>(3, 3) + (-point_crossmat) *
      //       state_propagat.cov.block<3, 3>(0, 0) *
      //       (-point_crossmat).transpose();

      // point_w cov (another_version)
      // var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ *
      // (state_propagat.rot_end * extR_).transpose() +
      //       state_propagat.cov.block<3, 3>(3, 3) - point_crossmat *
      //       state_propagat.cov.block<3, 3>(0, 0) * point_crossmat;

      // point_body cov
      var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ *
            (state_propagat.rot_end * extR_).transpose();

      double sigma_l = J_nq * ptpl_list_[i].plane_var_ * J_nq.transpose();

      R_inv(i) = 1.0 / (0.001 + sigma_l +
                        ptpl_list_[i].normal_.transpose() * var *
                            ptpl_list_[i].normal_);
      // R_inv(i) = 1.0 / (sigma_l + ptpl_list_[i].normal_.transpose() * var *
      // ptpl_list_[i].normal_);

      /*** calculate the Measuremnt Jacobian matrix H ***/
      V3D A(point_crossmat * state_.rot_end.transpose() *
            ptpl_list_[i].normal_);
      Hsub.row(i) << VEC_FROM_ARRAY(A), ptpl_list_[i].normal_[0],
          ptpl_list_[i].normal_[1], ptpl_list_[i].normal_[2];
      Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i),
          ptpl_list_[i].normal_[0] * R_inv(i),
          ptpl_list_[i].normal_[1] * R_inv(i),
          ptpl_list_[i].normal_[2] * R_inv(i);
      meas_vec(i) = -ptpl_list_[i].dis_to_plane_;
    }
    EKF_stop_flg = false;
    flg_EKF_converged = false;
    /*** Iterative Kalman Filter Update ***/
    MatrixXd K(DIM_STATE, effct_feat_num_);
    // auto &&Hsub_T = Hsub.transpose();
    auto &&HTz = Hsub_T_R_inv * meas_vec;
    // fout_dbg<<"HTz: "<<HTz<<endl;
    H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;
    // EigenSolver<Matrix<double, 6, 6>> es(H_T_H.block<6,6>(0,0));
    MD(DIM_STATE, DIM_STATE) &&K_1 =
        (H_T_H.block<DIM_STATE, DIM_STATE>(0, 0) +
         state_.cov.block<DIM_STATE, DIM_STATE>(0, 0).inverse())
            .inverse();
    G.block<DIM_STATE, 6>(0, 0) =
        K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
    auto vec = state_propagat - state_;
    VD(DIM_STATE)
    solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz +
               vec.block<DIM_STATE, 1>(0, 0) -
               G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
    int minRow, minCol;
    state_ += solution;
    auto rot_add = solution.block<3, 1>(0, 0);
    auto t_add = solution.block<3, 1>(3, 0);
    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015)) {
      flg_EKF_converged = true;
    }
    V3D euler_cur = state_.rot_end.eulerAngles(2, 1, 0);

    /*** Rematch Judgement ***/

    if (flg_EKF_converged ||
        ((rematch_num == 0) &&
         (iterCount == (config_setting_.max_iterations_ - 2)))) {
      rematch_num++;
    }

    /*** Convergence Judgements and Covariance Update ***/
    if (!EKF_stop_flg && (rematch_num >= 2 ||
                          (iterCount == config_setting_.max_iterations_ - 1))) {
      /*** Covariance Update ***/
      // _state.cov = (I_STATE - G) * _state.cov;
      state_.cov.block<DIM_STATE, DIM_STATE>(0, 0) =
          (I_STATE.block<DIM_STATE, DIM_STATE>(0, 0) -
           G.block<DIM_STATE, DIM_STATE>(0, 0)) *
          state_.cov.block<DIM_STATE, DIM_STATE>(0, 0);
      // total_distance += (_state.pos_end - position_last).norm();
      position_last_ = state_.pos_end;
      geoQuat_ = tf::createQuaternionMsgFromRollPitchYaw(
          euler_cur(0), euler_cur(1), euler_cur(2));

      // VD(DIM_STATE) K_sum  = K.rowwise().sum();
      // VD(DIM_STATE) P_diag = _state.cov.diagonal();
      EKF_stop_flg = true;
    }
    if (EKF_stop_flg) break;
  }

  // double t2 = omp_get_wtime();
  // scan_count++;
  // ekf_time = t2 - t0 - build_residual_time;

  // ave_build_residual_time = ave_build_residual_time * (scan_count - 1) /
  // scan_count + build_residual_time / scan_count; ave_ekf_time = ave_ekf_time
  // * (scan_count - 1) / scan_count + ekf_time / scan_count;

  // cout << "[ Mapping ] ekf_time: " << ekf_time << "s, build_residual_time: "
  // << build_residual_time << "s" << endl; cout << "[ Mapping ] ave_ekf_time: "
  // << ave_ekf_time << "s, ave_build_residual_time: " <<
  // ave_build_residual_time << "s" << endl;
}

void VoxelMapManager::StateEstimation2(StatesGroup &state_propagat) {
  // 预分配/复用中间量
  cross_mat_list_.resize(feats_down_size_);
  body_cov_list_.resize(feats_down_size_);

#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (size_t i = 0; i < feats_down_body_->size(); i++) {
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y,
                   feats_down_body_->points[i].z);
    if (point_this[2] == 0) point_this[2] = 0.001;

    M3D var;
    calcBodyCov(point_this, config_setting_.dept_err_,
                config_setting_.beam_err_, var);
    body_cov_list_[i] = var;

    // 外参到相机，再求叉乘矩阵
    V3D pc = extR_ * point_this + extT_;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(pc);
    cross_mat_list_[i] = point_crossmat;
  }

  if (pv_list_.size() != feats_down_size_) pv_list_.resize(feats_down_size_);

  int rematch_num = 0;
  MD(DIM_STATE, DIM_STATE) I_STATE;
  I_STATE.setIdentity();

  bool flg_EKF_converged = false;
  bool EKF_stop_flg = false;

  for (int iterCount = 0; iterCount < config_setting_.max_iterations_;
       iterCount++) {
    double total_residual = 0.0;

    // 当前先验协方差的姿态/位置子块（用于点协方差传播）
    const M3D rot_var = state_.cov.block<3, 3>(0, 0);
    const M3D t_var = state_.cov.block<3, 3>(3, 3);
    const M3D &Rwb = state_.rot_end;
    const V3D &twb = state_.pos_end;

    // 直接内联坐标变换（避免构造 world_lidar 点云）
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (size_t i = 0; i < feats_down_body_->size(); i++) {
      pointWithVar &pv = pv_list_[i];
      pv.point_b << feats_down_body_->points[i].x,
          feats_down_body_->points[i].y, feats_down_body_->points[i].z;

      // body -> cam(ext) -> world
      V3D pc = extR_ * pv.point_b + extT_;
      V3D pw = Rwb * pc + twb;
      pv.point_w = pw;

      // 点协方差传播：R*Σ_b*R^T + Jrot*Σ_rot*Jrot^T + Σ_trans
      const M3D &cov_b = body_cov_list_[i];
      const M3D &cross = cross_mat_list_[i];
      M3D cov = Rwb * cov_b * Rwb.transpose() +
                (-cross) * rot_var * (-cross.transpose()) + t_var;
      pv.var = cov;
      pv.body_var = cov_b;
    }

    // 建立点-平面约束
    ptpl_list_.clear();
    BuildResidualListOMPLRU(pv_list_, ptpl_list_);

    for (int i = 0; i < ptpl_list_.size(); i++) {
      total_residual += std::fabs(ptpl_list_[i].dis_to_plane_);
    }
    effct_feat_num_ = ptpl_list_.size();
    std::cout << "[ LIO ] Raw feature num: " << feats_undistort_->size()
              << ", downsampled feature num:" << feats_down_size_
              << " effective feature num: " << effct_feat_num_
              << " average residual: "
              << (effct_feat_num_ ? total_residual / effct_feat_num_ : 0.0)
              << std::endl;

    // 累加 HᵀR⁻¹H(6x6) 与 HᵀR⁻¹z(6x1)
    using Mat6 = Eigen::Matrix<double, 6, 6>;
    using Vec6 = Eigen::Matrix<double, 6, 1>;
    Mat6 HtRH = Mat6::Zero();
    Vec6 HTz = Vec6::Zero();

#ifdef MP_EN
#pragma omp parallel
    {
      Mat6 HtRH_local = Mat6::Zero();
      Vec6 HTz_local = Vec6::Zero();

#pragma omp for nowait
      for (int i = 0; i < effct_feat_num_; ++i) {
        const auto &ptpl = ptpl_list_[i];

        // body点经外参到相机
        V3D pc = extR_ * ptpl.point_b_ + extT_;
        M3D cross;
        cross << SKEW_SYM_MATRX(pc);

        // 单约束雅可比 1x6: [A^T, n^T]，其中 A = [pc]_x * R^T * n
        V3D A = cross * Rwb.transpose() * ptpl.normal_;
        double z = -ptpl.dis_to_plane_;

        // 噪声：Rinv = 1 / (0.001 + sigma_l + n^T var_w n)
        V3D pw_prop = state_propagat.rot_end * pc + state_propagat.pos_end;
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = pw_prop - ptpl.center_;
        J_nq.block<1, 3>(0, 3) = -ptpl.normal_;
        M3D var_b = state_propagat.rot_end * extR_ * ptpl.body_cov_ *
                    (state_propagat.rot_end * extR_).transpose();
        double sigma_l = (J_nq * ptpl.plane_var_ * J_nq.transpose())(0, 0);
        double Rinv = 1.0 / (0.001 + sigma_l +
                             ptpl.normal_.transpose() * var_b * ptpl.normal_);

        Vec6 h;
        h << A(0), A(1), A(2), ptpl.normal_(0), ptpl.normal_(1),
            ptpl.normal_(2);
        HtRH_local.noalias() += (h * h.transpose()) * Rinv;
        HTz_local.noalias() += h * (Rinv * z);
      }

#pragma omp critical
      {
        HtRH += HtRH_local;
        HTz += HTz_local;
      }
    }
#else
    for (int i = 0; i < effct_feat_num_; ++i) {
      const auto &ptpl = ptpl_list_[i];

      V3D pc = extR_ * ptpl.point_b_ + extT_;
      M3D cross;
      cross << SKEW_SYM_MATRX(pc);
      V3D A = cross * Rwb.transpose() * ptpl.normal_;
      double z = -ptpl.dis_to_plane_;

      V3D pw_prop = state_propagat.rot_end * pc + state_propagat.pos_end;
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = pw_prop - ptpl.center_;
      J_nq.block<1, 3>(0, 3) = -ptpl.normal_;
      M3D var_b = state_propagat.rot_end * extR_ * ptpl.body_cov_ *
                  (state_propagat.rot_end * extR_).transpose();
      double sigma_l = (J_nq * ptpl.plane_var_ * J_nq.transpose())(0, 0);
      double Rinv = 1.0 / (0.001 + sigma_l +
                           ptpl.normal_.transpose() * var_b * ptpl.normal_);

      Vec6 h;
      h << A(0), A(1), A(2), ptpl.normal_(0), ptpl.normal_(1), ptpl.normal_(2);
      HtRH.noalias() += (h * h.transpose()) * Rinv;
      HTz.noalias() += h * (Rinv * z);
    }
#endif

    // 信息形式：A = P^-1 + HᵀR⁻¹H，仅左上6x6有观测增量
    MD(DIM_STATE, DIM_STATE)
    P_inv = state_.cov.block<DIM_STATE, DIM_STATE>(0, 0).inverse();
    MD(DIM_STATE, DIM_STATE) A = P_inv;
    A.block<6, 6>(0, 0).noalias() += HtRH;

    // 解 x = A^{-1} * (P^{-1}(x_prop - x_prior) + HᵀR⁻¹z)
    Eigen::LLT<MD(DIM_STATE, DIM_STATE)> llt(A);
    auto vec = state_propagat - state_;
    VD(DIM_STATE) rhs = P_inv * vec;
    rhs.block<6, 1>(0, 0).noalias() += HTz;

    VD(DIM_STATE) solution = llt.solve(rhs);

    // 状态更新与收敛判定
    state_ += solution;
    auto rot_add = solution.block<3, 1>(0, 0);
    auto t_add = solution.block<3, 1>(3, 0);
    flg_EKF_converged =
        ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015));
    V3D euler_cur = state_.rot_end.eulerAngles(2, 1, 0);

    if (flg_EKF_converged ||
        ((rematch_num == 0) &&
         (iterCount == (config_setting_.max_iterations_ - 2)))) {
      rematch_num++;
    }

    // 协方差更新：P_new = (I - G)P，其中 (I-G) = A^{-1} P^{-1}
    if (!EKF_stop_flg && (rematch_num >= 2 ||
                          (iterCount == config_setting_.max_iterations_ - 1))) {
      MD(DIM_STATE, DIM_STATE) K1Pinv = llt.solve(P_inv);
      state_.cov.block<DIM_STATE, DIM_STATE>(0, 0) =
          K1Pinv * state_.cov.block<DIM_STATE, DIM_STATE>(0, 0);

      position_last_ = state_.pos_end;
      geoQuat_ = tf::createQuaternionMsgFromRollPitchYaw(
          euler_cur(0), euler_cur(1), euler_cur(2));
      EKF_stop_flg = true;
    }
    if (EKF_stop_flg) break;
  }
}
void VoxelMapManager::TransformLidar(
    const Eigen::Matrix3d rot, const Eigen::Vector3d t,
    const PointCloudXYZI::Ptr &input_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud) {
  pcl::PointCloud<pcl::PointXYZI>().swap(*trans_cloud);
  // trans_cloud->reserve(input_cloud->size());
  trans_cloud->resize(input_cloud->size());
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (size_t i = 0; i < input_cloud->size(); i++) {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR_ * p + extT_) + t);
    pcl::PointXYZI pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    // trans_cloud->points.emplace_back(pi);
    trans_cloud->points[i] = pi;
  }
}
void VoxelMapManager::BuildVoxelMapLRU(const PointCloudXYZI::Ptr &cloud_world) {
  float voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num =
      convertToIntVectorSafe(config_setting_.layer_init_num_);
  std::vector<pointWithVar> input_points;
  for (size_t i = 0; i < cloud_world->size(); i++) {
    pointWithVar pv;
    pv.point_w << cloud_world->points[i].x, cloud_world->points[i].y,
        cloud_world->points[i].z;
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y,
                   feats_down_body_->points[i].z);
    M3D var;
    calcBodyCov(point_this, config_setting_.dept_err_,
                config_setting_.beam_err_, var);
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    var =
        (state_.rot_end * extR_) * var * (state_.rot_end * extR_).transpose() +
        (-point_crossmat) * state_.cov.block<3, 3>(0, 0) *
            (-point_crossmat).transpose() +
        state_.cov.block<3, 3>(3, 3);
    pv.var = var;
    input_points.push_back(pv);
  }
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    const pointWithVar p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                            (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end()) {
      // 体素已存在
      voxel_map_[position]->second->temp_points_.push_back(p_v);
      voxel_map_[position]->second->new_points_++;
      // 更新的放至最前
      voxel_map_cache_.splice(voxel_map_cache_.begin(), voxel_map_cache_,
                              iter->second);
      iter->second = voxel_map_cache_.begin();
    } else {
      // 体素不存在
      VoxelOctoTree *octo_tree = new VoxelOctoTree(
          max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      voxel_map_cache_.push_front({position, {octo_tree}});
      voxel_map_.insert({position, voxel_map_cache_.begin()});
      voxel_map_[position]->second->quater_length_ = voxel_size / 4;
      voxel_map_[position]->second->voxel_center_[0] =
          (0.5 + position.x) * voxel_size;
      voxel_map_[position]->second->voxel_center_[1] =
          (0.5 + position.y) * voxel_size;
      voxel_map_[position]->second->voxel_center_[2] =
          (0.5 + position.z) * voxel_size;
      voxel_map_[position]->second->temp_points_.push_back(p_v);
      voxel_map_[position]->second->new_points_++;
      voxel_map_[position]->second->layer_init_num_ = layer_init_num;

      // LRU
      // 容量检查，删除尾部节点
      if (voxel_map_cache_.size() >= config_setting_.MAX_VOXEL_NUM) {
        while (voxel_map_cache_.size() >= config_setting_.MAX_VOXEL_NUM) {
          delete voxel_map_cache_.back().second;
          auto last_key = voxel_map_cache_.back().first;
          voxel_map_.erase(last_key);
          voxel_map_cache_.pop_back();
        }
      }
    }
  }

  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); ++iter) {
    iter->second->second->init_octo_tree();
  }
}
// void VoxelMapManager::BuildVoxelMap() {
//   float voxel_size = config_setting_.max_voxel_size_;
//   float planer_threshold = config_setting_.planner_threshold_;
//   int max_layer = config_setting_.max_layer_;
//   int max_points_num = config_setting_.max_points_num_;
//   std::vector<int> layer_init_num =
//       convertToIntVectorSafe(config_setting_.layer_init_num_);

//   std::vector<pointWithVar> input_points;

//   for (size_t i = 0; i < feats_down_world_->size(); i++) {
//     pointWithVar pv;
//     pv.point_w << feats_down_world_->points[i].x,
//         feats_down_world_->points[i].y, feats_down_world_->points[i].z;
//     V3D point_this(feats_down_body_->points[i].x,
//     feats_down_body_->points[i].y,
//                    feats_down_body_->points[i].z);
//     M3D var;
//     calcBodyCov(point_this, config_setting_.dept_err_,
//                 config_setting_.beam_err_, var);
//     M3D point_crossmat;
//     point_crossmat << SKEW_SYM_MATRX(point_this);
//     var =
//         (state_.rot_end * extR_) * var * (state_.rot_end * extR_).transpose()
//         +
//         (-point_crossmat) * state_.cov.block<3, 3>(0, 0) *
//             (-point_crossmat).transpose() +
//         state_.cov.block<3, 3>(3, 3);
//     pv.var = var;
//     input_points.emplace_back(pv);
//   }

//   uint plsize = input_points.size();
//   for (uint i = 0; i < plsize; i++) {
//     const pointWithVar p_v = input_points[i];
//     float loc_xyz[3];
//     for (int j = 0; j < 3; j++) {
//       loc_xyz[j] = p_v.point_w[j] / voxel_size;
//       if (loc_xyz[j] < 0) {
//         loc_xyz[j] -= 1.0;
//       }
//     }
//     VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
//                             (int64_t)loc_xyz[2]);
//     auto iter = voxel_map_.find(position);
//     if (iter != voxel_map_.end()) {
//       voxel_map_[position]->temp_points_.emplace_back(p_v);
//       voxel_map_[position]->new_points_++;
//     } else {
//       VoxelOctoTree *octo_tree = new VoxelOctoTree(
//           max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
//       voxel_map_[position] = octo_tree;
//       voxel_map_[position]->quater_length_ = voxel_size / 4;
//       voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) *
//       voxel_size; voxel_map_[position]->voxel_center_[1] = (0.5 + position.y)
//       * voxel_size; voxel_map_[position]->voxel_center_[2] = (0.5 +
//       position.z) * voxel_size;
//       voxel_map_[position]->temp_points_.emplace_back(p_v);
//       voxel_map_[position]->new_points_++;
//       voxel_map_[position]->layer_init_num_ = layer_init_num;
//     }
//   }
//   for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); ++iter) {
//     iter->second->init_octo_tree();
//   }
// }

V3F VoxelMapManager::RGBFromVoxel(const V3D &input_point) {
  int64_t loc_xyz[3];
  for (int j = 0; j < 3; j++) {
    loc_xyz[j] = floor(input_point[j] / config_setting_.max_voxel_size_);
  }

  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                          (int64_t)loc_xyz[2]);
  int64_t ind = loc_xyz[0] + loc_xyz[1] + loc_xyz[2];
  uint k((ind + 100000) % 3);
  V3F RGB((k == 0) * 255.0, (k == 1) * 255.0, (k == 2) * 255.0);
  // cout<<"RGB: "<<RGB.transpose()<<endl;
  return RGB;
}
void VoxelMapManager::UpdateVoxelMapLRU(
    const std::vector<pointWithVar> &input_points) {
  // 参数
  float voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num =
      convertToIntVectorSafe(config_setting_.layer_init_num_);
  size_t plsize = input_points.size();

  // 遍历处理
  for (size_t i = 0; i < plsize; i++) {
    const pointWithVar p_v = input_points[i];
    // 计算voxel坐标
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                            (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    // 如果点的位置已经存在voxel 那么更新
    if (iter != voxel_map_.end()) {
      iter->second->second->UpdateOctoTree(p_v);
      voxel_map_cache_.splice(voxel_map_cache_.begin(), voxel_map_cache_,
                              iter->second);  // 更新值并移动到头部
      iter->second = voxel_map_cache_.begin();
    } else {
      // 创建value
      VoxelOctoTree *octo_tree = new VoxelOctoTree(
          max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      octo_tree->quater_length_ = voxel_size / 4;
      octo_tree->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      octo_tree->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      octo_tree->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      octo_tree->temp_points_.push_back(p_v);
      octo_tree->new_points_++;
      octo_tree->layer_init_num_ = layer_init_num;

      // 插入新节点到头部
      voxel_map_cache_.emplace_front(position, octo_tree);
      voxel_map_.insert({position, voxel_map_cache_.begin()});
    }
  }

  // // 容量检查，删除尾部节点
  // std::cout << "[ LIO ] Voxel map size: " << voxel_map_cache_.size()
  //           << std::endl;
  if (voxel_map_cache_.size() >= config_setting_.MAX_VOXEL_NUM) {
    while (voxel_map_cache_.size() >= config_setting_.MAX_VOXEL_NUM) {
      delete voxel_map_cache_.back().second;
      auto last_key = voxel_map_cache_.back().first;
      voxel_map_.erase(last_key);
      voxel_map_cache_.pop_back();
    }
  }
}
// void VoxelMapManager::UpdateVoxelMap(
//     const std::vector<pointWithVar> &input_points) {
//   float voxel_size = config_setting_.max_voxel_size_;
//   float planer_threshold = config_setting_.planner_threshold_;
//   int max_layer = config_setting_.max_layer_;
//   int max_points_num = config_setting_.max_points_num_;
//   std::vector<int> layer_init_num =
//       convertToIntVectorSafe(config_setting_.layer_init_num_);
//   uint plsize = input_points.size();
//   for (uint i = 0; i < plsize; i++) {
//     const pointWithVar p_v = input_points[i];
//     float loc_xyz[3];
//     for (int j = 0; j < 3; j++) {
//       loc_xyz[j] = p_v.point_w[j] / voxel_size;
//       if (loc_xyz[j] < 0) {
//         loc_xyz[j] -= 1.0;
//       }
//     }
//     VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
//                             (int64_t)loc_xyz[2]);
//     auto iter = voxel_map_.find(position);
//     if (iter != voxel_map_.end()) {
//       voxel_map_[position]->UpdateOctoTree(p_v);
//     } else {
//       VoxelOctoTree *octo_tree = new VoxelOctoTree(
//           max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
//       voxel_map_[position] = octo_tree;
//       voxel_map_[position]->layer_init_num_ = layer_init_num;
//       voxel_map_[position]->quater_length_ = voxel_size / 4;
//       voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) *
//       voxel_size; voxel_map_[position]->voxel_center_[1] = (0.5 + position.y)
//       * voxel_size; voxel_map_[position]->voxel_center_[2] = (0.5 +
//       position.z) * voxel_size; voxel_map_[position]->UpdateOctoTree(p_v);
//     }
//   }
// }

void VoxelMapManager::BuildResidualListOMPLRU(
    std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list) {
  int max_layer = config_setting_.max_layer_;
  double voxel_size = config_setting_.max_voxel_size_;
  double sigma_num = config_setting_.sigma_num_;
  std::mutex mylock;
  ptpl_list.clear();
  std::vector<PointToPlane> all_ptpl_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_ptpl[i] = false;
  }
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < index.size(); i++) {
    pointWithVar &pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pv.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                            (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end()) {
      VoxelOctoTree *current_octo = iter->second->second;
      PointToPlane single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      build_single_residual(pv, current_octo, 0, is_sucess, prob, single_ptpl);
      if (!is_sucess) {
        VOXEL_LOCATION near_position = position;
        if (loc_xyz[0] >
            (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
          near_position.x = near_position.x + 1;
        } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
                                 current_octo->quater_length_)) {
          near_position.x = near_position.x - 1;
        }
        if (loc_xyz[1] >
            (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
          near_position.y = near_position.y + 1;
        } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
                                 current_octo->quater_length_)) {
          near_position.y = near_position.y - 1;
        }
        if (loc_xyz[2] >
            (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
          near_position.z = near_position.z + 1;
        } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
                                 current_octo->quater_length_)) {
          near_position.z = near_position.z - 1;
        }
        auto iter_near = voxel_map_.find(near_position);
        if (iter_near != voxel_map_.end()) {
          build_single_residual(pv, iter_near->second->second, 0, is_sucess,
                                prob, single_ptpl);
        }
      }
      if (is_sucess) {
        mylock.lock();
        useful_ptpl[i] = true;
        all_ptpl_list[i] = single_ptpl;
        mylock.unlock();
      } else {
        mylock.lock();
        useful_ptpl[i] = false;
        mylock.unlock();
      }
    }
  }
  for (size_t i = 0; i < useful_ptpl.size(); i++) {
    if (useful_ptpl[i]) {
      ptpl_list.emplace_back(all_ptpl_list[i]);
    }
  }
}

// void VoxelMapManager::BuildResidualListOMP(
//     std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list)
//     {
//   int max_layer = config_setting_.max_layer_;
//   double voxel_size = config_setting_.max_voxel_size_;
//   double sigma_num = config_setting_.sigma_num_;
//   std::mutex mylock;
//   ptpl_list.clear();
//   std::vector<PointToPlane> all_ptpl_list(pv_list.size());
//   std::vector<bool> useful_ptpl(pv_list.size());
//   std::vector<size_t> index(pv_list.size());
//   for (size_t i = 0; i < index.size(); ++i) {
//     index[i] = i;
//     useful_ptpl[i] = false;
//   }
// #ifdef MP_EN
//   omp_set_num_threads(MP_PROC_NUM);
// #pragma omp parallel for
// #endif
//   for (int i = 0; i < index.size(); i++) {
//     pointWithVar &pv = pv_list[i];
//     float loc_xyz[3];
//     for (int j = 0; j < 3; j++) {
//       loc_xyz[j] = pv.point_w[j] / voxel_size;
//       if (loc_xyz[j] < 0) {
//         loc_xyz[j] -= 1.0;
//       }
//     }
//     VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
//                             (int64_t)loc_xyz[2]);
//     auto iter = voxel_map_.find(position);
//     if (iter != voxel_map_.end()) {
//       VoxelOctoTree *current_octo = iter->second;
//       PointToPlane single_ptpl;
//       bool is_sucess = false;
//       double prob = 0;
//       build_single_residual(pv, current_octo, 0, is_sucess, prob,
//       single_ptpl); if (!is_sucess) {
//         VOXEL_LOCATION near_position = position;
//         if (loc_xyz[0] >
//             (current_octo->voxel_center_[0] + current_octo->quater_length_))
//             {
//           near_position.x = near_position.x + 1;
//         } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
//                                  current_octo->quater_length_)) {
//           near_position.x = near_position.x - 1;
//         }
//         if (loc_xyz[1] >
//             (current_octo->voxel_center_[1] + current_octo->quater_length_))
//             {
//           near_position.y = near_position.y + 1;
//         } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
//                                  current_octo->quater_length_)) {
//           near_position.y = near_position.y - 1;
//         }
//         if (loc_xyz[2] >
//             (current_octo->voxel_center_[2] + current_octo->quater_length_))
//             {
//           near_position.z = near_position.z + 1;
//         } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
//                                  current_octo->quater_length_)) {
//           near_position.z = near_position.z - 1;
//         }
//         auto iter_near = voxel_map_.find(near_position);
//         if (iter_near != voxel_map_.end()) {
//           build_single_residual(pv, iter_near->second, 0, is_sucess, prob,
//                                 single_ptpl);
//         }
//       }
//       if (is_sucess) {
//         mylock.lock();
//         useful_ptpl[i] = true;
//         all_ptpl_list[i] = single_ptpl;
//         mylock.unlock();
//       } else {
//         mylock.lock();
//         useful_ptpl[i] = false;
//         mylock.unlock();
//       }
//     }
//   }
//   for (size_t i = 0; i < useful_ptpl.size(); i++) {
//     if (useful_ptpl[i]) {
//       ptpl_list.emplace_back(all_ptpl_list[i]);
//     }
//   }
// }

void VoxelMapManager::build_single_residual(pointWithVar &pv,
                                            const VoxelOctoTree *current_octo,
                                            const int current_layer,
                                            bool &is_sucess, double &prob,
                                            PointToPlane &single_ptpl) {
  int max_layer = config_setting_.max_layer_;
  double sigma_num = config_setting_.sigma_num_;

  double radius_k = 3;
  Eigen::Vector3d p_w = pv.point_w;
  if (current_octo->plane_ptr_->is_plane_) {
    VoxelPlane &plane = *current_octo->plane_ptr_;
    Eigen::Vector3d p_world_to_center = p_w - plane.center_;
    float dis_to_plane =
        fabs(plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) +
             plane.normal_(2) * p_w(2) + plane.d_);
    float dis_to_center =
        (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
        (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
        (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
    float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

    if (range_dis <= radius_k * plane.radius_) {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
      J_nq.block<1, 3>(0, 3) = -plane.normal_;
      double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
      sigma_l += plane.normal_.transpose() * pv.var * plane.normal_;
      if (dis_to_plane < sigma_num * sqrt(sigma_l)) {
        is_sucess = true;
        double this_prob = 1.0 / (sqrt(sigma_l)) *
                           exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (this_prob > prob) {
          prob = this_prob;
          pv.normal = plane.normal_;
          single_ptpl.body_cov_ = pv.body_var;
          single_ptpl.point_b_ = pv.point_b;
          single_ptpl.point_w_ = pv.point_w;
          single_ptpl.plane_var_ = plane.plane_var_;
          single_ptpl.normal_ = plane.normal_;
          single_ptpl.center_ = plane.center_;
          single_ptpl.d_ = plane.d_;
          single_ptpl.layer_ = current_layer;
          single_ptpl.dis_to_plane_ = plane.normal_(0) * p_w(0) +
                                      plane.normal_(1) * p_w(1) +
                                      plane.normal_(2) * p_w(2) + plane.d_;
        }
        return;
      } else {
        // is_sucess = false;
        return;
      }
    } else {
      // is_sucess = false;
      return;
    }
  } else {
    if (current_layer < max_layer) {
      for (size_t leafnum = 0; leafnum < 8; leafnum++) {
        if (current_octo->leaves_[leafnum] != nullptr) {
          VoxelOctoTree *leaf_octo = current_octo->leaves_[leafnum];
          build_single_residual(pv, leaf_octo, current_layer + 1, is_sucess,
                                prob, single_ptpl);
        }
      }
      return;
    } else {
      return;
    }
  }
}
void VoxelMapManager::pubVoxelMapLRU() {
  double max_trace = 0.25;
  double pow_num = 0.2;
  rclcpp::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::msg::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<VoxelPlane> pub_plane_list;
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); iter++) {
    GetUpdatePlane(iter->second->second, config_setting_.max_layer_,
                   pub_plane_list);
  }
  for (size_t i = 0; i < pub_plane_list.size(); i++) {
    V3D plane_cov = pub_plane_list[i].plane_var_.block<3, 3>(0, 0).diagonal();
    double trace = plane_cov.sum();
    if (trace >= max_trace) {
      trace = max_trace;
    }
    trace = trace * (1.0 / max_trace);
    trace = pow(trace, pow_num);
    uint8_t r, g, b;
    mapJet(trace, 0, 1, r, g, b);
    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
    double alpha;
    if (pub_plane_list[i].is_plane_) {
      alpha = use_alpha;
    } else {
      alpha = 0;
    }
    pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);
  }
  voxel_map_pub_->publish(voxel_plane);
  loop.sleep();
}

// void VoxelMapManager::pubVoxelMap() {
//   double max_trace = 0.25;
//   double pow_num = 0.2;
//   rclcpp::Rate loop(500);
//   float use_alpha = 0.8;
//   visualization_msgs::msg::MarkerArray voxel_plane;
//   voxel_plane.markers.reserve(1000000);
//   std::vector<VoxelPlane> pub_plane_list;
//   for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); iter++) {
//     GetUpdatePlane(iter->second, config_setting_.max_layer_, pub_plane_list);
//   }
//   for (size_t i = 0; i < pub_plane_list.size(); i++) {
//     V3D plane_cov = pub_plane_list[i].plane_var_.block<3, 3>(0,
//     0).diagonal(); double trace = plane_cov.sum(); if (trace >= max_trace) {
//       trace = max_trace;
//     }
//     trace = trace * (1.0 / max_trace);
//     trace = pow(trace, pow_num);
//     uint8_t r, g, b;
//     mapJet(trace, 0, 1, r, g, b);
//     Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
//     double alpha;
//     if (pub_plane_list[i].is_plane_) {
//       alpha = use_alpha;
//     } else {
//       alpha = 0;
//     }
//     pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha,
//     plane_rgb);
//   }
//   voxel_map_pub_->publish(voxel_plane);
//   loop.sleep();
// }

void VoxelMapManager::GetUpdatePlane(const VoxelOctoTree *current_octo,
                                     const int pub_max_voxel_layer,
                                     std::vector<VoxelPlane> &plane_list) {
  if (current_octo->layer_ > pub_max_voxel_layer) {
    return;
  }
  if (current_octo->plane_ptr_->is_update_) {
    plane_list.emplace_back(*current_octo->plane_ptr_);
  }
  if (current_octo->layer_ < current_octo->max_layer_) {
    if (!current_octo->plane_ptr_->is_plane_) {
      for (size_t i = 0; i < 8; i++) {
        if (current_octo->leaves_[i] != nullptr) {
          GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer,
                         plane_list);
        }
      }
    }
  }
  return;
}

void VoxelMapManager::pubSinglePlane(
    visualization_msgs::msg::MarkerArray &plane_pub, const std::string plane_ns,
    const VoxelPlane &single_plane, const float alpha,
    const Eigen::Vector3d rgb) {
  visualization_msgs::msg::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = rclcpp::Time();
  plane.ns = plane_ns;
  plane.id = single_plane.id_;
  plane.type = visualization_msgs::msg::Marker::CYLINDER;
  plane.action = visualization_msgs::msg::Marker::ADD;
  plane.pose.position.x = single_plane.center_[0];
  plane.pose.position.y = single_plane.center_[1];
  plane.pose.position.z = single_plane.center_[2];
  geometry_msgs::msg::Quaternion q;
  CalcVectQuation(single_plane.x_normal_, single_plane.y_normal_,
                  single_plane.normal_, q);
  plane.pose.orientation = q;
  plane.scale.x = 3 * sqrt(single_plane.max_eigen_value_);
  plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value_);
  plane.scale.z = 2 * sqrt(single_plane.min_eigen_value_);
  plane.color.a = alpha;
  plane.color.r = rgb(0);
  plane.color.g = rgb(1);
  plane.color.b = rgb(2);
  plane.lifetime = rclcpp::Duration::from_seconds(0.01);
  plane_pub.markers.emplace_back(plane);
}

void VoxelMapManager::CalcVectQuation(const Eigen::Vector3d &x_vec,
                                      const Eigen::Vector3d &y_vec,
                                      const Eigen::Vector3d &z_vec,
                                      geometry_msgs::msg::Quaternion &q) {
  Eigen::Matrix3d rot;
  rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0),
      z_vec(1), z_vec(2);
  Eigen::Matrix3d rotation = rot.transpose();
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void VoxelMapManager::mapJet(double v, double vmin, double vmax, uint8_t &r,
                             uint8_t &g, uint8_t &b) {
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) {
    v = vmin;
  }

  if (v > vmax) {
    v = vmax;
  }

  double dr, dg, db;

  if (v < 0.1242) {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  } else if (v < 0.3747) {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  } else if (v < 0.6253) {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  } else if (v < 0.8758) {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  } else {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}

// void VoxelMapManager::mapSliding() {
//   if ((position_last_ - last_slide_position).norm() <
//       config_setting_.sliding_thresh) {
//     std::cout << RED << "[DEBUG]: Last sliding length "
//               << (position_last_ - last_slide_position).norm() << RESET <<
//               "\n";
//     return;
//   }

//   // get global id now
//   last_slide_position = position_last_;
//   double t_sliding_start = omp_get_wtime();
//   float loc_xyz[3];
//   for (int j = 0; j < 3; j++) {
//     loc_xyz[j] = position_last_[j] / config_setting_.max_voxel_size_;
//     if (loc_xyz[j] < 0) {
//       loc_xyz[j] -= 1.0;
//     }
//   }
//   // VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
//   // (int64_t)loc_xyz[2]);//discrete global
//   clearMemOutOfMap((int64_t)loc_xyz[0] + config_setting_.half_map_size,
//                    (int64_t)loc_xyz[0] - config_setting_.half_map_size,
//                    (int64_t)loc_xyz[1] + config_setting_.half_map_size,
//                    (int64_t)loc_xyz[1] - config_setting_.half_map_size,
//                    (int64_t)loc_xyz[2] + config_setting_.half_map_size,
//                    (int64_t)loc_xyz[2] - config_setting_.half_map_size);
//   double t_sliding_end = omp_get_wtime();
//   std::cout << RED << "[DEBUG]: Map sliding using "
//             << t_sliding_end - t_sliding_start << " secs" << RESET << "\n";
//   return;
// }

// void VoxelMapManager::clearMemOutOfMap(const int &x_max, const int &x_min,
//                                        const int &y_max, const int &y_min,
//                                        const int &z_max, const int &z_min) {
//   int delete_voxel_cout = 0;
//   // double delete_time = 0;
//   // double last_delete_time = 0;
//   for (auto it = voxel_map_.begin(); it != voxel_map_.end();) {
//     const VOXEL_LOCATION &loc = it->first;
//     bool should_remove = loc.x > x_max || loc.x < x_min || loc.y > y_max ||
//                          loc.y < y_min || loc.z > z_max || loc.z < z_min;
//     if (should_remove) {
//       // last_delete_time = omp_get_wtime();
//       delete it->second;
//       it = voxel_map_.erase(it);
//       // delete_time += omp_get_wtime() - last_delete_time;
//       delete_voxel_cout++;
//     } else {
//       ++it;
//     }
//   }
//   std::cout << RED << "[DEBUG]: Delete " << delete_voxel_cout << " root
//   voxels"
//             << RESET << "\n";
//   // std::cout<<RED<<"[DEBUG]: Delete "<<delete_voxel_cout<<" voxels using
//   // "<<delete_time<<" s"<<RESET<<"\n";
// }
