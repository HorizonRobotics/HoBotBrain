#include <Eigen/Dense>
#include <iostream>

#include "common_lib.h"
// 固定平面更新类
class GroundConstrain {
 public:
  GroundConstrain(const Eigen::Vector3d& n_w, double d)
      : n_w_(n_w.normalized()), d_(d) {}

  void update(StatesGroup& state,
              const Eigen::Vector3d& p_i_obs,  // IMU系下观测点
              const Eigen::Matrix<double, 1, 1>& R_meas) {
    // 世界到IMU旋转
    Eigen::Matrix3d R_wi = state.rot_end;
    Eigen::Vector3d p_wi = state.pos_end;

    // 预测值
    double z_pred = n_w_.transpose() * (p_wi + R_wi * p_i_obs) + d_;
    double residual = -z_pred;  // 希望约束满足 n_w^T * (...) + d = 0

    // 构造H (1x18)
    Eigen::Matrix<double, 1, DIM_STATE> H;
    H.setZero();

    // 对姿态偏差的偏导: -n_w^T * R_wi * [p_i_obs]_x
    Eigen::Matrix3d p_hat;
    p_hat << 0, -p_i_obs.z(), p_i_obs.y(), p_i_obs.z(), 0, -p_i_obs.x(),
        -p_i_obs.y(), p_i_obs.x(), 0;

    Eigen::RowVector3d H_theta = -n_w_.transpose() * R_wi * p_hat;
    H.block<1, 3>(0, 0) = H_theta;  // 0~2: 姿态

    // 对位置的偏导: n_w^T
    H.block<1, 3>(0, 3) = n_w_.transpose();  // 3~5: 位置

    // 其它状态对平面约束无直接影响，保持0

    // 卡尔曼增益
    Eigen::Matrix<double, 1, 1> S = H * state.cov * H.transpose() + R_meas;
    Eigen::Matrix<double, DIM_STATE, 1> K =
        state.cov * H.transpose() * S.inverse();

    // 状态更新
    Eigen::Matrix<double, DIM_STATE, 1> dx = K * residual;
    state += dx;  // 利用 StatesGroup::operator+=

    // 协方差更新
    Eigen::Matrix<double, DIM_STATE, DIM_STATE> I =
        Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
    state.cov = (I - K * H) * state.cov;
  }

 private:
  Eigen::Vector3d n_w_;  // 世界系平面法向
  double d_;             // 平面方程常数
};