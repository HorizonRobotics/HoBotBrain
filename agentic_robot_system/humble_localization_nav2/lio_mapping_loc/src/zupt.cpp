#include "zupt.h"

#include <Eigen/Dense>
#include <iostream>

#include "common_lib.h"

void ZUPT::applyZConstraint() {
  // 初始化变量
  Eigen::MatrixXd I =
      Eigen::MatrixXd::Identity(state_.cov.rows(), state_.cov.cols());
  bool is_converged = false;
  int max_iterations = 5;               // 最大迭代次数
  double convergence_threshold = 1e-2;  // 收敛阈值

  for (int iter = 0; iter < max_iterations; ++iter) {
    // 计算观测残差
    double residual = z_measurement_ - state_.pos_end.z();

    // 构造观测矩阵 H
    Eigen::MatrixXd H(1, state_.cov.rows());
    H.setZero();
    H(0, 5) = 1.0;  // Z 轴约束

    // 构造观测噪声矩阵 R
    Eigen::MatrixXd R(1, 1);
    R(0, 0) = measurement_noise_ * measurement_noise_;

    // 卡尔曼增益计算
    Eigen::MatrixXd S = H * state_.cov * H.transpose() + R;
    Eigen::MatrixXd K = state_.cov * H.transpose() * S.inverse();

    // 更新状态
    Eigen::VectorXd delta_x = K * residual;
    // std::cout << "[ZUPT] K: " << K.transpose() << std::endl;
    std::cout << "[ZUPT] Iteration " << iter << ", Residual: " << residual
              << ", Delta X: " << delta_x.transpose() << std::endl;
    // 只更新 Z 位置
    state_.pos_end.z() += delta_x(5);

    // 只更新与z相关的协方差（第5行和第5列）
    Eigen::MatrixXd cov_update = (I - K * H) * state_.cov;

    // 检查收敛条件
    if (delta_x.norm() < convergence_threshold) {
      is_converged = true;
      state_.cov.row(5) = cov_update.row(5);
      state_.cov.col(5) = cov_update.col(5);
      break;
    } else if (iter == max_iterations - 1) {
      state_.cov.row(5) = cov_update.row(5);
      state_.cov.col(5) = cov_update.col(5);
    }
  }

  if (!is_converged) {
    std::cerr << "[ZUPT] Warning: applyZConstraint did not converge within the "
                 "maximum iterations."
              << std::endl;
  }
}
