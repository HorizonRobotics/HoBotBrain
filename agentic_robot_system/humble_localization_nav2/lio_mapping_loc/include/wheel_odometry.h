#pragma once
#include <Eigen/Dense>

#include "common_lib.h"

// ---------------------------
// Wheel Odometry Constraint using extrinsic EKF (state in world frame)
// ---------------------------
class WheelOdometryConstraint {
 public:
  WheelOdometryConstraint() {}
  void initialize(const vector<double>& R_meas, const vector<double>& R_init_ex,
                  const vector<double>& p_init_ex) {
    wheel_meas_cov =
        Eigen::Map<const Eigen::Matrix<double, 3, 1>>(R_meas.data())
            .asDiagonal();
    wheel_meas_cov *= wheel_meas_cov;
    // 外参初始化
    R_ex_ << MAT_FROM_ARRAY(R_init_ex);
    p_ex_ << VEC_FROM_ARRAY(p_init_ex);
    // 初始化外参与交叉协方差
    P_ext_ext.setZero();
    P_x_ext_.setZero();
    // 角度(外参旋转)方差 ~ (5°)^2，平移(外参杠杆臂)方差 ~
    // (0.1m)^2，可按实际调整
    const double sig_rex = 5.0 * M_PI / 180.0;
    const double sig_pex = 0.1;
    P_ext_ext.block<3, 3>(0, 0) =
        (sig_rex * sig_rex) * Eigen::Matrix3d::Identity();
    P_ext_ext.block<3, 3>(3, 3) =
        (sig_pex * sig_pex) * Eigen::Matrix3d::Identity();
    P_x_ext_.setZero();

    ex_cov_inited_ = true;
  }
  // 联合估计：位姿/速度/零偏 + 外参（R_ex_, p_ex_）在同一滤波中更新
  // 模板参数：IDX_REX 为外参旋转3维块起始索引，IDX_PEX 为外参平移3维块起始索引
  void update_state_joint(StatesGroup& state,
                          const Eigen::Vector3d& vel_wheel) {
    if (!ex_cov_inited_) {
      // 若未显式 initialize，则做一次保底初始化
      P_ext_ext.setIdentity();
      P_ext_ext *= 1e-2;
      P_x_ext_.setZero();
      ex_cov_inited_ = true;
    }
    const double eps_dx = 1e-3;
    const int max_iter = 5;
    // IEKF：固定先验
    Eigen::Matrix<double, DIM_STATE, DIM_STATE> P_xx =
        0.5 * (state.cov + state.cov.transpose());

    // 组合增广先验 P_aug = [P_xx  P_xe; P_xe^T  P_ee]
    constexpr int NX = DIM_STATE;
    constexpr int NE = 6;
    Eigen::Matrix<double, NX + NE, NX + NE> P_aug;
    P_aug.setZero();
    P_aug.block(0, 0, NX, NX) = P_xx;
    P_aug.block(0, NX, NX, NE) = P_x_ext_;
    P_aug.block(NX, 0, NE, NX) = P_x_ext_.transpose();
    P_aug.block(NX, NX, NE, NE) = 0.5 * (P_ext_ext + P_ext_ext.transpose());

    // 预备
    Eigen::Matrix<double, 3, NX + NE> H;  // 增广雅可比
    Eigen::Vector3d r;
    double R_scale_last = 1.0;  // 记录最终使用的R放大因子
    for (int it = 0; it < max_iter; ++it) {
      // 计算状态相关量
      const Eigen::Matrix3d R_WI = state.rot_end;
      const Eigen::Vector3d omega_I = state.ang_vel_end - state.bias_g;
      const Eigen::Vector3d v_imu = R_WI.transpose() * state.vel_end;

      // 预测与残差（轮系）：h = R_ex_^T ( v_imu + ω×p_ex_ )
      const Eigen::Vector3d a = v_imu + omega_I.cross(p_ex_);
      const Eigen::Vector3d v_pred_wheel = R_ex_.transpose() * a;
      r = vel_wheel - v_pred_wheel;  // r = z - h(x)

      // 右扰动雅可比（H = ∂h/∂[x; extr]）
      H.setZero();
      // 整机状态块
      H.block<3, 3>(0, 0) = R_ex_.transpose() * skew(v_imu);  // ∂h/∂δθ_WI
      // 位置不观测 -> 0
      H.block<3, 3>(0, 7) = R_ex_.transpose() * R_WI.transpose();  // ∂h/∂v_W
      H.block<3, 3>(0, 10) = R_ex_.transpose() * skew(p_ex_);      // ∂h/∂b_g
      // 外参块：将其放在增广状态末尾
      // 外参旋转（右扰动）：∂h/∂δθ_ex = skew(v_pred_wheel)
      H.block<3, 3>(0, NX + 0) = skew(v_pred_wheel);
      // 外参平移：∂h/∂p_ex_ = R_ex_^T [ω]_x
      H.block<3, 3>(0, NX + 3) = R_ex_.transpose() * skew(omega_I);
      // 4.1) 基于马氏距离的自适应R（局部R_meas，不改成员wheel_meas_cov）
      Eigen::Matrix3d S0 = H * P_aug * H.transpose() + wheel_meas_cov;
      double maha = (r.transpose() * S0.inverse() * r)(0, 0);
      // χ²(3)阈值：95%≈7.815，99.9%≈16.27
      const double gate = 7.815, gate_hi = 16.27, scale_max = 100.0;
      double scale = 1.0;
      if (std::isfinite(maha) && maha > gate_hi) {
        // 简单鲁棒放大：按 maha/gate 放大，并做上限裁剪
        scale = std::min(maha / gate_hi, scale_max);
      }
      Eigen::Matrix3d R_meas = scale * wheel_meas_cov;
      R_scale_last = scale;
      std::cout << "[Wheel Odom] maha: " << maha << ", scale: " << scale
                << std::endl;
      std::cout << "[Wheel Odom] R_meas: " << R_meas << std::endl;
      // 创新协方差与增益（轮系噪声）
      // 5) Gain with prior covariance (IEKF)
      Eigen::Matrix3d S = H * P_aug * H.transpose() + R_meas;
      Eigen::Matrix<double, NX + NE, 3> K_aug =
          P_aug * H.transpose() * S.inverse();

      const Eigen::Matrix<double, NX + NE, 1> dx = K_aug * r;

      // 原始状态更新
      const Eigen::Vector3d dtheta = dx.block<3, 1>(0, 0);
      const Eigen::Vector3d dpos = dx.block<3, 1>(3, 0);
      const Eigen::Vector3d dvel = dx.block<3, 1>(7, 0);
      const Eigen::Vector3d dbg = dx.block<3, 1>(10, 0);
      const Eigen::Vector3d dba = dx.block<3, 1>(13, 0);
      const Eigen::Vector3d dg = dx.block<3, 1>(16, 0);

      // 外参
      const Eigen::Vector3d drex = dx.block<3, 1>(NX, 0);
      const Eigen::Vector3d dpex = dx.block<3, 1>(NX + 3, 0);

      // state.rot_end = state.rot_end * Exp(dtheta(0), dtheta(1), dtheta(2));
      // state.pos_end += dpos;
      // state.vel_end += dvel;
      // state.bias_g += dbg;
      state += dx.block<DIM_STATE, 1>(0, 0);
      std::cout << "[Wheel Odom] Iter " << it << " Residual: " << r.transpose()
                << std::endl;
      std::cout << "[Wheel Odom] dpos: " << dpos.transpose() << std::endl
                << "[Wheel Odom] dvel: " << dvel.transpose() << std::endl
                << "[Wheel Odom] dbg: " << dbg.transpose() << std::endl;
      std::cout << "[Wheel Odom] drex: " << drex.transpose() << std::endl
                << "[Wheel Odom] dpex: " << dpex.transpose() << std::endl;

      // 外参（右扰动）
      R_ex_ = R_ex_ * Exp(drex(0), drex(1), drex(2));
      p_ex_ += dpex;

      if (dvel.norm() < eps_dx) break;
    }

    // 末次线性化 + Joseph 协方差
    {
      const Eigen::Matrix3d R_WI = state.rot_end;
      const Eigen::Vector3d omega_I = state.ang_vel_end - state.bias_g;
      const Eigen::Vector3d v_imu = R_WI.transpose() * state.vel_end;
      const Eigen::Vector3d a = v_imu + omega_I.cross(p_ex_);
      const Eigen::Vector3d v_pred_wheel = R_ex_.transpose() * a;
      H.setZero();
      H.block<3, 3>(0, 0) = R_ex_.transpose() * skew(v_imu);       // ∂h/∂δθ_WI
      H.block<3, 3>(0, 7) = R_ex_.transpose() * R_WI.transpose();  // ∂h/∂v_W
      H.block<3, 3>(0, 10) = R_ex_.transpose() * skew(p_ex_);      // ∂h/∂b_g
      H.block<3, 3>(0, NX + 0) = skew(v_pred_wheel);
      H.block<3, 3>(0, NX + 3) = R_ex_.transpose() * skew(omega_I);
      // 使用最终的自适应R
      Eigen::Matrix3d R_meas = R_scale_last * wheel_meas_cov;
      Eigen::Matrix<double, NX + NE, NX + NE> I_aug =
          Eigen::Matrix<double, NX + NE, NX + NE>::Identity();
      Eigen::Matrix3d S = H * P_aug * H.transpose() + R_meas;
      Eigen::Matrix<double, NX + NE, 3> K_aug =
          P_aug * H.transpose() * S.inverse();
      Eigen::Matrix<double, NX + NE, NX + NE> P_post =
          (I_aug - K_aug * H) * P_aug * (I_aug - K_aug * H).transpose() +
          K_aug * R_meas * K_aug.transpose();
      P_post = 0.5 * (P_post + P_post.transpose());

      // 写回：拆分为整机、交叉、外参协方差
      state.cov = P_post.block(0, 0, NX, NX);
      P_x_ext_ = P_post.block(0, NX, NX, NE);
      P_ext_ext = P_post.block(NX, NX, NE, NE);
      std::cout << "[Wheel Odom] Post Extrinsic Cov: " << P_ext_ext
                << std::endl;
    }
  }

  void update_state(StatesGroup& state, const Eigen::Vector3d& vel_wheel) {
    const int max_iter = 5;
    const double eps_dx = 1e-2;

    // Keep prior covariance for IEKF; update P only once at the end
    const Eigen::Matrix<double, DIM_STATE, DIM_STATE> P_prior = state.cov;

    Eigen::Matrix<double, 3, DIM_STATE> H;
    Eigen::Vector3d r;
    Eigen::Matrix<double, DIM_STATE, 1> solution;
    // wheel to imu
    Eigen::Vector3d omega_I = state.ang_vel_end - state.bias_g;

    // 如果角速度大于10°/s，不进行融合
    // if (abs(omega_I[2]) > 10.0 * M_PI / 180.0) return;
    double R_scale_last = 1.0;  // 记录最终使用的R放大因子
    for (int it = 0; it < max_iter; ++it) {
      // 1) State-dependent terms
      Eigen::Matrix3d R_WI = state.rot_end;

      // 2) Predict measurement
      Eigen::Vector3d v_imu = R_WI.transpose() * state.vel_end;
      Eigen::Vector3d v_pred_wheel =
          R_ex_.transpose() * (v_imu + skew(omega_I) * p_ex_);
      std::cout << "[Wheel Odom] Iter " << it << " v_imu: " << v_imu.transpose()
                << std::endl
                << "v_pred_wheel: " << v_pred_wheel.transpose() << std::endl;

      // 3) Residual
      r = vel_wheel - v_pred_wheel;
      std::cout << "Residual: " << r.transpose() << std::endl;

      // 4) Jacobian
      H.setZero();
      H.block<3, 3>(0, 0) =
          R_ex_.transpose() * skew(v_imu);  // d wrt IMU rotation
      H.block<3, 3>(0, 7) =
          R_ex_.transpose() * R_WI.transpose();  // d wrt world velocity
      H.block<3, 3>(0, 10) = R_ex_.transpose() * skew(p_ex_);  // d wrt gyro

      // 4.1) 基于马氏距离的自适应R（局部R_meas，不改成员wheel_meas_cov）
      Eigen::Matrix3d S0 = H * P_prior * H.transpose() + wheel_meas_cov;
      double maha = (r.transpose() * S0.inverse() * r)(0, 0);
      // χ²(3)阈值：95%≈7.815，99.9%≈16.27
      const double gate = 7.815, gate_hi = 16.27, scale_max = 100.0;
      double scale = 1.0;
      if (std::isfinite(maha) && maha > 3 * gate_hi) {
        // 简单鲁棒放大：按 maha/gate 放大，并做上限裁剪
        scale = std::min(maha / gate_hi, scale_max);
      }
      Eigen::Matrix3d R_meas = scale * wheel_meas_cov;
      R_scale_last = scale;
      std::cout << "[Wheel Odom] maha: " << maha << ", scale: " << scale
                << std::endl;
      std::cout << "[Wheel Odom] R_meas: " << R_meas << std::endl;
      // 5) Gain with prior covariance (IEKF)
      Eigen::Matrix3d S = H * P_prior * H.transpose() + R_meas;
      Eigen::Matrix<double, DIM_STATE, 3> K =
          P_prior * H.transpose() * S.inverse();

      // 6) Increment and relinearize
      solution = K * r;
      auto vel_add = solution.block<3, 1>(7, 0);
      state += solution;
      if (vel_add.norm() < eps_dx) break;
    }

    // Covariance update (Joseph form) using final linearization
    {
      Eigen::Matrix3d R_WI = state.rot_end;
      Eigen::Vector3d v_imu = R_WI.transpose() * state.vel_end;

      H.setZero();
      H.block<3, 3>(0, 0) = R_ex_.transpose() * skew(v_imu);
      H.block<3, 3>(0, 7) = R_ex_.transpose() * R_WI.transpose();
      H.block<3, 3>(0, 10) = R_ex_.transpose() * skew(p_ex_);
      // 使用最终的自适应R
      Eigen::Matrix3d R_meas = R_scale_last * wheel_meas_cov;

      Eigen::Matrix3d S = H * P_prior * H.transpose() + R_meas;
      Eigen::Matrix<double, DIM_STATE, 3> K =
          P_prior * H.transpose() * S.inverse();

      Eigen::Matrix<double, DIM_STATE, DIM_STATE> I =
          Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
      Eigen::Matrix<double, DIM_STATE, DIM_STATE> KH = K * H;
      state.cov = (I - KH) * P_prior * (I - KH).transpose() +
                  K * R_meas * K.transpose();
    }
  }

  void update_vel(StatesGroup& state, const Eigen::Vector3d& vel_wheel) {
    const int max_iter = 3;
    const double eps_dx = 1e-2;

    // Keep prior covariance for IEKF; update P only once at the end
    Eigen::Matrix<double, DIM_STATE, DIM_STATE> P_prior = state.cov;

    Eigen::Matrix<double, 3, DIM_STATE> H;
    Eigen::Vector3d r;
    Eigen::Matrix<double, DIM_STATE, 1> solution;
    Eigen::Vector3d omega_I = state.ang_vel_end - state.bias_g;
    for (int it = 0; it < max_iter; ++it) {
      // 1) State-dependent terms
      Eigen::Matrix3d R_WI = state.rot_end;

      // 2) Predict measurement
      const Eigen::Vector3d v_pred_world =
          R_WI * (R_ex_ * vel_wheel - omega_I.cross(p_ex_));

      std::cout << "[Wheel Odom] Iter " << it
                << " v_pred_world: " << v_pred_world.transpose() << std::endl;
      std::cout << "Current vel: " << state.vel_end.transpose() << std::endl;

      // 3) Residual
      r = v_pred_world - state.vel_end;
      std::cout << "Residual: " << r.transpose() << std::endl;
      // 取正后计算去最小值
      double min_res = r.cwiseAbs().minCoeff();
      // TODO: 拐弯处降低权重
      // 验前卡方检测
      // for (int i = 0; i < r.size(); ++i) {
      //   wheel_meas_cov(i, i) =
      //       wheel_meas_cov(i, i) * pow(abs(r[i]) / min_res, 2);
      // }
      // 4) Jacobian
      H.setZero();
      H.block<3, 3>(0, 6) =
          Eigen::Matrix3d::Identity();  // d wrt world velocity
                                        // 5) Gain with prior covariance (IEKF)
      const Eigen::Matrix3d T = R_WI * R_ex_;
      wheel_cov_world = T * wheel_meas_cov * T.transpose();
      Eigen::Matrix3d S = H * P_prior * H.transpose() + wheel_cov_world;
      Eigen::Matrix<double, DIM_STATE, 3> K =
          P_prior * H.transpose() * S.inverse();

      // 6) Increment and relinearize
      solution = K * r;
      auto rot_add = solution.block<3, 1>(0, 0);
      std::cout << "rot_add: " << rot_add.transpose() << std::endl;
      auto pos_add = solution.block<3, 1>(3, 0);
      std::cout << "pos_add: " << pos_add.transpose() << std::endl;
      auto vel_add = solution.block<3, 1>(7, 0);
      std::cout << "vel_add: " << vel_add.transpose() << std::endl;
      state += solution;
      // // only update pose and velocity
      // state.pos_end += pos_add;
      // state.vel_end += vel_add;

      if (vel_add.norm() < eps_dx) {
        break;
      }
    }

    // Covariance update (Joseph form) using final linearization
    {
      Eigen::Matrix3d R_WI = state.rot_end;
      H.setZero();
      H.block<3, 3>(0, 7) =
          Eigen::Matrix3d::Identity();  // d wrt world velocity
      const Eigen::Matrix3d T = R_WI * R_ex_;
      wheel_cov_world = T * wheel_meas_cov * T.transpose();
      Eigen::Matrix3d S = H * P_prior * H.transpose() + wheel_cov_world;
      Eigen::Matrix<double, DIM_STATE, 3> K =
          P_prior * H.transpose() * S.inverse();
      Eigen::Matrix<double, DIM_STATE, DIM_STATE> I =
          Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
      state.cov = (I - K * H) * P_prior * (I - K * H).transpose() +
                  K * wheel_cov_world * K.transpose();
    }
  }

 private:
  Eigen::Matrix3d wheel_meas_cov, wheel_cov_world;  // 外参
  Eigen::Matrix3d R_ex_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p_ex_ = Eigen::Vector3d::Zero();
  // 增广外参协方差与交叉协方差（本类维护）
  Eigen::Matrix<double, 6, 6> P_ext_ext{
      Eigen::Matrix<double, 6, 6>::Zero()};  // [rex, pex]
  Eigen::Matrix<double, DIM_STATE, 6> P_x_ext_{
      Eigen::Matrix<double, DIM_STATE, 6>::Zero()};
  bool ex_cov_inited_{false};
  Eigen::Matrix3d skew(const Eigen::Vector3d& v) const {
    Eigen::Matrix3d S;
    S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return S;
  }
};
