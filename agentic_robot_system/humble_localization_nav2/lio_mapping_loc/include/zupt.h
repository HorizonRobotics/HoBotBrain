#include <Eigen/Dense>
#include <iostream>

#include "common_lib.h"

class ZUPT {
 public:
  ZUPT() {
    // ground_height = 0.0; // 默认地面高度
    measurement_noise_ = 0.05;  // 默认观测噪声
    z_measurement_ = 0.0;       // 默认地面高度
  }

  void setMeasurement(double measurement_noise, double z_measurement) {
    measurement_noise_ = measurement_noise;
    z_measurement_ = z_measurement;
  }
  void setState(const StatesGroup &state) { state_ = state; }
  void applyZConstraint();
  StatesGroup state_;
  double measurement_noise_;  // 观测噪声
  double z_measurement_;
};
