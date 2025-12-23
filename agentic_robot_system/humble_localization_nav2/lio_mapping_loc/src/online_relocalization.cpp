#include <thread>

#include "online-relo/pose_estimator.h"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  // ROS2 does not require AsyncSpinner; multithreading is handled internally.
  auto nh = std::make_shared<rclcpp::Node>("pose_estimator_node");
  auto pose_estimator_node = std::make_shared<pose_estimator>(
      nh);  // 假设 pose_estimator 继承自 rclcpp::Node

  RCLCPP_INFO(nh->get_logger(),
              "\033[1;32m----> Online Relocalization Started.\033[0m");
  // std::thread opt_thread(&pose_estimator::run, node.get());
  // std::thread pub_thread(&pose_estimator::publishThread, node.get());
  pose_estimator_node->run(nh);
  // pose_estimator_node->publish_cloud(nh);

  // rclcpp::executors::MultiThreadedExecutor executor;
  // executor.add_node(nh);
  // pose_estimator_node->start_localization();
  // std::thread opt_thread(&pose_estimator::run, pose_estimator_node);
  // std::thread pub_thread(&pose_estimator::publishThread,
  // pose_estimator_node);

  rclcpp::spin(nh);  // 使用 rclcpp::spin 来处理回调
  rclcpp::shutdown();
  return 0;
}
