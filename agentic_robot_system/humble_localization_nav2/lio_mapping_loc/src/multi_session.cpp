#include <algorithm>
#include <rclcpp/rclcpp.hpp>

#include "multi-session/Incremental_mapping.hpp"
#include "online-relo/pose_estimator.h"  // 若有 publishCloud 等工具函数

bool keyFrameSort(const std::pair<int, KeyFrame>& frame1,
                  const std::pair<int, KeyFrame>& frame2) {
  return frame1.first < frame2.first;
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("multi_session_node");

  std::string sessions_dir;
  std::string central_sess_name;
  std::string query_sess_name;
  std::string save_directory;
  int iteration;

  node->declare_parameter<std::string>("multi_session/sessions_dir", " ");
  node->declare_parameter<std::string>("multi_session/central_sess_name", " ");
  node->declare_parameter<std::string>("multi_session/query_sess_name", " ");
  node->declare_parameter<std::string>("multi_session/save_directory", " ");
  node->declare_parameter<int>("multi_session/iteration", 5);

  node->get_parameter("multi_session/sessions_dir", sessions_dir);
  node->get_parameter("multi_session/central_sess_name", central_sess_name);
  node->get_parameter("multi_session/query_sess_name", query_sess_name);
  node->get_parameter("multi_session/save_directory", save_directory);
  node->get_parameter("multi_session/iteration", iteration);

  RCLCPP_INFO(node->get_logger(), "----> multi-session starts.");

  MultiSession::IncreMapping multi_session(node,sessions_dir, central_sess_name,
                                           query_sess_name, save_directory);

  RCLCPP_INFO(node->get_logger(), "----> pose-graph optimization.");
  multi_session.run(iteration);

  RCLCPP_INFO(node->get_logger(), "----> publish cloud.");
  std::sort(multi_session.reloKeyFrames.begin(),
            multi_session.reloKeyFrames.end(), keyFrameSort);

  int total = multi_session.reloKeyFrames.size();
  int i = 0;
  rclcpp::Rate rate(0.5);

  while (rclcpp::ok() && i < total) {
    rclcpp::spin_some(node);

    publishCloud(multi_session.pubCentralGlobalMap, multi_session.centralMap_,
                 node->now(), "camera_init");
    publishCloud(multi_session.pubCentralTrajectory, multi_session.traj_central,
                 node->now(), "camera_init");
    publishCloud(multi_session.pubRegisteredTrajectory,
                 multi_session.traj_regis, node->now(), "camera_init");
    multi_session.visualizeLoopClosure();
    publishCloud(multi_session.pubReloCloud,
                 multi_session.reloKeyFrames[i].second.all_cloud, node->now(),
                 "camera_init");

    std::cout << "relo name(Idx): " << multi_session.reloKeyFrames[i].first
              << " target name(Idx): "
              << multi_session.reloKeyFrames[i].second.reloTargetIdx
              << " score: " << multi_session.reloKeyFrames[i].second.reloScore
              << std::endl;

    i++;
    rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}
