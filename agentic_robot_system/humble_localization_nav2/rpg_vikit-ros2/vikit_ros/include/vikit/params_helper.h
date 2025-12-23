/*
 * ros_params_helper.h
 *
 *  Created on: Feb 22, 2013
 *      Author: cforster
 *  Update on: Feb 08, 2025
 *      Author: StrangeFly
 *
 * from libpointmatcher_ros
 */

#ifndef ROS_PARAMS_HELPER_H_
#define ROS_PARAMS_HELPER_H_

#include <string>
#include <rclcpp/rclcpp.hpp>

namespace vk {

inline
bool hasParam(const rclcpp::Node::SharedPtr &nh, const std::string& name)
{
  return nh->has_parameter(name);
}

/* template<typename T>
T getParam(const rclcpp::Node::SharedPtr &nh, const std::string& name, const T& defaultValue)
{
  T v;
  if(nh->has_parameter(name))
  {
    v = nh->get_parameter(name).get_value<T>();
    RCLCPP_INFO(nh->get_logger(), "Found parameter: %s, value: %s", name.c_str(), std::to_string(v).c_str());
    return v;
  }
  else
    RCLCPP_WARN(nh->get_logger(), "Cannot find value for parameter: %s, assigning default: %s", name.c_str(), std::to_string(defaultValue).c_str());
	  nh->declare_parameter(name, defaultValue);
  return defaultValue;
} */

template<typename T>
T getParam(const rclcpp::Node::SharedPtr &nh, const std::string& name, const T& defaultValue)
{
  T v;
  if(nh->has_parameter(name))
  {
    v = nh->get_parameter(name).get_value<T>();
    RCLCPP_INFO(nh->get_logger(), "Found parameter: %s, value: %s", name.c_str(), std::to_string(v).c_str());
    return v;
  }
  else
  {
    RCLCPP_WARN(nh->get_logger(), "Cannot find value for parameter: %s, assigning default: %s", name.c_str(), std::to_string(defaultValue).c_str());
    nh->declare_parameter(name, defaultValue);
  }
  return defaultValue;
}
  
template<>
inline std::string getParam<std::string>(const rclcpp::Node::SharedPtr &nh, const std::string& name, const std::string& defaultValue)
{
  std::string v;
  if(nh->has_parameter(name))
  {
    v = nh->get_parameter(name).get_value<std::string>();
    RCLCPP_INFO(nh->get_logger(), "Found parameter: %s, value: %s", name.c_str(), v.c_str());
    return v;
  }
  else
  {
    RCLCPP_WARN(nh->get_logger(), "Cannot find value for parameter: %s, assigning default: %s", name.c_str(), defaultValue.c_str());
    nh->declare_parameter(name, defaultValue);
  }
  return defaultValue;
}

/* template<typename T>
T getParam(const rclcpp::Node::SharedPtr &nh, const std::string& name)
{
  T v;
  int i = 0;
  while(!nh->get_parameter(name, v))
  {
    RCLCPP_ERROR(nh->get_logger(), "Cannot find value for parameter: %s, will try again.", name.c_str());
    if ((i ++) >= 5) return T();
  }
  
  RCLCPP_INFO(nh->get_logger(), "Found parameter: %s, value: %s", name.c_str(), std::to_string(v).c_str());
  return v;
} */

	 
template<typename T>
typename std::enable_if<!std::is_same<T, std::string>::value, T>::type
getParam(const rclcpp::Node::SharedPtr &nh, const std::string& name)
{
  T v;
  int i = 0;
  while(!nh->get_parameter(name, v))
  {
    RCLCPP_ERROR(nh->get_logger(), "Cannot find value for parameter: %s, will try again.", name.c_str());
    if ((i++) >= 5) return T();
  }
  
  RCLCPP_INFO(nh->get_logger(), "Found parameter: %s, value: %s", name.c_str(), std::to_string(v).c_str());
  return v;
}
  
template<typename T>
typename std::enable_if<std::is_same<T, std::string>::value, T>::type
getParam(const rclcpp::Node::SharedPtr &nh, const std::string& name)
{
  T v;
  int i = 0;
  while(!nh->get_parameter(name, v))
  {
    RCLCPP_ERROR(nh->get_logger(), "Cannot find value for parameter: %s, will try again.", name.c_str());
    if ((i++) >= 5) return T();
  }
  
  RCLCPP_INFO(nh->get_logger(), "Found parameter: %s, value: %s", name.c_str(), v.c_str());
  return v;
}

} // namespace vk

#endif // ROS_PARAMS_HELPER_H_
