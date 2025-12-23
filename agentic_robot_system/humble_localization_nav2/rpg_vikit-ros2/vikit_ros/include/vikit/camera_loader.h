/*
 * camera_loader.h
 *
 *  Created on: Feb 11, 2014
 *      Author: cforster
 *  Update on: Feb 01, 2025
 *      Author: StrangeFly
 */

#ifndef VIKIT_CAMERA_LOADER_H_
#define VIKIT_CAMERA_LOADER_H_

#include <string>
#include <vikit/abstract_camera.h>
#include <vikit/pinhole_camera.h>
#include <vikit/atan_camera.h>
#include <vikit/omni_camera.h>
#include <vikit/equidistant_camera.h>
#include <vikit/polynomial_camera.h>
#include <vikit/params_helper.h>

namespace vk {
namespace camera_loader {

/// Load from ROS Namespace
inline bool loadFromRosNs(const rclcpp::Node::SharedPtr & nh, const std::string& ns, vk::AbstractCamera*& cam)
{
  auto ns_ = ns;
  if (ns_ != "") {
    ns_ = ns_ + ".";
  }
  bool res = true;
  std::string cam_model(getParam<std::string>(nh, ns_+"model"));
  if(cam_model == "Ocam")
  {
    cam = new vk::OmniCamera(getParam<std::string>(nh, ns_+"calib_file", ""));
  }
  else if(cam_model == "Pinhole")
  {
    cam = new vk::PinholeCamera(
        getParam<int>(nh, ns_+"width"),
        getParam<int>(nh, ns_+"height"),
        getParam<double>(nh, ns_+"scale", 1.0),
        getParam<double>(nh, ns_+"fx"),
        getParam<double>(nh, ns_+"fy"),
        getParam<double>(nh, ns_+"cx"),
        getParam<double>(nh, ns_+"cy"),
        getParam<double>(nh, ns_+"d0", 0.0),
        getParam<double>(nh, ns_+"d1", 0.0),
        getParam<double>(nh, ns_+"d2", 0.0),
        getParam<double>(nh, ns_+"d3", 0.0));
  }
  else if(cam_model == "EquidistantCamera")
  {
    cam = new vk::EquidistantCamera(
        getParam<int>(nh, ns_+"width"),
        getParam<int>(nh, ns_+"height"),
        getParam<double>(nh, ns_+"scale", 1.0),
        getParam<double>(nh, ns_+"fx"),
        getParam<double>(nh, ns_+"fy"),
        getParam<double>(nh, ns_+"cx"),
        getParam<double>(nh, ns_+"cy"),
        getParam<double>(nh, ns_+"k1", 0.0),
        getParam<double>(nh, ns_+"k2", 0.0),
        getParam<double>(nh, ns_+"k3", 0.0),
        getParam<double>(nh, ns_+"k4", 0.0));
  }
  else if(cam_model == "PolynomialCamera")
  {
    cam = new vk::PolynomialCamera(
        getParam<int>(nh, ns_+"width"),
        getParam<int>(nh, ns_+"height"),
        // getParam<double>(nh, ns_+"scale", 1.0),
        getParam<double>(nh, ns_+"fx"),
        getParam<double>(nh, ns_+"fy"),
        getParam<double>(nh, ns_+"cx"),
        getParam<double>(nh, ns_+"cy"),
        getParam<double>(nh, ns_+"skew"),
        getParam<double>(nh, ns_+"k2", 0.0),
        getParam<double>(nh, ns_+"k3", 0.0),
        getParam<double>(nh, ns_+"k4", 0.0),
        getParam<double>(nh, ns_+"k5", 0.0),
        getParam<double>(nh, ns_+"k6", 0.0),
        getParam<double>(nh, ns_+"k7", 0.0));
  }
  else if(cam_model == "ATAN")
  {
    cam = new vk::ATANCamera(
        getParam<int>(nh, ns_+"width"),
        getParam<int>(nh, ns_+"height"),
        getParam<double>(nh, ns_+"fx"),
        getParam<double>(nh, ns_+"fy"),
        getParam<double>(nh, ns_+"cx"),
        getParam<double>(nh, ns_+"cy"),
        getParam<double>(nh, ns_+"d0"));
  }
  else
  {
    cam = NULL;
    res = false;
  }
  return res;
}

inline bool loadFromRosNs(const rclcpp::Node::SharedPtr & nh, const std::string& ns, std::vector<vk::AbstractCamera*>& cam_list)
{
  auto ns_ = ns;
  if (ns_ != "") {
    ns_ = ns_ + ".";
  }
  bool res = true;
  std::string cam_model(getParam<std::string>(nh, ns+"model"));
  int cam_num = getParam<int>(nh, ns+"num");
  for (int i = 0; i < cam_num; i ++)
  {
    std::string cam_ns = ns_ + "cam_" + std::to_string(i);
    std::string cam_model(getParam<std::string>(nh, cam_ns+".model"));
    if(cam_model == "FishPoly")
    {
      cam_list.push_back(new vk::PolynomialCamera(
        getParam<int>(nh, cam_ns+"width"),
        getParam<int>(nh, cam_ns+"height"),
        // getParam<double>(nh, cam_ns+"/scale", 1.0),
        getParam<double>(nh, cam_ns+"A11"),  // cam_fx
        getParam<double>(nh, cam_ns+"A22"),  // cam_fy
        getParam<double>(nh, cam_ns+"u0"),  // cam_cx
        getParam<double>(nh, cam_ns+"v0"),  // cam_cy
        getParam<double>(nh, cam_ns+"A12"), // cam_skew
        getParam<double>(nh, cam_ns+"k2", 0.0),
        getParam<double>(nh, cam_ns+"k3", 0.0),
        getParam<double>(nh, cam_ns+"k4", 0.0),
        getParam<double>(nh, cam_ns+"k5", 0.0),
        getParam<double>(nh, cam_ns+"k6", 0.0),
        getParam<double>(nh, cam_ns+"k7", 0.0)));
    }
    else if(cam_model == "Pinhole")
    {
      cam_list.push_back(new vk::PinholeCamera(
          getParam<int>(nh, ns+"width"),
          getParam<int>(nh, ns+"height"),
          getParam<double>(nh, ns+"scale", 1.0),
          getParam<double>(nh, ns+"fx"),
          getParam<double>(nh, ns+"fy"),
          getParam<double>(nh, ns+"cx"),
          getParam<double>(nh, ns+"cy"),
          getParam<double>(nh, ns+"d0", 0.0),
          getParam<double>(nh, ns+"d1", 0.0),
          getParam<double>(nh, ns+"d2", 0.0),
          getParam<double>(nh, ns+"d3", 0.0)));
    }
    else 
    {
      // cam_list.clear();
      res = false;
    }
  }
  
  return res;
}

} // namespace camera_loader
} // namespace vk

#endif // VIKIT_CAMERA_LOADER_H_
