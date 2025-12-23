# cmake_minimum_required(VERSION 3.10)
cmake_minimum_required(VERSION 3.12)  # 确保 CMake 版本支持 CMP0074
cmake_policy(SET CMP0074 NEW)         # 启用 CMP0074 策略
project(fast_livo)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")
# set(CMAKE_BUILD_TYPE "Debug")
# add_compile_options(-g -O0)            # 禁止过度优化
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

IF("$ENV{ROS_DISTRO}" STRLESS "iron")
  add_definitions(-DPRE_ROS_IRON)
ENDIF()

# Set common compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
# Specific settings for Debug build
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

# Detect CPU architecture
message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")

# Specific settings for Release build
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 64-bit ARM optimizations (e.g., RK3588 and Jetson Orin NX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -ffast-math")
    message(STATUS "Using 64-bit ARM optimizations: -O3 -mcpu=native -mtune=native -ffast-math")
  else()
    # 32-bit ARM optimizations with NEON support
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
    message(STATUS "Using 32-bit ARM optimizations: -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
  endif()
  add_definitions(-DARM_ARCH)
else()
  # x86-64 (Intel/AMD) optimizations
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops") #-flto
  message(STATUS "Using general x86 optimizations: -O3 -march=native -mtune=native -funroll-loops") 
  add_definitions(-DX86_ARCH)
endif()

# Define project root directory
# add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
# Set ROOT_DIR variable to the root directory
add_definitions(-DROOT_DIR="/")

# Detect CPU core count for potential multithreading optimization
include(ProcessorCount)
ProcessorCount(N)
message(STATUS "Processor count: ${N}")

# Set the number of cores for multithreading
if(N GREATER 8)
  math(EXPR PROC_NUM "8")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
  message(STATUS "Multithreading enabled. Cores capped at ${PROC_NUM}")
elseif(N GREATER 1)
  math(EXPR PROC_NUM "${N}")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
  message(STATUS "Multithreading enabled. Cores: ${PROC_NUM}")
else()
  add_definitions(-DMP_PROC_NUM=1)
  message(STATUS "Single core detected. Multithreading disabled.")
endif()

# Check for OpenMP support
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
  add_compile_options(${OpenMP_CXX_FLAGS})
else()
  message(STATUS "OpenMP not found, proceeding without it")
endif()

# Check for mimalloc support
find_package(mimalloc QUIET)
if(mimalloc_FOUND)
  message(STATUS "mimalloc found")
else()
  message(STATUS "mimalloc not found, proceeding without it")
endif()

# Find ament and required dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
# find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
# find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(vikit_common REQUIRED)
find_package(vikit_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(GTSAM REQUIRED)
message(STATUS "GTSAM version: ${GTSAM_VERSION}")
find_package(PythonLibs REQUIRED)
# eigen_conversions
# find_package(eigen_conversions REQUIRED)
# Include directories for dependencies
include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${vikit_common_INCLUDE_DIRS}
  ${vikit_ros_INCLUDE_DIRS}
  include
)
include_directories(${PROJECT_SOURCE_DIR}/include/FRICP-toolkit)
include_directories(${CMAKE_BINARY_DIR}/rosidl_generator_cpp)
include_directories(
  include/fast_livo
)
rosidl_generate_interfaces(${PROJECT_NAME} "srv/SaveMap.srv" DEPENDENCIES std_msgs sensor_msgs)
set(dependencies
  rclcpp
  rclpy
  geometry_msgs
  nav_msgs
  std_msgs
  sensor_msgs
  visualization_msgs
  cv_bridge
  vikit_common
  vikit_ros
  image_transport
  # pcl_ros
  pcl_msgs
  pcl_conversions
  tf2_ros
  livox_ros_driver2
)

set(COMMON_DEPENDENCIES OpenMP::OpenMP_CXX)

# link_directories(${COMMON_DEPENDENCIES}
#   ${vikit_common_LIBRARIES}/libvikit_common.so
#   ${vikit_ros_LIBRARIES}/libvikit_ros.so
# )

# Add libraries
add_library(vio src/vio.cpp src/frame.cpp src/visual_point.cpp)
add_library(lio src/voxel_map.cpp)
add_library(pre src/preprocess.cpp)
add_library(imu_proc src/IMU_Processing.cpp)
add_library(laser_mapping src/LIVMapper.cpp include/sc-relo/Scancontext.cpp src/ground_detection.cpp src/zupt.cpp)
add_library(utils src/utils.cpp)
add_library(incremental_mapping include/multi-session/Incremental_mapping.cpp include/sc-relo/Scancontext.cpp)
add_library(pose_estimator include/online-relo/pose_estimator.cpp include/sc-relo/Scancontext.cpp)
# add_library(sc include/sc-relo/Scancontext.cpp)
ament_target_dependencies(vio ${dependencies} )
ament_target_dependencies(lio ${dependencies})
ament_target_dependencies(pre ${dependencies})
ament_target_dependencies(imu_proc ${dependencies})
ament_target_dependencies(laser_mapping ${dependencies})
ament_target_dependencies(incremental_mapping ${dependencies})
ament_target_dependencies(pose_estimator ${dependencies})
# linking libraries or executables to public dependencies
target_link_libraries(laser_mapping 
    ${CMAKE_SOURCE_DIR}/../../install/vikit_common/lib/libvikit_common.so 
    ${CMAKE_SOURCE_DIR}/../../install/vikit_ros/lib/libvikit_ros.so
    ${COMMON_DEPENDENCIES}
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(vio ${COMMON_DEPENDENCIES} "${cpp_typesupport_target}")
target_link_libraries(lio utils ${COMMON_DEPENDENCIES} "${cpp_typesupport_target}")
target_link_libraries(pre ${COMMON_DEPENDENCIES} "${cpp_typesupport_target}")
target_link_libraries(imu_proc ${COMMON_DEPENDENCIES} "${cpp_typesupport_target}")

add_executable(multi_session src/multi_session.cpp)
# ament_target_dependencies(multi_session ${dependencies})
target_link_libraries(multi_session ${COMMON_DEPENDENCIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} incremental_mapping gtsam stdc++fs "${cpp_typesupport_target}")

add_executable(online_relo src/online_relocalization.cpp)
# ament_target_dependencies(online_relo ${dependencies})
target_link_libraries(online_relo ${COMMON_DEPENDENCIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES}  ${OpenCV_LIBRARIES} pose_estimator incremental_mapping gtsam stdc++fs "${cpp_typesupport_target}")

# Add the main executable
add_executable(fastlivo_mapping src/main.cpp)

ament_target_dependencies(fastlivo_mapping ${dependencies})
# Link libraries to the executable
target_link_libraries(fastlivo_mapping
  laser_mapping
  vio
  lio
  pre
  imu_proc
  gtsam
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
  "${cpp_typesupport_target}"
  # ${GTSAM_LIBRARIES}  # 添加 GTSAM 库
)

# Link mimalloc if found
if(mimalloc_FOUND)
  target_link_libraries(fastlivo_mapping mimalloc)
endif()

# Install the executable
install(TARGETS
  fastlivo_mapping
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  online_relo
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  multi_session
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY config launch rviz_cfg
  DESTINATION share/${PROJECT_NAME}
)
install(
  DIRECTORY "include/"
  DESTINATION include
)
# Install the lib
install(TARGETS vio
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS lio
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS pre
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS imu_proc
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS laser_mapping
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS utils
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS incremental_mapping
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS pose_estimator
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_include_directories(include)
# Export dependencies
ament_export_dependencies(
  rclcpp
  rclpy
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  pcl_ros
  pcl_conversions
  tf2_ros
  livox_ros_driver2
  Eigen3
  PCL
  OpenCV 
  Sophus
  GTSAM
  rosidl_default_generators
)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()
ament_package()