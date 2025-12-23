cmake_minimum_required(VERSION 3.5)
project(vikit_ros)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(vikit_common REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
set(dependencies
  Eigen3
  rclcpp
  tf2_ros
  tf2_geometry_msgs
  vikit_common
  visualization_msgs
)

ament_export_dependencies(${dependencies})

# Link libraries
list(APPEND SOURCEFILES 
  src/output_helper.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCEFILES})
# Modify target_include_directories to include the project directory directly
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  rclcpp::rclcpp
  tf2_ros::tf2_ros
  vikit_common::vikit_common
  ${tf2_geometry_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
)

ament_target_dependencies(${PROJECT_NAME} ${dependencies})

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# Install library
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Export the include directory for the project
set_target_properties(${PROJECT_NAME} PROPERTIES
  PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Ensure the include directory is correctly set for downstream packages
ament_export_include_directories(include/${PROJECT_NAME})
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${dependencies})
ament_export_targets(${PROJECT_NAME})

ament_package()
