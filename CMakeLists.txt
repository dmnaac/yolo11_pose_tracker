cmake_minimum_required(VERSION 3.8)
project(yolo11_pose_tracker)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# ---- Custom messages ----
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Keypoint.msg"
  "msg/PersonPose.msg"
  "msg/PersonPoseArray.msg"
  DEPENDENCIES std_msgs
)

# ---- Python node script ----
# NOTE: do NOT call ament_python_install_package(${PROJECT_NAME}) here.
# rosidl_generate_interfaces already registers the python module named
# ${PROJECT_NAME}; a second registration duplicates the
# ament_cmake_python_copy_${PROJECT_NAME} target and fails the configure.
install(PROGRAMS
  scripts/pose_tracker_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  model
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
