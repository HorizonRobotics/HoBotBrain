#!/bin/bash

# run on hostmachine
SESSION_NAME="robot_sensors"
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux kill-session -t $SESSION_NAME
    echo "Session '$SESSION_NAME' has been deleted."
fi

# 创建新的 tmux 会话，不附加 (-d)
tmux new-session -d -s $SESSION_NAME -n camera_tab

# # --- Window 1: Camera ---
tmux send-keys -t $SESSION_NAME:camera_tab "1" C-m
# tmux send-keys -t $SESSION_NAME:camera_tab "ros2 launch realsense2_camera rs_new.py" C-m

# --- Window 2: IMU ---
tmux new-window -t $SESSION_NAME -n imu_tab
tmux send-keys -t $SESSION_NAME:imu_tab "1" C-m
tmux send-keys -t $SESSION_NAME:imu_tab "cd ~/code_vln/ros2_ws" C-m
tmux send-keys -t $SESSION_NAME:imu_tab "unset ASAN_OPTIONS" C-m
tmux send-keys -t $SESSION_NAME:imu_tab "source install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:imu_tab "ros2 run imu_publisher imu_extractor" C-m

# --- Window 3: LiDAR ---
tmux new-window -t $SESSION_NAME -n lidar_tab
tmux send-keys -t $SESSION_NAME:lidar_tab "1" C-m
tmux send-keys -t $SESSION_NAME:lidar_tab "unset ASAN_OPTIONS" C-m
tmux send-keys -t $SESSION_NAME:lidar_tab "ros2 launch livox_ros_driver2 msg_MID360_launch.py" C-m

# 附加到 tmux 会话
tmux attach-session -t $SESSION_NAME
