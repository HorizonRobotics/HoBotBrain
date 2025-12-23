#!/bin/bash

SESSION_NAME="robot_nav"

# run on hostmachine
# 删除旧 tmux 会话
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux kill-session -t $SESSION_NAME
    echo "Session '$SESSION_NAME' has been deleted."
fi

# 创建新的 tmux 会话
tmux new-session -d -s $SESSION_NAME -n nav

# -------------------
# Pane 0: 语音交互
# -------------------
tmux send-keys -t $SESSION_NAME:0 "1" C-m
tmux send-keys -t $SESSION_NAME:0 "source /home/unitree/agentic_robot_system/inside_docker_install/G1NavCode-Ros2Version/sem_nav/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0 "unset ASAN_OPTIONS" C-m
tmux send-keys -t $SESSION_NAME:0 "ros2 run chat_loc_python topic_chat_loc_pub" C-m

# -------------------
# Pane 1: 语义定位
# -------------------
tmux split-window -h -t $SESSION_NAME:0
tmux send-keys -t $SESSION_NAME:0.1 "1" C-m
tmux send-keys -t $SESSION_NAME:0.1 "source /home/unitree/agentic_robot_system/inside_docker_install/G1NavCode-Ros2Version/sem_nav/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0.1 "unset ASAN_OPTIONS" C-m
tmux send-keys -t $SESSION_NAME:0.1 "ros2 run goal_publisher goal_pose_publisher" C-m

# -------------------
# Pane 2: 管道写入 (g1_getvel_node)，读取Navigation发布速度写入管道
# -------------------
tmux split-window -v -t $SESSION_NAME:0
tmux send-keys -t $SESSION_NAME:0.2 "1" C-m
tmux send-keys -t $SESSION_NAME:0.2 "[ -p /tmp/vel_fifo ] && rm /tmp/vel_fifo" C-m
tmux send-keys -t $SESSION_NAME:0.2 "mkfifo /tmp/vel_fifo" C-m
tmux send-keys -t $SESSION_NAME:0.2 "source /home/unitree/agentic_robot_system/inside_docker_install/G1NavCode-Ros2Version/sem_nav/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0.2 "unset ASAN_OPTIONS" C-m
tmux send-keys -t $SESSION_NAME:0.2 "ros2 run g1_move g1_getvel_node" C-m

# -------------------
# Pane 3: 管道读取 (g1_pubvel_node) 控制运动
# -------------------
tmux split-window -v -t $SESSION_NAME:0
tmux send-keys -t $SESSION_NAME:0.3 "1" C-m
tmux send-keys -t $SESSION_NAME:0.3 "source /home/unitree/agentic_robot_system/inside_docker_install/G1NavCode-Ros2Version/sem_nav/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0.3 "unset ASAN_OPTIONS" C-m
tmux send-keys -t $SESSION_NAME:0.3 "ros2 run g1_move g1_pubvel_node" C-m

# 调整布局，让四个 pane 都可见
tmux select-layout -t $SESSION_NAME:0 tiled

# 附加到 tmux 会话
tmux attach-session -t $SESSION_NAME
