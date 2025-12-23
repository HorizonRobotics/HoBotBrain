# Agentic Robot System 

本项目包含一个完整的 **具身智能导航系统**（Agentic Robot System），包含定位、建图、导航、语义目标定位、语音交互、运控调用等模块，支持 Docker 与宿主机混合运行。

---

## 📁 项目结构
```text
agentic_robot_system/
├── humble_localization_nav2/        # Docker 内运行的本体定位 / 导航 & 避障模块
│   ├── g1_nav_bringup/              # 一键启动所有 launch 文件
│   ├── g1_navigation2/              # ROS 2 Navigation 参数接口
│   ├── lio_mapping_loc/             # FastLIVO2 + 重定位模块
│   ├── navigation2-humble/          # ROS 2 Navigation 2 核心功能包
│   ├── pubpose/                      # 接收 goal_publisher 目标位姿并转发给 Nav2 做全局导航与避障
│   └── rpg_vikit-ros2/              # FastLIVO2 第三方依赖
├── scripts/                         # 启动脚本（Docker / 宿主机）
│   ├── run_nav.sh                    # Docker 内一键启动所有算法模块
│   ├── run_sem_nav.sh                # 宿主机一键启动语义导航模块
│   └── run_sensors.sh                # 宿主机一键启动传感器
└── sem_nav_ctr/                      # 宿主机运行的语音 / 运控 / 目标语义定位模块
    ├── chat_loc_python/             # 语音交互客户端
    ├── g1_move/                     # G1 运控接口
    └── goal_publisher/              # 目标实例 / 区域的语义定位
``` 

## 🚀 功能概述

- ✅ **导航和避障（Nav2）**  
  基于 ROS 2 Navigation2 框架，支持全局路径规划、局部避障。

- ✅ **FastLIVO2 里程计 + 重定位**  
  实时点云里程计，并支持地图重定位。

- ✅ **语音交互控制**  
  通过本地语音客户端配合远程服务端实现语音导航任务，代码中仅包含客户端部分，建议自行实现语音交互模块, 或等下一步开源。

- ✅ **语义目标定位**  
  从目标名称（如"沙发"、"展厅"）解析为具体的三维空间目标位姿。

- ✅ **一键启动脚本**  
  提供 Docker / 宿主机 的一键启动方案。

---

## 🏃 启动方式

### Docker 构建与运行

确保已安装 Docker 与 NVIDIA Container Toolkit。

**基础镜像配置：**
- 自行构建 `ubuntu22.04 + ros2-humble` 基础镜像
- 在基础镜像中colcon build `humble_localization_nav2` 中的所有子模块

### 启动命令

**Docker 内启动导航模块：**
```bash
bash scripts/run_nav.sh
```

**宿主机启动语义导航模块：**
```bash
bash scripts/run_sem_nav.sh
```

**宿主机启动传感器：**
```bash
bash scripts/run_sensors.sh