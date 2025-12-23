from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.conditions import IfCondition
def generate_launch_description():
    pkg_dir = get_package_share_directory('fast_livo')
    
    # 声明参数
    rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Whether to start RViz'
    )

    # 加载参数文件
    param_file = os.path.join(pkg_dir, 'config', 'mid360_online_reloc.yaml')

    # 主节点
    online_relo_node = Node(
        package='fast_livo',
        executable='online_relo',
        name='online_relo',
        output='screen',
        parameters=[param_file]
    )

    # RViz节点
    rviz_node = Node(
        # condition=LaunchConfiguration('rviz'),
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_dir, 'rviz_cfg', 'loc_new.rviz')],
        prefix=['nice']
    )

    return LaunchDescription([
        rviz_arg,
        online_relo_node,
        rviz_node
    ])