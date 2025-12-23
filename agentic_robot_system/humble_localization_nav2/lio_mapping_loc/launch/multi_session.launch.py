from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('fast_lio_sam')
    
    # 声明参数
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to start RViz'
    )

    # 加载参数文件
    param_file = os.path.join(pkg_dir, 'config', 'multi_session.yaml')

    # 主节点
    multi_session_node = Node(
        package='fast_lio_sam',
        executable='multi_session',
        name='multi_session',
        output='screen',
        parameters=[param_file]
    )

    # RViz节点
    rviz_node = Node(
        condition=LaunchConfiguration('rviz'),
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', os.path.join(pkg_dir, 'rviz_cfg', 'sc_relo.rviz')],
        prefix=['nice']
    )

    return LaunchDescription([
        rviz_arg,
        multi_session_node,
        rviz_node
    ])