from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # 包含第一个Launch文件
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('fast_livo'),
                    'launch',
                    'online_relo.launch.py'
                ])
            ]),
            # 可以传递参数给被包含的Launch文件
            #launch_arguments={
            #    'arg_name': 'value',
            #    'another_arg': 'value2'
            #}.items()
        ),

        # 包含第二个Launch文件
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('fast_livo'),
                    'launch',
                    'mapping_g1.launch.py'
                ])
            ])
        ),
        
        # 包含第3个Launch文件
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('g1_navigation2'),
                    'launch',
                    'navigation2.launch.py'
                ])
            ])
        ),

        # 启动独立节点1
        #Node(
        #    package='tfpub',
        #    executable='tfpub',
        #    name='tfpub',
        #    namespace='tfpub',
        #    #parameters=[{'param_name': 'param_value'}],
        #    #remappings=[('topic1', 'topic2')]
        #)
    ])
