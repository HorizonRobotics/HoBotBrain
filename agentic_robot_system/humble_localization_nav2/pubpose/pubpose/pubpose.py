#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from std_msgs.msg import String
import tf_transformations as transform
import math

class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')
        print('waiting for start...')
        # 订阅启动导航的主题
        self.multipoint_navigation_sub = self.create_subscription(
            String,
            'chat_signal_pub',
            self.multi_navigation_callback,
            10
        )
        
        self.onepoint_navigation_sub = self.create_subscription(
            PoseStamped,
            'object_pose',
            self.one_navigation_callback,
            10
        )
        

        # 创建到达航点的发布者
        self.waypoint_reached_pub = self.create_publisher(String, 'waypoint_reached', 10)

        # 初始化导航器
        self.navigator = BasicNavigator()

        # 定义航点列表 [x, y, orientation_z, orientation_w, waypoint_name]
        self.waypoints = [
            [-109.52, -54.59, "Waypoint_1"],
            [-114.62, -54.56, "Waypoint_2"],
            [-117.42, -58.44, "Waypoint_3"]
        ]

        self.current_waypoint_index = 0
        self.total_waypoints = len(self.waypoints)
        self.navigation_active = False
        #self.nav2_ready = False

        # 等待Nav2完全启动
        #self.wait_for_nav2_timer = self.create_timer(1.0, self.wait_for_nav2)
    '''
    def wait_for_nav2(self):
        """等待Nav2系统准备就绪"""
        if not self.nav2_ready:
            try:
                self.navigator.waitUntilNav2Active()
                self.get_logger().info("Nav2 is ready!")
                self.nav2_ready = True
                self.destroy_timer(self.wait_for_nav2_timer)
            except Exception as e:
                self.get_logger().warn(f"Waiting for Nav2 to become active: {e}")
    '''
    def one_navigation_callback(self, msg):
        self.get_logger().info(f"Navigating to target...")
        self.navigator.goToPose(msg)
    
    def multi_navigation_callback(self, msg):
        print(msg.data)
        """接收到启动导航命令时的回调函数"""
        #if not self.nav2_ready:
        #    self.get_logger().warn("Nav2 is not ready yet. Ignoring navigation command.")
        #    return
        if msg.data == 'stop':
            print('stop...')
            self.navigator.cancelTask()
            
            return
        if self.navigation_active:
            self.get_logger().info("Navigation is already active. Ignoring duplicate command.")
            return

        if msg.data == 'horizon':
            print('start...')
            self.current_waypoint_index = 0

        if self.current_waypoint_index >= self.total_waypoints:
            self.get_logger().info("All waypoints completed!")
            return

        self.get_logger().info(f"Received navigation command: {msg.data}")

        # 开始导航到第一个航点
        self.navigate_to_next_waypoint()
        
        if self.current_waypoint_index == 0:
            # 创建定时器检查导航状态
            self.navigation_timer = self.create_timer(0.5, self.check_navigation_status)

    def create_pose_stamped(self, position_x, position_y):
        """创建一个PoseStamped消息"""
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        goal_pose.pose.position.x = position_x
        goal_pose.pose.position.y = position_y
        return goal_pose

    def publish_waypoint_reached(self, waypoint_name):
        """发布航点到达消息"""
        msg = String()
        msg.data = f"{waypoint_name}"
        self.waypoint_reached_pub.publish(msg)
        self.get_logger().info(f"Published waypoint reached: {waypoint_name}")

    def navigate_to_next_waypoint(self):
        """导航到下一个航点"""
        self.navigation_active = True
        # 获取当前航点
        wp = self.waypoints[self.current_waypoint_index]
        goal_pose = self.create_pose_stamped(wp[0], wp[1])

        self.get_logger().info(f"Navigating to {wp[2]}...")
        self.navigator.goToPose(goal_pose)

        return True

    def check_navigation_status(self):
        """检查导航状态"""
        if not self.navigation_active:
            return

        if not self.navigator.isTaskComplete():
            # 任务仍在进行中
            feedback = self.navigator.getFeedback()
            # 可以在这里添加更多反馈处理逻辑
            return

        # 任务已完成，检查结果
        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            # 发布航点到达消息
            wp_name = self.waypoints[self.current_waypoint_index][2]
            self.publish_waypoint_reached(wp_name)

            self.current_waypoint_index += 1
            self.get_logger().info(f"Successfully reached {wp_name}")
            self.navigation_active = False
            
            if self.current_waypoint_index >= self.total_waypoints:
                self.get_logger().info("All waypoints completed! Destory timer!")
                self.destroy_timer(self.navigation_timer)
            '''
            # 导航到下一个航点
            if not self.navigate_to_next_waypoint():
                # 所有航点已完成
                self.navigation_active = False
                self.destroy_timer(self.navigation_timer)
            '''
        else:
            self.get_logger().error(f"Navigation failed with result code: {result}")
            self.navigation_active = False
            self.destroy_timer(self.navigation_timer)

def main(args=None):
    rclpy.init(args=args)

    # 创建节点
    waypoint_navigator = WaypointNavigator()

    # 运行节点
    rclpy.spin(waypoint_navigator)

    # 关闭节点
    waypoint_navigator.navigator.lifecycleShutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
