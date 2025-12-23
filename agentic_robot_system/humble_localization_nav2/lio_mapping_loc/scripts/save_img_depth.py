#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import os

class ImagePoseSaver(Node):
    def __init__(self):
        super().__init__('image_pose_saver')


        # 顶层输出目录
        self.output_dir = '/map/image_depth_pose'
        # 清空目录下的内容
        
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.rgb_dir = os.path.join(self.output_dir, 'rgb')
        self.depth_dir = os.path.join(self.output_dir, 'depth')
        self.pose_dir = os.path.join(self.output_dir, 'pose')

        # 创建目录
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.pose_dir, exist_ok=True)

        # 初始化 cv_bridge
        self.bridge = CvBridge()

        # 订阅话题
        self.sub_rgb = self.create_subscription(
            Image, '/rgb_img', self.rgb_callback, 10)
        self.sub_depth = self.create_subscription(
            Image, '/depth_img', self.depth_callback, 10)
        self.sub_pose = self.create_subscription(
            PoseStamped, '/camera_pose', self.pose_callback, 10)

        # 打开姿态文件
        self.pose_file_path = os.path.join(self.pose_dir, 'pose_log.txt')
        self.pose_file = open(self.pose_file_path, 'w')

        self.img_count = 0
        self.get_logger().info('✅ ImagePoseSaver node started.')
        self.get_logger().info(f'Saving RGB -> {self.rgb_dir}')
        self.get_logger().info(f'Saving Depth -> {self.depth_dir}')
        self.get_logger().info(f'Saving Pose -> {self.pose_file_path}')

    def rgb_callback(self, msg):
        self.save_image(msg, img_type='rgb')

    def depth_callback(self, msg):
        self.save_image(msg, img_type='depth')

    def pose_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        line = f"{timestamp:.6f} {msg.pose.position.x:.6f} {msg.pose.position.y:.6f} {msg.pose.position.z:.6f} " \
               f"{msg.pose.orientation.x:.6f} {msg.pose.orientation.y:.6f} " \
               f"{msg.pose.orientation.z:.6f} {msg.pose.orientation.w:.6f}\n"
        self.pose_file.write(line)
        self.pose_file.flush()

    def save_image(self, msg, img_type):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if img_type == 'rgb':
                filename = os.path.join(self.rgb_dir, f"{timestamp:.6f}.png")
            else:
                filename = os.path.join(self.depth_dir, f"{timestamp:.6f}.png")

            cv2.imwrite(filename, cv_image)
            self.img_count += 1

            if self.img_count % 50 == 0:
                self.get_logger().info(f"Saved {self.img_count} images so far...")
        except Exception as e:
            self.get_logger().error(f"Failed to save {img_type} image: {e}")

    def destroy_node(self):
        self.pose_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImagePoseSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
