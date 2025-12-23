import asyncio
import time
from loguru import logger
import numpy as np
from queue import Queue
import threading
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .drobotc_g1 import DRobotC

class DRobotCNode(Node):
    def __init__(self, name:str = "drobotc", host:str = "180.76.187.170", port:int = 10071, device_name:str = "ReSpeaker"):
        super().__init__(name)

        # 初始化
        self.drobotc = DRobotC(host = host, port = port, device_name = device_name)

        # drobotc队列
        self.text_queue = self.drobotc.get_text_queue()
        self.control_queue = self.drobotc.get_control_queue()

        # 创建发布者对象(消息类型, 话题名, 队列长度)
        self.loc_pub = self.create_publisher(String, "chat_loc_pub", 100)
        self.signal_pub = self.create_publisher(String, "chat_signal_pub", 100)
        self.qa_pub = self.create_publisher(String, "chat_qa_pub", 100)

        # 订阅消息
        self.signal_sub = self.create_subscription(String, "waypoint_reached", self.waypoint_callback, 100)

        # 创建websocket连接线程
        self.ws_thread = threading.Thread(target=self._run_websocket)
        self.ws_thread.daemon = True  # 设置为守护线程，这样主程序退出时会自动结束
        self.ws_thread.start()

        # 创建定时器
        self.timer = self.create_timer(0.04, self.loc_callback)

    def _run_websocket(self):
        """在单独的线程中运行websocket连接"""
        try:
            asyncio.run(self.drobotc.connect())
        except Exception as e:
            logger.error(f"Websocket connection error: {e}")
        finally:
            self.drobotc.is_recording = False
            self.drobotc.p.terminate()

    def loc_callback(self):
        if not self.text_queue.empty():
            msg_str = self.text_queue.get_nowait()
            msg_type, msg_data, msg_chat_id = msg_str.split("::")
            if msg_type == "loc":
                pub_data = String()
                pub_data.data = msg_data
                self.loc_pub.publish(pub_data)
            elif msg_type == "signal":
                pub_data = String()
                pub_data.data = msg_data
                self.signal_pub.publish(pub_data)
            elif msg_type == "qa":
                pub_data = String()
                pub_data.data = msg_data
                self.qa_pub.publish(pub_data)
        else:
            time.sleep(0.04)

    def waypoint_callback(self, msg):
        """获得订阅的字符串
        
        Args:
            msg (String): 从waypoint_reached话题接收到的消息
        """
        # 获取消息内容
        received_data = msg.data
        # 在这里处理接收到的数据
        logger.info(f"ROS 订阅到导航点信息: {received_data}")
        self.control_queue.put(received_data)

def main(args=None):
    rclpy.init(args=args)
    node = DRobotCNode("topic_chat_loc_pub")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
