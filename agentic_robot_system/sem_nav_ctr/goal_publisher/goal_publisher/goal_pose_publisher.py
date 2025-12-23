#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import math
from ament_index_python.packages import get_package_share_directory
from copy import deepcopy
import sys
from std_msgs.msg import String

from hovsg.graph.graph import Graph
import hydra
import open3d as o3d
from omegaconf import DictConfig
import time
import numpy as np

from rclpy.action import ActionClient
from nav2_msgs.action import FollowWaypoints

# pylint: disable=all
          
class GoalPosePublisher(Node):
    
    def __init__(self, cfg: DictConfig):
        super().__init__('goal_pose_publisher')

        # 创建发布者，消息类型为PoseStamped，话题名为/goal_pose，队列大小为10
        self.publisher_ = self.create_publisher(PoseStamped, '/object_pose', 10)
        # 订阅String话题
        self.subscription = self.create_subscription(
            String,
            '/chat_loc_pub',  
            self.hovsggetgoal_callback,
            10)
        self._action_client = ActionClient(self, FollowWaypoints, '/follow_waypoints')
        # 设置定时器，每1秒发布一次目标位姿
        #timer_period = 1.0  # 秒
        #self.timer = self.create_timer(timer_period, self.timer_callback)
        self.count = 0
        self.params = cfg
        self.graph = Graph(cfg)
        self.T_switch_axis = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], dtype=np.float64) # g1_navi
        self.T_tomap = np.linalg.inv(self.T_switch_axis)
        self.hovsgcreate()
        self.use_gpt = 0
        #self.hovsggetgoal()

        # 初始化计数器
        

        self.get_logger().info('GoalPosePublisher 节点已启动，正在发布 /goal_pose 话题...')
        #print(f"This node is running with Python at: {sys.executable}")

    def pubpose(self, x, y):
        # 创建PoseStamped消息
        msg = PoseStamped()

        # 设置消息头
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'  # 假设目标位姿在map坐标系中

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0

        # 设置目标朝向 - 这里使用简单的四元数表示
        # 让机器人始终朝向圆心
        #msg.pose.orientation = self.get_quaternion_from_euler(0, 0, angle + math.pi)

        # 发布消息
        self.publisher_.publish(msg)

        # 记录日志
        self.get_logger().info(f'发布第 {self.count} 个目标位姿: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}')

        # 增加计数器
        self.count += 1
    def hovsgcreate(self):
        # Load graph
        hovsg = self.graph
        hovsg.load_graph(self.params.main.graph_path)
        self.use_gpt = self.params.main.use_gpt
        # 自主判断房间类型和名字
        hovsg.generate_room_names(
            generate_method="view_embedding",
            # digua_demo room_types
            default_room_types=[                
                "地瓜实验室",                               
                "地平线展厅",
                "地平线小邮局",
                "长走廊",
                "转角走廊",
                "电梯间",                
                "电梯",             
            ]
        )
        # 人为设定房间类型和名字
        designated_room_names_digua = [        
        "none", 
        "none",
        "展厅",  
        "none", 
        "转角走廊",
        "走廊",
        "地瓜电梯间接待区",]
        '''
        designated_room_names_ic7f_demo = [      
        "none", 
        "none",
        "办公区",
        "餐厅",
        "电梯间走廊",
        "茶水间",
        "办公休息区",
    ]
    '''
        designated_room_names_1014 = [      
        "转角走廊",
        "none",
        "长走廊",
        "地平线展厅", 
        "none", 
        "none", 
        "长走廊",
        "none",
        "none",
        "地瓜机器人办公区",]
        hovsg.set_room_names(room_names=designated_room_names_1014)
    def hovsggetgoal_callback(self, msg):
        hovsg = self.graph
        query_instruction = '来自语音查找'
        ans = msg.data
        print(ans)
        start_time = time.time()
        floor, room, obj, object_query = hovsg.query_hierarchy_protected(query_instruction, ans, top_k=5, use_gpt=self.use_gpt)
        end_time = time.time()
        print(f"运行时间: {end_time - start_time:.4f} 秒")
        
        # handle the J6 芯片 badcase， top3能精准找到
        '''if "芯片" in query_instruction or "chip" in query_instruction.lower() or "mirror" in query_instruction.lower():
            if len(obj) > 1:
                obj = [obj[1]]
        else:
            if len(obj) > 1:
                obj = [obj[0]]'''

        best_room_id = room[0].room_id
        if best_room_id == "0_9":
            if object_query == "":
                if len(obj) > 4:
                    obj = [obj[4]]
        # object_query_lower = object_query.lower()
        # if "rest" in object_query_lower:
        #     if len(obj) > 1:
        #         obj = [obj[1]]
        # if "car" in object_query_lower:
        #     if len(obj) > 1:
        #         obj = [obj[1]]
        # if "television" in object_query_lower:
        #     if len(obj) > 1:
        #         obj = [obj[1]]
        
        
        # visualize the query
        print(floor.floor_id, [(r.room_id, r.name) for r in room], [o.object_id for o in obj])
        # use open3d to visualize room.pcd and color the points where obj.pcd is
        print("len(obj): ", len(obj))
        for i in range(len(obj)):
            obj_pcd = obj[i].pcd.paint_uniform_color([1, 0, 0]) # rgb
            obj_pcd = deepcopy(obj[i].pcd)
            obj_center = obj_pcd.get_center()
            print("obj_center in scenegraph: ", obj_center)
            obj_center_h = np.hstack((obj_center, 1.0))  # 齐次坐标 (4,)
            obj_center_in_map = (self.T_tomap @ obj_center_h)[:3]  
            print("obj_center in lidarmap: ", obj_center_in_map) 
            self.pubpose(obj_center_in_map[0],obj_center_in_map[1])  
                    
    def hovsggetgoal(self):
        # loop forever and ask for query, until user click 'q'
        hovsg = self.graph
        while True:
            query = input("Enter query: ")
            if query == "q":
                break
            print(query)
            ans = ''
            start_time = time.time()
            floor, room, obj = hovsg.query_hierarchy(query, ans, top_k=1)
            end_time = time.time()
            print(f"运行时间: {end_time - start_time:.4f} 秒")
            # visualize the query
            print(floor.floor_id, [(r.room_id, r.name) for r in room], [o.object_id for o in obj])
            # use open3d to visualize room.pcd and color the points where obj.pcd is
            print("len(obj): ", len(obj))
            for i in range(len(obj)):
                obj_pcd = obj[i].pcd.paint_uniform_color([1, 0, 0]) # rgb
                obj_pcd = deepcopy(obj[i].pcd)
                obj_center = obj_pcd.get_center()
                print("obj_center in scenegraph: ", obj_center)
                obj_center_h = np.hstack((obj_center, 1.0))  # 齐次坐标 (4,)
                obj_center_in_map = (self.T_tomap @ obj_center_h)[:3]  
                print("obj_center in lidarmap: ", obj_center_in_map) 
                self.pubpose(obj_center_in_map[0],obj_center_in_map[1])  
                
                    

@hydra.main(version_base=None, config_path=get_package_share_directory('goal_publisher')+"/config", config_name="visualize_query_graph_demo")           
def main(params: DictConfig, args=None):
  
    # 初始化ROS2 Python客户端库
    rclpy.init(args=args)

    # 创建节点
    goal_pose_publisher = GoalPosePublisher(params)

    try:
        # 运行节点
        rclpy.spin(goal_pose_publisher)
    except KeyboardInterrupt:
        # 处理Ctrl+C信号
        pass
    finally:
        # 销毁节点
        goal_pose_publisher.destroy_node()
        # 关闭ROS2 Python客户端库
        rclpy.shutdown()

if __name__ == '__main__':
    main()
