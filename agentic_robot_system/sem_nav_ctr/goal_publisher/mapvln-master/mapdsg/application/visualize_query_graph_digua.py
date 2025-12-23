from copy import deepcopy

from hovsg.graph.graph import Graph
import hydra
import open3d as o3d
from omegaconf import DictConfig
import time
import numpy as np
import os
# pylint: disable=all

# 🏢 一、单层常见房间类型
    # Office / Workspace —— 办公室/工作区
    # Private office -- 单人办公室
    # Open-plan office / Open workspace 开放式工位区
    # Meeting Room / Conference Room —— 会议室
    # Small meeting room / Huddle room 小会议室
    # Large conference hall / Multipurpose room -- 大会议室 / 多功能厅
    # Break Room / Pantry / Kitchenette —— 茶水间 / 小厨房
    # Restroom / Toilet / Washroom —— 卫生间
    # Storage / Utility Room —— 储藏间 / 设备间
    # Server Room / IT Room / Data Center —— 机房 / 数据中心
    # Reception Area —— 接待区

# 🏢 二、跨楼层或公共区域
    # Lobby / Entrance Hall —— 大堂 / 入口大厅
    # Atrium —— 中庭（通常是多层贯通的挑空空间，有玻璃顶采光）
    # Corridor / Hallway —— 走廊 / 过道
    # Stairwell —— 楼梯间
    # Elevator Lobby —— 电梯厅
    # Mechanical / Service Floor —— 机电层（跨楼层设置，空调、水电管道等）
    # Auditorium —— 报告厅 / 大型会议厅（可能跨层）
    # Cafeteria / Dining Hall —— 餐厅 / 员工食堂（常跨层挑高）
    # Gym / Fitness Center —— 健身房（部分高档办公楼会有）

'''
def visualize_and_save(room_pcd, obj_pcd, end_sphere, save_path="scene.png"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)  # 设置 False 可后台渲染
    vis.add_geometry(room_pcd)
    vis.add_geometry(obj_pcd)
    vis.add_geometry(end_sphere)

    vis.poll_events()
    vis.update_renderer()

    # 计算房间和物体的中心
    room_center = np.array(room_pcd.get_center())
    obj_center = np.array(obj_pcd.get_center())

    # 相机位置：在房间中心 + y方向上方 5m
    cam_pos = room_center + np.array([0, 5.0, 0.0])

    # 设置相机参数
    ctr = vis.get_view_control()
    ctr.set_lookat(obj_center)                     # 看向物体中心
    ctr.set_front((cam_pos - obj_center) /
                  np.linalg.norm(cam_pos - obj_center))  # 相机朝向
    ctr.set_up([0, 1, 0])                          # 这里假设 z 作为水平参考，上方向定为 z

    ctr.set_zoom(0.7)  # 缩放调节

    vis.poll_events()
    vis.update_renderer()

    # 保存截图
    vis.capture_screen_image(save_path)
    vis.destroy_window()
    print(f"Saved visualization to {save_path}")
'''
@hydra.main(version_base=None, config_path="../config", config_name="visualize_query_graph_demo")
def main(params: DictConfig):
    # Load graph
    #scene_id = params.main.scene_id
    use_gpt = params.main.use_gpt
    # params.main.scene_id = scene_id
    # Create save directory
    #params.main.dataset_path = os.path.join(params.main.dataset_path, scene_id) # params.main.scene_id
    #save_dir = os.path.join(params.main.save_path, params.main.dataset, scene_id) # params.main.scene_id
    #params.main.save_path = save_dir
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir, exist_ok=True)
    #print("dataset_path: ", params.main.dataset_path)
    #print("save_path: ", save_dir)

    hovsg = Graph(params)
    hovsg.load_graph(params.main.graph_path)
    
    # 自主判断房间类型和名字
    hovsg.generate_room_names(
            generate_method="view_embedding",
            # generate_method="label",
            # generate_method="obj_embedding",
            # digua_demo room_types 0807
            default_room_types=[                
                "Hallway",                               
                # "Exhibition Hall",
                "Office Pantry",
                "Elevator Lobby",                
                # "Lift",
                "Office",
                # "Office-Pantry",
                # "none",
                "Cafeteria", 
                # "Reception Area",
            ]
    )
    designated_room_names_ic7f_demo = [      
        "none", 
        "none",
        "办公区",
        "餐厅",
        "电梯间走廊",
        "茶水间",
        "办公休息区",
    ]
    hovsg.set_room_names(room_names=designated_room_names_ic7f_demo)
    # import pdb; pdb.set_trace()
    
    T_switch_axis = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], dtype=np.float64) # map to dsg
    T_tomap = np.linalg.inv(T_switch_axis) # dsg to map
    # print("T_tomap: ", T_tomap)
    # loop forever and ask for query, until user click 'q'
    while True:
        query_instruction = input("Enter query: ")
        if query_instruction == "q":
            break
        # query_instruction = "Find me a plants in the 地平线展厅"
        print(query_instruction)
        hovsg.curr_query_save_dir = os.path.join(hovsg.vln_result_dir, query_instruction)
        if not os.path.exists(hovsg.curr_query_save_dir):
            os.makedirs(hovsg.curr_query_save_dir)

        start_time = time.time()
        ans = ''
        floor, room, obj = hovsg.query_hierarchy_protected(query_instruction, ans, top_k=1, use_gpt=use_gpt)
        end_time = time.time()
        print(f"运行时间: {end_time - start_time:.4f} 秒")
        # visualize the query
        print(floor.floor_id, [(r.room_id, r.name) for r in room], [o.object_id for o in obj])
        # use open3d to visualize room.pcd and color the points where obj.pcd is
        print("len(obj): ", len(obj))
        for i in range(len(obj)):
            obj_pcd = obj[i].pcd.paint_uniform_color([0, 1, 0]) # rgb
            room_pcd = room[i].pcd
            obj_pcd = deepcopy(obj[i].pcd)
            room_pcd = deepcopy(room[i].pcd)
            obj_center = obj_pcd.get_center()
            print("obj_center in scenegraph: ", obj_center)
            obj_center_h = np.hstack((obj_center, 1.0))  # 齐次坐标 (4,)
            obj_center_in_map = (T_tomap @ obj_center_h)[:3]  
            print("obj_center in lidarmap: ", obj_center_in_map)

            #end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
            #end_sphere.translate(obj_center)
            #end_sphere.paint_uniform_color([1, 0, 0])  
            # o3d.visualization.draw_geometries([room_pcd, obj_pcd, end_sphere])
            # 合并点云
            #mesh_pcd = end_sphere.sample_points_uniformly(number_of_points=500)
            #combined_pcd = room_pcd + obj_pcd + mesh_pcd
            # 保存为单个文件
            #pcd_save_path = os.path.join(hovsg.curr_query_save_dir, f"scene_{i}.ply")
            #pcd_render_save_path = os.path.join(hovsg.curr_query_save_dir, f"scene_{i}.png")
            #o3d.io.write_point_cloud(pcd_save_path, combined_pcd)
            #visualize_and_save(room_pcd, obj_pcd, end_sphere, save_path=pcd_render_save_path)
            #print(f"Saved {pcd_save_path}")



if __name__ == "__main__":
    main()
