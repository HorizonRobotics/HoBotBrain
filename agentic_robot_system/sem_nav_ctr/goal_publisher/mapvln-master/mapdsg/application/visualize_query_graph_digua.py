from copy import deepcopy

from hovsg.graph.graph import Graph
import hydra
import open3d as o3d
from omegaconf import DictConfig
import time
import numpy as np
import os
import json

# pylint: disable=all

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

instruction_templelate_ic3f_demo = [ # 27
    
    # 4 west mixed-use space
    # "white table with dark legs in the hallway",
    # "green chairs with a simple design in the hallway",
    "带我去地瓜电梯间找盆栽",
    "带我去地瓜电梯间找电视",
    
    # #0 west pantry
    "带我去地瓜接待区找瓶水",
    "带我去地瓜接待区找椅子",
    "带我去地瓜接待区找纸杯",

    # # 8 east mixed-use space
    "展厅找J6芯片",
      
    # #3/5 cafeteria
    "展厅找镜子",
    "展厅找把椅子",
    "展厅找百事可乐",
    "展厅找可口可乐"
]

@hydra.main(version_base=None, config_path="../config", config_name="visualize_query_graph_icra_ic3f_diguademo") # obj-embedding
# @hydra.main(version_base=None, config_path="../config", config_name="visualize_query_graph_icra_ic4f") # label, obj-embedding
# @hydra.main(version_base=None, config_path="../config", config_name="visualize_query_graph_icra_sh3f") #label, obj-embedding, view-embedding
# @hydra.main(version_base=None, config_path="../config", config_name="visualize_query_graph_icra_ic7f_demo")
# @hydra.main(version_base=None, config_path="../config", config_name="visualize_query_graph_1014_demo")
def main(params: DictConfig):
    # Load graph
    scene_id = params.main.scene_id
    use_gpt = params.main.use_gpt
    # params.main.scene_id = scene_id
    # Create save directory
    params.main.dataset_path = os.path.join(params.main.dataset_path, scene_id) # params.main.scene_id
    save_dir = os.path.join(params.main.save_path, params.main.dataset, scene_id) # params.main.scene_id
    params.main.save_path = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print("dataset_path: ", params.main.dataset_path)
    print("save_path: ", save_dir)

    hovsg = Graph(params)
    hovsg.load_graph_new(params.main.graph_path)
    # 自主判断房间类型和名字
    hovsg.generate_room_names(
            generate_method="view_embedding",
            # digua_demo room_types
            default_room_types=[                
                "Hallway",                               
                "Reception area",
                "Exhibition Hall",
                "Pantry",
                "Corner Hallway",
                "Elevator Lobby",                
                "Lift",
                "Office",
                "Cafeteria",             
            ]
    )
    # 人为设定房间类型和名字
    # designated_room_names = [        
    #     "厕所",         
    #     "地平线展厅",        
    #     "长走廊",
    #     "地瓜电梯间",]    

    designated_room_names_digua = [        
        "none", 
        "地平线走廊",
        "地平线展厅",
        "none",
        "转角走廊",
        # "转角走廊",
        "长走廊",
        "地瓜电梯间",]
    designated_room_names_ic3f = [      
        "none", #Exhibition Hall",
        "none",
        "Exhibition Hall",  
        "none", 
        "Corner Hallway",
        "HallWay",
        "Elevator Lobby Reception Area",
    ]
    designated_room_names_ic3f_demo = [      
        "none", #Exhibition Hall",
        "none",
        "展厅",  
        "none", 
        "转角走廊",
        "走廊",
        "地瓜电梯间接待区",
    ]
    designated_room_names_ic4f = [      
        "pantry", #Exhibition Hall",
        "office",
        "Hallway",
    ]
    designated_room_names_sh3f = [      
        "office pantry", #Exhibition Hall",
        "office pantry",
        "office",
    ]
    designated_room_names_ic7f = [      
        "west pantry", 
        "hallway",
        "hallway",
        "cafeteria",
        "west mixed-use space",
        "cafeteria",
        "elevator lobby",
        "hallyway office",
        "east mixed-use space",        
    ]
    designated_room_names_1014demo = [      
        "转角走廊",
        "none",
        "长走廊",
        "地平线展厅", 
        "none", 
        "none", 
        "长走廊",
        "接待区",
        "none",
        "地瓜办公区电梯间",]
    hovsg.set_room_names(room_names=designated_room_names_ic3f_demo)
    # hovsg.set_room_names(room_names=designated_room_names_1014demo)
    # import pdb; pdb.set_trace()
    
    T_switch_axis = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], dtype=np.float64) # map to dsg
    T_tomap = np.linalg.inv(T_switch_axis) # dsg to map
    json_save_path = os.path.join(hovsg.vln_result_dir, "all_results.json")
    all_results = []  # 存放每条 query 的结果
    # print("T_tomap: ", T_tomap)
    # loop forever and ask for query, until user click 'q'
    while True:
        query_instruction = input("Enter query: ")
        if query_instruction == "q":
            break
        # query_instruction = "带我在展厅找芯片RDK"
        print(query_instruction)
        hovsg.curr_query_save_dir = os.path.join(hovsg.vln_result_dir, query_instruction)
        if not os.path.exists(hovsg.curr_query_save_dir):
            os.makedirs(hovsg.curr_query_save_dir)

        start_time = time.time()
        floor, room, obj, res_dict = hovsg.query_hierarchy_protected(query_instruction, top_k=5, use_gpt=False)
        end_time = time.time()
        query_time = end_time - start_time
        print(f"运行时间: {query_time:.4f} 秒")

        # save log for debug
        # 构建要写入 JSON 的数据
        query_result = {
            "query": query_instruction,
            "room_query": res_dict["room_query"],
            "object_query": res_dict["object_query"],
            "time_seconds": query_time,
            "floor_id": floor.floor_id,
            "rooms": [{"room_id": r.room_id, "name": r.name} for r in room],
            "objects": [{"object_id": o.object_id} for o in obj],
            "objects_scores": res_dict["object_scores"]
        }        


        # handle the J6 芯片 badcase FOR 0901
        if "芯片" in query_instruction or "chip" in query_instruction.lower() or "镜子" in query_instruction or "mirror" in query_instruction.lower():
            if len(obj) > 1:
                obj = [obj[1]]
                # room = [room[0]]
        else:
            if len(obj) > 1:
                obj = [obj[0]]
                # room = [room[0]]

        # # FOR 1014DEMO map
        # object_query_lower = res_dict["object_query"].lower()
        # if "chip" in object_query_lower and "dk" in object_query_lower and "rdk" not in object_query_lower:
        #     if len(obj) > 3:
        #         obj = [obj[2]]
        # else:
        #     if len(obj) > 1:
        #         obj = [obj[0]]
        
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

            end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            end_sphere.translate(obj_center)
            end_sphere.paint_uniform_color([1, 0, 0])  
            # o3d.visualization.draw_geometries([room_pcd, obj_pcd, end_sphere])
            # 合并点云
            mesh_pcd = end_sphere.sample_points_uniformly(number_of_points=500)
            combined_pcd = room_pcd + obj_pcd + mesh_pcd
            # 保存为单个文件
            pcd_save_path = os.path.join(hovsg.curr_query_save_dir, f"scene_{i}.ply")
            pcd_render_save_path = os.path.join(hovsg.curr_query_save_dir, f"scene_{i}.png")
            o3d.io.write_point_cloud(pcd_save_path, combined_pcd)
            visualize_and_save(room_pcd, obj_pcd, end_sphere, save_path=pcd_render_save_path)
            print(f"Saved {pcd_save_path}")
        
        all_results.append(query_result)
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"All results saved to {json_save_path}")

if __name__ == "__main__":
    main()
