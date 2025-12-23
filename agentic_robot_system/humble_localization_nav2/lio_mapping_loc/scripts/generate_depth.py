import numpy as np
import cv2
import argparse
import logging
import os
import open3d as o3d
from scipy.spatial import KDTree
# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
import cv2
import numpy as np
def read_pcd_poses(file_path):
    """
    读取 TUM 格式的位姿文件，并返回位姿和对应的时间序列。

    Args:
        file_path (str): TUM 格式的位姿文件路径。

    Returns:
        tuple: 位姿列表 (np.ndarray) 和时间序列列表 (list)。
    """
    poses = []
    timestamps = []
    with open(file_path, 'r') as f:
        for idx,line in enumerate(f):
            parts = line.strip().split()
            if len(parts) == 7:  # 确保每行有 8 个字段
                # timestamp = f"{int(parts[0]):06d}"  # 时间戳在第一列，转换为6位整数
                tx, ty, tz = map(float, parts[0:3])
                qw,qx, qy, qz= map(float, parts[3:7])
                poses.append([tx, ty, tz])
                timestamps.append(idx)
    return np.array(poses), timestamps
def load_scan_poses(file_path):
    """
    从 scan_pos.txt 文件中加载位姿信息。

    Args:
        file_path (str): 位姿文件路径。

    Returns:
        dict: 键为索引，值为 4x4 齐次变换矩阵的字典。
    """
    poses = {}
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            values = list(map(float, line.strip().split()))
            if len(values) != 7:
                print(f"Warning: Invalid pose format at line {idx}")
                continue
            # print("index: ", idx)
            # import pdb
            # pdb.set_trace()
            tx, ty, tz, qw, qx, qy, qz = values
            rotation = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            translation = np.array([tx, ty, tz])
            # 构造 4x4 齐次变换矩阵
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = translation
            poses[idx] = pose
    return poses
def whether_occluded_deocc(uvs, img_input, image_scale, dilate_view_path=""):
    """
    判断点是否被遮挡的函数。

    参数:
        uvs (list of tuple): 包含 (x, y, z) 的点列表。
        img_input (numpy.ndarray): 输入图像，用于获取图像尺寸。
        image_scale (int): 图像缩放比例，用于调整膨胀操作的核大小。
        dilate_view_path (str): 可选，保存膨胀结果的路径。

    返回:
        list of bool: 每个点是否被遮挡的标志。
    """
    imgH, imgW = img_input.shape[:2]
    occlusion_flag = np.zeros(len(uvs), dtype=bool)
    depth = np.full((imgH, imgW), np.inf, dtype=np.float32)
    fb = 100.0

    # 多对一取最近的点（前景）
    for x, y, z in uvs:
        if z > 0 and 0 <= x < imgW and 0 <= y < imgH:
            col, row = int(x), int(y)
            depth[row, col] = min(depth[row, col], z)

    # 计算反深度图并膨胀
    inv_depth = np.where(depth < np.inf, fb / depth, 0)
    kernel_size = max(1, 2 // image_scale)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    inv_dilate_depth = cv2.dilate(inv_depth, element, iterations=4)

    # 保存膨胀结果（可选）
    if dilate_view_path:
        inv_depth_u8 = (inv_depth * 8).astype(np.uint8)
        inv_dilate_depth_u8 = (inv_dilate_depth * 8).astype(np.uint8)
        disp_vis = cv2.applyColorMap(inv_depth_u8, cv2.COLORMAP_JET)
        disp_dilate_vis = cv2.applyColorMap(inv_dilate_depth_u8, cv2.COLORMAP_JET)
        result = np.hstack((disp_vis, disp_dilate_vis))
        cv2.imwrite(dilate_view_path, result)

    # 判断遮挡
    for idx, (x, y, z) in enumerate(uvs):
        if z > 0 and 0 <= x < imgW and 0 <= y < imgH:
            col, row = int(x), int(y)
            pesudo_disp = inv_dilate_depth[row, col]
            px_disp = fb / z
            if pesudo_disp == 0 or abs(px_disp - pesudo_disp) >= 3.0:
                occlusion_flag[idx] = True
        else:
            occlusion_flag[idx] = True

    return occlusion_flag.tolist()
def whether_occluded_deoccfast(uvs, img_input, image_scale, dilate_view_path=""):
    """
    判断点是否被遮挡的函数。
    
    参数:
        uvs (list of tuple): 包含 (x, y, z) 的点列表。
        img_input (numpy.ndarray): 输入图像，用于获取图像尺寸。
        image_scale (int): 图像缩放比例，用于调整膨胀操作的核大小。
        dilate_view_path (str): 可选，保存膨胀结果的路径。
    
    返回:
        list of bool: 每个点是否被遮挡的标志。
    """
    imgH, imgW = img_input.shape[:2]
    occlusion_flag = [False] * len(uvs)
    depth = np.zeros((imgH, imgW), dtype=np.float32)
    inv_depth = np.zeros((imgH, imgW), dtype=np.float32)
    min_depth = np.full((imgH, imgW), 1000.0, dtype=np.float32)
    fb = 20.0

    # 多对一取最近的点（前景）
    for k, (x, y, z) in enumerate(uvs):
        if z <= 0 or x + 0.5 < 0 or x + 0.5 >= imgW or y + 0.5 < 0 or y + 0.5 >= imgH:
            occlusion_flag[k] = True
            continue

        col = int(x + 0.5)
        row = int(y + 0.5)
        # col= int(x)
        # row= int(y)
        cur_depth = z
        if min_depth[row, col] > cur_depth:  # 当前点距离更近
            min_depth[row, col] = cur_depth
            depth[row, col] = cur_depth
            inv_depth[row, col] = fb / cur_depth

    # 膨胀操作
    kernel_size = max(1, 4 // image_scale)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    inv_dilate_depth = cv2.dilate(inv_depth, element, iterations=4)
    inv_dilate_depth_s16 = np.int16(inv_dilate_depth)
    # inv_dilate_depth_s16 = np.uint8(np.clip(inv_dilate_depth_s16, 0, 255))
    cv2.filterSpeckles(inv_dilate_depth_s16, 0, 1000, 1)
    # 保存膨胀结果（可选）
    # if dilate_view_path:
    #     # inv_depth_u8 = np.uint8(inv_depth * 8)
    #     # inv_dilate_depth_u8 = np.uint8(inv_dilate_depth * 8)
    #     inv_dilate_depth_u8 = inv_dilate_depth.astype(np.uint8)
    #     inv_depth_u8 = inv_depth.astype(np.uint8)
    #     cv2.filterSpeckles(inv_dilate_depth_u8, 0, 1000, 1)

    #     disp_vis = cv2.applyColorMap(inv_depth_u8*8, cv2.COLORMAP_JET)
    #     disp_dilate_vis = cv2.applyColorMap(inv_dilate_depth_u8*8, cv2.COLORMAP_JET)
    #     result = np.hstack((disp_vis, disp_dilate_vis))
    #     cv2.imwrite(dilate_view_path, result)

    for k, (x, y, z) in enumerate(uvs):
        if z <= 0 or x + 0.5 < 0 or x + 0.5 >= imgW or y + 0.5 < 0 or y + 0.5 >= imgH:
            occlusion_flag[k] = True
            continue

        col = int(x + 0.5)
        row = int(y + 0.5)        
        # col= int(x)
        # row= int(y)
        # noise = inv_dilate_depth_s16[row, col]
        noise = inv_dilate_depth_s16[row, col]
        if noise == 0:
            # print(f"Warning: noise is zero at index {k}, setting occlusion_flag to True")
            occlusion_flag[k] = True
            continue
        pesudo_disp = inv_dilate_depth_s16[row, col]
        px_disp = fb / z
        if abs(px_disp - pesudo_disp) >= 3.0:
            occlusion_flag[k] = True

    return occlusion_flag
# 读取相机内参
def read_camera_intrinsics(camera_file):
    try:
        with open(camera_file, 'r') as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue  # 跳过注释行和空行
                parts = line.strip().split()
                if len(parts) < 8:
                    logging.warning(f"Invalid line format: {line.strip()}")  # 警告日志
                    continue  # 跳过无效行
                # 解析相机参数
                camera_id, model = parts[0], parts[1]
                width, height = int(parts[2]), int(parts[3])
                fx, fy, cx, cy = map(float, parts[4:8])
                intrinsics = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                return intrinsics, width, height
        raise ValueError(f"Invalid camera file format in file: {camera_file}")
    except Exception as e:
        logging.error(f"Error reading camera intrinsics: {e}")
        raise

# 读取点云文件
def read_point_cloud(point_cloud_file):
    try:
        points = []
        with open(point_cloud_file, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue  # 跳过注释行
                parts = line.strip().split()
                if len(parts) < 8:
                    continue  # 跳过无效行
                x, y, z = map(float, parts[1:4])
                points.append([x, y, z])
        return np.array(points)
    except Exception as e:
        logging.error(f"Error reading point cloud: {e}")
        raise

# 读取图像轨迹文件
def read_image_trajectories(images_file):
    try:
        trajectories = []
        # trajectories2 = []
        with open(images_file, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 1):  # 每两行处理一次
                if lines[i].startswith("#") or lines[i].strip() == "":
                    # print(f"Skipping comment or empty line: {lines[i]}")  # 调试信息
                    continue  # 跳过注释行和空行
                parts = lines[i].strip().split()
                # print(f"Processing line: {lines[i]}")  # 调试信息
                # print(f"Processing line: {lines[i].strip()}")  # 调试信息
                if len(parts) < 8:  # 检查是否有足够的字段
                    # logging.warning(f"Invalid line format in images file: {lines[i].strip()}")
                    continue
                # 解析四元数和平移向量
                # print(parts)
                qw, qx, qy, qz = map(float, parts[1:5])  # 四元数
                tx, ty, tz = map(float, parts[5:8])  # 平移向量
                # camera to world
                # rotation = quaternion_to_rotation_matrix(qw, qx, qy, qz)
                # world to camera
                rotation = quaternion_to_rotation_matrix(qw, qx, qy, qz)
                translation = np.array([tx, ty, tz])
                # camera to world
                # rotation2 = np.linalg.inv(rotation)
                # translation2 = -np.dot(rotation, translation)
                trajectories.append((rotation, translation))
                # trajectories2.append((rotation2, translation2))
        return trajectories
    except Exception as e:
        logging.error(f"Error reading image trajectories: {e}")
        raise
def read_image_tum_trajectories(images_file):
    poses = {}
    with open(images_file, 'r') as f:
        for idx, line in enumerate(f):
            values = list(map(float, line.strip().split()))
            if len(values) != 8:
                print(f"Warning: Invalid pose format at line {idx}")
                continue
            timestamp = f"{float(values[0]):.4f}"  # 时间戳在第一列，保留四位小数
            tx, ty, tz, qx, qy, qz, qw = values[1:8]
            rotation = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            translation = np.array([tx, ty, tz])
            # 构造 4x4 齐次变换矩阵
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = translation
            poses[timestamp] = pose
    return poses    
# 四元数转旋转矩阵
def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]
    ])

# 投影点云
def project_points(points, rotation, translation, intrinsics, img_width, img_height):
    # 将点云转换到相机坐标系
    points_camera = np.dot(rotation, points.T) + translation.reshape(3, 1)

    # 剔除 z <= 0 的点（相机后方的点）
    valid_mask = points_camera[2, :] > 0
    points_camera = points_camera[:, valid_mask]

    # 投影到图像平面
    points_image = np.dot(intrinsics, points_camera / points_camera[2, :])

    # 剔除超出图像范围的点
    x, y = points_image[0, :], points_image[1, :]
    valid_mask = (x >= 0) & (x < img_width) & (y >= 0) & (y < img_height)
    points_camera = points_camera[:, valid_mask]
    points_image = points_image[:, valid_mask]

    return points_image, points_camera

def generate_occ_depth(points_image, points_camera, img_width, img_height, dilate_view_path, depth_output_path, visualization_output_path, depth_factor, img_input, image_scale):
    """
    生成深度图，并对点云进行去遮挡处理。
    :param points_image: 投影到图像上的点云坐标 (numpy array)
    :param points_camera: 相机坐标系下的点云坐标 (numpy array)
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :param depth_factor: 深度缩放因子
    :param img_input: 输入图像，用于遮挡判断
    :param image_scale: 图像缩放比例
    """
    # 将点云转换为 (x, y, z) 格式
    uvs = np.vstack((points_image[0, :], points_image[1, :], points_camera[2, :])).T
    # print(f"uvs size: {uvs.shape}")

    # 调用 whether_occluded_deoccfast 进行去遮挡
    occlusion_flags = np.array(whether_occluded_deoccfast(uvs, img_input, image_scale, dilate_view_path))

    # 筛选未被遮挡的点
    valid_mask = (uvs[:, 2] > 0) & ~occlusion_flags
    # valid_mask= (uvs[:, 2] > 0)
    valid_uvs = uvs[valid_mask]

    # 提取有效点的坐标和深度
    x = valid_uvs[:, 0].astype(int)
    y = valid_uvs[:, 1].astype(int)
    z = valid_uvs[:, 2]

    # 筛选在图像范围内的点
    in_bounds_mask = (0 <= x) & (x < img_width) & (0 <= y) & (y < img_height)
    x, y, z = x[in_bounds_mask], y[in_bounds_mask], z[in_bounds_mask]

    # 初始化深度图
    depth_map = np.zeros((img_height, img_width), dtype=np.float32)

    # 填充深度图
    depth_map[y, x] = z * depth_factor

    # 生成颜色映射
    colors = depth2color(valid_uvs, min_depth=0.2, max_depth=100.0)
    # # 可视化点云颜色（采样加速 + 向量化绘制）
    # if len(x) > 0:
    #     # 将颜色与 in-bounds 的点对齐
    #     colors_in_bounds = colors[0][in_bounds_mask]

    #     # 子采样，限制最大绘制点数
    #     max_draw = 80000  # 可根据需求调整
    #     if len(x) > max_draw:
    #         sel = np.random.choice(len(x), max_draw, replace=False)
    #         xs = x[sel]
    #         ys = y[sel]
    #         cols_sel = colors_in_bounds[sel]
    #     else:
    #         xs = x
    #         ys = y
    #         cols_sel = colors_in_bounds

    #     # 去重相同像素（减少重复绘制）
    #     lin_idx = ys * img_width + xs
    #     _, uniq_idx = np.unique(lin_idx, return_index=True)
    #     xs = xs[uniq_idx]
    #     ys = ys[uniq_idx]
    #     cols_sel = cols_sel[uniq_idx]

    #     # 向量化给像素着色（BGR）
    #     img_input[ys, xs] = cols_sel.astype(np.uint8)
    # 可视化点云颜色
    for i, (col, row) in enumerate(zip(x, y)):
        if i%4==0:
            continue
        color = colors[0][i]
        cv2.circle(img_input, (col, row), 1, (int(color[0]), int(color[1]), int(color[2])), -1)
    # 保存结果
    cv2.imwrite(visualization_output_path, img_input)
    cv2.imwrite(depth_output_path, depth_map.astype(np.uint16))
    logging.info(f"Saved depth map to: {depth_output_path}")
    # logging.info(f"Saved visualization to: {visualization_output_path}")

def depth2color(points, min_depth=0.2, max_depth=1000.0):
    """
    Convert depth values to a color map representation.
    
    :param points: List of (x, y, z) tuples representing 3D points.
    :param min_depth: Minimum depth value for normalization.
    :param max_depth: Maximum depth value for normalization.
    :return: Color map as a numpy array.
    """
    if points.size == 0:  # 检查输入是否为空
        return np.zeros((1, 0, 3), dtype=np.uint8)  # 返回空的颜色映射
    N = len(points)
    dist_gray = np.zeros((1, N), dtype=np.uint8)
    min_depth = np.min(points[:, 2])
    max_depth = np.max(points[:, 2])
    for i in range(N):
        dist = points[i][2]  # Assuming points is a list of (x, y, z) tuples
        dist = np.clip(dist, min_depth, max_depth)
        ratio = (dist - min_depth) / (max_depth - min_depth)  # 归一化到 [0, 1]
        ratio = np.sqrt(ratio)  # 使用平方根增强近处的颜色变化
        dist_gray[0, i] = int(ratio * 255)
    # print(f"dist_gray: {dist_gray}")
    dist_color = cv2.applyColorMap(dist_gray, cv2.COLORMAP_JET)
    # print(f"dist_color: {dist_color}")

    return dist_color


def load_keyframe_clouds(keyframe_dir, nearest_timestamps, kf_cloud_poses):
    """
    加载关键帧点云，将其转换到世界坐标系，并拼接成局部地图。

    Args:
        keyframe_dir (str): 关键帧点云文件所在目录。
        nearest_timestamps (list): 最近关键帧的时间戳列表。
        kf_cloud_poses (np.ndarray): 关键帧的位姿列表，每行包含 [tx, ty, tz, qw, qx, qy, qz]。

    Returns:
        np.ndarray: 拼接后的局部地图点云。
    """
    local_map = []
    for timestamp in nearest_timestamps:
        # 构造点云文件路径
        pcd_file = os.path.join(keyframe_dir, f"{timestamp}.pcd")
        if not os.path.exists(pcd_file):
            logging.warning(f"Point cloud file not found: {pcd_file}")
            continue

        # 加载点云文件
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(pcd.points)

        # 添加到局部地图
        local_map.append(points)

    # 拼接所有点云
    if local_map:
        local_map = np.vstack(local_map)
    else:
        local_map = np.empty((0, 3))  # 如果没有点云，返回空数组

    return local_map
# 主函数
# def main():

#     # 自动定位文件
#     # input_dir = args.input_dir
#     input_dir = "/home/users/tingyang.xiao/3D_Recon/datasets/G1/rosbag2_2025_07_18-19_16_33_new/sfm/sparse/0"
#     output_dir= "/home/users/tingyang.xiao/3D_Recon/datasets/G1/rosbag2_2025_07_18-19_16_33_new/depth"
#     vis_output_dir= "/home/users/tingyang.xiao/3D_Recon/datasets/G1/rosbag2_2025_07_18-19_16_33_new/vis_depth"
#     images_file = os.path.join(input_dir, "poses.txt")
#     camera_file = os.path.join(input_dir, "cameras.txt")
#     keyframe_dir = "/home/users/tingyang.xiao/3D_Recon/datasets/G1/rosbag2_2025_07_18-19_16_33_new/PCD"
#     traj_kf_file="/home/users/tingyang.xiao/3D_Recon/datasets/G1/rosbag2_2025_07_18-19_16_33_new/PCD/scans_pos.txt"
#     # keyframe_dir = "/home/users/tingyang.xiao/3D_Recon/datasets/G1/rosbag2_2025_04_08-21_29_56_fast_livo/PCD"
#     # traj_kf_file="/home/users/tingyang.xiao/3D_Recon/datasets/G1/rosbag2_2025_04_08-21_29_56_fast_livo/PCD/scans_pos.txt"
#     # 检查文件是否存在
#     # if not (os.path.exists(images_file) and os.path.exists(point_cloud_file) and os.path.exists(camera_file)):
#     #     logging.error("One or more required files (images.txt, points3D.txt, cameras.txt) are missing in the input directory.")
#     #     return

#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(vis_output_dir, exist_ok=True)

#     intrinsics, img_width, img_height = read_camera_intrinsics(camera_file)

#     # trajectories = read_image_trajectories(images_file)
#     trajectories = read_image_tum_trajectories(images_file)

#     # print(f"trajectories size: {len(trajectories)}")
#     kf_cloud_poses, timestamps = read_pcd_poses(traj_kf_file)
#     # 提取平移向量
#     kf_translations = np.array(kf_cloud_poses)
#     # 构建 KDTree 进行快速邻近搜索
#     kf_tree = KDTree(kf_translations)
#     # print(f"KDTree size: {kf_tree.data.shape}")
#     # 设置半径范围
#     radius = 4.0  # 半径范围（单位与点云坐标一致）
#     for image_timestamp, pose in trajectories.items():
#         # pose 是 4x4 齐次矩阵
#         rotation = pose[:3, :3]
#         translation = pose[:3, 3]
#         print(f"Processing frame {image_timestamp} with rotation: {rotation}, translation: {translation}")
#         # ...后续处理...
#         cam2world_rot = np.linalg.inv(rotation)
#         cam2world_trans= -np.dot(cam2world_rot, translation)
#         # print(f"cam2world_trans: {cam2world_trans}")
#         # 使用 KDTree 搜索最近的关键帧位姿
#         indices = kf_tree.query_ball_point(cam2world_trans, r=radius)
#         # nearest_points = kf_cloud_poses[indices]
#         nearest_timestamps = [timestamps[idx+1] for idx in indices]
#         # print(f"Frame {i}: Found {len(nearest_points)} points within radius {radius}.")

#         # 加载并拼接局部地图点云
#         local_map = load_keyframe_clouds(keyframe_dir, nearest_timestamps, kf_cloud_poses)
#         # 对局部地图点云进行下采样
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(local_map)
#         pcd = pcd.voxel_down_sample(voxel_size=0.01)  # 体素下采样
#         local_map = np.asarray(pcd.points)

#         if local_map.size == 0:
#             logging.warning(f"No valid point clouds found for frame {i}. Skipping.")
#             continue

#         # # 投影点云到图像
#         points_image, points_camera = project_points(local_map, rotation, translation, intrinsics,img_width, img_height)

#         # # 读取对应的 RGB 图像作为输入图像
#         rgb_image_path = '/home/users/tingyang.xiao/3D_Recon/datasets/G1/rosbag2_2025_07_18-19_16_33_new/images'
#         rgb_image_path = os.path.join(rgb_image_path, f"{image_timestamp}.png")   
#         img_input = cv2.imread(rgb_image_path)
#         if img_input is None:
#             logging.error(f"Failed to read RGB image: {rgb_image_path}")
#             continue
#         dilate_view_path= os.path.join(output_dir, f"{image_timestamp}_dilate.png")
#         depth_output_path = os.path.join(output_dir, f"{image_timestamp}.png")
#         visualization_output_path = os.path.join(vis_output_dir, f"{image_timestamp}_overlay.png")
#         generate_occ_depth(points_image, points_camera, img_width, img_height, dilate_view_path,depth_output_path,visualization_output_path,
#                                    depth_factor=1000, img_input=img_input, image_scale=1)
# ...existing code...
import concurrent.futures

def process_frame(args):
    image_timestamp, pose, kf_tree, kf_cloud_poses, timestamps, keyframe_dir, intrinsics, img_width, img_height, output_dir, vis_output_dir,rgb_image_path = args

    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    cam2world_rot = np.linalg.inv(rotation)
    cam2world_trans = -np.dot(cam2world_rot, translation)
    radius = 4.0

    indices = kf_tree.query_ball_point(cam2world_trans, r=radius)
    nearest_timestamps = [timestamps[idx] for idx in indices]

    local_map = load_keyframe_clouds(keyframe_dir, nearest_timestamps, kf_cloud_poses)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(local_map)
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    local_map = np.asarray(pcd.points)

    if local_map.size == 0:
        logging.warning(f"No valid point clouds found for frame {image_timestamp}. Skipping.")
        return

    points_image, points_camera = project_points(local_map, rotation, translation, intrinsics, img_width, img_height)
    rgb_image_path = os.path.join(rgb_image_path, f"{image_timestamp}.png")
    img_input = cv2.imread(rgb_image_path)
    if img_input is None:
        logging.error(f"Failed to read RGB image: {rgb_image_path}")
        return

    dilate_view_path = os.path.join(output_dir, f"{image_timestamp}_dilate.png")
    depth_output_path = os.path.join(output_dir, f"{image_timestamp}.png")
    visualization_output_path = os.path.join(vis_output_dir, f"{image_timestamp}_overlay.png")
    generate_occ_depth(points_image, points_camera, img_width, img_height, dilate_view_path, depth_output_path, visualization_output_path,
                       depth_factor=1000, img_input=img_input, image_scale=1)

def main():
    input_dir = "/home/users/tingyang.xiao/VLN/dataset/1014_demo//Colmap/sparse/0"
    output_dir = "/home/users/tingyang.xiao/VLN/dataset/1014_demo/Colmap/depth_dense"
    rgb_image_path = '/home/users/tingyang.xiao/VLN/dataset/1014_demo/Colmap/images'
    vis_output_dir = "/home/users/tingyang.xiao/VLN/dataset/1014_demo/Colmap/vis_depth_dense"
    images_file = os.path.join(input_dir, "poses.txt")
    camera_file = os.path.join(input_dir, "cameras.txt")
    keyframe_dir = "/home/users/tingyang.xiao/VLN/dataset/1014_demo/PCD"
    traj_kf_file = "/home/users/tingyang.xiao/VLN/dataset/1014_demo/PCD/scans_pos.txt"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_output_dir, exist_ok=True)

    intrinsics, img_width, img_height = read_camera_intrinsics(camera_file)
    trajectories = read_image_tum_trajectories(images_file)
    kf_cloud_poses, timestamps = read_pcd_poses(traj_kf_file)
    kf_translations = np.array(kf_cloud_poses)
    kf_tree = KDTree(kf_translations)

    # 并行处理每一帧
    args_list = [
        (image_timestamp, pose, kf_tree, kf_cloud_poses, timestamps, keyframe_dir, intrinsics, img_width, 
         img_height, output_dir, vis_output_dir, rgb_image_path)
        for image_timestamp, pose in trajectories.items()
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(process_frame, args_list))

if __name__ == "__main__":
    main()
