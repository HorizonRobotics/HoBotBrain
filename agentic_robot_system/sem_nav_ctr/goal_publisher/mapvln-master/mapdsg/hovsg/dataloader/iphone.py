
import os 
import numpy as np
from PIL import Image
import torchvision
import open3d as o3d
import yaml
from hovsg.dataloader.generic import RGBDDataset
from scipy.spatial.transform import Rotation as R
import cv2
from typing import Any, Dict, List, Optional
import json

class IPhoneDataset(RGBDDataset):
    """
    Dataset class for the ScanNet dataset.

    This class provides an interface to load RGB-D data samples from the ScanNet
    dataset. The dataset format is assumed to follow the ScanNet v2 dataset format.
    """
    
    def __init__(self, cfg):
        """
        Args:
            root_dir: Path to the root directory containing the dataset.
            mode: "train", "val", or "test" depending on the data split.
            transforms: Optional transformations to apply to the data.
        """
        super(IPhoneDataset, self).__init__(cfg)
        self.root_dir = cfg["root_dir"]
        self.transforms = cfg["transforms"]
        pose_name = "colmap_pose"
        camera_config_path = "orbslam3_rgbd.yaml"
        if os.path.exists(camera_config_path):
            self.rgb_intrinsics, self.depth_intrinsics = self.load_camera_params(os.path.join(self.root_dir, camera_config_path))
            self.frames = None
        else:
            self.frames = self.load_camera_config(os.path.join(self.root_dir, "transforms.json"))

        self.scale = 1000.0
        print("self.root_dir: ", self.root_dir)
        if pose_name is not None and os.path.exists(
            os.path.join(self.root_dir, f"{pose_name}.txt")
        ):
            print("use pose file: ", pose_name)
            camtoworlds, ts_list = self.load_tum_pose(
                os.path.join(self.root_dir, f"{pose_name}.txt")
            )
        elif os.path.exists(os.path.join(self.root_dir, "CameraTrajectory.txt")):
            camtoworlds, ts_list = self.load_tum_pose(
                os.path.join(self.root_dir, "CameraTrajectory.txt")
            )
            print("use pose file: ", "CameraTrajectory.txt")
        else:
            assert False, "No pose file found in the directory"
        
        if int(ts_list[0]) != ts_list[0]:
            self.ts_list = ["{:.4f}".format(ts) for ts in ts_list]
        else:
            self.ts_list = [str(int(ts)) for ts in ts_list]

        print("self.ts_list: ", self.ts_list)
        print("len(self.frames) : ", len(self.frames))
        # import pdb; pdb.set_trace()
        
        
        self.camtoworlds = camtoworlds
        self.indices = np.arange(len(self.ts_list))
        # re-range ts_list for abs ts
        # self.ts_list = [str(int(ts)) for ts in range(len(ts_list))]
        # self.ts_list = self.ts_list[self.start_index : self.end_index]
        self.image_paths = [
                    os.path.join(self.root_dir, "images_2",  f"frame_{int(ts):05d}.jpg")
                    for ts in self.ts_list
                ]
        self.depth_paths = [
                    os.path.join(self.root_dir, "depth_2",  f"frame_{int(ts):05d}.png")
                    for ts in self.ts_list
                ]
        print("iphone dataset init success!")

    def __len__(self):
            return len(self.indices)
    
    def load_camera_config(self, json_path: str) -> Dict:
        # 存在性检查
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"路径不是文件: {json_path}")
        
        # 读取配置
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        # 数据解析
        frames = []
        for idx, frame in enumerate(config["frames"]):
            # 使用os.path处理路径
            rgb_path = os.path.normpath(frame["file_path"]).replace("images", "images_2")
            depth_path = os.path.normpath(frame.get("depth_file_path", "")).replace("depth", "depth_2")
            
            frames.append({
                "K": [
                    [frame["fl_x"]/2, 0, frame["cx"]/2],
                    [0, frame["fl_y"]/2, frame["cy"]/2],
                    [0, 0, 1]
                ],
                "image_size": (frame["w"] // 2, frame["h"] // 2),
                "distortion": [
                    frame.get("k1", 0),
                    frame.get("k2", 0),
                    frame.get("p1", 0),
                    frame.get("p2", 0)
                ],
                "transform": frame["transform_matrix"],
                "rgb_path": rgb_path,
                "depth_path": depth_path
            })
        
        return frames

    
    def load_camera_params(
        self, config_path: str, camera_name: str = None
    ) -> np.ndarray:

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # load camera intrinsics
        K = np.eye(3)
        if "Camera.fx" in config.keys() and type(config["Camera.fx"]) == set:
            K[0, 0] = next(iter(config["Camera.fx"]))
            K[1, 1] = next(iter(config["Camera.fy"]))
            K[0, 2] = next(iter(config["Camera.cx"]))
            K[1, 2] = next(iter(config["Camera.cy"]))
            image_size = next(iter(config["Camera.width"])), next(
                iter(config["Camera.height"])
            )
        else:
            K[0, 0] = config["Camera1.fx"]
            K[1, 1] = config["Camera1.fy"]
            K[0, 2] = config["Camera1.cx"]
            K[1, 2] = config["Camera1.cy"]
            image_size = (config["Camera.width"]), (config["Camera.height"])

        distortion = np.zeros(8)
        depth_K = K.copy()

        return K, depth_K
    
    def get_frame_pose(self, idx: int) -> np.ndarray:
        return self.camtoworlds[idx, :]
    
    def load_tum_pose(self, path: str) -> np.ndarray:
        """Load ego pose from file.

        Args:
            path (str): Path to ego pose file.

        Returns:
            np.ndarray: Ego pose, tum format, ts tx ty tz qx qy qz qw
        """
        tum_pose_raw = np.loadtxt(path)
        # sort by ts
        tum_pose_raw = tum_pose_raw[tum_pose_raw[:, 0].argsort()]
        tum_pose = tum_pose_raw
        # transform to (n,4,4) matrix
        ts_list = []
        T_list = []
        for pose in tum_pose:
            # Extract translation and quaternion
            ts, tx, ty, tz, qx, qy, qz, qw = pose
            # Create rotation matrix from quaternion
            quat = [qx, qy, qz, qw]
            rot_matrix = R.from_quat(
                quat
            ).as_matrix()  # Convert quaternion to 3x3 rotation matrix
            # Create the homogeneous transformation matrix (4x4)
            T = np.eye(4)
            T[:3, :3] = rot_matrix  # Rotation part
            T[:3, 3] = [tx, ty, tz]  # Translation part

            # Append to list
            T_list.append(T)
            ts_list.append(ts)

        camtoworlds = np.array(T_list)
        return camtoworlds, ts_list
        
    def __getitem__(self, idx):
        """
        Get a data sample based on the given index.

        Args:
            idx: Index of the data sample.

        Returns:
            RGB image and depth image as numpy arrays.
        """
        # rgb_path, depth_path, pose_path = self.data_list[idx]
        image_id = self.indices[idx]
        # rgb_path = self.frames[image_id][""]
        # rgb_path = self.image_paths[image_id]
        # depth_path = self.depth_paths[image_id]
        rgb_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]

        pose = self.get_frame_pose(idx)
        # T_switch_axis = np.array([[1,0,0,-2.5],[0,0,1,0],[0,-1,0,2.5],[0,0,0,1]], dtype=np.float64) scannet0518
        T_switch_axis = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], dtype=np.float64)
        pose = T_switch_axis @ pose
        rgb_image = self._load_image(rgb_path)
        depth_image = self._load_depth(depth_path)
        # add depth clip preprocess
        depth = np.array(depth_image)
        clip_depth_mask = depth > 3.0*1000
        depth[clip_depth_mask] = 0
        # 计算梯度（Sobel 算子）
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        # 设置深度变化阈值（如 0.05 米）
        threshold = 0.1 * 1e3
        edge_mask = grad_mag > threshold
        depth[edge_mask] = 0

        depth_image = Image.fromarray(depth)

        if self.transforms is not None:
            # convert to Tensor
            rgb_image = self.transforms(rgb_image)
            depth_image = self.transforms(depth_image)
            
        return rgb_image, depth_image, pose, self.rgb_intrinsics, self.depth_intrinsics
    
    def _get_data_list(self):
        """
        Get a list of RGB-D data samples based on the dataset format and mode.

        Returns:
            List of RGB-D data samples (RGB image path, depth image path).
        """
        rgb_data_list = []
        depth_data_list = []
        pose_data_list = []
        pass

    def _load_image(self, path):
        """
        Load the RGB image from the given path.

        Args:
            path: Path to the RGB image file.

        Returns:
            RGB image as a numpy array.
        """
        # Load the RGB image using PIL
        rgb_image = Image.open(path)
        return rgb_image

    def _load_depth(self, path):
        """
        Load the depth image from the given path.

        Args:
            path: Path to the depth image file.

        Returns:
            Depth image as a numpy array.
        """
        # Load the depth image using OpenCV
        depth_image = Image.open(path)
        return depth_image
    

    def create_pcd(self, rgb, depth, camera_pose=None, idx = None, mask_img=False, filter_distance=np.inf):
        """
        Create Open3D point cloud from RGB and depth images, and camera pose. filter_distance is used to filter out
        points that are further than a certain distance.
        :param rgb (pil image): RGB image
        :param depth (pil image): Depth image
        :param camera_pose (np.array): Camera pose
        :param mask_img (bool): Mask image
        :param filter_distance (float): Filter distance
        :return: Open3D point cloud
        """
        # convert rgb and depth images to numpy arrays
        image_id = self.indices[idx]
        rgb = np.array(rgb).astype(np.uint8)
        depth = np.array(depth)
        # resize rgb image to match depth image size if needed
        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        # load depth camera intrinsics
        H = rgb.shape[0]
        W = rgb.shape[1]
        # camera_matrix = self.depth_intrinsics
        camera_matrix = np.array(self.frames[image_id-1]["K"]) # ? todo
        # print("image_id:", image_id)
        # print("camera_matrix: ", camera_matrix)
        scale = self.scale
        # create point cloud
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        depth = depth.astype(np.float32) / scale

        # # depth visualize
        # # 自动将深度图归一化到 [0, 255]
        # depth_norm = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # depth_norm = depth_norm.astype(np.uint8)
        # depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)  # 你也可以用 COLORMAP_TURBO 等
        # cv2.imshow(str(image_id), depth_colored)
        # cv2.waitKey(1)

        # depth 超过3m的全部置为0
        # clip_depth_mask = depth > 3.0
        # depth[clip_depth_mask] = 0
        if mask_img:
            depth = depth * rgb
        mask = depth > 0
        x = x[mask]
        y = y[mask]
        depth = depth[mask]
        # convert to 3D
        X = (x - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
        Y = (y - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
        Z = depth
        # 该帧相机坐标系的平均深度
        if Z.mean() > filter_distance:
            return o3d.geometry.PointCloud()
        # convert to open3d point cloud
        points = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if not mask_img:
            colors = rgb[mask]
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        # 相机系点云转换到世界坐标系
        pcd.transform(camera_pose)
        return pcd
    
    def create_3d_masks(self, masks, depth, full_pcd, full_pcd_tree, camera_pose, idx=None, down_size=0.02, filter_distance=None):
        """
        create 3d masks from 2D masks
        Args:
            masks: list of 2D masks
            depth: depth image
            full_pcd: full point cloud
            full_pcd_tree: KD-Tree of full point cloud
            camera_pose: camera pose
            down_size: voxel size for downsampling
        Returns:
            list of 3D masks as Open3D point clouds
        """
        pcd_list = []
        pcd = np.asarray(full_pcd.points)
        depth = np.array(depth)
        for i in range(len(masks)):
            # get the mask
            mask = masks[i]["segmentation"]
            mask = np.array(mask)
            # create pcd from mask
            pcd_masked = self.create_pcd(mask, depth, camera_pose, idx=idx, mask_img=True, filter_distance=filter_distance)
            # using KD-Tree to find the nearest points in the point cloud
            pcd_masked = np.asarray(pcd_masked.points)
            dist, indices = full_pcd_tree.query(pcd_masked, k=1, workers=-1)
            pcd_masked = pcd[indices]
            pcd_mask = o3d.geometry.PointCloud()
            pcd_mask.points = o3d.utility.Vector3dVector(pcd_masked)
            colors = np.asarray(full_pcd.colors)
            colors = colors[indices]
            pcd_mask.colors = o3d.utility.Vector3dVector(colors)
            pcd_mask = pcd_mask.voxel_down_sample(voxel_size=down_size)
            pcd_list.append(pcd_mask)
        return pcd_list
    
    