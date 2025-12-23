import os 
import cv2
import numpy as np
from PIL import Image
import torchvision
import open3d as o3d
import yaml
from hovsg.dataloader.generic import RGBDDataset
from scipy.spatial.transform import Rotation as R


class HorizonDataset(RGBDDataset):
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
        super(HorizonDataset, self).__init__(cfg)
        self.root_dir = cfg["root_dir"]
        self.transforms = cfg["transforms"]
        self.depth_cut = float(cfg["depth_cut"])
        # pose_name = "colmap_pose"
        # camera_config_path = "orbslam3_rgbd.yaml"
        pose_name = "poses"
        camera_config_path = "d435i.yaml"
        self.rgb_intrinsics, self.depth_intrinsics = self.load_camera_params(os.path.join(self.root_dir, camera_config_path))
        self.scale = 1000.0
        print("self.root_dir: ", self.root_dir)
        if pose_name is not None and os.path.exists(
            os.path.join(self.root_dir, f"{pose_name}.txt")
        ):
            print("use pose file: ", pose_name)
            camtoworlds, ts_list = self.load_tum_pose_w2c(
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
        
        self.camtoworlds = camtoworlds
        self.indices = np.arange(len(self.ts_list))

        # re-range ts_list for abs ts
        # self.ts_list = [str(int(ts)) for ts in range(len(ts_list))]
        # self.ts_list = self.ts_list[self.start_index : self.end_index]
        if int(ts_list[0]) != ts_list[0]:
            self.image_paths = [
                        os.path.join(self.root_dir, "images", f"{float(ts):.4f}.png")
                        for ts in self.ts_list
                    ]
            self.depth_paths = [
                        os.path.join(self.root_dir, "depth", f"{float(ts):.4f}.png")
                        for ts in self.ts_list
                    ]
        else:
            self.image_paths = [
                        os.path.join(self.root_dir, "color", f"{int(ts):05d}.png")
                        for ts in self.ts_list
                    ]
            self.depth_paths = [
                        os.path.join(self.root_dir, "depth", f"{int(ts):05d}.png")
                        for ts in self.ts_list
                    ]
        self.frameId2imgPath = self.image_paths
        print("horizon dataset init success!")

    def __len__(self):
        return len(self.indices)
    
    def get_camera_intrinsics(self):
        return self.depth_intrinsics
    
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
    
    def load_tum_pose_w2c(self, path: str) -> np.ndarray:
        """
        Load ego pose from file.

        Args:
            path (str): Path to ego pose file.

        Returns:
            np.ndarray: Ego pose, tum format, ts tx ty tz qx qy qz qw
        """
        tum_pose_raw = np.loadtxt(path)
        # sort by ts
        tum_pose_raw = tum_pose_raw[tum_pose_raw[:, 0].argsort()]
        # pt = PoseTransformer()
        # pt.loadarray(tum_pose_raw)
        # pt.normalize2origin()
        # tum_pose = pt.dumparray()
        tum_pose = tum_pose_raw
        # transform to (n,4,4) matrix
        ts_list = []
        T_list = []
        for pose in tum_pose:
            # Extract translation and quaternion
            ts, tx, ty, tz, qx, qy, qz, qw = pose
            # ts, tx, ty, tz, qw, qx, qy, qz = pose
            # Create rotation matrix from quaternion
            quat = [qx, qy, qz, qw]
            rot_matrix = R.from_quat(
                quat
            ).as_matrix()  # Convert quaternion to 3x3 rotation matrix
            # Create the homogeneous transformation matrix (4x4)
            T = np.eye(4)
            T[:3, :3] = rot_matrix  # Rotation part
            T[:3, 3] = [tx, ty, tz]  # Translation part
            c2w = np.linalg.inv(T)
            # c2w = T

            # Append to list
            T_list.append(c2w)
            ts_list.append(ts)

        camtoworlds = np.array(T_list)
        return camtoworlds, ts_list

    def load_tum_pose(self, path: str) -> np.ndarray:
        """
        Load ego pose from file.

        Args:
            path (str): Path to ego pose file.

        Returns:
            np.ndarray: Ego pose, tum format, ts tx ty tz qx qy qz qw
        """
        tum_pose_raw = np.loadtxt(path)
        # sort by ts
        tum_pose_raw = tum_pose_raw[tum_pose_raw[:, 0].argsort()]
        # pt = PoseTransformer()
        # pt.loadarray(tum_pose_raw)
        # pt.normalize2origin()
        # tum_pose = pt.dumparray()
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
        rgb_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]

        pose = self.get_frame_pose(idx)
        # T_switch_axis = np.array([[1,0,0,-2.5],[0,0,1,0],[0,-1,0,2.5],[0,0,0,1]], dtype=np.float64) # scannet0518
        # T_switch_axis = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], dtype=np.float64) # kitchen
        # T_switch_axis = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float64) # go2_navi
        T_switch_axis = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], dtype=np.float64) # g1_navi fastlivo2
        pose = T_switch_axis @ pose
        rgb_image = self._load_image(rgb_path)
        depth_image = self._load_depth(depth_path)

        # # # use sensor-depth, add depth clip preprocess
        # depth = np.array(depth_image)
        # clip_depth_mask = depth > 3.0*1000
        # depth[clip_depth_mask] = 0
        # # 计算梯度（Sobel 算子）
        # grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        # grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        # grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        # # 设置深度变化阈值（如 0.05 米）
        # threshold = 0.1 * 1e3
        # edge_mask = grad_mag > threshold
        # depth[edge_mask] = 0
        # depth_image = Image.fromarray(depth)

        #use fastlivo2 depth, add depth clip preprocess
        depth = np.array(depth_image)
        clip_depth_mask = depth > self.depth_cut*1000
        depth[clip_depth_mask] = 0
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
    

    def create__pcd(self, rgb, depth, camera_pose=None):
        """
        This method should be implemented by subclasses to create a point cloud 
        from RGB-D images.
        """
        rgb = np.array(rgb)
        depth = np.array(depth)
        rgb = np.array(Image.fromarray(rgb).resize((depth.shape[1], depth.shape[0])))
        depth_scale = 1000.0
        camera_matrix = self.depth_intrinsics
        depth_img = depth.astype(np.float32) / depth_scale
        x, y = np.meshgrid(np.arange(depth_img.shape[1]), np.arange(depth_img.shape[0]))
        mask = depth_img > 0 
        x = x[mask]
        y = y[mask]
        depth_img = depth_img[mask]
        X = (x - camera_matrix[0, 2]) * depth_img / camera_matrix[0, 0]
        Y = (y - camera_matrix[1, 2]) * depth_img / camera_matrix[1, 1]
        Z = depth_img
        pcd = np.hstack(([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1), np.ones((X.shape[0], 1))]))

        # apply projection matrix
        if camera_pose is not None:
            pcd = np.dot(camera_pose, pcd.T).T
            pcd = pcd[:, :3] / pcd[:, 3:]
        colors = rgb.reshape(-1, 3) / 255
        colors = colors[mask.reshape(-1)]
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcd)
        pcd1.colors = o3d.utility.Vector3dVector(colors)
        return pcd1


