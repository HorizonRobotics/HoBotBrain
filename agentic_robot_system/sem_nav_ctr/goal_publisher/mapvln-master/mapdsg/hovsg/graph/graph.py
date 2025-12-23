"""
    Class to represent the HOV-SG graph
"""
from copy import deepcopy
from datetime import datetime
import os
import copy
from typing import Any, Dict, List, Set, Tuple, Union
from pathlib import Path

import cv2
from omegaconf import DictConfig
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')  # 设置为非GUI后端
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from scipy.ndimage import binary_erosion, median_filter
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d

from sklearn.cluster import DBSCAN
from tqdm import tqdm
import networkx as nx

import json
import torch
import open_clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from hovsg.graph.object import Object
from hovsg.graph.room import Room
from hovsg.graph.view import View
from hovsg.graph.floor import Floor

from hovsg.dataloader.hm3dsem import HM3DSemDataset
from hovsg.dataloader.scannet import ScannetDataset
from hovsg.dataloader.horizon import HorizonDataset
from hovsg.dataloader.iphone import IPhoneDataset
from hovsg.dataloader.replica import ReplicaDataset

from hovsg.utils.clip_utils import get_img_feats, get_text_feats_multiple_templates
from hovsg.models.sam_clip_feats_extractor import extract_feats_per_pixel
from hovsg.utils.graph_utils import (
    seq_merge,
    pcd_denoise_dbscan,
    pcd_denoise_statistical,
    pcd_denoise_dbscan_vis,
    feats_denoise_dbscan,
    distance_transform,
    map_grid_to_point_cloud,
    compute_room_embeddings,
    find_intersection_share,
    hierarchical_merge,
    check_object_in_view,
    visualize_pcd_on_image,
)

from hovsg.graph.navigation_graph import NavigationGraph
from hovsg.utils.label_feats import get_label_feats
from hovsg.utils.constants import MATTERPORT_GT_LABELS, CLIP_DIM
from hovsg.utils.llm_utils import (
    parse_floor_room_object_gpt35,
    parse_hier_query,
    parse_hier_query_use_prompt_insentence_parse,
    parse_hier_query_use_prompt_insentence_parse_icra,
    infer_floor_id_from_query,
)

from functools import lru_cache
import os
import re
import time
from typing import Dict, List, Tuple, Union
import openai
from openai import AzureOpenAI, OpenAI
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

# pylint: disable=all


class Graph:
    """
    Class to represent the HOV-SG graph
    :param cfg: Config file
    :param inf_params: Inference parameters
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.full_pcd = o3d.geometry.PointCloud()
        self.mask_feats = []
        self.mask_feats_d = []
        self.mask_pcds = []
        self.mask_weights = []
        self.objects = []
        self.rooms = []
        self.floors = []
        self.views = []
        self.full_feats_array = []
        self.graph = nx.Graph()
        self.graph.add_node(0, name="building", type="building")
        self.room_masks = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # load CLIP model
        if self.cfg.models.clip.type == "ViT-L/14@336px":
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14",
                pretrained=str(self.cfg.models.clip.checkpoint),
                device=self.device,
            )
            self.clip_feat_dim = CLIP_DIM["ViT-L-14"]
            # self.clip_feat_dim = constants.clip_feat_dim[self.cfg.models.clip.type]
        elif self.cfg.models.clip.type == "ViT-L/14":
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14",
                pretrained=str(self.cfg.models.clip.checkpoint),
                device=self.device,
            )
            self.clip_feat_dim = CLIP_DIM["ViT-L-14"]
        elif self.cfg.models.clip.type == "ViT-H-14":
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14",
                pretrained=str(self.cfg.models.clip.checkpoint),
                device=self.device,
            )
            self.clip_feat_dim = CLIP_DIM["ViT-H-14"]
        elif self.cfg.models.clip.type == "ViT-B-32":
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained=str(self.cfg.models.clip.checkpoint),
                device=self.device,
                precision='fp16',
            )
            self.clip_feat_dim = CLIP_DIM["ViT-B-32"]
        
        self.clip_model.eval()
        
       
        if self.cfg.main.use_gpt:
            self.graph_path = self.cfg.main.graph_path
            # add using llm interface
            azure_endpoint="xxxx"
            azure_api_key="xxxx"
            azure_api_version="xxxx"
            self.gpt_model="xxxx"

            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
                )
            
            # load the dataset
            dataset_cfg = {"root_dir": self.cfg.main.dataset_path, "transforms": None, "depth_cut": self.cfg.main.depth_cut}
            # import pdb; pdb.set_trace()
            if self.cfg.main.dataset == "hm3dsem":
                self.dataset = HM3DSemDataset(dataset_cfg)
            elif self.cfg.main.dataset == "scannet":
                self.dataset = ScannetDataset(dataset_cfg)
            elif self.cfg.main.dataset == "horizon":
                self.dataset = HorizonDataset(dataset_cfg)
            elif self.cfg.main.dataset == "iphone":
                self.dataset = IPhoneDataset(dataset_cfg)
            elif self.cfg.main.dataset == "replica":
                self.dataset = ReplicaDataset(dataset_cfg)
            else:
                print("Dataset not supported")
                return
                
            self.graph_tmp_folder = os.path.join(cfg.main.save_path, "tmp")
            if not os.path.exists(self.graph_tmp_folder):
                os.makedirs(self.graph_tmp_folder)

            self.vln_result_dir = os.path.join(cfg.main.save_path, "vln_result_presentation")
            if not os.path.exists(self.vln_result_dir):
                os.makedirs(self.vln_result_dir)
            self.curr_query_save_dir = self.vln_result_dir
            if not hasattr(self.cfg, "pipeline"):
                print("-- entering querying and evaluation mode")
                return

        else:    
            self.graph_tmp_folder = os.path.join(cfg.main.save_path, "tmp")
            if not os.path.exists(self.graph_tmp_folder):
                os.makedirs(self.graph_tmp_folder)

            self.vln_result_dir = os.path.join(cfg.main.save_path, "vln_result_presentation")
            if not os.path.exists(self.vln_result_dir):
                os.makedirs(self.vln_result_dir)

            self.curr_query_save_dir = self.vln_result_dir
            if not hasattr(self.cfg, "pipeline"):
                print("-- entering querying and evaluation mode")
                return


            # load the SAM model
            model_type = self.cfg.models.sam.type
            self.sam = sam_model_registry[model_type](
                checkpoint=str(self.cfg.models.sam.checkpoint)
            )
            self.sam.to(device=self.device)
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=self.cfg.models.sam.points_per_side,
                pred_iou_thresh=self.cfg.models.sam.pred_iou_thresh,
                points_per_batch=self.cfg.models.sam.points_per_batch,
                stability_score_thresh=self.cfg.models.sam.stability_score_thresh,
                crop_n_layers=self.cfg.models.sam.crop_n_layers,
                min_mask_region_area=self.cfg.models.sam.min_mask_region_area,
            )
            self.sam.eval()
            
            # load the dataset
            dataset_cfg = {"root_dir": self.cfg.main.dataset_path, "transforms": None, "depth_cut": self.cfg.main.depth_cut}
            if self.cfg.main.dataset == "hm3dsem":
                self.dataset = HM3DSemDataset(dataset_cfg)
            elif self.cfg.main.dataset == "scannet":
                self.dataset = ScannetDataset(dataset_cfg)
            elif self.cfg.main.dataset == "horizon":
                self.dataset = HorizonDataset(dataset_cfg)
            elif self.cfg.main.dataset == "iphone":
                self.dataset = IPhoneDataset(dataset_cfg)
            elif self.cfg.main.dataset == "replica":
                self.dataset = ReplicaDataset(dataset_cfg)
            else:
                print("Dataset not supported")
                return
            
        
        
    def generate_object_querys(self, instruction):    
        # prompt = f"""
        # You are an AI assistant for visual navigation.  
        # Given a navigation instruction, extract the main target object(s) mentioned or implied.
        # If the instruction does not explicitly mention an object, infer the most likely target object(s) based on common sense and the user's intent.  
        # Generate a diverse bullet list of English phrases for CLIP-based image retrieval, including synonyms and descriptive variants.  
        # If no clear object is mentioned, output an empty list.

        # Instruction: {instruction}
        # """
        prompt = f"""
        You are an AI assistant for visual navigation, and your name is Digua. Please ignore all occurrences of the word Digua in the input instructions, as they do not represent navigation targets.
        Given a navigation instruction, extract the main target object(s) mentioned or implied.
        If the instruction does not explicitly mention an object, infer the most likely target object(s) based on common sense and the user's intent.
        Generate a diverse bullet list of English phrases for CLIP-based image retrieval, including synonyms and descriptive variants.
        If no clear object is mentioned, output an empty list.
        
        Instruction: {instruction}
        """

        response_flag = False
        while not response_flag:
            try:
                print("Sending request stage 1 ...")
                response = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    seed=123,
                )
                response_flag = True
            except Exception as e:
                print(e)
                time.sleep(1)
                print("Retrying ...")
        response = response.choices[0].message.content
        text_probes = re.findall(r"-(.*?)\n", response)
        text_probes = [item.strip(' "-') for item in text_probes]
        text_probes = [item for item in text_probes if len(item) > 0]
        return text_probes

    def create_feature_map(self, save_path=None):
        """
        Create the feature map of the HOV-SG (full point cloud + feature map point level + feature map mask level)
        :param save_path : str, optional, The path to save the feature map
        """

        if self.dataset is None:
            print("No dataset loaded")
            return

        # save images for mobilityvln
        # Directory and video setup
        # # Video writer setup
        self.run_mobilityvln = False
        if self.run_mobilityvln:
            from scipy.spatial.transform import Rotation as R
            now_str = datetime.now().strftime("%Y%m%d%H%M%S")
            self.episode_root = f"/home/unitree/code_vln/results/{now_str}"
            os.makedirs(self.episode_root, exist_ok=True)
            self.colmap_data_dir = os.path.join(self.episode_root, "colmap_data")
            self.colmap_img_dir = os.path.join(self.colmap_data_dir, "images")
            self.colmap_depth_dir = os.path.join(self.colmap_data_dir, "depth")
            os.makedirs(self.colmap_data_dir, exist_ok=True)
            os.makedirs(self.colmap_img_dir, exist_ok=True)
            os.makedirs(self.colmap_depth_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                os.path.join(self.episode_root, "output_video.mp4"),
                fourcc,
                5,
                (640, 480))  
            self.camera_id = 1
            self.frame_count = 0
            self.fp = open(os.path.join(self.colmap_data_dir, "images.txt"), "w+")
            self.T_switch_axis = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], dtype=np.float64) # g1_navi
            self.T_tomap = np.linalg.inv(self.T_switch_axis)
            for i in tqdm(range(0, len(self.dataset), self.cfg.pipeline.skip_frames), desc="Generating tourvideo for mobilityvln"):
                rgb_image, depth_image, pose, _, depth_intrinsics = self.dataset[i]
                pose = self.T_tomap @ pose
                rgb = np.array(rgb_image)
                depth = np.array(depth_image)
                img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                # 保存帧到视频
                self.video_writer.write(rgb)            
                # 保存图像
                image_name = f"image{self.frame_count:05}.png"
                depth_name = f"depth{self.frame_count:05}.png"                                
                cv2.imwrite(os.path.join(self.colmap_img_dir, image_name), img_bgr)
                cv2.imwrite(os.path.join(self.colmap_depth_dir, depth_name), depth)
                #  Extract pose information
                tx, ty, tz = pose[0, 3], pose[1, 3], pose[2, 3]
                rot_mat = pose[:3, :3]           
                r = R.from_matrix(rot_mat) # 转四元数（格式：x, y, z, w）
                qx, qy, qz, qw = r.as_quat()
                # qw = self.current_pose.orientation.w
                # qx = self.current_pose.orientation.x
                # qy = self.current_pose.orientation.y
                # qz = self.current_pose.orientation.z
                # tx = self.current_pose.position.x
                # ty = self.current_pose.position.y
                # tz = self.current_pose.position.z
                r = R.from_quat([qx, qy, qz, qw])
                yaw, pitch, roll = r.as_euler("zyx", degrees=False)
                # Write pose to file
                self.fp.write(f"{self.frame_count} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {yaw} {self.camera_id} {image_name}\n\n")
                # self.fp.write(f"{self.frame_count} {tx} {ty} {tz} {yaw} {image_name}\n\n")
                self.frame_count += 1        

        # import pdb; pdb.set_trace()
        # exit(1)
            
        # create the RGB-D point cloud
        for i in tqdm(range(0, len(self.dataset), self.cfg.pipeline.skip_frames), desc="Creating RGB-D point cloud"):
            rgb_image, depth_image, pose, _, depth_intrinsics = self.dataset[i]
            self.full_pcd += self.dataset.create_pcd(rgb_image, depth_image, pose, idx=i)

        # filter point cloud
        self.full_pcd = self.full_pcd.voxel_down_sample(
            voxel_size=self.cfg.pipeline.voxel_size
        )
        # self.full_pcd = pcd_denoise_dbscan_vis(self.full_pcd, eps=0.05, min_points=50, visualize=True)
        self.full_pcd = pcd_denoise_dbscan(self.full_pcd, eps=0.01, min_points=100)
        # self.full_pcd = pcd_denoise_statistical(self.full_pcd)
        cl, ind = self.full_pcd.remove_radius_outlier(nb_points=1000, radius=1.0) # 0.05, 
        inlier_cloud = self.full_pcd.select_by_index(ind)
        self.full_pcd = inlier_cloud
        self.save_full_pcd(path=self.cfg.main.save_path)
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # exit(1)

        # create tree from full point cloud
        locs_in = np.array(self.full_pcd.points)
        print("full_pcd point num: ", locs_in.shape)
        tree_pcd = cKDTree(locs_in)
        n_points = locs_in.shape[0]
        counter = torch.zeros((n_points, 1), device="cpu")
        sum_features = torch.zeros((n_points, self.clip_feat_dim), device="cpu")

        # extract features for each frame
        frames_pcd = []
        frames_feats = []
        for i in tqdm(range(0, len(self.dataset), self.cfg.pipeline.skip_frames), desc="Extracting features"):
            rgb_image, depth_image, pose, _, _ = self.dataset[i]
            if rgb_image.size != depth_image.size:
                rgb_image = rgb_image.resize(depth_image.size)
            F_2D, F_masks, masks, F_g = extract_feats_per_pixel(
                np.array(rgb_image),
                self.mask_generator,
                self.clip_model,
                self.preprocess,
                clip_feat_dim=self.clip_feat_dim,
                bbox_margin=self.cfg.pipeline.clip_bbox_margin,
                maskedd_weight=self.cfg.pipeline.clip_masked_weight,
            )
            F_2D = F_2D.cpu()
            pcd = self.dataset.create_pcd(rgb_image, depth_image, pose, idx=i)
            masks_3d = self.dataset.create_3d_masks(
                masks,
                depth_image,
                self.full_pcd,
                tree_pcd,
                pose,
                i,
                down_size=self.cfg.pipeline.voxel_size,
                filter_distance=self.cfg.pipeline.max_mask_distance,
            )
            frames_pcd.append(masks_3d)
            frames_feats.append(F_masks)
            # fuse features for each point in the full pcd
            mask = np.array(depth_image) > 0
            mask = torch.from_numpy(mask)
            F_2D = F_2D[mask]
            # using cKdtree to find the closest point in the full pcd for each point in frame pcd
            dis, idx = tree_pcd.query(np.asarray(pcd.points), k=1, workers=-1)
            sum_features[idx] += F_2D
            counter[idx] += 1
        # compute the average features
        counter[counter == 0] = 1e-5
        sum_features = sum_features / counter
        self.full_feats_array = sum_features.cpu().numpy()
        self.full_feats_array: np.ndarray

        print("self.full_feats_array.shape : ", self.full_feats_array.shape)
        
        # free memory
        del sum_features, counter
        torch.cuda.empty_cache() 

        # merging the masks
        if self.cfg.pipeline.merge_type == "hierarchical":
            tqdm.write("Merging 3d masks hierarchically")
            self.mask_pcds = hierarchical_merge(
                frames_pcd, 
                self.cfg.pipeline.init_overlap_thresh, 
                self.cfg.pipeline.overlap_thresh_factor, 
                self.cfg.pipeline.voxel_size, 
                self.cfg.pipeline.iou_thresh,
            )
        elif self.cfg.pipeline.merge_type == "sequential":
            tqdm.write("Merging 3d masks sequentially") 
            self.mask_pcds = seq_merge(
                frames_pcd, 
                self.cfg.pipeline.init_overlap_thresh, 
                self.cfg.pipeline.voxel_size, 
                self.cfg.pipeline.iou_thresh
            )

        # remove any small pcds
        # for i, pcd in enumerate(self.mask_pcds):
        for i, pcd in reversed(list(enumerate(self.mask_pcds))):
            # if pcd.is_empty() or len(pcd.points) < 100:
            if pcd.is_empty() or len(pcd.points) < 10:
                self.mask_pcds.pop(i)
        # fuse point features in every 3d mask
        # self.mask_pcds, finally merged 3d instances
        masks_feats = []
        for i, mask_3d in tqdm(enumerate(self.mask_pcds), desc="Fusing features"):
            # find the points in the mask
            # mask_3d = mask_3d.voxel_down_sample(self.cfg.pipeline.voxel_size * 2)
            mask_3d = mask_3d.voxel_down_sample(self.cfg.pipeline.voxel_size)
            points = np.asarray(mask_3d.points)
            dist, idx = tree_pcd.query(points, k=1, workers=-1)
            # 根据距离阈值过滤掉“太远”的点
            valid_mask = dist <= 0.8
            n_total = len(points)
            n_valid = int(valid_mask.sum())
            n_removed = n_total - n_valid
            if n_removed > 0:
                tqdm.write(f"mask {i}: removed {n_removed}/{n_total} points with dist > {0.1}")
            # if n_valid == 0:
            #     # 全部点都太远，按原逻辑插入一个零向量
            #     masks_feats.append(
            #         np.zeros((1, self.clip_feat_dim), dtype=self.full_feats_array.dtype)
            #     )
            #     continue
             # 只保留有效的索引
            valid_idx = idx[valid_mask]
            feats = self.full_feats_array[valid_idx]   # shape = (n_valid, clip_feat_dim)
            feats = np.nan_to_num(feats)
            # feats = self.full_feats_array[idx]
            # feats = np.nan_to_num(feats)
            # filter feats with dbscan
            if feats.shape[0] == 0:
                masks_feats.append(
                    np.zeros((1, self.clip_feat_dim), dtype=self.full_feats_array.dtype)
                )
                continue
            feats = feats_denoise_dbscan(feats, eps=0.01, min_points=100)
            # feats = feats_denoise_dbscan(feats, eps=1.0, min_points=50) # set one single feature-vector for each merged-3d-segment
            # print("feats.shape: ", feats.shape)
            masks_feats.append(feats)
        self.mask_feats = masks_feats
        print("number of masks: ", len(self.mask_feats))
        print("number of pcds in hovsg: ", len(self.mask_pcds))
        assert len(self.mask_pcds) == len(self.mask_feats)

    def segment_floors(self, path, flip_zy=False):
        """
        Segment the floors from the full point cloud
        :param path: str, The path to save the intermediate results
        """
        # downsample the point cloud
        downpcd = self.full_pcd.voxel_down_sample(voxel_size=0.05)
        # flip the z and y axis
        if flip_zy:
            downpcd.points = o3d.utility.Vector3dVector(
                np.array(downpcd.points)[:, [0, 2, 1]]
            )
            downpcd.transform(np.eye(4) * np.array([1, 1, -1, 1]))
        # rotate the point cloud to align floor with the y axis
        T1 = np.eye(4)
        T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        downpcd = np.asarray(downpcd.points)
        print("downpcd", downpcd.shape)

        # divide z axis range into 0.01m bin
        reselotion = 0.01
        bins = np.abs(np.max(downpcd[:, 1]) - np.min(downpcd[:, 1])) / reselotion
        print("min, max",np.min(downpcd[:, 1]), np.max(downpcd[:, 1]))
        print("bins", bins)
        z_hist = np.histogram(downpcd[:, 1], bins=int(bins))
        # smooth the histogram
        z_hist_smooth = gaussian_filter1d(z_hist[0], sigma=2)
        # Find the peaks in this histogram.
        distance = 0.2 / reselotion
        print("distance", distance)
        # set the min peak height based on the histogram
        print(np.mean(z_hist_smooth))
        min_peak_height = np.percentile(z_hist_smooth, 90)
        print("min_peak_height", min_peak_height)
        peaks, _ = find_peaks(z_hist_smooth, distance=distance, height=min_peak_height)

        # plot the histogram
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(
                min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r"
            )
            plt.savefig(os.path.join(self.graph_tmp_folder, "floor_histogram.png"))

        # cluster the peaks using DBSCAN
        peaks_locations = z_hist[1][peaks]
        clustering = DBSCAN(eps=1, min_samples=1).fit(peaks_locations.reshape(-1, 1))
        labels = clustering.labels_

        # plot the histogram
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(
                min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r"
            )
            # plot the clusters
            for i in range(len(np.unique(labels))):
                plt.plot(
                    z_hist[1][peaks[labels == i]],
                    z_hist_smooth[peaks[labels == i]],
                    "o",
                )
            plt.savefig(
                os.path.join(self.graph_tmp_folder, "floor_histogram_cluster.png")
            )

        # for each cluster find the top 2 peaks
        clustred_peaks = []
        for i in range(len(np.unique(labels))):
            # for first and last cluster, find the top 1 peak
            if i == 0 or i == len(np.unique(labels)) - 1:
                p = peaks[labels == i]
                top_p = p[np.argsort(z_hist_smooth[p])[-1:]].tolist()
                top_p = [z_hist[1][p] for p in top_p]
                clustred_peaks.append(top_p)
                continue
            p = peaks[labels == i]
            top_p = p[np.argsort(z_hist_smooth[p])[-2:]].tolist()
            top_p = [z_hist[1][p] for p in top_p]
            clustred_peaks.append(top_p)
        clustred_peaks = [item for sublist in clustred_peaks for item in sublist]
        clustred_peaks = np.sort(clustred_peaks)
        print("clustred_peaks", clustred_peaks)

        floors = []
        # for every two consecutive peaks with 2m distance, assign floor level
        for i in range(0, len(clustred_peaks) - 1, 2):
            floors.append([clustred_peaks[i], clustred_peaks[i + 1]])
            print("computed floors: ", floors)
        if not floors:
            floors.append([z_hist[1].min().item(), z_hist[1].max().item()])
            print("priors floors", floors)
        # for the first floor extend the floor to the ground
        floors[0][0] = (floors[0][0] + np.min(downpcd[:, 1])) / 2
        # for the last floor extend the floor to the ceiling
        floors[-1][1] = (floors[-1][1] + np.max(downpcd[:, 1])) / 2
        # floors[-1][1] = np.max(downpcd[:, 1])
        print("number of floors: ", len(floors))
        

        floors_pcd = []
        for i, floor in enumerate(floors):
            floor_obj = Floor(str(i), name="floor_" + str(i))
            floor_pcd = self.full_pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(-np.inf, floor[0], -np.inf),
                    max_bound=(np.inf, floor[1], np.inf),
                )
            )
            bbox = floor_pcd.get_axis_aligned_bounding_box()
            floor_obj.vertices = np.asarray(bbox.get_box_points())
            floor_obj.pcd = floor_pcd
            floor_obj.floor_zero_level = np.min(np.array(floor_pcd.points)[:, 1])
            floor_obj.floor_height = floor[1] - floor_obj.floor_zero_level
            self.floors.append(floor_obj)
            floors_pcd.append(floor_pcd)
        print("final floors: ", floors)
        return floors
    
    def segment_floors_new(self, path, flip_zy=False):
        """
        Segment the floors from the full point cloud
        :param path: str, The path to save the intermediate results
        """
        # downsample the point cloud
        downpcd = self.full_pcd.voxel_down_sample(voxel_size=0.05)
        # flip the z and y axis
        if flip_zy:
            downpcd.points = o3d.utility.Vector3dVector(
                np.array(downpcd.points)[:, [0, 2, 1]]
            )
            downpcd.transform(np.eye(4) * np.array([1, 1, -1, 1]))
        # rotate the point cloud to align floor with the y axis
        T1 = np.eye(4)
        T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        downpcd = np.asarray(downpcd.points)
        print("downpcd", downpcd.shape)

        # divide z axis range into 0.01m bin
        reselotion = 0.01
        bins = np.abs(np.max(downpcd[:, 1]) - np.min(downpcd[:, 1])) / reselotion
        print("min, max",np.min(downpcd[:, 1]), np.max(downpcd[:, 1]))
        print("bins", bins)
        z_hist = np.histogram(downpcd[:, 1], bins=int(bins))
        # smooth the histogram
        z_hist_smooth = gaussian_filter1d(z_hist[0], sigma=2)
        # Find the peaks in this histogram.
        distance = 0.2 / reselotion
        print("distance", distance)
        # set the min peak height based on the histogram
        print(np.mean(z_hist_smooth))
        min_peak_height = np.percentile(z_hist_smooth, 90)
        print("min_peak_height", min_peak_height)
        peaks, _ = find_peaks(z_hist_smooth, distance=distance, height=min_peak_height)

        # plot the histogram
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(
                min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r"
            )
            plt.savefig(os.path.join(self.graph_tmp_folder, "floor_histogram.png"))

        # cluster the peaks using DBSCAN
        peaks_locations = z_hist[1][peaks]
        clustering = DBSCAN(eps=1, min_samples=1).fit(peaks_locations.reshape(-1, 1))
        labels = clustering.labels_

        # plot the histogram
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(
                min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r"
            )
            # plot the clusters
            for i in range(len(np.unique(labels))):
                plt.plot(
                    z_hist[1][peaks[labels == i]],
                    z_hist_smooth[peaks[labels == i]],
                    "o",
                )
            plt.savefig(
                os.path.join(self.graph_tmp_folder, "floor_histogram_cluster.png")
            )

        # for each cluster find the top 2 peaks
        clustred_peaks = []
        for i in range(len(np.unique(labels))):
            # for first and last cluster, find the top 1 peak
            if i == 0 or i == len(np.unique(labels)) - 1:
                p = peaks[labels == i]
                top_p = p[np.argsort(z_hist_smooth[p])[-1:]].tolist()
                top_p = [z_hist[1][p] for p in top_p]
                clustred_peaks.append(top_p)
                continue
            p = peaks[labels == i]
            top_p = p[np.argsort(z_hist_smooth[p])[-2:]].tolist()
            top_p = [z_hist[1][p] for p in top_p]
            clustred_peaks.append(top_p)
        clustred_peaks = [item for sublist in clustred_peaks for item in sublist]
        clustred_peaks = np.sort(clustred_peaks)
        print("clustred_peaks", clustred_peaks)

        floors = []
        # for every two consecutive peaks with 2m distance, assign floor level
        for i in range(0, len(clustred_peaks) - 1, 2):
            floors.append([clustred_peaks[i], clustred_peaks[i + 1]])
            print("computed floors: ", floors)
        if not floors:
            floors.append([z_hist[1].min().item(), z_hist[1].max().item()])
            print("priors floors", floors)
        # for the first floor extend the floor to the ground
        floors[0][0] = (floors[0][0] + np.min(downpcd[:, 1])) / 2
        # for the last floor extend the floor to the ceiling
        floors[-1][1] = (floors[-1][1] + np.max(downpcd[:, 1])) / 2
        # floors[-1][1] = np.max(downpcd[:, 1])
        print("number of floors: ", len(floors))
        

        floors_pcd = []
        for i, floor in enumerate(floors):
            floor_obj = Floor(str(i), name="floor_" + str(i))
            floor_pcd = self.full_pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(-np.inf, floor[0], -np.inf),
                    max_bound=(np.inf, floor[1], np.inf),
                )
            )
            bbox = floor_pcd.get_axis_aligned_bounding_box()
            floor_obj.vertices = np.asarray(bbox.get_box_points())
            floor_obj.pcd = floor_pcd
            floor_obj.floor_zero_level = np.min(np.array(floor_pcd.points)[:, 1])
            floor_obj.floor_height = floor[1] - floor_obj.floor_zero_level
            self.floors.append(floor_obj)
            floors_pcd.append(floor_pcd)
        print("final floors: ", floors)
        return floors
       
    def segment_floors_manually(self, path, flip_zy=False):
        """
        Segment the floors from the full point cloud
        :param path: str, The path to save the intermediate results
        """
        # downsample the point cloud
        # downpcd = o3d.io.read_point_cloud(path).voxel_down_sample(voxel_size=0.05)
        downpcd = self.full_pcd.voxel_down_sample(voxel_size=0.05)
        # flip the z and y axis
        if flip_zy:
            downpcd.points = o3d.utility.Vector3dVector(
                np.array(downpcd.points)[:, [0, 2, 1]]
            )
            downpcd.transform(np.eye(4) * np.array([1, 1, -1, 1]))
        # rotate the point cloud to align floor with the y axis
        T1 = np.eye(4)
        T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        downpcd = np.asarray(downpcd.points)
        print("downpcd", downpcd.shape)

        # divide z axis range into 0.01m bin
        reselotion = 0.01
        bins = np.abs(np.max(downpcd[:, 1]) - np.min(downpcd[:, 1])) / reselotion
        print("min, max",np.min(downpcd[:, 1]), np.max(downpcd[:, 1]))
        print("bins", bins)
        z_hist = np.histogram(downpcd[:, 1], bins=int(bins))
        # smooth the histogram
        z_hist_smooth = gaussian_filter1d(z_hist[0], sigma=2)
        # Find the peaks in this histogram.
        distance = 0.2 / reselotion
        print("distance", distance)
        # set the min peak height based on the histogram
        print(np.mean(z_hist_smooth))
        min_peak_height = np.percentile(z_hist_smooth, 90)
        print("min_peak_height", min_peak_height)
        peaks, _ = find_peaks(z_hist_smooth, distance=distance, height=min_peak_height)

        # plot the histogram
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(
                min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r"
            )
            plt.savefig(os.path.join(self.graph_tmp_folder, "floor_histogram.png"))

        # cluster the peaks using DBSCAN
        peaks_locations = z_hist[1][peaks]
        clustering = DBSCAN(eps=1, min_samples=1).fit(peaks_locations.reshape(-1, 1))
        labels = clustering.labels_

        # plot the histogram
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(
                min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r"
            )
            # plot the clusters
            for i in range(len(np.unique(labels))):
                plt.plot(
                    z_hist[1][peaks[labels == i]],
                    z_hist_smooth[peaks[labels == i]],
                    "o",
                )
            plt.savefig(
                os.path.join(self.graph_tmp_folder, "floor_histogram_cluster.png")
            )

        # for each cluster find the top 2 peaks
        clustred_peaks = []
        for i in range(len(np.unique(labels))):
            # for first and last cluster, find the top 1 peak
            if i == 0 or i == len(np.unique(labels)) - 1:
                p = peaks[labels == i]
                top_p = p[np.argsort(z_hist_smooth[p])[-1:]].tolist()
                top_p = [z_hist[1][p] for p in top_p]
                clustred_peaks.append(top_p)
                continue
            p = peaks[labels == i]
            top_p = p[np.argsort(z_hist_smooth[p])[-2:]].tolist()
            top_p = [z_hist[1][p] for p in top_p]
            clustred_peaks.append(top_p)
        clustred_peaks = [item for sublist in clustred_peaks for item in sublist]
        clustred_peaks = np.sort(clustred_peaks)
        print("clustred_peaks", clustred_peaks)

        # 检查相邻峰值之间的距离是否超过或等于2.9m
        adjusted_peaks = []
        for i in range(len(clustred_peaks) - 1):
            adjusted_peaks.append(clustred_peaks[i])
            if clustred_peaks[i + 1] - clustred_peaks[i] >= 2.9:
                # 在两个峰值之间插入一个虚拟边界
                mid_point = clustred_peaks[i + 1]-0.5
                # mid_point = clustred_peaks[i + 1]-1.0
                adjusted_peaks.append(mid_point)
        adjusted_peaks.append(clustred_peaks[-1])

        # 如果最后一个峰值距离点云最大值较远，插入天花板边界
        # max_z = np.max(downpcd[:, 1])
        # if max_z - adjusted_peaks[-1] > 1.0:
        #    adjusted_peaks.append(max_z)

        clustred_peaks = np.array(adjusted_peaks)
        print("adjusted_peaks", clustred_peaks)
        floors = []
        # 根据调整后的峰值生成楼层范围
        for i in range(len(clustred_peaks) - 1):
            floors.append([clustred_peaks[i], clustred_peaks[i + 1]])
        print("computed floors: ", floors)

        if not floors:
            floors.append([z_hist[1].min().item(), z_hist[1].max().item()])
            print("priors floors", floors)

        # 对第一个楼层和最后一个楼层进行扩展
        floors[0][0] = (floors[0][0] + np.min(downpcd[:, 1])) / 2
        floors[-1][1] = (floors[-1][1] + np.max(downpcd[:, 1])) / 2

        # 调试输出：打印 clustred_peaks 和 adjusted_peaks
        print("Original clustred_peaks:", clustred_peaks)
        print("Adjusted clustred_peaks after inserting virtual boundaries:", adjusted_peaks)

        # 调试输出：打印楼层范围
        print("Generated floor ranges:", floors)
        # 确认楼层数量
        print("Total number of floors detected:", len(floors))
        print("number of floors: ", len(floors))
        
        floors_pcd = []
        for i, floor in enumerate(floors):
            floor_obj = Floor(str(i), name="floor_" + str(i))
            floor_pcd = self.full_pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(-np.inf, floor[0], -np.inf),
                    max_bound=(np.inf, floor[1], np.inf),
                )
            )
            bbox = floor_pcd.get_axis_aligned_bounding_box()
            floor_obj.vertices = np.asarray(bbox.get_box_points())
            floor_obj.pcd = floor_pcd
            floor_obj.floor_zero_level = np.min(np.array(floor_pcd.points)[:, 1])
            floor_obj.floor_height = floor[1] - floor_obj.floor_zero_level
            self.floors.append(floor_obj)
            floors_pcd.append(floor_pcd)
        print("final floors: ", floors)
        return floors
  
    
    def segment_rooms_new(self, floor: Floor, path):
        """
        Segment the rooms from the floor point cloud
        :param floor: Floor, The floor object
        :param path: str, The path to save the intermediate results
        """

        tmp_floor_path = os.path.join(self.graph_tmp_folder, floor.floor_id)
        if not os.path.exists(tmp_floor_path):
            os.makedirs(tmp_floor_path, exist_ok=True)

        floor_pcd = floor.pcd
        xyz = np.asarray(floor_pcd.points)
        xyz_full = xyz.copy()
        # print("xyz.shape: ", xyz.shape)
        # import pdb; pdb.set_trace()
        floor_zero_level = floor.floor_zero_level
        floor_height = floor.floor_height
        print("floor_zero_level, floor_height = ", floor_zero_level, floor_height)
        # import pdb; pdb.set_trace()
        ## Slice below the ceiling ##
        xyz = xyz[xyz[:, 1] < floor_zero_level + floor_height - 0.3]
        # xyz = xyz[xyz[:, 1] >= floor_zero_level + 1.5]
        xyz = xyz[xyz[:, 1] >= floor_zero_level + 1.0]
        # xyz = xyz[xyz[:, 1] >= floor_zero_level + 0.5]
        xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + floor_height - 0.2]
        ## Slice above the floor and below the ceiling ##
        # xyz = xyz[xyz[:, 1] < floor_zero_level + 1.8]
        # xyz = xyz[xyz[:, 1] > floor_zero_level + 0.8]
        # xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + 1.8]

        # project the point cloud to 2d
        pcd_2d = xyz[:, [0, 2]]
        xyz_full = xyz_full[:, [0, 2]]

        # print("pcd_2d.shape: ", pcd_2d.shape)
        # import pdb; pdb.set_trace()

        # define the grid size and resolution based on the 2d point cloud
        grid_size = (
            int(np.max(pcd_2d[:, 0]) - np.min(pcd_2d[:, 0])),
            int(np.max(pcd_2d[:, 1]) - np.min(pcd_2d[:, 1])),
        )
        grid_size = (grid_size[0] + 1, grid_size[1] + 1)
        resolution = self.cfg.pipeline.grid_resolution
        print("grid_size: ", grid_size)

        # calc 2d histogram of the floor using the xyz point cloud to extract the walls skeleton
        num_bins = (int(grid_size[0] // resolution), int(grid_size[1] // resolution))
        num_bins = (num_bins[1] + 1, num_bins[0] + 1)
        hist, _, _ = np.histogram2d(pcd_2d[:, 1], pcd_2d[:, 0], bins=num_bins)
        if self.cfg.pipeline.save_intermediate_results:
            # plot the histogram
            plt.figure()
            plt.imshow(hist, interpolation="nearest", cmap="jet", origin="lower")
            plt.colorbar()
            plt.savefig(os.path.join(tmp_floor_path, "2D_histogram.png"))

        # applythresholding
        hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist = cv2.GaussianBlur(hist, (5, 5), 1)
        hist_threshold = 0.25 * np.max(hist)
        _, walls_skeleton = cv2.threshold(hist, hist_threshold, 255, cv2.THRESH_BINARY)

        # create a bigger image to avoid losing the walls
        walls_skeleton = cv2.copyMakeBorder(
            walls_skeleton, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )

        # apply closing to the walls skeleton
        kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        walls_skeleton = cv2.morphologyEx(
            walls_skeleton, cv2.MORPH_CLOSE, kernal, iterations=1
        )

        # extract outside boundary from histogram of xyz_full
        hist_full, _, _ = np.histogram2d(xyz_full[:, 1], xyz_full[:, 0], bins=num_bins)
        hist_full = cv2.normalize(hist_full, hist_full, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        hist_full = cv2.GaussianBlur(hist_full, (21, 21), 2)
        _, outside_boundary = cv2.threshold(hist_full, 0, 255, cv2.THRESH_BINARY)

        # create a bigger image to avoid losing the walls
        outside_boundary = cv2.copyMakeBorder(
            outside_boundary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )

        # apply closing to the outside boundary
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        outside_boundary = cv2.morphologyEx(
            outside_boundary, cv2.MORPH_CLOSE, kernal, iterations=3
        )

        # extract the outside contour from the outside boundary
        contours, _ = cv2.findContours(
            outside_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outside_boundary = np.zeros_like(outside_boundary)
        cv2.drawContours(outside_boundary, contours, -1, (255, 255, 255), -1)
        outside_boundary = outside_boundary.astype(np.uint8)

        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.imshow(walls_skeleton, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "walls_skeleton.png"))

            plt.figure()
            plt.imshow(outside_boundary, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "outside_boundary.png"))

        # combine the walls skelton and outside boundary
        full_map = cv2.bitwise_or(walls_skeleton, cv2.bitwise_not(outside_boundary))

        # apply closing to the full map
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        full_map = cv2.morphologyEx(full_map, cv2.MORPH_CLOSE, kernal, iterations=2)

        if self.cfg.pipeline.save_intermediate_results:
            # plot the full map
            plt.figure()
            plt.imshow(full_map, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "full_map.png"))
        # apply distance transform to the full map
        room_vertices = distance_transform(full_map, resolution, tmp_floor_path)
        # room_vertices = [room_vertices[0]] # one room case

        # using the 2D room vertices, map the room back to the original point cloud using KDTree
        room_pcds = []
        room_masks = []
        room_2d_points = []
        floor_tree = cKDTree(np.array(floor_pcd.points))
        for i in tqdm(range(len(room_vertices)), desc="Assign floor points to rooms"):
            print("idx = ", i)
            room = np.zeros_like(full_map)
            room[room_vertices[i][0], room_vertices[i][1]] = 255
            room_masks.append(room)
            room_m = map_grid_to_point_cloud(room, resolution, pcd_2d)
            room_2d_points.append(room_m)
            # extrude the 2D room to 3D room by adding z value from floor zero level to floor zero level + floor height, step by 0.1m
            z_levels = np.arange(
                floor_zero_level, floor_zero_level + floor_height, 0.05
            )
            z_levels = z_levels.reshape(-1, 1)
            z_levels *= -1
            room_m3dd = []
            for z in z_levels:
                room_m3d = np.hstack((room_m, np.ones((room_m.shape[0], 1)) * z))
                room_m3dd.append(room_m3d)
            room_m3d = np.concatenate(room_m3dd, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(room_m3d)
            # rotate floor pcd to align with the original point cloud
            T1 = np.eye(4)
            T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
            pcd.transform(T1)
            # find the nearest point in the original point cloud # 很慢
            _, idx = floor_tree.query(np.array(pcd.points), k=1, workers=-1)
            pcd = floor_pcd.select_by_index(idx)
            room_pcds.append(pcd)
            # room_pcds.append(floor_pcd) # one room case
        self.room_masks[floor.floor_id] = room_masks

        # compute the features of room: input a list of poses and images, output a list of embeddings list
        rgb_list = []
        pose_list = []
        F_g_list = []

        all_global_clip_feats = dict()
        for i, img_id in tqdm(enumerate(range(0, len(self.dataset), self.cfg.pipeline.skip_frames)), desc="Computing room features"):
            rgb_image, _, pose, _, _ = self.dataset[img_id]
            F_g = get_img_feats(np.array(rgb_image), self.preprocess, self.clip_model)
            all_global_clip_feats[str(img_id)] = F_g
            rgb_list.append(rgb_image)
            pose_list.append(pose)
            F_g_list.append(F_g)
        np.savez(
            os.path.join(self.graph_tmp_folder, "room_views.npz"),
            **all_global_clip_feats,
        )

        pcd_min = np.min(np.array(floor_pcd.points), axis=0)
        pcd_max = np.max(np.array(floor_pcd.points), axis=0)
        assert pcd_min.shape[0] == 3

        repr_embs_list, repr_img_ids_list, room_id2img_id, room_clip_embeddings_list= compute_room_embeddings(
            room_pcds, pose_list, F_g_list, pcd_min, pcd_max, 24, tmp_floor_path
        )
        assert len(repr_embs_list) == len(room_2d_points)
        assert len(repr_img_ids_list) == len(room_2d_points)
        assert len(room_id2img_id) == len(room_2d_points)
        self.room_id2img_ids = room_id2img_id
       
        room_index = 0
        for i in range(len(room_2d_points)):
            room = Room(
                str(floor.floor_id) + "_" + str(room_index),
                floor.floor_id,
                name="room_" + str(room_index),
            )
            room.pcd = room_pcds[i]
            room.vertices = room_2d_points[i]
            self.floors[int(floor.floor_id)].add_room(room)
            room.room_height = floor_height
            room.room_zero_level = floor.floor_zero_level
            room.embeddings = repr_embs_list[i]
            room.represent_images = [int(k * self.cfg.pipeline.skip_frames) for k in repr_img_ids_list[i]]
            room.sample_images = [int(k * self.cfg.pipeline.skip_frames) for k in room_id2img_id[i]]
            room.clip_embeddings = room_clip_embeddings_list[i]
            self.rooms.append(room)
            room_index += 1
        print(
            "number of rooms in floor {} is {}".format(
                floor.floor_id, len(self.floors[int(floor.floor_id)].rooms)
            )
        )
        # import pdb; pdb.set_trace()
        # 构建view层级
        view_index = 0
        for room_id in range(len(room_id2img_id)):
            # all_view_cnt += len(room_id2img_id[room_id])
            for i, img_id in enumerate(room_id2img_id[room_id]):
                retarget_img_id = img_id * self.cfg.pipeline.skip_frames
                img_path = self.dataset.frameId2imgPath[retarget_img_id]
                view = View(str(floor.floor_id) + "_" + str(room_id) + "_" + str(view_index), 
                            room_id, retarget_img_id)
                view.img_path = img_path
                # view.embedding = room_clip_embeddings_list[room_id][i]
                self.views.append(view)
                view_index += 1
                self.rooms[room_id].views.append(view)


    def segment_rooms(self, floor: Floor, path):
        """
        Segment the rooms from the floor point cloud
        :param floor: Floor, The floor object
        :param path: str, The path to save the intermediate results
        """

        tmp_floor_path = os.path.join(self.graph_tmp_folder, floor.floor_id)
        if not os.path.exists(tmp_floor_path):
            os.makedirs(tmp_floor_path, exist_ok=True)

        floor_pcd = floor.pcd
        xyz = np.asarray(floor_pcd.points)
        xyz_full = xyz.copy()
        # print("xyz.shape: ", xyz.shape)
        # import pdb; pdb.set_trace()
        floor_zero_level = floor.floor_zero_level
        floor_height = floor.floor_height
        print("floor_zero_level, floor_height = ", floor_zero_level, floor_height)
        # import pdb; pdb.set_trace()
        ## Slice below the ceiling ##
        xyz = xyz[xyz[:, 1] < floor_zero_level + floor_height - 0.3]
        # xyz = xyz[xyz[:, 1] >= floor_zero_level + 1.5]
        xyz = xyz[xyz[:, 1] >= floor_zero_level + 1.0]
        # xyz = xyz[xyz[:, 1] >= floor_zero_level + 0.5]
        xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + floor_height - 0.2]
        ## Slice above the floor and below the ceiling ##
        # xyz = xyz[xyz[:, 1] < floor_zero_level + 1.8]
        # xyz = xyz[xyz[:, 1] > floor_zero_level + 0.8]
        # xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + 1.8]

        # project the point cloud to 2d
        pcd_2d = xyz[:, [0, 2]]
        xyz_full = xyz_full[:, [0, 2]]

        # print("pcd_2d.shape: ", pcd_2d.shape)
        # import pdb; pdb.set_trace()

        # define the grid size and resolution based on the 2d point cloud
        grid_size = (
            int(np.max(pcd_2d[:, 0]) - np.min(pcd_2d[:, 0])),
            int(np.max(pcd_2d[:, 1]) - np.min(pcd_2d[:, 1])),
        )
        grid_size = (grid_size[0] + 1, grid_size[1] + 1)
        resolution = self.cfg.pipeline.grid_resolution
        print("grid_size: ", grid_size)

        # calc 2d histogram of the floor using the xyz point cloud to extract the walls skeleton
        num_bins = (int(grid_size[0] // resolution), int(grid_size[1] // resolution))
        num_bins = (num_bins[1] + 1, num_bins[0] + 1)
        hist, _, _ = np.histogram2d(pcd_2d[:, 1], pcd_2d[:, 0], bins=num_bins)
        if self.cfg.pipeline.save_intermediate_results:
            # plot the histogram
            plt.figure()
            plt.imshow(hist, interpolation="nearest", cmap="jet", origin="lower")
            plt.colorbar()
            plt.savefig(os.path.join(tmp_floor_path, "2D_histogram.png"))

        # applythresholding
        hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist = cv2.GaussianBlur(hist, (5, 5), 1)
        hist_threshold = 0.25 * np.max(hist)
        _, walls_skeleton = cv2.threshold(hist, hist_threshold, 255, cv2.THRESH_BINARY)

        # create a bigger image to avoid losing the walls
        walls_skeleton = cv2.copyMakeBorder(
            walls_skeleton, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )

        # apply closing to the walls skeleton
        kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        walls_skeleton = cv2.morphologyEx(
            walls_skeleton, cv2.MORPH_CLOSE, kernal, iterations=1
        )

        # extract outside boundary from histogram of xyz_full
        hist_full, _, _ = np.histogram2d(xyz_full[:, 1], xyz_full[:, 0], bins=num_bins)
        hist_full = cv2.normalize(hist_full, hist_full, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        hist_full = cv2.GaussianBlur(hist_full, (21, 21), 2)
        _, outside_boundary = cv2.threshold(hist_full, 0, 255, cv2.THRESH_BINARY)

        # create a bigger image to avoid losing the walls
        outside_boundary = cv2.copyMakeBorder(
            outside_boundary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )

        # apply closing to the outside boundary
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        outside_boundary = cv2.morphologyEx(
            outside_boundary, cv2.MORPH_CLOSE, kernal, iterations=3
        )

        # extract the outside contour from the outside boundary
        contours, _ = cv2.findContours(
            outside_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outside_boundary = np.zeros_like(outside_boundary)
        cv2.drawContours(outside_boundary, contours, -1, (255, 255, 255), -1)
        outside_boundary = outside_boundary.astype(np.uint8)

        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.imshow(walls_skeleton, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "walls_skeleton.png"))

            plt.figure()
            plt.imshow(outside_boundary, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "outside_boundary.png"))

        # combine the walls skelton and outside boundary
        full_map = cv2.bitwise_or(walls_skeleton, cv2.bitwise_not(outside_boundary))

        # apply closing to the full map
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        full_map = cv2.morphologyEx(full_map, cv2.MORPH_CLOSE, kernal, iterations=2)

        if self.cfg.pipeline.save_intermediate_results:
            # plot the full map
            plt.figure()
            plt.imshow(full_map, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "full_map.png"))
        # apply distance transform to the full map
        room_vertices = distance_transform(full_map, resolution, tmp_floor_path)
        # room_vertices = [room_vertices[0]] # one room case

        # using the 2D room vertices, map the room back to the original point cloud using KDTree
        room_pcds = []
        room_masks = []
        room_2d_points = []
        floor_tree = cKDTree(np.array(floor_pcd.points))
        for i in tqdm(range(len(room_vertices)), desc="Assign floor points to rooms"):
            print("idx = ", i)
            room = np.zeros_like(full_map)
            room[room_vertices[i][0], room_vertices[i][1]] = 255
            room_masks.append(room)
            room_m = map_grid_to_point_cloud(room, resolution, pcd_2d)
            room_2d_points.append(room_m)
            # extrude the 2D room to 3D room by adding z value from floor zero level to floor zero level + floor height, step by 0.1m
            z_levels = np.arange(
                floor_zero_level, floor_zero_level + floor_height, 0.05
            )
            z_levels = z_levels.reshape(-1, 1)
            z_levels *= -1
            room_m3dd = []
            for z in z_levels:
                room_m3d = np.hstack((room_m, np.ones((room_m.shape[0], 1)) * z))
                room_m3dd.append(room_m3d)
            room_m3d = np.concatenate(room_m3dd, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(room_m3d)
            # rotate floor pcd to align with the original point cloud
            T1 = np.eye(4)
            T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
            pcd.transform(T1)
            # find the nearest point in the original point cloud # 很慢
            _, idx = floor_tree.query(np.array(pcd.points), k=1, workers=-1)
            pcd = floor_pcd.select_by_index(idx)
            room_pcds.append(pcd)
            # room_pcds.append(floor_pcd) # one room case
        self.room_masks[floor.floor_id] = room_masks

        # compute the features of room: input a list of poses and images, output a list of embeddings list
        rgb_list = []
        pose_list = []
        F_g_list = []

        all_global_clip_feats = dict()
        for i, img_id in tqdm(enumerate(range(0, len(self.dataset), self.cfg.pipeline.skip_frames)), desc="Computing room features"):
            rgb_image, _, pose, _, _ = self.dataset[img_id]
            F_g = get_img_feats(np.array(rgb_image), self.preprocess, self.clip_model)
            all_global_clip_feats[str(img_id)] = F_g
            rgb_list.append(rgb_image)
            pose_list.append(pose)
            F_g_list.append(F_g)
        np.savez(
            os.path.join(self.graph_tmp_folder, "room_views.npz"),
            **all_global_clip_feats,
        )

        pcd_min = np.min(np.array(floor_pcd.points), axis=0)
        pcd_max = np.max(np.array(floor_pcd.points), axis=0)
        assert pcd_min.shape[0] == 3

        repr_embs_list, repr_img_ids_list, room_id2img_id, room_clip_embeddings = compute_room_embeddings(
            room_pcds, pose_list, F_g_list, pcd_min, pcd_max, 10, tmp_floor_path
        )
        assert len(repr_embs_list) == len(room_2d_points)
        assert len(repr_img_ids_list) == len(room_2d_points)

        room_index = 0
        for i in range(len(room_2d_points)):
            room = Room(
                str(floor.floor_id) + "_" + str(room_index),
                floor.floor_id,
                name="room_" + str(room_index),
            )
            room.pcd = room_pcds[i]
            room.vertices = room_2d_points[i]
            self.floors[int(floor.floor_id)].add_room(room)
            room.room_height = floor_height
            room.room_zero_level = floor.floor_zero_level
            room.embeddings = repr_embs_list[i]
            room.represent_images = [int(k * self.cfg.pipeline.skip_frames) for k in repr_img_ids_list[i]]
            self.rooms.append(room)
            room_index += 1
        print(
            "number of rooms in floor {} is {}".format(
                floor.floor_id, len(self.floors[int(floor.floor_id)].rooms)
            )
        )

    def identify_object(self, object_feat, text_feats, classes):
        """
        Identify the object class by computing the similarity between the object feature and the text features
        we use COCO-Stuff dataset classes as the text features (183) class.
        :param object_feat: np.ndarray, The object feature
        :param text_feats: np.ndarray, The text features
        :param classes: List, The list of classes
        :return: str, The object class
        """
        similarity = np.dot(object_feat.reshape(1, -1), text_feats.T)
        # find the class with the highest similarity
        return classes[np.argmax(similarity)]

    def segment_objects(self, save_dir: str = None):
        """
        Per floor, assign each object to the room with the highest overlap.
        :param save_dir: str, optional, The path to save the intermediate results
        """
        for i, pcd in enumerate(self.mask_pcds):
            self.mask_pcds[i] = pcd_denoise_dbscan(pcd, eps=0.05, min_points=10)
        text_feats, classes = get_label_feats(
            self.clip_model,
            self.clip_feat_dim,
            self.cfg.pipeline.obj_labels,
            self.cfg.main.save_path,
        )
        # print("self.clip_feat_dim: ", self.clip_feat_dim)
        # print("text_feats.shape: ", text_feats.shape)

        pbar = tqdm(enumerate(self.floors), total=len(self.floors), desc="Floor: ")
        margin = 0.2
        for f_idx, floor in pbar:
            pbar.set_description(f"Floor: {f_idx}")
            floor_pcd = floor.pcd
            objects_inside_floor = list()
            # assign objects to rooms
            for i, pcd in enumerate(self.mask_pcds):
                # if len(pcd.points) == 0:
                if len(pcd.points) < 10:
                    continue
                min_z = np.min(np.asarray(pcd.points)[:, 1])
                max_z = np.max(np.asarray(pcd.points)[:, 1])
                if min_z > floor.floor_zero_level - margin and max_z < (
                    floor.floor_zero_level + floor.floor_height + margin
                ):
                    objects_inside_floor.append(i)

            # show the second layer of pbar with tqdm
            obj_pbar = tqdm(
                enumerate(objects_inside_floor),
                total=len(objects_inside_floor),
                desc="Object: ",
                leave=False,
            )
            for obj_floor_idx, mask_idx in obj_pbar:
                room_assoc = list()
                for r_idx, room in enumerate(floor.rooms):
                    room_assoc.append(
                        find_intersection_share(
                            room.vertices,
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]],
                            0.2,
                        )
                    )
                # for outlier objects, utilize Euclidean distance between room centers and mask centers
                if np.sum(room_assoc) == 0:
                    for r_idx, room in enumerate(floor.rooms):
                        # use negative distance to align with the similarity metric
                        room_assoc[r_idx] = -1 * np.linalg.norm(
                            np.mean(room.vertices, axis=0)
                            - np.mean(
                                np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]],
                                axis=0,
                            )
                        )
                    if self.cfg.pipeline.save_intermediate_results:
                        plt.clf()
                        fig, ax = plt.subplots()
                        for r_idx, room in enumerate(floor.rooms):
                            if np.argmax(room_assoc) == r_idx:
                                plt.scatter(
                                    room.vertices[:, 0],
                                    room.vertices[:, 1],
                                    color="red",
                                )
                            else:
                                continue
                                # plt.scatter(room.vertices[:, 0], room.vertices[:, 1])
                        plt.scatter(
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]][:, 0],
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]][:, 1],
                            s=0.05,
                            alpha=0.5,
                            color="green",
                        )
                        ax.set_aspect("equal")

                        debug_objects_dir = os.path.join(
                            self.graph_tmp_folder, "objects"
                        )
                        os.makedirs(debug_objects_dir, exist_ok=True)
                        plt.savefig(
                            os.path.join(
                                debug_objects_dir,
                                f"{floor.rooms[np.argmax(room_assoc)].room_id}_{floor.rooms[np.argmax(room_assoc)].object_counter}.png",
                            )
                        )

                closest_room_idx = np.argmax(room_assoc)

                name = self.identify_object(
                    self.mask_feats[mask_idx], text_feats, classes
                )
                # if [i for i in ["wall", "floor", "ceiling", "window", "door", "roof", "railing"] if i in name]:
                #     continue
                parent_room = floor.rooms[closest_room_idx]
                object = Object(
                    parent_room.room_id + "_" + str(parent_room.object_counter),
                    parent_room.room_id,
                )
                parent_room.object_counter += 1
                object.name = name
                obj_pbar.set_description(
                    f"object name: {object.name}, {object.object_id}"
                )
                object.pcd = self.mask_pcds[mask_idx]
                object.vertices = np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]]
                object.embedding = self.mask_feats[mask_idx]
                floor.rooms[closest_room_idx].add_object(object)
                self.objects.append(object)


    def segment_objects_new(self, save_dir: str = None):
        """
        Per floor, assign each object to the room with the highest overlap.
        :param save_dir: str, optional, The path to save the intermediate results
        """
        for i, pcd in enumerate(self.mask_pcds):
            self.mask_pcds[i] = pcd_denoise_dbscan(pcd, eps=0.05, min_points=10)
        text_feats, classes = get_label_feats(
            self.clip_model,
            self.clip_feat_dim,
            self.cfg.pipeline.obj_labels,
            self.cfg.main.save_path,
        )
        # print("self.clip_feat_dim: ", self.clip_feat_dim)
        # print("text_feats.shape: ", text_feats.shape)

        pbar = tqdm(enumerate(self.floors), total=len(self.floors), desc="Floor: ")
        margin = 0.2
        for f_idx, floor in pbar:
            pbar.set_description(f"Floor: {f_idx}")
            floor_pcd = floor.pcd
            objects_inside_floor = list()
            # assign objects to rooms
            for i, pcd in enumerate(self.mask_pcds):
                # if len(pcd.points) == 0:
                if len(pcd.points) < 10:
                    continue
                min_z = np.min(np.asarray(pcd.points)[:, 1])
                max_z = np.max(np.asarray(pcd.points)[:, 1])
                if min_z > floor.floor_zero_level - margin and max_z < (
                    floor.floor_zero_level + floor.floor_height + margin
                ):
                    objects_inside_floor.append(i)

            # show the second layer of pbar with tqdm
            obj_pbar = tqdm(
                enumerate(objects_inside_floor),
                total=len(objects_inside_floor),
                desc="Object: ",
                leave=False,
            )
            for obj_floor_idx, mask_idx in obj_pbar:
                room_assoc = list()
                for r_idx, room in enumerate(floor.rooms):
                    room_assoc.append(
                        find_intersection_share(
                            room.vertices,
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]],
                            0.2,
                        )
                    )
                # for outlier objects, utilize Euclidean distance between room centers and mask centers
                if np.sum(room_assoc) == 0:
                    for r_idx, room in enumerate(floor.rooms):
                        # use negative distance to align with the similarity metric
                        room_assoc[r_idx] = -1 * np.linalg.norm(
                            np.mean(room.vertices, axis=0)
                            - np.mean(
                                np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]],
                                axis=0,
                            )
                        )
                    if self.cfg.pipeline.save_intermediate_results:
                        plt.clf()
                        fig, ax = plt.subplots()
                        for r_idx, room in enumerate(floor.rooms):
                            if np.argmax(room_assoc) == r_idx:
                                plt.scatter(
                                    room.vertices[:, 0],
                                    room.vertices[:, 1],
                                    color="red",
                                )
                            else:
                                continue
                                # plt.scatter(room.vertices[:, 0], room.vertices[:, 1])
                        plt.scatter(
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]][:, 0],
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]][:, 1],
                            s=0.05,
                            alpha=0.5,
                            color="green",
                        )
                        ax.set_aspect("equal")

                        debug_objects_dir = os.path.join(
                            self.graph_tmp_folder, "objects"
                        )
                        os.makedirs(debug_objects_dir, exist_ok=True)
                        plt.savefig(
                            os.path.join(
                                debug_objects_dir,
                                f"{floor.rooms[np.argmax(room_assoc)].room_id}_{floor.rooms[np.argmax(room_assoc)].object_counter}.png",
                            )
                        )

                closest_room_idx = np.argmax(room_assoc)

                name = self.identify_object(
                    self.mask_feats[mask_idx], text_feats, classes
                )
                # if [i for i in ["wall", "floor", "ceiling", "window", "door", "roof", "railing"] if i in name]:
                #     continue
                parent_room = floor.rooms[closest_room_idx]                
                object = Object(
                    parent_room.room_id + "_" + str(parent_room.object_counter),
                    parent_room.room_id,
                )
                parent_room.object_counter += 1
                object.name = name
                obj_pbar.set_description(
                    f"object name: {object.name}, {object.object_id}"
                )
                object.pcd = self.mask_pcds[mask_idx]
                object.vertices = np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]]
                object.embedding = self.mask_feats[mask_idx]
                # floor.rooms[closest_room_idx].add_object(object)
                # self.objects.append(object)                
                # build view-object topology graph 
                best_view_id = None
                best_depth = float("inf")
                all_views_in_room = parent_room.views
                camera_matrix = self.dataset.get_camera_intrinsics()
                for view in all_views_in_room:
                    img, _, pose, _, _ = self.dataset[view.img_id]
                    obj_in_view, mean_depth= check_object_in_view(
                        np.array(img).shape[1], 
                        np.array(img).shape[0], 
                        camera_matrix, 
                        np.linalg.inv(pose), 
                        np.array(self.mask_pcds[mask_idx].points),
                        return_depth=True   # 修改 check_object_in_view，支持返回深度
                    )
                    if obj_in_view:                        
                        object.view_ids.append(view.view_id)
                        view.object_ids.append(object.object_id)
                        view.text_discription.append(object.name)
                        # 找最优视角（平均深度最小）
                        if mean_depth < best_depth:
                            best_depth = mean_depth
                            best_view_id = view.view_id
                object.best_view_id = best_view_id                    
                floor.rooms[closest_room_idx].add_object(object)
                self.objects.append(object)

    def create_graph(self):
        """
        Create the full HOV-SG graph as a networkx graph
        """
        # add nodes to the graph
        for floor in self.floors:
            self.graph.add_node(floor, name="floor", type="floor")
            self.graph.add_edge(0, floor)
            for room in floor.rooms:
                self.graph.add_node(room, name="room", type="room")
                self.graph.add_edge(floor, room)
                for object in room.objects:
                    self.graph.add_node(object, name=object.name, type="object")
                    self.graph.add_edge(room, object)

    def create_graph_new(self):
        """
        Create the full HOV-SG graph as a networkx graph
        """
        # add nodes to the graph
        for floor in self.floors:
            self.graph.add_node(floor, name="floor", type="floor")
            self.graph.add_edge(0, floor)
            for room in floor.rooms:
                self.graph.add_node(room, name="room", type="room")
                self.graph.add_edge(floor, room)
                for object in room.objects:
                    self.graph.add_node(object, name=object.name, type="object")
                    self.graph.add_edge(room, object)
        
        for view in self.views:
            self.graph.add_node(view, name="view", type="view")
            for floor in self.floors:
                for room in floor.rooms:
                    if room.room_id == view.room_id:
                        self.graph.add_edge(room, view)
                        break
            for obj in self.objects:
                if obj.object_id in view.object_ids:
                    self.graph.add_edge(view, obj)
                    
               

    def save_graph(self, path):
        """
        Save the HOV-SG graph
        :param path: str, The path to save the graph
        """
        # create a folder for the graph
        if not os.path.exists(path):
            os.makedirs(path)
        # create a folder for floors, rooms and objects
        if not os.path.exists(os.path.join(path, "floors")):
            os.makedirs(os.path.join(path, "floors"))
        if not os.path.exists(os.path.join(path, "rooms")):
            os.makedirs(os.path.join(path, "rooms"))
        if not os.path.exists(os.path.join(path, "objects")):
            os.makedirs(os.path.join(path, "objects"))
        if not os.path.exists(os.path.join(path, "views")):
            os.makedirs(os.path.join(path, "views"))        
        # save the graph
        for i, node in enumerate(self.graph.nodes(data=True)):
            topo_obj, node_dict = node
            if type(topo_obj) == Floor:
                topo_obj.save(os.path.join(path, "floors"))
            elif type(topo_obj) == Room:
                topo_obj.save(os.path.join(path, "rooms"))
            elif type(topo_obj) == Object:
                topo_obj.save(os.path.join(path, "objects"))

    
    def save_graph_new(self, path):
        """
        Save the HOV-SG graph
        :param path: str, The path to save the graph
        """
        # create a folder for the graph
        if not os.path.exists(path):
            os.makedirs(path)
        # create a folder for floors, rooms and objects
        if not os.path.exists(os.path.join(path, "floors")):
            os.makedirs(os.path.join(path, "floors"))
        if not os.path.exists(os.path.join(path, "rooms")):
            os.makedirs(os.path.join(path, "rooms"))
        if not os.path.exists(os.path.join(path, "objects")):
            os.makedirs(os.path.join(path, "objects"))
        if not os.path.exists(os.path.join(path, "views")):
            os.makedirs(os.path.join(path, "views"))        # save the graph
        for i, node in enumerate(self.graph.nodes(data=True)):
            topo_obj, node_dict = node
            if type(topo_obj) == Floor:
                topo_obj.save(os.path.join(path, "floors"))
            elif type(topo_obj) == Room:
                topo_obj.save(os.path.join(path, "rooms"))
            elif type(topo_obj) == Object:
                topo_obj.save(os.path.join(path, "objects"))
            elif type(topo_obj) == View:
                topo_obj.save(os.path.join(path, "views"))
    
    # def build_view_topo_graph(self):
    #     """
    #     Build the view-topology graph for each room
    #     """
    #     for room in self.rooms:
    #         room.build_view_topo_graph(self.room_id2img_ids[room.room_id], self.cfg.pipeline.view_graph_radius)

    def load_graph(self, path):
        """
        Load the HOV-SG graph
        :param path: str, The path to load the graph
        """
        print(".. loading predicted graph")
        # load floors
        floor_files = os.listdir(os.path.join(path, "floors"))
        floor_files.sort()
        floor_files = sorted([f for f in floor_files if f.endswith(".ply")])
        for floor_file in floor_files:
            floor_file = floor_file.split(".")[0]
            floor = Floor(str(floor_file), name="floor_" + str(floor_file))
            floor.load(os.path.join(path, "floors"))
            self.floors.append(floor)
            self.graph.add_node(floor, name="floor_" + str(floor_file), type="floor")
            self.graph.add_edge(0, floor)
        print("# pred floors: ", len(self.floors))
        # load rooms
        room_files = os.listdir(os.path.join(path, "rooms"))
        room_files.sort()
        room_files = [f for f in room_files if f.endswith(".ply")]
        for room_file in room_files:
            room_file = room_file.split(".")[0]
            room = Room(str(room_file), room_file.split("_")[0])
            room.load(os.path.join(path, "rooms"))
            self.rooms.append(room)
            self.graph.add_node(room, name="room_" + str(room_file), type="room")
            self.graph.add_edge(self.floors[int(room_file.split("_")[0])], room)
            if isinstance(self.floors[int(room.floor_id)].rooms[0], str):
                self.floors[int(room.floor_id)].rooms = []
            self.floors[int(room.floor_id)].rooms.append(room)
        print("# pred rooms: ", len(self.rooms))
        # load objects
        object_files = os.listdir(os.path.join(path, "objects"))
        object_files.sort()
        object_files = [f for f in object_files if f.endswith(".ply")]
        for object_file in object_files:
            object_file = object_file.split(".")[0]
            room_id = "_".join(object_file.split("_")[:2])
            parent_room = None
            for room in self.rooms:
                if room.room_id == room_id:
                    parent_room = room
                    break
            assert (
                parent_room is not None
            ), f"Couldn't find the room with room id {room_id}"
            objectt = Object(
                str(object_file), room_id, name="object_" + str(object_file)
            )
            objectt.load(os.path.join(path, "objects"))
            objectt.room_id = room_id  # object_file.split("_")[1]
            self.objects.append(objectt)
            self.graph.add_node(
                objectt, name="object_" + str(object_file), type="object"
            )
            self.graph.add_edge(parent_room, objectt)
            # add object to the room
            parent_room.add_object(objectt)
        print("# pred objects: ", len(self.objects))
        print("-------------------")

    def load_graph_new(self, path):
        """
        Load the HOV-SG graph
        :param path: str, The path to load the graph
        """
        print(".. loading predicted graph")
        self.graph_path = path
        # load floors
        floor_files = os.listdir(os.path.join(path, "floors"))
        floor_files.sort()
        floor_files = sorted([f for f in floor_files if f.endswith(".ply")])
        for floor_file in floor_files:
            floor_file = floor_file.split(".")[0]
            floor = Floor(str(floor_file), name="floor_" + str(floor_file))
            floor.load(os.path.join(path, "floors"))
            self.floors.append(floor)
            self.graph.add_node(floor, name="floor_" + str(floor_file), type="floor")
            self.graph.add_edge(0, floor)
        print("# pred floors: ", len(self.floors))
        # load rooms
        room_files = os.listdir(os.path.join(path, "rooms"))
        room_files.sort()
        room_files = [f for f in room_files if f.endswith(".ply")]
        for room_file in room_files:
            room_file = room_file.split(".")[0]
            room = Room(str(room_file), room_file.split("_")[0])
            room.load_new(os.path.join(path, "rooms"))
            self.rooms.append(room)
            self.graph.add_node(room, name="room_" + str(room_file), type="room")
            self.graph.add_edge(self.floors[int(room_file.split("_")[0])], room)
            if isinstance(self.floors[int(room.floor_id)].rooms[0], str):
                self.floors[int(room.floor_id)].rooms = []
            self.floors[int(room.floor_id)].rooms.append(room)
        print("# pred rooms: ", len(self.rooms))
        # load objects
        object_files = os.listdir(os.path.join(path, "objects"))
        object_files.sort()
        object_files = [f for f in object_files if f.endswith(".ply")]
        for object_file in object_files:
            object_file = object_file.split(".")[0]
            room_id = "_".join(object_file.split("_")[:2])
            parent_room = None
            for room in self.rooms:
                if room.room_id == room_id:
                    parent_room = room
                    break
            assert (
                parent_room is not None
            ), f"Couldn't find the room with room id {room_id}"
            objectt = Object(
                str(object_file), room_id, name="object_" + str(object_file)
            )
            objectt.load_new(os.path.join(path, "objects"))
            objectt.room_id = room_id  # object_file.split("_")[1]
            self.objects.append(objectt)
            self.graph.add_node(
                objectt, name="object_" + str(object_file), type="object"
            )
            self.graph.add_edge(parent_room, objectt)
            # add object to the room
            parent_room.add_object(objectt)        
        print("# pred objects: ", len(self.objects))
        
        # load views
        view_files = os.listdir(os.path.join(path, "views"))
        view_files.sort()
        for view_file in view_files:
            view_file = view_file.split(".")[0]
            room_id = "_".join(view_file.split("_")[:2])
            parent_room = None
            for room in self.rooms:
                if room.room_id == room_id:
                    parent_room = room
                    break
            assert (
                parent_room is not None
            ), f"Couldn't find the room with room id {room_id}"
            vieww = View(
                str(view_file), room_id, img_id=None, name="view_" + str(view_file)
            )
            vieww.load(os.path.join(path, "views"))
            vieww.room_id = room_id 
            self.views.append(vieww)
            self.graph.add_node(vieww, name="view_" + str(view_file), type="view")
            self.graph.add_edge(parent_room, vieww)
                    
        print("-------------------")

    def build_graph(self, save_path=None):
        """
        Build the HOV-SG, by segmenting the floors, rooms, and objects and creating the graph.
        :param save_path: str, The path to save the intermediate results
        """
        print("segmenting floors...")
        self.segment_floors_manually(save_path)

        # import pdb; pdb.set_trace()
        print("segmenting rooms...")
        for floor in self.floors:
            # self.segment_rooms_new(floor, save_path)
            self.segment_rooms(floor, save_path)

        # import pdb; pdb.set_trace()
        print("segmenting/identifying objects...")
        self.segment_objects(save_path)

        print("number of objects: ", len(self.objects))
        if self.cfg.pipeline.merge_objects_graph:
            # merge objects that close to each other with same name
            for room in tqdm(self.rooms):
                print("room: ", room.room_id)
                print(" number of objects before merging: ", len(room.objects))
                room.merge_objects()
                print(" number of objects after merging: ", len(room.objects))

        print("creating graph...")
        self.create_graph()

        # create navigation graph for each floor
        print("createing nav_graph...")
        self.create_nav_graph()

        # save the graph
        self.save_graph(os.path.join(save_path, "graph"))

        print("# floors: ", len(self.floors))
        print("# rooms: ", len(self.rooms))
        print("# objects: ", len(self.objects))
        print("--> HOV-SG representation successfully built")

    def build_graph_new(self, save_path=None):
        """
        Build the HOV-SG, by segmenting the floors, rooms, views and objects and creating the graph.
        :param save_path: str, The path to save the intermediate results
        """
        print("segmenting floors...")
        self.segment_floors(save_path)

        # import pdb; pdb.set_trace()
        print("segmenting rooms...")
        for floor in self.floors:
            self.segment_rooms_new(floor, save_path)

        # import pdb; pdb.set_trace()
        print("segmenting/identifying objects...")
        self.segment_objects_new(save_path)

        print("number of objects: ", len(self.objects))
        if self.cfg.pipeline.merge_objects_graph:
            # merge objects that close to each other with same name
            for room in tqdm(self.rooms):
                print("room: ", room.room_id)
                print(" number of objects before merging: ", len(room.objects))
                room.merge_objects()
                print(" number of objects after merging: ", len(room.objects))

        print("creating graph...")
        self.create_graph_new()

        # create navigation graph for each floor
        print("createing nav_graph...")
        self.create_nav_graph()

        # save the graph
        now_str = datetime.now().strftime("%Y%m%d%H%M%S")
        self.save_graph_new(os.path.join(save_path, "graph_"+now_str))

        print("# floors: ", len(self.floors))
        print("# rooms: ", len(self.rooms))
        print("# views: ", len(self.views))
        print("# objects: ", len(self.objects))
        print("--> HOV-SG representation successfully built")

    def create_nav_graph(self):
        """
        Create the navigation graph for each floor and connect the floors together
        """
        last_nav_graph = None
        global_voronoi = None

        # create a folder for the resulting navigation graph
        nav_dir = os.path.join(self.cfg.main.save_path, "graph", "nav_graph")
        os.makedirs(nav_dir, exist_ok=True)

        # get pose list
        poses_list = []
        for i in range(0, len(self.dataset), self.cfg.pipeline.skip_frames):
            _, _, pose, _, _ = self.dataset[i]
            poses_list.append(pose)

        for floor_id, floor in enumerate(self.floors):
            nav_graph = NavigationGraph(floor.pcd, cell_size=0.03)
            upperbound = None
            if floor_id + 1 < len(self.floors):
                upperbound = self.floors[floor_id + 1].floor_zero_level
            # import pdb; pdb.set_trace()
            floor_poses_list = nav_graph.get_floor_poses(floor, poses_list, upperbound)
            # import pdb; pdb.set_trace()
            sparse_stairs_voronoi = nav_graph.get_stairs_graph_with_poses_v2(
                floor, floor_id, poses_list, nav_dir
            )
            sparse_floor_voronoi = nav_graph.get_floor_graph(
                floor, floor_poses_list, nav_dir
            )
            if sparse_stairs_voronoi is not None:
                print(f"connecting stairs and floor {floor_id}")
                sparse_floor_voronoi = nav_graph.connect_stairs_and_floor_graphs(
                    sparse_stairs_voronoi, sparse_floor_voronoi, nav_dir
                )
            NavigationGraph.save_voronoi_graph(
                sparse_floor_voronoi, nav_dir, "sparse_voronoi"
            )

            if last_nav_graph is not None and last_nav_graph.has_stairs:
                print(f"connecting two floors {floor_id}")
                global_voronoi = nav_graph.connect_voronoi_graphs(
                    last_nav_graph.sparse_floor_voronoi, nav_graph.sparse_floor_voronoi
                )
            last_nav_graph = nav_graph

        if global_voronoi is None:
            global_voronoi = last_nav_graph.sparse_floor_voronoi

        NavigationGraph.save_voronoi_graph(global_voronoi, nav_dir, "global_nav_graph")

    def set_room_names(self, room_names: List[str]):
        """Set the name for each room node

        Args:
            room_names (List[str]): a list of room names with the same length as self.rooms
        """
        assert len(room_names) == len(
            self.rooms
        ), "The length of room_names should be the same as the number of rooms in the graph"
        for i in range(len(self.rooms)):
            self.rooms[i].name = room_names[i]
            vertices = self.rooms[i].vertices
            center = np.mean(vertices, axis=0)
            self.rooms[i].room_center_pos = center
            # self.rooms[i].set_txt_embeddings(room_names[i], self.clip_model, self.clip_feat_dim)


    def generate_room_names(
        self,
        generate_method: str = "label",
        default_room_types: List[str] = None,
    ):
        """Generate a name for each room node based on children nodes' embedding

        Args:
            generate_method (str): "label" or "obj_embedding" or "view_embedding"
            default_room_types (List[str]): optionally provide a list of default room types so that the
                                            room names can only be one of the provided options. When the
                                            generate_method is set to "embedding", this list is mandatory.
            clip_model (Any): when the generate_method is set to "embedding", a clip model needs to be
                              provided to the method.
            clip_feat_dim (int): when the generate_method is set to "embedding", the clip features dimension
                                 needs to be provided to this method
        """
        for i in range(len(self.rooms)):
            if generate_method in ["obj_embedding", "view_embedding"]:
                # print(default_room_types)
                assert (
                    default_room_types is not None
                ), "You should provide a list of default room types"
                assert self.clip_model is not None, "You should provide a clip model"
                assert (
                    self.clip_feat_dim is not None
                ), "You should provide the clip features dimension"
            self.rooms: List[Room]
            if generate_method in ["obj_embedding", "label"]:
                self.rooms[i].infer_room_type_from_objects(
                    infer_method=generate_method,
                    default_room_types=default_room_types,
                    clip_model=self.clip_model,
                    clip_feat_dim=self.clip_feat_dim,
                )
            elif generate_method in ["view_embedding"]:
                self.rooms[i].infer_room_type_from_view_embedding(
                    default_room_types, self.clip_model, self.clip_feat_dim
                )
            else:
                return NotImplementedError

    def query_graph(self, query):
        """
        search in graph of the openmap with a text query
        """
        text_feats = get_text_feats_multiple_templates(
            [query], self.clip_model, self.clip_feat_dim
        )
        # compute similarity between the text query and the objects embeddings in the graph
        similarity = np.dot(text_feats, np.array([o.embedding for o in self.objects]).T)
        # similarity = compute_similarity(text_feats, np.array([o.embedding for o in self.objects]))
        # find top 5 similar objects
        top_index = np.argsort(similarity[0])[::-1][:5]
        # print the top 5 similar objects
        for i in top_index:
            print(self.objects[i].name, similarity[0])
            print("room: ", self.objects[i].room_id)
            obj_pcd = self.objects[i].pcd.paint_uniform_color([1, 0, 0])
            # find the room with a room id that matches the object's room id
            for room in self.rooms:
                if room.room_id == self.objects[i].room_id:
                    room_pcd = room.pcd
                    break
            o3d.visualization.draw_geometries([room_pcd, obj_pcd])

        # return the object with the highest similarity
        return self.objects[top_index[0]]

    def query_floor(self, query: str, query_method: str = "clip") -> int:
        """search a floor based on the number of the text query

        Args:
            query (str): a number in text format
            query_method (str): "clip" match the clip embeddings of the query text and the text description of all floors.
                                "gpt" provide the floor ids in the graph and the text query to a gpt agent, and ask for the
                                matching floor id.

        Returns:
            int: The target floor id in self.floors
        """
        # TODO: assume that the self.floors are ordered according to the floor level in ascending order. Check again.
        zero_levels_list = [x.floor_zero_level for x in self.floors]
        # print("zero_levels_list: ", zero_levels_list)
        zero_level_order_ids = np.argsort(zero_levels_list)

        # check whether the query is a number that is an integer
        try:
            return zero_level_order_ids[int(query) - 1]
        except:
            if query_method == "clip":
                text_feats = get_text_feats_multiple_templates(
                    [query], self.clip_model, self.clip_feat_dim
                )
                floor_names = ["floor " + str(i) for i in range(len(self.floors))]
                floor_embs = get_text_feats_multiple_templates(
                    floor_names, self.clip_model, self.clip_feat_dim
                )
                sim_mat = np.dot(text_feats, floor_embs.T)
                # sim_mat = compute_similarity(text_feats, floor_embs)
                # print(sim_mat)
                top_index = np.argsort(sim_mat[0])[::-1][0]
                return zero_level_order_ids[top_index]

            elif query_method == "gpt":
                floor_ids_list = [i + 1 for i in range(len(self.floors))]
                floor_id = infer_floor_id_from_query(floor_ids_list, query)
                return zero_level_order_ids[floor_id - 1]

    def upload2oss(self, retrieved_img_list: list):
        self.upload_flag = True
        self.force_reupload = False
        self.src_img_root = os.path.dirname(retrieved_img_list[0])
        img_dir_prefix = f"{os.path.basename(self.src_img_root)}/images"    
        img_list = retrieved_img_list  # 不做 sorted    
        # img_list = sorted(
        #     retrieved_img_list,
        #     key=lambda x: float(os.path.splitext(os.path.basename(x))[0])
        # )           
        n_images = len(img_list)
        self.downsampeld_img_list = img_list
        auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
        bucket = oss2.Bucket(auth, "oss-cn-beijing.aliyuncs.com", "mapvln")
        self.oss_img_list = []
        for file in tqdm(self.downsampeld_img_list):
            file_name = os.path.basename(file)
            self.oss_img_list.append(
                f"https://mapvln.oss-cn-beijing.aliyuncs.com/{img_dir_prefix}/{file_name}"
            )
            oss_url = f"{img_dir_prefix}/{file_name}"
            if self.upload_flag:
                if not bucket.object_exists(oss_url) or self.force_reupload:
                    bucket.put_object_from_file(
                        oss_url,
                        file,
                    )
                else:
                    print(f"{oss_url} already exists in Aliyun OSS, skipping.")
        print(
            f"Uploaded {len(self.downsampeld_img_list)} images to Aliyun OSS."
        )

    def vlm_choose(self, video_image_local_paths: list, instruction: str):
        system_prompt = """
            You are a robot operating in an indoor environment and your task is to respond to the user command about going 
            to a specific location by finding the closest frame in the provided locations to navigate to.
        """
        # import pdb; pdb.set_trace()
        self.upload2oss(video_image_local_paths)
        # import pdb; pdb.set_trace()
        video_prompt = []
        for i, img_url in enumerate(self.oss_img_list):
            video_prompt.append({"type": "text", "text": f"Frame:{i}"})
            video_prompt.append(
                {"type": "image_url", "image_url": {"url": img_url}}
            )
        instruction_prompt = f"User says: {instruction}. Can you find the closet frame in the provided locations to navigate to?"
        rules_prompt = """
        Rules to follow:
        1. Output the frame id (integer) wrapped in the <frame_id> tag.
        2. Carefully compare the candidate locations with the user instruction and select the closest one.  Describe the image you choose in detail to justify your choice after you respond the frame_id.
        """

        messages = [
            {
                "role": "user",
                "content": [
                    *video_prompt,
                    {
                        "type": "text",
                        "text": system_prompt,
                    },
                    {
                        "type": "text",
                        "text": instruction_prompt,
                    },
                    {
                        "type": "text",
                        "text": rules_prompt,
                    },
                ],
            }
        ]
        response_flag = False
        while not response_flag:
            try:
                print("Sending request stage 2 ...")
                response = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=messages,
                    seed=123,
                )
                response_flag = True
            except Exception as e:
                print(e)
                time.sleep(1)
                print("Retrying ...")
        response = response.choices[0].message.content
        return response


    def detect_and_select_best_gpt(self, imglist: List[str], query: str, score_threshold: float = 0.5):
        """
        使用 GPT Vision 判断 object 是否在每张图中出现，并返回最佳图像。
        Args:
            imglist: 图片 URL 列表
            query: 查询目标对象
            score_threshold: 分数阈值，低于此值即判定为 False
        Returns:
            results: List[bool]，每张图是否包含 object
            best_image: str，最符合 query 的图像（如果没有则 None）
        """
        self.upload2oss(imglist)
        results, scores = [], []

        for img in self.oss_img_list:
            # Step1: yes/no 判断
            prompt_yesno = f"Does this image contain a '{query}'? Answer strictly with 'yes' or 'no'."
            messages_yesno = [
                {"role": "system", "content": "You are an object detector. Answer only 'yes' or 'no', no explanation."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_yesno},
                        {"type": "image_url", "image_url": {"url": img}},
                    ],
                },
            ]
            resp_yesno = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=messages_yesno,
            )
            ans_raw = resp_yesno.choices[0].message.content.strip().lower()
            has_object = (ans_raw == "yes")  # 严格匹配

            score = 0.0
            if has_object:
                # Step2: 打分
                prompt_score = f"On a scale from 0 to 1, how strongly does this image contain a '{query}'? Respond only with a single number (e.g., 0.73)."
                # prompt_score = (
                # f"On a scale from 0 to 1, how strongly and prominently does this image show a '{query}'? "
                # f"Consider both the confidence of presence and the relative size/visual prominence of the object in the image. "
                # f"If the object is large and visually central, score closer to 1. "
                # f"If the object is small, partially hidden, or in the background, score lower to 0.5. "
                # f"Respond only with a single number (e.g., 0.85)."
                # )
                messages_score = [
                    {"role": "system", "content": "You are an object detector. Answer only with a single number between 0 and 1, no text."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_score},
                            {"type": "image_url", "image_url": {"url": img}},
                        ],
                    },
                ]
                resp_score = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=messages_score,
                )
                ans_score_raw = resp_score.choices[0].message.content.strip()
                try:
                    score = float(ans_score_raw)
                    if not (0.0 <= score <= 1.0):
                        score = 0.0
                except Exception:
                    score = 0.0

                # 分数低于阈值则否定
                if score < score_threshold:
                    has_object = False

            results.append(has_object)
            scores.append(score)
            print(f"[GPT] Image: {img} → raw_yesno='{ans_raw}', score={score:.3f}, has_object={has_object}, query={query}")

        # Step3: 选出 best（如果都没有，则返回 None）
        if any(results):
            best_idx = int(np.argmax(scores))
            best_image = imglist[best_idx]
        else:
            best_image = None

        return results, best_image

    def detect_object_in_image(self, img_path: str, query: str, score_threshold: float = 0.3) -> bool:
        # 上传图片到 OSS
        self.upload2oss([img_path])
        img_url = self.oss_img_list[0]
        # 一步到位：直接输出 0~1
        prompt = (
            f"On a scale from 0 to 1, does this image contain a '{query}'? "
            "Respond only with a single number between 0 and 1."
        )
        messages = [
            {"role": "system", "content": "You are an object detector. Answer only with a single number, no text."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=messages,
        )
        ans_raw = resp.choices[0].message.content.strip()

        try:
            score = float(ans_raw)
            if not (0.0 <= score <= 1.0):
                score = 0.0
        except Exception:
            score = 0.0

        has_object = score >= score_threshold
        print(f"[GPT] Image: {img_url} → score={score:.3f}, has_object={has_object}")
        return has_object


    def visualize_goal_images(self, mean_depth, goal_image_path_online, goal_image_path_by_clip, goal_image_path_by_gpt, save_name="goal_compare.png"):
        # 读取两张图片
        img_online = cv2.imread(goal_image_path_online)
        img_gpt_best = cv2.imread(goal_image_path_by_clip)
        img_gpt = cv2.imread(goal_image_path_by_gpt)

        if img_gpt_best is None or img_gpt is None or img_online is None:            
            raise FileNotFoundError("其中一张图片路径无效，请检查路径是否正确")

        # 保证两张图大小一致（都缩放到640x480）
        img_gpt_best = cv2.resize(img_gpt_best, (640, 480))
        img_gpt = cv2.resize(img_gpt, (640, 480))
        img_online = cv2.resize(img_online, (640, 480))

        # 在左上角添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (0, 255, 0)  # 绿色

        cv2.putText(img_gpt_best, "BEST", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(img_gpt, "GPT", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(img_online, "ObjBestView", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(img_online, f"{mean_depth:.2f}", (10, 300), font, font_scale, color, thickness, cv2.LINE_AA)
        # 横向拼接
        combined = np.hstack((img_online, img_gpt_best, img_gpt))
        # 保存结果
        save_path = os.path.join(self.curr_query_save_dir, save_name)
        cv2.imwrite(save_path, combined)
        # 可视化
        cv2.imshow("Goal Image Comparison", combined)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    def get_object_info(self, target_obj_id):
        target_object = self.objects[target_obj_id]
        target_object_id = target_object.object_id
        target_object_best_view_id = target_object.best_view_id
        best_view = None
        for view in self.views:
            if view.view_id == target_object_best_view_id:
                best_view = view
                break
        assert best_view is not None, "best view is None"
        # best_view_image_path = self.dataset.frameId2imgPath[best_view.img_id]
        best_view_image_path = best_view.img_path
        best_view_img_id = best_view.img_id
        return best_view_image_path, best_view_img_id, target_object_id
    
    def get_object_best_view(self, target_object):
        target_object_id = target_object.object_id
        target_object_best_view_id = target_object.best_view_id
        best_view = None
        for view in self.views:
            if view.view_id == target_object_best_view_id:
                best_view = view
                break
        # assert best_view is not None, "best view is None"
        # best_view_image_path = self.dataset.frameId2imgPath[best_view.img_id]
        if best_view is None:
            return ""
        best_view_image_path = best_view.img_path
        return best_view_image_path
    
    def find_view_by_imgpath(self, img_path):
        for view in self.views:
            if view.img_path == img_path:
                return view, view.img_id
        return None, None

    def find_object_by_object_id(self, object_id):
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None        


    def query_room_obj_text_and_clip(self, instruction, room_query, object_query, negative_prompt, room_query_method="label", object_query_method="clip", update_flag=True):
        '''
        Query the graph with text input for room and object
        '''
        offline_start_time = time.time()
        query_time_consumer = dict()
        query_time_consumer["room_query"] = room_query
        query_time_consumer["object_query" ] = object_query
        query_time_consumer["negative_prompt"] = negative_prompt
        is_dectect_room = "unknown" not in room_query.lower()
        if room_query is None or room_query == "":
            is_dectect_room = False
        
        is_dectect_obj = "unknown" not in object_query.lower()
        if object_query is None or object_query == "":
            is_dectect_obj = False
        
        print("is_dectect_room: ", is_dectect_room, "is_dectect_obj: ", is_dectect_obj)

        # query room
        start_time = time.time()
        if room_query_method == "label" and is_dectect_room:
            rooms_list = self.rooms
            print("query room use label")
            for room in rooms_list:
                assert (
                    room.name is not None
                ), "The name attribute for the room has not been generated"
            room_names_list = [room.name for room in rooms_list]
            room_embs = get_text_feats_multiple_templates(
                room_names_list, self.clip_model, self.clip_feat_dim
            )
            query_room_text_feats = get_text_feats_multiple_templates(
                [room_query], self.clip_model, self.clip_feat_dim
            )
            similarity = np.dot(query_room_text_feats, room_embs.T)
            top_index = np.argsort(similarity[0])[::-1]
            for i in top_index[:3]:
                print("room: ", rooms_list[i].room_id, rooms_list[i].name, similarity[0][i])
            same_sim_indices = []
            tar_sim = similarity[0, top_index[0]]
            same_sim_indices.append(top_index[0])
            for i in top_index[1:]:
                if np.abs(similarity[0, i] - tar_sim) < 1e-3:
                    same_sim_indices.append(i)

            target_rooms = [rooms_list[i] for i in same_sim_indices]
            target_room_ids = [target_room.room_id for target_room in target_rooms]
            target_ids = [
                i for i, x in enumerate(self.rooms) if x.room_id in target_room_ids
            ]

        else:
            query_room_text_feats = get_text_feats_multiple_templates(
                [room_query], self.clip_model, self.clip_feat_dim
            )
            room2query_sim = dict()
            room2query_feat = dict()   # 保存对应的特征向量
            room2query_id = dict()     # 保存对应的embedding索引
            for room in self.rooms:
                embeddings = np.stack(room.embeddings)   # [view_num, 768]
                sims = np.dot(query_room_text_feats, embeddings.T)  # [1, view_num]，query和每个view的相似度
                max_idx = np.argmax(sims)                # 找到最大相似度的位置
                max_sim = sims[0, max_idx]               # 最大相似度值
                max_feat = embeddings[max_idx]           # 对应的特征向量 (768,)
                
                room2query_sim[room.room_id] = max_sim
                room2query_feat[room.room_id] = max_feat
                room2query_id[room.room_id] = max_idx

            room2query_sim_sorted = {
                int(k.split("_")[-1]): v
                for k, v in sorted(
                    room2query_sim.items(), key=lambda item: item[1], reverse=True
                )
            }
            target_ids = list(room2query_sim_sorted.keys())[
                0 : min(len(room2query_sim_sorted), 10)
            ]
        room_retrival_time = time.time() - start_time

        # print query room result
        print("target_room_ids: ", target_ids)
        query_time_consumer["room_retrieval_by_clip"] = room_retrival_time
        
        # query object
        # goal_room_id = target_ids[0]
        if not is_dectect_obj:
            print("not found object, use llm to find intention object")
            object_query =self.generate_object_querys(instruction)
        
        if object_query in negative_prompt:
            query_id = negative_prompt.index(object_query)
        else:
            query_id = None

        if query_id is None:
            object_query = [object_query, *negative_prompt]
            query_id = 0
        else:
            object_query = negative_prompt



        query_object_text_feats = get_text_feats_multiple_templates(
            object_query, self.clip_model, self.clip_feat_dim
        )  # (len(categories), feat_dim)

        room_ids_list = []
        for obj in self.objects:
            for i, room in enumerate(self.rooms):
                if obj.room_id == room.room_id:
                    room_ids_list.append(i)
                    break

        if object_query_method == "clip":
            if len(target_ids) != 0:
                objects_list = []
                room_ids_list = []
                for i in target_ids:
                    objects_list.extend(self.rooms[i].objects)
                    room_ids_list.extend([i] * len(self.rooms[i].objects))
            objects_list: List[Object]
            object_embs = np.array([obj.embedding for obj in objects_list])
            sim_mat = np.dot(query_object_text_feats, object_embs.T)
            top_index = np.argsort(sim_mat[query_id])[::-1][:10] # top-10
            # import pdb; pdb.set_trace()
            for i in top_index:
                print("object name, score: ", objects_list[i].name, sim_mat[0][i])
                print("object id: ", objects_list[i].object_id)

            top_k = 5
            top_index = np.argsort(sim_mat[query_id])[::-1][:top_k]
            if len(negative_prompt) > 0:
                cls_ids = np.argmax(sim_mat, axis=0)  # category id for each object
                print(f"cls_ids: {cls_ids}")
                max_scores = np.max(sim_mat, axis=0)  # max scores for each object
                obj_ids = np.where(cls_ids == query_id)[
                    0
                ]  # find the obj ids that assign max score to the target category
                if len(obj_ids) > 0:
                    obj_scores = max_scores[obj_ids]
                    resort_ids = np.argsort(
                        -obj_scores
                    )  # sort the obj ids based on max score (descending)
                    top_index = obj_ids[resort_ids]  # get the top index
                    top_index = top_index[:top_k]
            target_object_score = [sim_mat[query_id][i] for i in top_index]
            target_object_id = [objects_list[i].object_id for i in top_index]
            target_room_id = [room_ids_list[i] for i in top_index]
            target_id = []
            for ti in target_object_id:
                target_id.append(
                    [i for i, x in enumerate(self.objects) if x.object_id == ti][0]
                )
            FastMatching_time = time.time() - start_time
            query_time_consumer["FastMatching_time"] = FastMatching_time

        
##########################################################################################################
        save_json_path = os.path.join(self.curr_query_save_dir, "query_time_consumer.json")
        # elif object_query_method == "gpt":                
        best_object = self.objects[target_id[0]]
        best_object_best_view_id = best_object.best_view_id
        best_view = None
        for view in self.views:
            if view.view_id == best_object_best_view_id:
                best_view = view
                break
        if best_view is None:
            total_online_query_time = FastMatching_time
            query_time_consumer["total_query_time"] = f"{total_online_query_time:.4f} 秒"
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(query_time_consumer, f, ensure_ascii=False, indent=4)
            res_dict = dict()
            res_dict["FastMatching"] = FastMatching_time
            res_dict["ObjectInImageCheck"] = 0.0
            res_dict["VLM_Rethinking"] = 0.0
            res_dict["Re_Matching"] = 0.0
            res_dict["Total_Time"] = total_online_query_time
            return res_dict, target_id, target_room_id
            
        # # assert best_view is not None, "best view is None"
        # if best_view is None:
        #     best_object = self.objects[target_id[1]]
        #     best_object_best_view_id = best_object.best_view_id
        #     best_view = None
        #     for view in self.views:
        #         if view.view_id == best_object_best_view_id:
        #             best_view = view
        #             break
        # if best_view is None:
        #     best_object = self.objects[target_id[2]]
        #     best_object_best_view_id = best_object.best_view_id
        #     best_view = None
        #     for view in self.views:
        #         if view.view_id == best_object_best_view_id:
        #             best_view = view
        #             break
        

        # assert len(target_id) == 5
        # top2_best_view_image_path, top2_best_view_img_id, top2_best_object_id = self.get_object_info(target_id[1])
        # top3_best_view_image_path, top3_best_view_img_id, top3_best_object_id = self.get_object_info(target_id[2])
        # top4_best_view_image_path, top4_best_view_img_id, top4_best_object_id = self.get_object_info(target_id[3])
        # top5_best_view_image_path, top5_best_view_img_id, top5_best_object_id = self.get_object_info(target_id[4])

        # best_view_image_path = self.dataset.frameId2imgPath[best_view.img_id]
        best_view_image_path = best_view.img_path
        best_view_img_id = best_view.img_id
        best_view_view_id = best_view.view_id
        print("online_best_view_image_path: ", best_view_image_path)
        # print("online_top2_best_view_image_path: ", top2_best_view_image_path)
        # print("online_top3_best_view_image_path: ", top3_best_view_image_path)
        # print("online_top4_best_view_image_path: ", top4_best_view_image_path)
        # print("online_top5_best_view_image_path: ", top5_best_view_image_path)
        goal_image_path_online = best_view_image_path
        query_time_consumer["top1_image_path_online_object_best_view"] = goal_image_path_online
        # query_time_consumer["top2_image_path_online_object_best_view"] = top2_best_view_image_path
        # query_time_consumer["top3_image_path_online_object_best_view"] = top3_best_view_image_path
        # query_time_consumer["top4_image_path_online_object_best_view"] = top4_best_view_image_path
        # query_time_consumer["top5_image_path_online_object_best_view"] = top5_best_view_image_path
##########################################################################################################
        start_time = time.time()       
        Object_in_goal_view_check = self.detect_object_in_image(best_view_image_path, object_query[query_id])
        Object_in_goal_view_check_time = time.time() - start_time
        query_time_consumer["Object_in_goal_view_check_time"] = f"{Object_in_goal_view_check_time:.4f} 秒"
        query_time_consumer["Object_in_goal_view_check_res"] = Object_in_goal_view_check
        if Object_in_goal_view_check:
            total_online_query_time = FastMatching_time + Object_in_goal_view_check_time
            query_time_consumer["total_query_time"] = f"{total_online_query_time:.4f} 秒"
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(query_time_consumer, f, ensure_ascii=False, indent=4)
            res_dict = dict()
            res_dict["FastMatching"] = FastMatching_time
            res_dict["ObjectInImageCheck"] = Object_in_goal_view_check_time
            res_dict["VLM_Rethinking"] = 0.0
            res_dict["Re_Matching"] = 0.0
            res_dict["Total_Time"] = total_online_query_time
            return res_dict, target_id, target_room_id
        
        else: # run gpt-refine
            total_online_query_time = FastMatching_time + Object_in_goal_view_check_time
            all_image_incides = []
            all_image_embedding = []
            for room in self.rooms:
                img_ids = room.sample_images  # list of images
                embs = room.clip_embeddings  # shape [view, 768]            
                # 确保长度对齐
                assert len(img_ids) == len(embs), \
                    f"Number of images ({len(img_ids)}) != embeddings ({len(embs)})"
                all_image_incides.extend(img_ids)
                all_image_embedding.extend(embs)  # 每个embedding对应一张图                
            print("all_image_incides: ", len(all_image_incides))
            print("all_image_embedding: ", len(all_image_embedding))        
            room_ids = target_ids
            for room_id in room_ids[:1]:
                print(f"find goal image in room {room_id}")
                start_time = time.time()
                # find goal image by clip            
                # room_img_indices = self.rooms[room_id].represent_images[0] # use repr images in curr room
                # room_img_indices = self.rooms[room_id].sample_images # use all images in curr room
                # room_image_local_paths = [self.dataset.frameId2imgPath[idx] for idx in room_img_indices]            
                # room_image_local_feature = self.rooms[room_id].embeddings
                # room_image_local_feature = self.rooms[room_id].clip_embeddings
                # room_embeddings = np.stack(room_image_local_feature)   # [view_num, 768]
                gloal_embedding =np.stack(all_image_embedding)  # [total_view_num, 768]
                sims = np.dot(query_object_text_feats[0], gloal_embedding.T)  # [1, view_num]，query和每个view的相似度
                clip_max_idx = np.argmax(sims) # 找到最大相似度的位置

                # 计算 top_k，确保不超过 sims 的长度
                top_k = min(24, sims.shape[0])            
                top_idx = np.argsort(sims)[-top_k:][::-1]  # 从大到小取前 top_k 索引

                # find goal image by clip
                goal_image_path_by_clip = self.dataset.frameId2imgPath[all_image_incides[clip_max_idx]]
                print(f"goal_image_path_by_clip: {goal_image_path_by_clip}")
                end_time = time.time()
                query_time_consumer[f"goal_image_reterival_by_clip_{room_id}"] = end_time - start_time
                print(f"find goal image by clip 运行时间: {end_time - start_time:.4f} 秒")
                query_time_consumer["goal_image_path_by_clip"] = goal_image_path_by_clip
                start_time = time.time()
        
                # find goal image by gpt
                room_clip_refined_topk_image_local_paths = [self.dataset.frameId2imgPath[all_image_incides[idx]] for idx in top_idx]
                room_image_local_paths = room_clip_refined_topk_image_local_paths
                print("room_image_local_paths: ", room_image_local_paths)

                response = self.vlm_choose(
                    room_image_local_paths, instruction
                )
                print(response)
                match = re.findall(r"\d+", response)
                if match:
                    frame_id = match[0]
                    goal_img_path = self.downsampeld_img_list[int(frame_id)]
                else:
                    print("No frame id found in response text.")
                    goal_img_path = None
                goal_image_path_by_gpt = goal_img_path
                end_time = time.time()
                query_time_consumer[f"goal_image_reterival_by_gpt_{room_id}"] = end_time - start_time
                query_time_consumer["goal_image_path_by_gpt"] = goal_image_path_by_gpt
                print(f"find goal image by gpt 运行时间: {end_time - start_time:.4f} 秒")
                print("goal_image_path_by_gpt: ", goal_image_path_by_gpt)

                # judge whether object in goal image
                # goal_index = all_image_incides.index(best_view_img_id)
                # goal_image_clip_embedding = np.array(all_image_embedding[goal_index])
                # goal_sim_mat = np.dot(query_object_text_feats[0], goal_image_clip_embedding.T)
                # select_imgs = [goal_image_path_online, top2_best_view_image_path, top3_best_view_image_path,
                #                top4_best_view_image_path, top5_best_view_image_path, goal_image_path_by_gpt]
                select_imgs = [goal_image_path_online, goal_image_path_by_clip, goal_image_path_by_gpt]
                                
                gpt_check_start_time = time.time()
                gpt_check_result, best_image_path  = self.detect_and_select_best_gpt(select_imgs, object_query[query_id])
                gpt_check_time = time.time() - gpt_check_start_time
                query_time_consumer["gpt_check_time"] = gpt_check_time
                # print("goal_sim_mat:" , goal_sim_mat)
                # if goal_sim_mat[0, query_id] > 0.3:
                print("检测结果:", gpt_check_result)      # [True, False]
                print("最佳图像:", best_image_path)
                query_time_consumer["检测结果"] = gpt_check_result
                query_time_consumer["最佳图像"] = best_image_path
                # query_time_consumer["goal_sim_mat"] = goal_sim_mat.tolist()
                print(best_image_path != goal_image_path_online)
                print("update_flat: ", update_flag)
                avg_distance_in_gptview = -1.0
                gpt_refine_time_start = time.time()
                if gpt_check_result[0] is False and update_flag and best_image_path != goal_image_path_online and best_image_path is not None:
                    print("performing gpt refineing..............................")
                    objs_embedding_in_view = []
                    gpt_refine_best_view, gpt_refine_best_view_img_id = self.find_view_by_imgpath(best_image_path)
                    assert gpt_refine_best_view is not None
                    object_ids_in_view = gpt_refine_best_view.object_ids
                    for object_id in object_ids_in_view:
                        object_target = self.find_object_by_object_id(object_id)
                        assert object_target is not None
                        objs_embedding_in_view.append(object_target.embedding)
                    if len(objs_embedding_in_view) > 0:
                        objs_embedding_in_view = np.stack(objs_embedding_in_view)
                        obj_sims = np.dot(query_object_text_feats[0], objs_embedding_in_view.T)  # [1, obj_num]，query和这个view下每个obj的相似度
                        max_obj_idx = np.argmax(obj_sims)                # 找到最大相似度的位置
                        max_obj_sim = obj_sims[max_obj_idx]           # 最大相似度值
                        print(f"max_obj_sim: {max_obj_sim}")
                        max_sim_object_id = object_ids_in_view[max_obj_idx]
                        final_object = self.find_object_by_object_id(max_sim_object_id)
                        final_obj_pcd = final_object.pcd
                        camera_matrix = self.dataset.get_camera_intrinsics()
                        img, _, pose, _, _ = self.dataset[gpt_refine_best_view_img_id]
                        if not isinstance(img, np.ndarray):
                            img = np.array(img)
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            # save_path = os.path.join(self.curr_query_save_dir, save_name)
                        avg_distance_in_gptview = visualize_pcd_on_image(final_obj_pcd, img, camera_matrix, np.linalg.inv(pose), 
                                                                        save_path=os.path.join(self.curr_query_save_dir, f"gpt_refine_object_id_{final_object.object_id}.png"))
                        # replace and save obect feature embedding?
                        # final_object.embedding = final_object.embedding * 0.2 + query_object_text_feats[0] * 0.8
                        new_objects_path = os.path.join(self.graph_path, "objects_update")
                        if not os.path.exists(new_objects_path):
                            os.makedirs(new_objects_path)
                        final_object.save(os.path.join(self.graph_path, "objects_update"))
                gpt_refine_time= time.time() - gpt_refine_time_start
                query_time_consumer["gpt_refine_time"] = gpt_refine_time

                # 计算距离
                obj_pcd = deepcopy(best_object.pcd)
                obj_center = obj_pcd.get_center()
                camera_matrix = self.dataset.get_camera_intrinsics()
                img, _, pose, _, _ = self.dataset[best_view_img_id]
                obj_in_view, mean_depth_online= check_object_in_view(
                    np.array(img).shape[1], 
                    np.array(img).shape[0], 
                    camera_matrix, 
                    np.linalg.inv(pose), 
                    np.array(obj_pcd.points),
                    return_depth=True   # 修改 check_object_in_view，支持返回深度
                )
                # gpt_index = all_image_incides.index()
                if best_image_path is not None:               
                    self.visualize_goal_images(mean_depth_online, goal_image_path_online, goal_image_path_by_clip, goal_image_path_by_gpt, save_name=f"goal_compare_room_{room_id}.png")
                else:
                    best_image_path = goal_image_path_by_gpt
                    self.visualize_goal_images(mean_depth_online, goal_image_path_online, goal_image_path_by_clip, goal_image_path_by_gpt, save_name=f"goal_compare_room_{room_id}.png")

                # 保存为 JSON 文件
            total_query_time_offline = total_online_query_time + gpt_check_time + gpt_refine_time
            query_time_consumer["total_query_time"] = f"{total_query_time_offline:.4f} 秒"
            query_time_consumer["online_object_distance_in_online_view"] = mean_depth_online
            query_time_consumer["gptref_object_distance_in_ofline_view"] = avg_distance_in_gptview        
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(query_time_consumer, f, ensure_ascii=False, indent=4)
            res_dict = dict()
            res_dict["FastMatching"] = FastMatching_time
            res_dict["ObjectInImageCheck"] = Object_in_goal_view_check_time
            res_dict["VLM_Rethinking"] = gpt_check_time
            res_dict["Re_Matching"] = gpt_refine_time
            res_dict["Total_Time"] = total_query_time_offline
            return res_dict, target_id, target_room_id
                                           
        
        
    def query_object_new(
        self,
        query: str,
        floor_id: int = -1,
        room_ids: List[int] = [],
        query_method: str = "clip",
        top_k: int = 1,
        negative_prompt: List[str] = [],
    ) -> Tuple[List[int], List[int]]:
        """search an object (from a room) with a text query

        Args:
            query (str): a description of the object
            room_ids (List[int], optional): The room ids. Defaults to [], which means search from all rooms.
            query_method (str, optional): "clip" means using clip features of the objects and the query to search.
                                          Defaults to "clip".
            top_k (int, optional): The number of top results to return. Default to 1.
            negative_prompt (List[str], optional): A list of categories used as negative prompt.


        Returns:
            Tuple[int, int]: The target object id in self.objects and the corresponding room id in self.rooms.
        """

        if query in negative_prompt:
            query_id = negative_prompt.index(query)
        else:
            query_id = None

        if query_id is None:
            query = [query, *negative_prompt]
            query_id = 0
        else:
            query = negative_prompt

        print(f"query_id: {query_id}")
        print(f"categories list: {query}")

        if query is None or query == "":
            is_object_text_valid = False
        
        print(f"object query: {query}")

        query_text_feats = get_text_feats_multiple_templates(
            query, self.clip_model, self.clip_feat_dim
        )  # (len(categories), feat_dim)

        room_ids_list = []
        idd = 0
        for obj in self.objects:
            for i, room in enumerate(self.rooms):
                if obj.room_id == room.room_id:
                    room_ids_list.append(i)
                    break

        if len(room_ids) != 0:
            objects_list = []
            room_ids_list = []
            for i in room_ids:
                 if floor_id != -1:
                    objects_list.extend(self.floors[floor_id].rooms[i].objects)
                    room_ids_list.extend([i] * len(self.floors[floor_id].rooms[i].objects))
                 else:
                    objects_list.extend(self.rooms[i].objects)
                    room_ids_list.extend([i] * len(self.rooms[i].objects))
                    
        objects_list: List[Object]
        if query_method == "clip":
            object_embs = np.array([obj.embedding for obj in objects_list])
            sim_mat = np.dot(query_text_feats, object_embs.T) # shape [2,37]
            top_index = np.argsort(sim_mat[query_id])[::-1][:10]
            for i in top_index:
                print("object id, object name, score: ",  objects_list[i].object_id, objects_list[i].name, sim_mat[0][i])
                # print("object id: ", objects_list[i].object_id)

            top_index = np.argsort(sim_mat[query_id])[::-1][:top_k]
            if len(negative_prompt) > 0:
                cls_ids = np.argmax(sim_mat, axis=0)  # category id for each object
                # print(f"cls_ids: {cls_ids}")
                max_scores = np.max(sim_mat, axis=0)  # max scores for each object
                obj_ids = np.where(cls_ids == query_id)[
                    0
                ]  # find the obj ids that assign max score to the target category
                if len(obj_ids) > 0:
                    obj_scores = max_scores[obj_ids]
                    resort_ids = np.argsort(
                        -obj_scores
                    )  # sort the obj ids based on max score (descending)
                    top_index = obj_ids[resort_ids]  # get the top index
                    top_index = top_index[:top_k]

            target_object_id = [objects_list[i].object_id for i in top_index]
            target_object_score = [sim_mat[0][i] for i in top_index]
            target_room_id = [room_ids_list[i] for i in top_index]
            target_id = []
            for ti in target_object_id:
                target_id.append(
                    [i for i, x in enumerate(self.objects) if x.object_id == ti][0]
                )

            return target_id, target_room_id, target_object_score
        return NotImplementedError    

    def query_room_new(
        self, query: str, floor_id: int = -1, query_method: str = "view_embedding"
    ) -> List[int]:
        """search a room node with a text query

        Args:
            query (str): a text describing the room
            floor_id (int): -1 means global search. 0-(max_floor - 1) means searching the target room on a specific floor.
            query_method (str): "label" use pre-defined label stored in the room node. "view_embedding" use the room embedding
                                stored in the room node. "children_embedding" use all children objects' embedding and find
                                the most representative one for the room.
        Returns:
            (Room): the target room ids in self.rooms which matches th equery the best
        """
        is_room_text_valid = "unknown" not in query.lower()
        if query is None or query == "":
            is_room_text_valid = False
        query_text_feats = get_text_feats_multiple_templates(
            [query], self.clip_model, self.clip_feat_dim
        )

        rooms_list = self.rooms
        if floor_id != -1:
            rooms_list = self.floors[floor_id].rooms
            # for room_id in self.floors[floor_id].rooms:
            #     for room in self.rooms:
            #         if room.room_id == room_id:
            #             rooms_list.append(room)
        rooms_list: List[Room]
        if query_method == "label" and is_room_text_valid:
            print("query room use label")
            for room in rooms_list:
                assert (
                    room.name is not None
                ), "The name attribute for the room has not been generated"
            room_names_list = [room.name for room in rooms_list]
            # print(room_names_list)
            room_embs = get_text_feats_multiple_templates(
                room_names_list, self.clip_model, self.clip_feat_dim
            )
            similarity = np.dot(query_text_feats, room_embs.T)
            # similarity = compute_similarity(query_text_feats, room_embs)
            top_index = np.argsort(similarity[0])[::-1]
            # print the top 3 matching rooms
            for i in top_index[:3]:
                print("room: ", rooms_list[i].room_id, rooms_list[i].name, similarity[0][i])
                # print("room: ", rooms_list[i].room_id)

            same_sim_indices = []
            tar_sim = similarity[0, top_index[0]]
            same_sim_indices.append(top_index[0])
            for i in top_index[1:]:
                if np.abs(similarity[0, i] - tar_sim) < 1e-3:
                    same_sim_indices.append(i)

            target_rooms = [rooms_list[i] for i in same_sim_indices]
            target_room_ids = [target_room.room_id for target_room in target_rooms]
            target_ids = [
                #i for i, x in enumerate(self.rooms) if x.room_id in target_room_ids
                i for i, x in enumerate(rooms_list) if x.room_id in target_room_ids
            ]
            
            return target_ids
        else:
            print("query room use view embedding") 
            # room2query_sim = dict()
            # for room in self.rooms:
            #     # find best goal-img id for each room
            #     room_query_sim_median = np.max(
            #         np.dot(query_text_feats, np.stack(room.embeddings).T)
            #     )
            #     # room_query_sim_median = np.max(compute_similarity(query_text_feats, np.stack(room.embeddings)))
            #     room2query_sim[room.room_id] = room_query_sim_median
            room2query_sim = dict()
            room2query_feat = dict()   # 保存对应的特征向量
            room2query_id = dict()     # 保存对应的embedding索引

            #for room in self.rooms:
            for room in rooms_list:
                embeddings = np.stack(room.embeddings)   # [view_num, 768]
                sims = np.dot(query_text_feats, embeddings.T)  # [1, view_num]，query和每个view的相似度
                max_idx = np.argmax(sims)                # 找到最大相似度的位置
                max_sim = sims[0, max_idx]               # 最大相似度值
                max_feat = embeddings[max_idx]           # 对应的特征向量 (768,)
                
                room2query_sim[room.room_id] = max_sim
                room2query_feat[room.room_id] = max_feat
                room2query_id[room.room_id] = max_idx

            room2query_sim_sorted = {
                int(k.split("_")[-1]): v
                for k, v in sorted(
                    room2query_sim.items(), key=lambda item: item[1], reverse=True
                )
            }
            if is_room_text_valid:
                return list(room2query_sim_sorted.keys())[
                    0 : min(len(room2query_sim_sorted), 5)
                ]  # return three highest-ranking rooms
            else:
                return list(room2query_sim_sorted.keys())[
                    0 : min(len(room2query_sim_sorted), 10)
                ]  # return three highest-ranking rooms
        
        # elif query_method == "children_embedding":
        #     return NotImplementedError

    def query_room(
        self, query: str, floor_id: int = -1, query_method: str = "view_embedding"
    ) -> List[int]:
        """search a room node with a text query

        Args:
            query (str): a text describing the room
            floor_id (int): -1 means global search. 0-(max_floor - 1) means searching the target room on a specific floor.
            query_method (str): "label" use pre-defined label stored in the room node. "view_embedding" use the room embedding
                                stored in the room node. "children_embedding" use all children objects' embedding and find
                                the most representative one for the room.
        Returns:
            (Room): the target room ids in self.rooms which matches th equery the best
        """
        is_room_text_valid = "unknown" not in query.lower()
        if query is None or query == "":
            is_room_text_valid = False
        query_text_feats = get_text_feats_multiple_templates(
            [query], self.clip_model, self.clip_feat_dim
        )

        rooms_list = self.rooms
        if floor_id != -1:
            rooms_list = self.floors[floor_id].rooms
            # for room_id in self.floors[floor_id].rooms:
            #     for room in self.rooms:
            #         if room.room_id == room_id:
            #             rooms_list.append(room)
        rooms_list: List[Room]
        if query_method == "label" and is_room_text_valid:
            print("query room use label")
            for room in rooms_list:
                assert (
                    room.name is not None
                ), "The name attribute for the room has not been generated"
            room_names_list = [room.name for room in rooms_list]
            # print(room_names_list)
            room_embs = get_text_feats_multiple_templates(
                room_names_list, self.clip_model, self.clip_feat_dim
            )
            similarity = np.dot(query_text_feats, room_embs.T)
            # similarity = compute_similarity(query_text_feats, room_embs)
            top_index = np.argsort(similarity[0])[::-1]
            # print the top 3 matching rooms
            for i in top_index[:3]:
                print("room: ", rooms_list[i].room_id, rooms_list[i].name, similarity[0][i])
                # print("room: ", rooms_list[i].room_id)

            same_sim_indices = []
            tar_sim = similarity[0, top_index[0]]
            same_sim_indices.append(top_index[0])
            for i in top_index[1:]:
                if np.abs(similarity[0, i] - tar_sim) < 1e-3:
                    same_sim_indices.append(i)

            target_rooms = [rooms_list[i] for i in same_sim_indices]
            target_room_ids = [target_room.room_id for target_room in target_rooms]
            target_ids = [
                #i for i, x in enumerate(self.rooms) if x.room_id in target_room_ids
                i for i, x in enumerate(rooms_list) if x.room_id in target_room_ids
            ]
            return target_ids
        else:
            print("query room use view embedding")
            room2query_sim = dict()
            #for room in self.rooms:
            for room in rooms_list:
                room_query_sim_median = np.max(
                    np.dot(query_text_feats, np.stack(room.embeddings).T)
                )
                # room_query_sim_median = np.max(compute_similarity(query_text_feats, np.stack(room.embeddings)))
                room2query_sim[room.room_id] = room_query_sim_median
            room2query_sim_sorted = {
                int(k.split("_")[-1]): v
                for k, v in sorted(
                    room2query_sim.items(), key=lambda item: item[1], reverse=True
                )
            }
            return list(room2query_sim_sorted.keys())[
                0 : min(len(room2query_sim_sorted), 3)
            ]  # return three highest-ranking rooms
        # elif query_method == "children_embedding":
        #     return NotImplementedError

    def query_object(
        self,
        query: str,
        floor_id: int = -1,
        room_ids: List[int] = [],
        query_method: str = "clip",
        top_k: int = 1,
        negative_prompt: List[str] = [],
    ) -> Tuple[List[int], List[int]]:
        """search an object (from a room) with a text query

        Args:
            query (str): a description of the object
            room_ids (List[int], optional): The room ids. Defaults to [], which means search from all rooms.
            query_method (str, optional): "clip" means using clip features of the objects and the query to search.
                                          Defaults to "clip".
            top_k (int, optional): The number of top results to return. Default to 1.
            negative_prompt (List[str], optional): A list of categories used as negative prompt.


        Returns:
            Tuple[int, int]: The target object id in self.objects and the corresponding room id in self.rooms.
        """

        if query in negative_prompt:
            query_id = negative_prompt.index(query)
        else:
            query_id = None

        if query_id is None:
            query = [query, *negative_prompt]
            query_id = 0
        else:
            query = negative_prompt

        print(f"query_id: {query_id}")
        print(f"categories list: {query}")

        if query is None or query == "":
            is_object_text_valid = False
        
        print(f"object query: {query}")

        query_text_feats = get_text_feats_multiple_templates(
            query, self.clip_model, self.clip_feat_dim
        )  # (len(categories), feat_dim)
        # print(f"text_feats.shape: {query_text_feats.shape}")
        # print(query_text_feats[:, :10])

        room_ids_list = []
        idd = 0
        for obj in self.objects:
            # print(idd)
            # print(obj.object_id)
            # idd = idd + 1
            for i, room in enumerate(self.rooms):
                if obj.room_id == room.room_id:
                    room_ids_list.append(i)
                    break

        # import pdb; pdb.set_trace()

        if len(room_ids) != 0:
            objects_list = []
            room_ids_list = []
            for i in room_ids:
                if floor_id != -1:
                    objects_list.extend(self.floors[floor_id].rooms[i].objects)
                    room_ids_list.extend([i] * len(self.floors[floor_id].rooms[i].objects))
                else:
                    objects_list.extend(self.rooms[i].objects)
                    room_ids_list.extend([i] * len(self.rooms[i].objects))
        objects_list: List[Object]
        if query_method == "clip":
            object_embs = np.array([obj.embedding for obj in objects_list])
            sim_mat = np.dot(query_text_feats, object_embs.T)
            # sim_mat = compute_similarity(query_text_feats, object_embs)  # (len(categories), len(objects_list))
            top_index = np.argsort(sim_mat[query_id])[::-1][:10]
            # top_index = np.argsort(sim_mat[query_id])[::1][:10]
            for i in top_index:
                print("object name, score: ", objects_list[i].name, sim_mat[0][i])
                print("object id: ", objects_list[i].object_id)

            # plt.hist(sim_mat.flatten(), bins=100)
            # plt.show()

            top_index = np.argsort(sim_mat[query_id])[::-1][:top_k]
            if len(negative_prompt) > 0:
                cls_ids = np.argmax(sim_mat, axis=0)  # category id for each object
                print(f"cls_ids: {cls_ids}")
                max_scores = np.max(sim_mat, axis=0)  # max scores for each object
                obj_ids = np.where(cls_ids == query_id)[
                    0
                ]  # find the obj ids that assign max score to the target category
                if len(obj_ids) > 0:
                    obj_scores = max_scores[obj_ids]
                    resort_ids = np.argsort(
                        -obj_scores
                    )  # sort the obj ids based on max score (descending)
                    top_index = obj_ids[resort_ids]  # get the top index
                    top_index = top_index[:top_k]

            target_object_id = [objects_list[i].object_id for i in top_index]
            target_room_id = [room_ids_list[i] for i in top_index]
            target_id = []
            for ti in target_object_id:
                target_id.append(
                    [i for i, x in enumerate(self.objects) if x.object_id == ti][0]
                )

            return target_id, target_room_id
        return NotImplementedError

    def query_hierarchy_protected_icra(
        self, query_instruction: str, top_k: int = 1, use_gpt: bool = False
    ) -> Tuple[Floor, Room, List[Object]]:
        """return the target floor, room, object.

        Args:
            query (str): the long query like "object X in room Y on floor Z"
            top_k (int, optional): The number of top results to return. Default to 1.

        Returns:
            Tuple[Floor, List[Room], List[Object]]: return a floor object, a room object and a list of object objects
        """
        negative_labels = ["background", "wall"]
        #negative_labels = ["background"]
        
        start_time = time.time()
        floor_query, room_query, object_query = parse_hier_query_use_prompt_insentence_parse_icra(self.cfg, query_instruction)    
        llm_parse_time = time.time() - start_time
        print("llm_parse_time: " , llm_parse_time)
        # log these in a txt file
        # with open("room_obj_query_log.txt", "a") as f:
        #     f.write(f"query: {query_instruction} -- {floor_query}, {room_query}, {object_query}\n")
        # print((f"query: {query_instruction} -- {floor_query}, {room_query}, {object_query}\n"))
        
        if "Exhibition" in room_query:
            negative_labels = ["wall"]

        floor_id = self.query_floor(floor_query) if floor_query is not None else -1
        print(f"floor id: {floor_id}")

        is_dectect_room = "unknown" not in room_query.lower()
        if room_query is None or room_query == "":
            is_dectect_room = False

        
        is_dectect_obj = "unknown" not in object_query.lower()
        if object_query is None or object_query == "":
            is_dectect_obj = False
        
        print("is_dectect_room: ", is_dectect_room, "is_dectect_obj: ", is_dectect_obj)

        # ## offline use gpt to check and update object-reterival
        if use_gpt:
            res_dict, object_ids, room_ids = self.query_room_obj_text_and_clip(query_instruction, room_query, object_query, negative_prompt=negative_labels, room_query_method="label", object_query_method="clip", update_flag=True)
            res_dict["LLM_Parse_Time"] = llm_parse_time
            res_dict["room_query"] = room_query
            res_dict["object_query"] = object_query
            res_dict["negative_labels"] = negative_labels
            return (
                self.floors[floor_id] if floor_id != -1 else None,
                [self.rooms[k] for k in room_ids],
                [self.objects[i] for i in object_ids],
                res_dict,
            )        
        room_ids = (
            # self.query_room_new(room_query, floor_id=floor_id, query_method="label")
            self.query_room_new(room_query, floor_id=floor_id, query_method="label")
            # self.query_room(room_query, floor_id=floor_id)
            if room_query is not None
            else []
        )
        print("room_ids: ", room_ids)

        best_room_id = room_ids[0] if len(room_ids) > 0 else -1
        # best room, best object
        print(f"room ids: {room_ids}")
        object_ids, room_ids, object_scores = (
            self.query_object_new(
                object_query,
                floor_id=floor_id,
                room_ids=room_ids,
                top_k=top_k,
                negative_prompt=negative_labels,
            )
            if object_query is not None
            else ([], [])
        )
        # print(f"object ids: {object_ids}")

        res_dict = dict()
        res_dict["room_query"] = room_query
        res_dict["object_query"] = object_query
        res_dict["negative_labels"] = negative_labels
        res_dict["LLM_Parse_Time"] = llm_parse_time
        res_dict["FastMatching"] = 0.0
        res_dict["ObjectInImageCheck"] = 0.0
        res_dict["VLM_Rethinking"] = 0.0
        res_dict["Re_Matching"] = 0.0
        res_dict["Total_Time"] = 0.0
        
        

        print("query_hierarchy_protected_icra fun cost: ", time.time()-start_time)
        return (
            self.floors[floor_id] if floor_id != -1 else None,
            [self.rooms[k] for k in room_ids],
            [self.objects[i] for i in object_ids],
            res_dict,
        )



    def query_hierarchy_protected(
        self, query_instruction: str, ans: str = '', top_k: int = 1, use_gpt: bool = False
    ) -> Tuple[Floor, Room, List[Object]]:
        """return the target floor, room, object.

        Args:
            query (str): the long query like "object X in room Y on floor Z"
            top_k (int, optional): The number of top results to return. Default to 1.

        Returns:
            Tuple[Floor, List[Room], List[Object]]: return a floor object, a room object and a list of object objects
        """
        #negative_labels = ["background", "wall"]
        negative_labels = ["background", "wall", "divider", "ledge", "pillar", "tape", "stairs", "door", "doors", "stair", "window", "glass", "railing", "glass doors", "whiteboard", "sliding door", "carpet", "picture", "projector screen", "speaker", "cart", "light switch", "bathroom stall", "tissue box"]
        start_time = time.time()
        if ans == '':
            floor_query, room_query, object_query = parse_hier_query_use_prompt_insentence_parse(self.cfg, query_instruction)
        else:
            floor_query = ans.split(",")[0].strip()
            room_query = ans.split(",")[1].strip()
            object_query = ans.split(",")[2].strip()

        llm_parse_time = time.time() - start_time
        # log these in a txt file
        with open("room_obj_query_log.txt", "a") as f:
            f.write(f"query: {query_instruction} -- {floor_query}, {room_query}, {object_query}\n")
        print((f"query: {query_instruction} -- {floor_query}, {room_query}, {object_query}\n"))
        # if "展厅" in room_query:
        #     negative_labels = ["wall"]

        res_dict = dict()
        res_dict["object_query"] = object_query
        res_dict["room_query"] = room_query
        res_dict["negative_labels"] = negative_labels
                
        
        # import pdb; pdb.set_trace()

        floor_id = self.query_floor(floor_query) if floor_query is not None else -1
        print(f"floor id: {floor_id}")

        is_dectect_room = "unknown" not in room_query.lower()
        if room_query is None or room_query == "":
            is_dectect_room = False

        
        is_dectect_obj = "unknown" not in object_query.lower()
        if object_query is None or object_query == "":
            is_dectect_obj = False
        
        print("is_dectect_room: ", is_dectect_room, "is_dectect_obj: ", is_dectect_obj)

        if use_gpt:
            res_dict, object_ids, room_ids = self.query_room_obj_text_and_clip(query_instruction, room_query, object_query, negative_prompt=negative_labels, room_query_method="label", object_query_method="clip", update_flag=True)
            res_dict["LLM_Parse_Time"] = llm_parse_time
            res_dict["object_query"] = object_query
            res_dict["room_query"] = room_query
            res_dict["negative_labels"] = negative_labels
            return (
                self.floors[floor_id] if floor_id != -1 else None,
                [self.rooms[k] for k in room_ids],
                [self.objects[i] for i in object_ids],
                res_dict,
            )        
        
        room_ids = (
            self.query_room_new(room_query, floor_id=floor_id, query_method="label")
            # self.query_room_new(room_query, floor_id=floor_id, query_method="view_embedding")
            # self.query_room(room_query, floor_id=floor_id)
            if room_query is not None
            else []
        )

        best_room_id = room_ids[0] if len(room_ids) > 0 else -1
        # best room, best object

        print(f"room ids: {room_ids}")
        print("negative_labels: ", negative_labels)
        object_ids, room_ids, object_scores = (
            self.query_object_new(
                object_query,
                floor_id=floor_id,
                room_ids=room_ids,
                top_k=top_k,
                negative_prompt=negative_labels,
            )
            if object_query is not None
            else ([], [])
        )
        res_dict["object_scores"] = object_scores
        # save best view
        # best_view_image_path, _, _ = self.get_object_info(object_ids[0])
        # print(f"object ids: {object_ids}")        
        return (
            self.floors[floor_id] if floor_id != -1 else None,
            [self.floors[floor_id].rooms[k] for k in room_ids] if floor_id != -1 else [self.rooms[k] for k in room_ids],
            [self.objects[i] for i in object_ids],
            res_dict,
        )


    def query_hierarchy(
        self, query: str, top_k: int = 1
    ) -> Tuple[Floor, Room, List[Object]]:
        """return the target floor, room, and the list of top k objects

        Args:
            query (str): the long query like "object X in room Y on floor Z"
            top_k (int, optional): The number of top results to return. Default to 1.

        Returns:
            Tuple[Floor, List[Room], List[Object]]: return a floor object, a room object and a list of object objects
        """

        negative_labels = ["background"]

        floor_query, room_query, object_query = parse_hier_query(self.cfg, query)
        # log these in a txt file
        with open("room_obj_query_log.txt", "a") as f:
            f.write(f"query: {query} -- {floor_query}, {room_query}, {object_query}\n")

        print((f"query: {query} -- {floor_query}, {room_query}, {object_query}\n"))
        floor_id = self.query_floor(floor_query) if floor_query is not None else -1
        print(f"floor id: {floor_id}")
        room_ids = (
            # self.query_room(room_query, floor_id=floor_id, query_method="label")
            self.query_room(room_query, floor_id=floor_id)
            if room_query is not None
            else []
        )
        print(f"room ids: {room_ids}")
        object_ids, room_ids = (
            self.query_object(
                object_query,
                floor_id=floor_id,
                room_ids=room_ids,
                top_k=top_k,
                negative_prompt=negative_labels,
            )
            if object_query is not None
            else ([], [])
        )
        # print(f"object ids: {object_ids}")
        # import pdb; pdb.set_trace()
        return (
            self.floors[floor_id] if floor_id != -1 else None,
            [self.floors[floor_id].rooms[k] for k in room_ids] if floor_id != -1 else [self.rooms[k] for k in room_ids],
            #[self.rooms[k] for k in room_ids],
            [self.objects[i] for i in object_ids],
        )

    def save_full_pcd(self, path):
        """
        Save the full pcd to disk
        :param path: str, The path to save the full pcd
        """
        if not os.path.exists(path):
            os.makedirs(path)
        o3d.io.write_point_cloud(os.path.join(path, "full_pcd.ply"), self.full_pcd)
        print("full pcd saved to disk in {}".format(path))
        return None

    def load_full_pcd(self, path):
        """
        Load the full pcd from disk
        :param path: str, The path to load the full pcd
        """
        if not os.path.exists(path):
            print("full pcd not found in {}".format(path))
            return None
        self.full_pcd = o3d.io.read_point_cloud(os.path.join(path, "full_pcd.ply"))
        print(
            "full pcd loaded from disk with shape {}".format(
                np.asarray(self.full_pcd.points).shape
            )
        )
        return self.full_pcd

    def save_full_pcd_feats(self, path):
        """
        Save the full pcd with feats to disk
        :param path: str, The path to save the full pcd feats
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        valid_mask_pcds = []
        valid_mask_feats = []

        for pcd, feat in zip(self.mask_pcds, self.mask_feats):
            if len(pcd.points) > 0:
                valid_mask_pcds.append(pcd)
                valid_mask_feats.append(feat)

        self.mask_pcds = valid_mask_pcds
        self.mask_feats = valid_mask_feats


        for i, feat in enumerate(self.mask_feats):
            print(f"mask_feat[{i}] shape: {np.array(feat).shape}")

        # check if the full pcd feats is empty list
        if len(self.mask_feats) != 0:
            self.mask_feats = np.array(self.mask_feats)
            torch.save(
                torch.from_numpy(self.mask_feats), os.path.join(path, "mask_feats.pt")
            )
        if len(self.full_feats_array) != 0:
            torch.save(
                torch.from_numpy(self.full_feats_array),
                os.path.join(path, "full_feats.pt"),
            )
        print("full pcd feats saved to disk in {}".format(path))
        return None

    def load_full_pcd_feats(self, path, full_feats=False, normalize=True):
        """
        Load the full pcd with feats from disk
        :param path: str, The path to load the full pcd feats
        :param full_feats: bool, Whether to load the full feats or the mask feats
        :param normalize: bool, Whether to normalize the feats
        """
        if not os.path.exists(path):
            print("full pcd feats not found in {}".format(path))
            return None
        if full_feats:
            self.full_feats_array = torch.load(
                os.path.join(path, "full_feats.pt")
            ).float()
            if normalize:
                self.full_feats_array = (
                    torch.nn.functional.normalize(self.full_feats_array, p=2, dim=-1)
                    .cpu()
                    .numpy()
                )
            else:
                self.full_feats_array = self.full_feats_array.cpu().numpy()
            print(
                "full pcd feats loaded from disk with shape {}".format(
                    self.full_feats_array.shape
                )
            )
            return self.full_feats_array
        else:
            self.mask_feats = torch.load(os.path.join(path, "mask_feats.pt")).float()
            if normalize:
                self.mask_feats = (
                    torch.nn.functional.normalize(self.mask_feats, p=2, dim=-1)
                    .cpu()
                    .numpy()
                )
            else:
                self.mask_feats = self.mask_feats.cpu().numpy()
            print(
                "full pcd feats loaded from disk with shape {}".format(
                    self.mask_feats.shape
                )
            )
            return self.mask_feats

    def print_details(self):
        """
        Print the details of the graph
        """
        print("number of floors: ", len(self.floors))
        print("number of rooms: ", len(self.rooms))
        print("number of objects: ", len(self.objects))
        return None

    def save_masked_pcds(self, path, state="both"):
        """
        Save the masked pcds to disk
        :params state: str 'both' or 'objects' or 'full' to save the full masked pcds or only the objects.
        """
        # # remove any small pcds
        tqdm.write("-- removing small and empty masks --")
        # for i, pcd in enumerate(self.mask_pcds):
        for i, pcd in reversed(list(enumerate(self.mask_pcds))):
            if len(pcd.points) < 10:
                self.mask_pcds.pop(i)
                self.mask_feats.pop(i)

        # for i, pcd in enumerate(self.mask_pcds):
        for i, pcd in reversed(list(enumerate(self.mask_pcds))):
            if pcd.is_empty():
                self.mask_pcds.pop(i)
                self.mask_feats.pop(i)

        if state == "both":
            if not os.path.exists(path):
                os.makedirs(path)
            objects_path = os.path.join(path, "objects")
            if not os.path.exists(objects_path):
                os.makedirs(objects_path)
            print("number of masked pcds: ", len(self.mask_pcds))
            print("number of mask_feats: ", len(self.mask_feats))
            for i, pcd in enumerate(self.mask_pcds):
                o3d.io.write_point_cloud(
                    os.path.join(objects_path, "pcd_{}.ply".format(i)), pcd
                )

            masked_pcd = o3d.geometry.PointCloud()
            for pcd in self.mask_pcds:
                pcd.paint_uniform_color(np.random.rand(3))
                masked_pcd += pcd
            o3d.io.write_point_cloud(os.path.join(path, "masked_pcd.ply"), masked_pcd)
            print("masked pcds saved to disk in {}".format(path))

        elif state == "objects":
            if not os.path.exists(path):
                os.makedirs(path)
            for i, pcd in enumerate(self.mask_pcds):
                o3d.io.write_point_cloud(
                    os.path.join(objects_path, "pcd_{}.ply".format(i)), pcd
                )
            print("masked pcds saved to disk in {}".format(path))

        elif state == "full":
            if not os.path.exists(path):
                os.makedirs(path)
            masked_pcd = o3d.geometry.PointCloud()
            for pcd in self.mask_pcds:
                pcd.paint_uniform_color(np.random.rand(3))
                masked_pcd += pcd
            o3d.io.write_point_cloud(os.path.join(path, "masked_pcd.ply"), masked_pcd)
            print("masked pcds saved to disk in {}".format(path))

    def load_masked_pcds_new(self, path):
        """
        Load the masked pcds from disk
        """
        # make sure that self.mask_feats is already loaded
        if len(self.mask_feats) == 0:
            print("load full pcd feats first")
            return None
        if os.path.exists(os.path.join(path, "objects")):
            self.mask_pcds = []
            number_of_pcds = len(os.listdir(os.path.join(path, "objects")))
            not_found = []
            for i in range(number_of_pcds):
                if os.path.exists(
                    os.path.join(path, "objects", "pcd_{}.ply".format(i))
                ):
                    self.mask_pcds.append(
                        o3d.io.read_point_cloud(
                            os.path.join(path, "objects", "pcd_{}.ply".format(i))
                        )
                    )
                else:
                    print("masked pcd {} not found in {}".format(i, path))
                    not_found.append(i)
            # # new add
            # for i, pcd in reversed(list(enumerate(self.mask_pcds))):
            #     if len(pcd.points) < 100:
            #         self.mask_pcds.pop(i)
            print(
                "number of masked pcds loaded from disk {}".format(len(self.mask_pcds))
            )                        
            # remove masks_feats that are not found
            not_found = [i for i in not_found if i < len(self.mask_feats)]            
            self.mask_feats = np.delete(self.mask_feats, not_found, axis=0)
            print(
                "number of mask_feats loaded from disk {}".format(len(self.mask_feats))
            )
            # import pdb; pdb.set_trace()

            # # # new
            # for i, pcd in enumerate(self.mask_pcds):
            #     if len(pcd.points) < 10:
            #         self.mask_pcds.pop(i)
            #         self.mask_feats.pop(i)

            return self.mask_pcds
        else:
            print("masked pcds for objects not found in {}".format(path))
            return None

    def load_masked_pcds(self, path):
        """
        Load the masked pcds from disk
        """
        # make sure that self.mask_feats is already loaded
        if len(self.mask_feats) == 0:
            print("load full pcd feats first")
            return None
        if os.path.exists(os.path.join(path, "objects")):
            self.mask_pcds = []
            number_of_pcds = len(os.listdir(os.path.join(path, "objects")))
            not_found = []
            for i in range(number_of_pcds):
                if os.path.exists(
                    os.path.join(path, "objects", "pcd_{}.ply".format(i))
                ):
                    self.mask_pcds.append(
                        o3d.io.read_point_cloud(
                            os.path.join(path, "objects", "pcd_{}.ply".format(i))
                        )
                    )
                else:
                    print("masked pcd {} not found in {}".format(i, path))
                    not_found.append(i)
            print(
                "number of masked pcds loaded from disk {}".format(len(self.mask_pcds))
            )
            # remove masks_feats that are not found
            self.mask_feats = np.delete(self.mask_feats, not_found, axis=0)
            print(
                "number of mask_feats loaded from disk {}".format(len(self.mask_feats))
            )
            # # # new
            # for i, pcd in enumerate(self.mask_pcds):
            #     if len(pcd.points) < 10:
            #         self.mask_pcds.pop(i)
            #         self.mask_feats.pop(i)

            return self.mask_pcds
        else:
            print("masked pcds for objects not found in {}".format(path))
            return None

    def transform(self, transform):
        """
        Transform the openmap full pcd and masked pcds
        :param transform: np.ndarray, The transformation matrix
        """
        self.full_pcd.transform(transform)
        for i, pcd in enumerate(self.mask_pcds):
            self.mask_pcds[i].transform(transform)
        return None

    def visualize_instances(self):
        """
        visualize the instance of obejcts in the graph
        """
        all_objects_pcd = o3d.geometry.PointCloud()
        number_of_objects = 0
        for i, node in enumerate(self.graph.nodes):
            if type(node) == Object:
                print("object name: ", node.name, node.object_id)
                print("number of points: ", len(node.pcd.points))
                all_objects_pcd += node.pcd
                number_of_objects += 1
        print("number of objects: ", number_of_objects)
        o3d.visualization.draw_geometries([all_objects_pcd])
        return None
    
