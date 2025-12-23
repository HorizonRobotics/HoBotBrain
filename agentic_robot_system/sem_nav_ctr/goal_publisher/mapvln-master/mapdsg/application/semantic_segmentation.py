import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from hovsg.graph.graph import Graph

# pylint: disable=all

scene_ids = [
    # "demo_iphone_4",
    # "demo_iphone_3",
    "room0",
]

# @hydra.main(version_base=None, config_path="../config", config_name="semantic_segmentation_horizon")
@hydra.main(version_base=None, config_path="../config", config_name="semantic_segmentation")
def main(params: DictConfig):
    
    for scene_id in scene_ids:
        scene_id = params.main.scene_id
        # params.main.scene_id = scene_id
        # Create save directory
        params.main.dataset_path = os.path.join(params.main.dataset_path, scene_id) # params.main.scene_id
        save_dir = os.path.join(params.main.save_path, params.main.dataset, scene_id) # params.main.scene_id
        params.main.save_path = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        print("dataset_path: ", params.main.dataset_path)
        print("save_path: ", save_dir)
        # Create graph
        hovsg = Graph(params)

        # Create feature map
        hovsg.create_feature_map()

        # Save full point cloud, features, and masked point clouds (pcd for all objects)
        hovsg.save_masked_pcds(path=save_dir, state="both")
        hovsg.save_full_pcd(path=save_dir)
        hovsg.save_full_pcd_feats(path=save_dir)

        # # # # for debugging: load preconstructed map as follows
        # hovsg.load_full_pcd(path=save_dir)
        # hovsg.load_full_pcd_feats(path=save_dir)
        # hovsg.load_masked_pcds_new(path=save_dir)

        # import pdb; pdb.set_trace()
        
        # create graph, only if dataset is not Replia or ScanNet
        print(params.main.dataset)
        # import pdb; pdb.set_trace()
        if True or params.main.dataset != "replica" and params.main.dataset != "scannet" and params.pipeline.create_graph:
            hovsg.build_graph(save_path=save_dir)
        else:
            print("Skipping hierarchical scene graph creation for Replica and ScanNet datasets.")


if __name__ == "__main__":
    main()