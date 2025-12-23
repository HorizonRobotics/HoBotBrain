"""
    Class to represent a floor in a HOV-Sgraph.
"""

import json
import os

import numpy as np
import open3d as o3d




class View:
    """
    Class to represent a View/image in room.
    :param view_id: Unique identifier for the view
    :param name: Name of the floor (e.g., "First", "Second")
    """
    def __init__(self, view_id, room_id, img_id, name=None):
        self.view_id = view_id  # Unique identifier for the view
        self.room_id = room_id # room id of the view belongs to
        self.img_id = img_id # image index of the view in dataset        
        self.object_ids = []  # List of objects inside the view   
        self.name = name  # Name of the floor (e.g., "First", "Second")
        self.img_path = None # image path of the view in dataset       
        self.embedding = None  # CLIP Embedding of the view             
        self.text_discription = []
        # self.clip_embedding = []  # List of tensors of visual embedd
        # self.txt_descriptions = []  # List of text descriptions of the view

    def add_object(self, objectt_id):
        """
        Method to add objects to the room
        :param objectt: Object object to be added to the room
        """
        self.object_ids.append(objectt_id)  # Method to add objects to the view
    
    def save(self, path):
        metadata = {
        "view_id": int(self.view_id) if isinstance(self.view_id, np.integer) else self.view_id,
        "room_id": int(self.room_id) if isinstance(self.room_id, np.integer) else self.room_id,
        "img_id": int(self.img_id) if isinstance(self.img_id, np.integer) else self.img_id,
        "object_ids": [int(x) if isinstance(x, np.integer) else x for x in self.object_ids],
        "img_path": self.img_path,
        "text_discription": [str(x) for x in self.text_discription],  # 👈 强制转成 str
        }
        with open(os.path.join(path, str(self.view_id) + ".json"), "w") as outfile:
            json.dump(metadata, outfile)

    # def save(self, path):
    #     """
    #     Save the floor in folder as ply for the point cloud
    #     and json for the metadata
    #     """
    #     # save the point cloud
    #     # o3d.io.write_point_cloud(os.path.join(path, str(self.floor_id) + ".ply"), self.pcd)
    #     # save the metadata
    #     metadata = {
    #         "view_id": self.view_id,
    #         "room_id": self.room_id,
    #         "img_id": self.img_id,
    #         "object_ids": self.object_ids,
    #         "img_path": self.img_path,
    #         # "embedding": self.embedding,
    #         # "name": self.name,
    #         "text_discription": self.text_discription,            
    #     }
    #     with open(os.path.join(path, str(self.view_id) + ".json"), "w") as outfile:
    #         json.dump(metadata, outfile)

    def load(self, path):
        """
        Load the floor from folder as ply for the point cloud
        and json for the metadata
        """
        # load the metadata
        with open(path + "/" + str(self.view_id) + ".json") as json_file:
            metadata = json.load(json_file)            
            self.room_id = metadata["room_id"]
            self.img_id = metadata["img_id"]
            self.img_path = metadata["img_path"]
            self.object_ids = metadata["object_ids"]
            # self.embeddings = np.asarray(metadata["embedding"])
            # self.name = metadata["name"]
            self.text_discription = metadata["text_discription"]

    def __str__(self):
        return f"View ID: {self.view_id}, Name: {self.name}, Img ID: {self.img_id}, Img Path: {self.img_path}, Number of Objects: {len(self.objects)}"
