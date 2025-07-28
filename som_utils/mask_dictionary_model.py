import numpy as np
import json
import torch
import copy
import os
import cv2
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment


def calculate_volume(dimensions):
    length, width, height = dimensions
    volume = length * width * height
    return volume

def euclidean_distance_2d(loc1, loc2):
    loc1_2d = np.array(loc1[:2])
    loc2_2d = np.array(loc2[:2])
    distance_2d = np.linalg.norm(loc1_2d - loc2_2d)
    return distance_2d

def hungarian_matching(tracks, detections, cost_matrix, next_track_id):
    updated_masks = {}
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_tracks = list(range(len(tracks)))
    unmatched_detections = list(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1.2: #TODO: need to be increased
            matches.append((r, c))
            unmatched_tracks.remove(r)
            unmatched_detections.remove(c)
    
    for r, c in matches:
        new_mask_copy = ObjectInfo()
        new_mask_copy.mask = tracks[r]['mask']
        new_mask_copy.instance_id = tracks[r]['instance_id']
        new_mask_copy.class_name = tracks[r]['class_name']
        new_mask_copy.location = tracks[r]['location']
        updated_masks[tracks[r]['instance_id']] = new_mask_copy

    for det_idx in unmatched_detections:
        new_mask_copy = ObjectInfo()
        new_track = detections[det_idx].copy()
        next_track_id += 1
        new_mask_copy.mask = new_track['mask']
        new_mask_copy.instance_id = next_track_id
        new_mask_copy.class_name = new_track['class_name']
        new_mask_copy.location = new_track['location']
        updated_masks[next_track_id] = new_mask_copy

    return updated_masks, next_track_id

@dataclass
class MaskDictionaryModel:
    mask_name:str = ""
    mask_height: int = 900
    mask_width:int = 1600
    promote_type:str = "mask"
    labels:dict = field(default_factory=dict)

    def add_new_frame_annotation(self, mask_list, box_list, label_list, location_list = [], background_value = 0):
        mask_img = torch.zeros(mask_list.shape[-2:])
        anno_2d = {}
        for idx, (mask, box, label) in enumerate(zip(mask_list, box_list, label_list)):
            try:
                final_index = background_value + label
            except:
                final_index = background_value + idx + 1

            if mask.shape[0] != mask_img.shape[0] or mask.shape[1] != mask_img.shape[1]:
                raise ValueError("The mask shape should be the same as the mask_img shape.")
            # mask = mask
            mask_img[mask == True] = final_index
            # print("label", label)
            name = label
            box = box # .numpy().tolist()
            loc = None
            if len(location_list) > 0:
                loc = location_list[idx]
            new_annotation = ObjectInfo(instance_id = final_index, mask = mask, class_name = name, x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3], location = loc)
            anno_2d[final_index] = new_annotation

        # np.save(os.path.join(output_dir, output_file_name), mask_img.numpy().astype(np.uint16))
        self.mask_height = mask_img.shape[0]
        self.mask_width = mask_img.shape[1]
        self.labels = anno_2d


    def update_masks(self, tracking_annotation_dict, iou_threshold=0.8, objects_count=0):
        updated_masks = {}

        tracks, detections = [], []

        for seg_obj_id, seg_mask in self.labels.items():  # tracking_masks 
            if seg_mask.mask.sum() == 0:
                    continue
            detections.append({
                "bbox": [seg_mask.x1, seg_mask.y1, seg_mask.x2, seg_mask.y2],
                "instance_id": None,
                "class_name": seg_mask.class_name,
                "mask": seg_mask.mask,
                "location": seg_mask.location,
                }
            )
        
        for object_id, object_info in tracking_annotation_dict.labels.items():
            tracks.append({
                "bbox": [object_info.x1, object_info.y1, object_info.x2, object_info.y2],
                "instance_id": object_info.instance_id,
                "class_name": object_info.class_name,
                "mask": object_info.mask,
                "location": object_info.location,
                }
            )

        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self.calculate_iou(track['mask'], det['mask'])  # tensor, numpy
                class_match = 1 if track['class_name'] == det['class_name'] else 0  
                try:
                    dist = euclidean_distance_2d(track['location'], det['location'])
                except:
                    dist = 0.0    
                size = calculate_volume(det['location'][-4:-1])
                if size <= 1:
                    alpha = 0.05 # pedestrain
                elif size > 1 and size <= 2:
                    alpha = 0.03 # cyclelist
                elif size > 2:
                    alpha = 0.01 # vehicle
                cost_matrix[i, j] = 1 - iou + dist * alpha  + (1 - class_match) * 10 #TODO: add distance
        
        self.labels, objects_count = hungarian_matching(tracks, detections, cost_matrix, objects_count)
    
        return objects_count

    def get_target_class_name(self, instance_id):
        return self.labels[instance_id].class_name

    def get_target_location(self, instance_id):
        return self.labels[instance_id].location

    def get_target_logit(self, instance_id):
        return self.labels[instance_id].logit
    
    @staticmethod
    def calculate_iou(mask1, mask2):
        # Convert masks to float tensors for calculations
        mask1 = mask1.to(torch.float32)
        mask2 = mask2.to(torch.float32)
        
        # Calculate intersection and union
        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum() - intersection
        
        # Calculate IoU
        iou = intersection / union
        return iou

    def to_dict(self):
        return {
            "mask_name": self.mask_name,
            "mask_height": self.mask_height,
            "mask_width": self.mask_width,
            "promote_type": self.promote_type,
            "labels": {k: v.to_dict() for k, v in self.labels.items()}
        }
    
    def to_json(self, json_file):
        with open(json_file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
            
    def from_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
            self.mask_name = data["mask_name"]
            self.mask_height = data["mask_height"]
            self.mask_width = data["mask_width"]
            self.promote_type = data["promote_type"]
            self.labels = {int(k): ObjectInfo(**v) for k, v in data["labels"].items()}
        return self


@dataclass
class ObjectInfo:
    instance_id:int = 0
    mask: any = None
    class_name:str = ""
    x1:int = 0
    y1:int = 0
    x2:int = 0
    y2:int = 0
    location: any = None
    logit:float = 0.0

    def get_mask(self):
        return self.mask
    
    def get_id(self):
        return self.instance_id

    def get_location(self):
        return self.location

    def update_box(self):
        nonzero_indices = torch.nonzero(self.mask)
        
    
        if nonzero_indices.size(0) == 0:
            # print("nonzero_indices", nonzero_indices)
            return []
        

        y_min, x_min = torch.min(nonzero_indices, dim=0)[0]
        y_max, x_max = torch.max(nonzero_indices, dim=0)[0]
        

        bbox = [x_min, y_min, x_max, y_max]        
        self.x1 = bbox[0].item()
        self.y1 = bbox[1].item()
        self.x2 = bbox[2].item()
        self.y2 = bbox[3].item()
    
    def to_dict(self):
        try:
            outptus = {
                "instance_id": self.instance_id,
                "class_name": self.class_name,
                "x1": self.x1.item(),
                "y1": self.y1.item(),
                "x2": self.x2.item(),
                "y2": self.y2.item(),
            }
        except:
            outptus = {
                "instance_id": self.instance_id,
                "class_name": self.class_name,
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
            }
        return outptus