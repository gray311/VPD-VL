# import os
# import json
# import torch
# import random
# from tqdm import tqdm

# import transformers
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from rouge_score import rouge_scorer
# import pandas as pd
# import argparse


# random.seed(233)

# def parse_args():
#     parser = argparse.ArgumentParser(description="Testing Script")
#     parser.add_argument('--data_file', type=str, default=None)
#     parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
#     parser.add_argument('--lora_path', type=str, default=None)
#     args = parser.parse_args()
#     return args

# args = parse_args()


# def load_model(args):
#     model_path = args.model_path
#     # model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
#     # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
#     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#         model_path,
#         torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2",
#         # device_map="auto",
#     )


#     if args.lora_path is not None:
#         from peft import LoraConfig, get_peft_model
#         from huggingface_hub import hf_hub_download
#         # checkpoint_path = hf_hub_download(args.model_path, args.lora_path)
        
#         checkpoint_path = hf_hub_download("gray311/case", args.lora_path)
#         target_modules=r'.*model.*\.(up_proj|k_proj|down_proj|v_proj|q_proj|o_proj|gate_proj)'
#         config = LoraConfig(
#             r=128, 
#             lora_alpha=256, 
#             target_modules=target_modules, 
#             lora_dropout=0.05,
#             bias="none", 
#             task_type="CAUSAL_LM"
#         )
#         model = get_peft_model(model, config)
#         model.load_state_dict(torch.load(checkpoint_path), strict=False)
#         model = model.merge_and_unload()


#     # default processer
#     processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
#     model.cuda(1)

#     model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

#     pipeline = transformers.pipeline(
#         "text-generation",
#         model=model_id,
#         model_kwargs={"torch_dtype": torch.bfloat16},
#         # device_map="auto",
#     )

#     return model

# """
# python nuplan_test.py \
#     --data_file ./workspace/data/collision/nuplan_mini_val.csv \
#     --model_path gray311/VPD-Driver \
#     --lora_path checkpoint.pt

# """

# if __name__ == "__main__":
#     # model = load_model(args)
#     data = pd.read_csv(args.data_file)
    
#     collision_examples = []
#     for i in tqdm(range(data.shape[0])):
#         line = data.iloc[i, :]
#         if line['Collision'] == True:
#             collision_examples.append(line)
            
#     print(len(collision_examples))
#     print(collision_examples[:2])





import os
import sys
import colorsys
import functools
import random
from enum import IntEnum
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import math
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PltPolygon
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from shapely.ops import unary_union, nearest_points

from PIL import Image
import descartes
import numpy.typing as npt
from pyquaternion import Quaternion
import numpy as np
from tqdm import tqdm

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario, CameraChannel
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.common.actor_state.oriented_box import OrientedBoxPointType
from nuplan.database.utils.boxes.box3d import Box3D, BoxVisibility, box_in_image
from nuplan.database.nuplan_db_orm.frame import Frame
from nuplan.database.utils.geometry import quaternion_yaw, view_points
from nuplan.database.nuplan_db_orm.utils import get_boxes

from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.database.maps_db.map_api import NuPlanMapWrapper
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer



DEFAULT_SENSOR_CHANNEL = 'CAM_F0'
NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', '/home/ec2-user/_Yingzi/VPD-Driver/workspace/data/nuplan/dataset')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', '/home/ec2-user/_Yingzi/VPD-Driver/workspace/data/nuplan/dataset/maps')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', "/home/ec2-user/_Yingzi/VPD-Driver/workspace/data/nuplan/dataset/nuplan-v1.1/splits/mini")
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')


def get_ego_car_speed_and_turn_angle(velocity):
    x, y = velocity.x, velocity.y
    speed = math.sqrt(x ** 2 + y ** 2)
    theta_rad = math.atan2(y, x)
    theta_deg = math.degrees(theta_rad)
    return speed, theta_deg

def get_distance(box: Box3D):
    center_point = box.center
    return math.sqrt(center_point[0] ** 2 + center_point[1] ** 2)

def get_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)

    inter_width = max(0, inter_max_x - inter_min_x)
    inter_height = max(0, inter_max_y - inter_min_y)

    intersection_area = inter_width * inter_height
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    if box1_area > box2_area:
        union_area = box2_area
    else:
        union_area = box1_area

    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou

def get_sensor_coord(points, ego_pose, cs_record, near_plane=1e-8):
    mark = True
     # Transform into the ego vehicle frame for the timestamp of the image.
    points = points - np.array(ego_pose.translation_np).reshape((-1, 1))
    points = np.dot(ego_pose.quaternion.rotation_matrix.T, points)

    # Transform into the camera.
    points = points - np.array(cs_record.translation).reshape((-1, 1))
    points = np.dot(Quaternion(cs_record.rotation).rotation_matrix.T, points)

    # Remove points that are partially behind the camera.
    depths = points[2, :]
    behind = depths < near_plane
    if np.all(behind):
        mark = False

    # if render_behind_cam:
        # Perform clipping on polygons that are partially behind the camera.
    points = NuScenesMapExplorer._clip_points_behind_camera(points, near_plane)
    # elif np.any(behind):
    #     # Otherwise ignore any polygon that is partially behind the camera.
    #     continue

    # Ignore polygons with less than 3 points after clipping.
    if len(points) == 0 or points.shape[1] < 3:
        mark = False

    return points, mark
        


def get_box_in_image(
    box: Box3D,
    intrinsic: npt.NDArray[np.float64],
    imsize: Tuple[float, float],
    vis_level: int = BoxVisibility.ANY,
    front: int = 2,
    min_front_th: float = 0.1,
    with_velocity: bool = False,
) -> bool:
    corners_3d = box.corners()

    # Add the velocity vector endpoint if it is not nan.
    if with_velocity and not np.isnan(box.velocity_endpoint).any():
        corners_3d = np.concatenate((corners_3d, box.velocity_endpoint), axis=1)

    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]
    

    # True if a corner is at least min_front_th meters in front of camera.
    in_front = corners_3d[front, :] > min_front_th
    corners_img = corners_img[:, in_front]
    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    
    max_x, max_y, min_x, min_y = -1, -1, 10000, 10000
    x1, x2, y1, y2 = 0, imsize[0], 0, imsize[1]
    for i in range(corners_img.shape[1]):
        x, y = corners_img[0][i], corners_img[1][i]
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    if min_x < 0: min_x = 0
    if max_x > imsize[0]: max_x = imsize[0]
    if min_y < 0: min_y = 0
    if max_y > imsize[1]: max_y = imsize[1]
        
    # print("="*50)
    # print([min_x, min_y, max_x, max_y])
    # print(corners_img)
    return [min_x, min_y, max_x, max_y]
    
def get_merged_polygon(polygon1, polygon2):
    polygon1_boundary = list(polygon1.exterior.coords)
    polygon2_boundary = list(polygon2.exterior.coords)

    intersection = polygon1.intersection(polygon2)

    if isinstance(intersection, LineString) or isinstance(intersection, Point):
        return Polygon(polygon1_boundary + polygon2_boundary)

    polygon1_unique = [point for point in polygon1_boundary if point not in list(intersection.exterior.coords)]
    polygon2_unique = [point for point in polygon2_boundary if point not in list(intersection.exterior.coords)]

    merged_boundary = polygon1_unique + polygon2_unique
    merged_polygon = Polygon(merged_boundary)

    return merged_polygon

def get_ego_centric_waypoint(points, past=False):
    try:
        if past:
            x0, y0 = points[-1]
            flag = -1
        else:
            x0, y0 = points[0]
            flag = 1
        translated_points = [(x - x0, y - y0) for x, y in points]

        dx, dy = translated_points[1]
        angle = np.arctan2(dy, dx)

        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)]
        ])

        rotated_points = [np.dot(rotation_matrix, np.array([x, y])) for x, y in translated_points]
        flipped_points = [(-y, x * flag) for x, y in rotated_points]

        waypoint = []
        for i, (x, y) in enumerate(flipped_points):
            waypoint.append([x, y])
        sdc_fut_traj_all = np.array(waypoint)
        sdc_fut_traj_mask_all = np.ones_like(sdc_fut_traj_all)

        return sdc_fut_traj_all, sdc_fut_traj_mask_all, False
    except:
        sdc_fut_traj_all = np.zeros((6,2))
        sdc_fut_traj_mask_all = np.zeros((6,2))

        return sdc_fut_traj_all, sdc_fut_traj_mask_all, True



class NuPlanScenarioLoader:
    def __init__(self, scenario_id, frequency=10):
        self.nuplandb_wrapper = NuPlanDBWrapper(
            data_root=NUPLAN_DATA_ROOT,
            map_root=NUPLAN_MAPS_ROOT,
            db_files=NUPLAN_DB_FILES,
            map_version=NUPLAN_MAP_VERSION,
        )
        self.frequency = frequency
        self.scenario_id = scenario_id
        print(scenario_id)
        self.log_db = self.nuplandb_wrapper.get_log_db(scenario_id)
        self.scenes = {}
        for scene in self.log_db.scene:
            initial_lidar_token = scene.lidar_pcs[0].token
            initial_timestamp = scene.lidar_pcs[0].timestamp
            end_lidar_token = scene.lidar_pcs[-1].token
            end_timestamp = scene.lidar_pcs[-1].timestamp

            self.scenes[scene.name] = {
                'scene_token': scene.token,
                'lidar_token': initial_lidar_token,
                'timestamp': initial_timestamp,
                'end_lidar_token': end_lidar_token,
                'end_timestamp': end_timestamp,
                "map_name": self.log_db.map_name,
                'lidar_pcs': scene.lidar_pcs,
            }

        self.camera = {}
        for item in self.log_db.camera:
            self.camera[item.channel] = {
                "translation": item.translation,
                "intrinsic": item.intrinsic,
                "width": item.width,
                "height": item.height,
                "rotation": item.rotation,
                "distortion": item.distortion,
                "token": item.token
            }

        
    def load_scenario(self, scene_name, scenario_duration=20):
        """Loads the scenario based on the scenario ID and initial parameters."""
        lidar_pcs = self.scenes[scene_name]['lidar_pcs']
        self.map_name = self.scenes[scene_name]['map_name']
        self.scene_name = scene_name
        self.scenario = NuPlanScenario(
            data_root=f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini",
            log_file_load_path=f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/{self.scenario_id}",
            initial_lidar_token=self.scenes[scene_name]['lidar_token'],
            initial_lidar_timestamp=self.scenes[scene_name]['timestamp'],
            scenario_type="scenario_type",
            map_root=NUPLAN_MAPS_ROOT,
            map_version=NUPLAN_MAP_VERSION,
            map_name=self.map_name,
            scenario_extraction_info=ScenarioExtractionInfo(
                scenario_name=self.scenario_id,
                scenario_duration=scenario_duration,
                extraction_offset=0.5,
                subsample_ratio=1
            ),
            ego_vehicle_parameters=get_pacifica_parameters(),
            sensor_root=NUPLAN_DATA_ROOT + "/nuplan-v1.1/sensor_blobs",
        )
        self.number_of_frames = self.scenario.get_number_of_iterations()
        print(self.number_of_frames)
        self.velocity_list, self.turn_angle_list = [], []
        # self.ego_pose_token_list = [item.ego_pose_token for item in lidar_pcs]
        # self.lidar_pc_token2ego_pose_token = {item.token: item.ego_pose_token for item in lidar_pcs}
        self.object_infos = self.get_object_infos(self.scenes[scene_name]['timestamp'], self.scenes[scene_name]['end_timestamp'])

        sample_descriptions = {}
        index = 0
        self.frame_num = 0
        self.is_end = False
        while True:
            if index % self.frequency == 0:
                descriptions, sample_token = self.get_sample_description(index)
                if self.is_end == False:
                    sample_descriptions[self.frame_num] = descriptions
                    self.frame_num += 1
                else:
                    index = self.number_of_frames + 1
                    break
            index += 1
            if index >= self.number_of_frames:
                break

        metadata = {
            'sample_descriptions': sample_descriptions
        }
        return metadata

    def get_sample_description(self, index):
        sample_token = self.scenario.get_scenario_tokens()[index]
        sample_description = self.get_ego_information(index)
        sample_description['sample_token'] = sample_token
        return sample_description, sample_token

    def _get_map_segmentation(self, lane, ego_pose, camera, intersection_threshold=1000, connector=False):
        render_outside_im = True

        points = np.array(lane.polygon.exterior.xy)
        poly = Polygon(points.T)
        points = np.vstack((points, np.ones((1, points.shape[1])) * ego_pose.z))

        points, mark = get_sensor_coord(points, ego_pose, camera)
        if mark == False:
            return None

        points = view_points(np.array(points), camera.intrinsic_np, normalize=True)
        inside = np.ones(points.shape[1], dtype=bool)
        inside = np.logical_and(inside, points[0, :] > 1)
        inside = np.logical_and(inside, points[0, :] < camera.width - 1)
        inside = np.logical_and(inside, points[1, :] > 1)
        inside = np.logical_and(inside, points[1, :] < camera.height - 1)
        if render_outside_im:
            if np.all(np.logical_not(inside)):
                return None
        else:
            if np.any(np.logical_not(inside)):
                return None

        points = points[:2, :]
        # points[0, :] = np.clip(points[0, :], 0, im.size[0] - 1)
        # points[1, :] = np.clip(points[1, :], 0, im.size[1] - 1)
        if connector:
            points_2d = [(p0, p1) for (p0, p1) in zip(points[0], points[1])] # to make the connector and lane segmentation overlap 
        else:
            points_2d = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
        
        polygon_proj = Polygon(points_2d)
        polygon_im = Polygon([(0, 0), (0, camera.height), (camera.width, camera.height), (camera.width, 0)])
    
        intersection_polygon = polygon_proj.intersection(polygon_im)
        if intersection_polygon.area < intersection_threshold:
            return None

        point = np.array(intersection_polygon.centroid.coords)

        return (intersection_polygon, point[0])


    def get_ego_information(self, iteration):
        """Retrieves ego vehicle information for a specific timestamp (iteration)."""
        ego_state = self.scenario.get_ego_state_at_iteration(iteration)
        lidar_pc_token = self.scenario._lidarpc_tokens[iteration]
        speed, turn_angle = get_ego_car_speed_and_turn_angle(ego_state.agent.velocity)
        self.velocity_list.append(speed)
        self.turn_angle_list.append(turn_angle)
        timestamp = ego_state.agent.metadata.timestamp_us
        ego_location = ego_state.agent.box.corner(point=OrientedBoxPointType.FRONT_BUMPER)
      
        past_trajectory = self.scenario.get_ego_past_trajectory(iteration, time_horizon=3, num_samples=6) # The duration (in seconds) for how far back in time you want to retrieve ego states
        future_trajectory = self.scenario.get_ego_future_trajectory(iteration, time_horizon=3, num_samples=6)
        
        sdc_fut_traj_all = [(item.waypoint._oriented_box.corner(point=OrientedBoxPointType.FRONT_BUMPER).x, item.waypoint._oriented_box.corner(point=OrientedBoxPointType.FRONT_BUMPER).y) for item in list(future_trajectory)]
        sdc_past_traj_all = [(item.waypoint._oriented_box.corner(point=OrientedBoxPointType.FRONT_BUMPER).x, item.waypoint._oriented_box.corner(point=OrientedBoxPointType.FRONT_BUMPER).y) for item in list(past_trajectory)]
        
        sdc_fut_traj_all, sdc_fut_traj_mask_all, is_end = get_ego_centric_waypoint(sdc_fut_traj_all)
        # sdc_past_traj_all = get_ego_centric_waypoint(sdc_past_traj_all, past=True)

        if is_end:
            self.is_end = True
            return {}

        if int(timestamp / 100000) not in self.object_infos.keys():
            self.is_end = True
            return {}

        scene_image = self.get_scene_image_at_timestamp(iteration)
        # Save image to a folder
        image_save_path = f"images/nuplan-{self.scenario_id}_{self.scene_name}"
        os.makedirs(image_save_path, exist_ok=True)
        image_filename = os.path.join(image_save_path, f"{str(self.frame_num).zfill(6)}.jpg")
        if not os.path.exists(image_filename):
            plt.imsave(image_filename, scene_image)
            print(f"Image saved to {image_filename}")

        tracked_objects = self.scenario.get_tracked_objects_at_iteration(iteration)
        tracked_objects = list(tracked_objects.tracked_objects)
        camera_token = self.camera[DEFAULT_SENSOR_CHANNEL]['token']
        

        object_infos = self.object_infos[int(timestamp / 100000)]
        object_descriptions = {}
        object_num = 0
        ego_pose, camera = None, None
        lane_ids, lane_connector_ids = [], []
        lane_id, lane_dist = self.scenario.map_api.get_distance_to_nearest_map_object(ego_location, SemanticMapLayer.LANE)
        lane_ids.append(lane_id)
        ego_lane_id = lane_id
        for obj in tracked_objects:
            if obj.track_token in object_infos.keys():
                if obj.metadata.category_name not in ['vehicle', 'pedestrain', 'bicycle']: continue
                lane_id, lane_dist = self.scenario.map_api.get_distance_to_nearest_map_object(obj.box.all_corners()[0], SemanticMapLayer.LANE)
                lane = self.scenario.map_api._get_lane(lane_id)
                left_lane, right_lane = lane.adjacent_edges
                lane_ids.append(lane_id)
        
                lane_connector_ids.extend([item.id for item in lane.incoming_edges])
                lane_connector_ids.extend([item.id for item in lane.outgoing_edges])

                object_descriptions[object_num] = object_infos[obj.track_token]
                object_descriptions[object_num]['category_name'] = obj.metadata.category_name
                object_descriptions[object_num]['lane_id'] = lane_id
                ego_pose = object_infos[obj.track_token]['ego_pose']
                camera = object_infos[obj.track_token]['camera']
                object_num += 1

        # lane_ids_tmp = []
        # lane_ids = list(set(lane_ids))
        # lane_connector_ids = list(set(lane_connector_ids))

        # for lane_id in lane_ids:
        #     lane = self.scenario.map_api._get_lane(lane_id)
        #     left_lane, right_lane = lane.adjacent_edges
        #     for lane_tmp in [lane, left_lane, right_lane]:
        #         if lane_tmp is None: continue
        #         lane_ids_tmp.append(lane_tmp.id)
        #         for lane_connector in lane_tmp.incoming_edges:
        #             lane_connector_ids.append(lane_connector.id)
        #             lane_ids_tmp.extend([item.id for item in lane_connector.incoming_edges])
        #             lane_ids_tmp.extend([item.id for item in lane_connector.outgoing_edges])
        #         for lane_connector in lane_tmp.outgoing_edges:
        #             lane_connector_ids.append(lane_connector.id)
        #             lane_ids_tmp.extend([item.id for item in lane_connector.incoming_edges])
        #             lane_ids_tmp.extend([item.id for item in lane_connector.outgoing_edges])
            
        # lane_ids.extend(lane_ids_tmp)
        # lane_ids = list(set(lane_ids))
        # lane_connector_ids = list(set(lane_connector_ids))

        # lane_polygons, lane_connector_polygons, lane_points, lane_labels = [], [], [], []
        # for lane_id in lane_ids:
        #     lane = self.scenario.map_api._get_lane(lane_id)
        #     lane_info = self._get_map_segmentation(lane, ego_pose, camera)
        #     if lane_info is None: continue
        #     lane_polygon, points = lane_info
        #     lane_polygons.append(lane_polygon)
        #     lane_points.append(points)
        #     if lane_id == ego_lane_id:
        #         lane_labels.append("ego")
        #     else:
        #         lane_labels.append("object")
        
        # for lane_connector_id in lane_connector_ids:
        #     lane_connector = self.scenario.map_api._get_lane_connector(lane_connector_id)
        #     lane_info = self._get_map_segmentation(lane_connector, ego_pose, camera, intersection_threshold=200, connector=True)
        #     if lane_info is None: continue
        #     lane_polygon, points = lane_info
        #     lane_connector_polygons.append(lane_polygon)
        
        sample_token = None
        sample_description = {
            "ego_vehicle_location": ego_location,
            "ego_vehicle_heading": turn_angle,
            "ego_vehicle_velocity": speed,
            "ego_future_trajectory": sdc_fut_traj_all,
            "ego_future_trajectory_mask": sdc_fut_traj_mask_all,
            "timestamp": timestamp,
            "filepath": image_filename,
            "sample_annotations": object_descriptions,
            "city": self.map_name,
            "ego_lane_id": ego_lane_id,
            # "lane_polygons": lane_polygons,
            # "lane_points": lane_points,
            # "lane_labels": lane_labels
        }


        return sample_description

    def get_object_infos(self, timestamp_l, timestamp_r):
        object_infos = {}
        for item in tqdm(self.log_db.image):
            if item.timestamp < timestamp_l: continue
            if item.timestamp > timestamp_r: continue
            timestamp = int(item.timestamp / 100000)
            if DEFAULT_SENSOR_CHANNEL in item.filename_jpg:
                
                object_info = []
                for obj_box, obj_box_vehicle in zip(item.boxes(Frame.SENSOR), item.boxes(Frame.VEHICLE)):
                    if box_in_image(
                        obj_box, item.camera.intrinsic_np, (item.camera.width, item.camera.height), vis_level=BoxVisibility.ANY
                    ):
                        points_2d = get_box_in_image(obj_box, item.camera.intrinsic_np, (item.camera.width, item.camera.height), vis_level=BoxVisibility.ANY)
                        dist = get_distance(obj_box_vehicle)
                        overlap = False
                        for i, obj_info in enumerate(object_info):
                            iou =  get_iou(points_2d, obj_info[1])
                            if iou > 0.7:  ### Remove obscured objects
                                if dist > obj_info[2]:
                                    overlap = True
                                else:
                                    del object_info[i]
                        
                        if overlap: continue
                        object_info.append((
                            obj_box.track_token, 
                            points_2d, 
                            dist, 
                            (obj_box_vehicle.center[0], obj_box_vehicle.center[1], obj_box_vehicle.center[2]),
                            (obj_box_vehicle.size[0], obj_box_vehicle.size[1], obj_box_vehicle.size[2]),
                            item.ego_pose,
                            item.camera
                        ))
                
                object_infos[timestamp] = {line[0]: {
                    "instance_token": line[0], 
                    "bounding_box": line[1], 
                    "distance": line[2], 
                    "location": list(line[3]), 
                    "size_in_meter": list(line[4]),
                    "ego_pose": line[5],
                    "camera": line[6]} for i, line in enumerate(object_info)}
                
                if timestamp == 16334196007:
                    print(item.timestamp)
                    print(object_info)
                
        return object_infos

    def get_scene_image_at_timestamp(self, iteration, channel=CameraChannel.CAM_F0):
        """Retrieves scene image for a specific timestamp (iteration)."""
        sensors = self.scenario.get_sensors_at_iteration(iteration, [channel])
        img = sensors.images[channel]
        return img.as_numpy


    def process_scenario(self, save_images=False, image_save_path="images"):
        """Processes the entire scenario across all timestamps and extracts images, ego and object information."""
        num_iterations = self.scenario.get_number_of_iterations()

        # Prepare a container to store the scenario information
        scenario_data = []
        
        for i in range(num_iterations):
            print(f"Processing iteration {i + 1}/{num_iterations}...")

            # Get scene image
            scene_image = self.get_scene_image_at_timestamp(i)
            if save_images:
                # Save image to a folder
                os.makedirs(image_save_path, exist_ok=True)
                image_filename = os.path.join(image_save_path, f"scene_image_{i}.png")
                plt.imsave(image_filename, scene_image)
                print(f"Image saved to {image_filename}")

            # Get ego information
            ego_info = self.get_ego_information_at_timestamp(i)

            # Get tracked object information
            object_info = self.get_tracked_objects_at_timestamp(i)

            # Save the iteration's data
            scenario_data.append({
                "timestamp": self.scenario.get_iteration_time_us(i),
                "ego_info": ego_info,
                "object_info": object_info,
                "scene_image": scene_image if not save_images else image_filename  # Save the path or image array
            })

        def _get_map_segmentation(self, sample_token):
            return None

        return scenario_data


if __name__ == "__main__":
    # Example usage:
    TEST_SCENARIO_ID = "2021.10.05.07.10.04_veh-52_01442_01802"
    scene_name = "scene-0001"
    
    for i in ["12"]:
        scene_name = f"scene-00{str(i)}"
        nuplan_loader = NuPlanScenarioLoader(TEST_SCENARIO_ID)
        metadata = nuplan_loader.load_scenario(
            scene_name=scene_name,
        )

        print(metadata['sample_descriptions'][30])



    

    # _map_db = GPKGMapsDB(
    #     map_version="nuplan-maps-v1.0",
    #     map_root="/data/yingzi_ma/Visual-Prompt-Driving/workspace/nuplan/dataset/maps"
    # )

    # map_api = NuPlanMapWrapper(
    #     maps_db = _map_db,
    #     map_name = "us-nv-las-vegas-strip"
    # )

    # print(map_api)

  
            



