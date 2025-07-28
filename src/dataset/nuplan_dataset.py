import numpy as np
import torch
import os
import json

import pandas as pd
from tqdm import tqdm
import math
import numpy.typing as npt
import numpy as np
from scipy.spatial.transform import Rotation as R

from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Affine2D
import matplotlib.patches as patches
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

from nuplan.database.utils.geometry import view_points
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import (
    NuPlanScenario,
    CameraChannel,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioExtractionInfo,
)
from nuplan.database.nuplan_db_orm.frame import Frame
from nuplan.database.utils.boxes.box3d import Box3D, BoxVisibility, box_in_image
from nuplan.database.nuplan_db_orm.utils import get_boxes


from blg.datasets.nuplan_dataset import get_sensor_coord
from blg.model.occupancy_model import OccupancyModel
from blg.model.behavior_model import BehaviorModel
from blg.datasets.trajdata_dataloader import get_dataloader
from blg.utils.general_helpers import check_folder, load_yaml_config, CPU
from blg.utils.plot_helpers import plot_map, plot_planning, plot_agent_hist
from blg.generate_from_map import get_lane_graph_from_map, convert_world_to_local

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



def get_3d_box_corners(center, size, heading):
    x, y, z = center
    l, w, h = size

    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    z_corners = [ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2]

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    R = np.array([
        [cos_h, -sin_h, 0],
        [sin_h,  cos_h, 0],
        [0,      0,     1]
    ])

    corners = np.array([x_corners, y_corners, z_corners])

    corners_rotated = R @ corners
    corners = corners_rotated + np.array([[x], [y], [z]])

    return corners
 


def get_relative_direction(agent_points, ego_pose, front_thresh=2.0, side_thresh=1.0):
    agent_center = agent_points[-1][:3]
    ego_position = np.array([ego_pose.x, ego_pose.y, ego_pose.z])
    ego_quat = [ego_pose.qx, ego_pose.qy, ego_pose.qz, ego_pose.qw]
    relative_vec_global = agent_center - ego_position
    rot = R.from_quat(ego_quat)
    relative_ego = rot.inv().apply(relative_vec_global)
    
    return relative_ego

def is_visible_object(relative_xyz, max_distance=20, max_view_angle=80.0):
    x, y, _ = relative_xyz
    distance = np.linalg.norm([x, y])

    if distance > max_distance:
        return False

    angle_deg = np.degrees(np.arctan2(y, x))  
    if angle_deg < -max_view_angle or angle_deg > max_view_angle:
        return False

    return True

def get_box_in_image(
    corners_3d: npt.NDArray[np.float64],
    intrinsic: npt.NDArray[np.float64],
    imsize: Tuple[float, float],
    vis_level: int = BoxVisibility.ANY,
    front: int = 2,
    min_front_th: float = 0.1,
    with_velocity: bool = False,
) -> bool:
    # corners_3d = box.corners()

    # # Add the velocity vector endpoint if it is not nan.
    # if with_velocity and not np.isnan(box.velocity_endpoint).any():
    #     corners_3d = np.concatenate((corners_3d, box.velocity_endpoint), axis=1)

    # print(corners_3d)
    
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
    
        
    return [min_x, min_y, max_x, max_y]

def get_traj_from_behavior(lane_graph, ego_xyhv, lon_cand, lat_cand):
    bm = BehaviorModel(dt=0.1, horizon=30)  # 3 seconds
    # ego_xyhv[3] = 4  # set v to 4 m/s

    # convert the ego point to the frenet coordinate
    # the current pose should be the same for all current lanes, so we just take the first one
    curr_lane = lane_graph.curr.center[0]

    # show all combinations of behaviors
    longitudinal_bev = {
        "keep speed": 0.0,
        "accelerate": 4.0,
        "decelerate": -1.0,
    }
    lateral_bev = {
        "keep lane": lane_graph.curr.center,
        # assume the vehicle can only change one lane at a time
        "left lane-change": lane_graph.left_group[0].center,
        "right lane-change": lane_graph.right_group[0].center,
    }
    beh_traj = []
    exist_last_pt = []

    assert len(lon_cand) == len(lat_cand)
    for lon_i, lat_i in zip(lon_cand, lat_cand):
        target_acc = longitudinal_bev[lon_i]
        target_lanes = lateral_bev[lat_i]
        for target_lane in target_lanes:
            new_traj_xyhv, new_lane = bm.combined_behavior(
                ego_xyhv, target_acc, curr_lane, target_lane
            )

            # to make the plot clear, we remove traj that are too close
            if len(exist_last_pt) > 0:
                new_pt = new_traj_xyhv[-1, 0:2]
                dist = torch.norm(new_pt - torch.stack(exist_last_pt), dim=1)
                if torch.min(dist) < 1:
                    continue
            exist_last_pt.append(new_traj_xyhv[-1, 0:2].clone())
            beh_traj.append(new_traj_xyhv)

    beh_traj = torch.stack(beh_traj, dim=0)
    return beh_traj


    
    

def load_scene_camera(scene_ids, scene_ts, agent_names, agent_world_locs, desired_dt, DEFUALT_SENSOR=None, only_ego_pose=False):
    NUPLAN_DATA_ROOT = "/home/ec2-user/_Yingzi/VPD-Driver/workspace/data/nuplan/dataset"
    NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
    NUPLAN_MAPS_ROOT = "/home/ec2-user/_Yingzi/VPD-Driver/workspace/data/nuplan/dataset/maps"
    NUPLAN_SENSOR_ROOT = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/sensor_blobs"
    NUPLAN_DB_FILES = f"{NUPLAN_DATA_ROOT}/mini"

    SCENARIO_DURATION = 20.0  # [s] duration of the scenario (e.g. extract 20s from when the event occurred)
    EXTRACTION_OFFSET = 0.0  # [s] offset of the scenario (e.g. start at -5s from when the event occurred)
    SUBSAMPLE_RATIO = 1.0  # ratio used sample the scenario (e.g. a 0.1 ratio means sample from 20Hz to 2Hz)
    NUPLAN_ORIGINAL_DT = 0.05

    # get the name of log file and scene token
    unique_token = scene_ids
    scene_ids = scene_ids.split("=")
    logfile = scene_ids[0]
    scene_token = scene_ids[1]
    # scene_ts represents the trajdata index of current step
    curr_frame = int((scene_ts * desired_dt) / NUPLAN_ORIGINAL_DT)

    nuplandb_wrapper = NuPlanDBWrapper(
        data_root=NUPLAN_DATA_ROOT,
        map_root=NUPLAN_MAPS_ROOT,
        db_files=NUPLAN_DB_FILES,
        map_version=NUPLAN_MAP_VERSION,
    )
    print(unique_token, scene_ts)
    log_db = nuplandb_wrapper.get_log_db(logfile)

    # collect all camera info
    camera_dict = {}
    for item in log_db.camera:
        camera_dict[item.channel] = item

    curr_scene = None
    lidar_pcs = None
    for scene in log_db.scene:
        # only select the scene with the same token
        if scene_token != scene.lidar_pcs[0].scene_token:
            continue

        curr_scene = {
            "scene_token": scene_token,
            "lidar_token": scene.lidar_pcs[0].token,
            "timestamp": scene.lidar_pcs[0].timestamp,
            "end_lidar_token": scene.lidar_pcs[-1].token,
            "end_timestamp": scene.lidar_pcs[-1].timestamp,
            "map_name": log_db.map_name,
            "camera": camera_dict["CAM_F0"],  # only need the front camera
            "curr_ego_pose": scene.lidar_pcs[curr_frame].ego_pose,
        }
        lidar_pcs = scene.lidar_pcs
    assert curr_scene is not None, f"Scene {scene_token} not found in log {logfile}"

    if only_ego_pose:
        return curr_scene['curr_ego_pose']
    
    # get the current scenario
    scenario = NuPlanScenario(
        data_root=NUPLAN_DB_FILES,
        log_file_load_path=logfile,
        initial_lidar_token=curr_scene["lidar_token"],
        initial_lidar_timestamp=curr_scene["timestamp"],
        scenario_type="scenario_type",
        map_root=NUPLAN_MAPS_ROOT,
        map_version=NUPLAN_MAP_VERSION,
        map_name=curr_scene["map_name"],
        scenario_extraction_info=ScenarioExtractionInfo(
            scenario_name=logfile,
            scenario_duration=SCENARIO_DURATION,
            extraction_offset=EXTRACTION_OFFSET,
            subsample_ratio=SUBSAMPLE_RATIO,
        ),
        ego_vehicle_parameters=get_pacifica_parameters(),
        sensor_root=NUPLAN_SENSOR_ROOT,
    )
    number_of_frames = scenario.get_number_of_iterations() - 1

    # retrieve scene image for a specific timestamp (iteration)
    channels = DEFUALT_SENSOR
    img_list = []
    front_img_list = []
    for t_i in range(curr_frame, number_of_frames):
        sensors = scenario.get_sensors_at_iteration(t_i, channels)
        img_front = sensors.images[channels[0]].as_numpy
        
        if len(DEFUALT_SENSOR) > 2:
            if channels[1] in sensors.images.keys():
                img_left = sensors.images[channels[1]].as_numpy
            else:
                img_left = np.zeros_like(img_front)
        
            if channels[2] in sensors.images.keys():
                img_right = sensors.images[channels[2]].as_numpy
            else:
                img_right = np.zeros_like(img_front)
                
            # add an offset of left and right camera
            img_left = img_left[:, :-400]
            img_right = img_right[:, 400:]
            img = np.concatenate([img_left, img_front, img_right], axis=1)
        else:
            img = img_front
            
        img_list.append(img)
    
  
    data_root = f"./workspace/data/vpd-sft/nuplan/{unique_token}_s{scene_ts}/"
    image_path_list = []
    timestamp_list = []
    object_list = []
    ego_pose_list = []
    
    
    try:
        if not os.path.exists(data_root):
            os.mkdir(data_root)
        for t_i in range(max(0, curr_frame - 80), curr_frame, 10):
            sensors = scenario.get_sensors_at_iteration(t_i, channels)
            img_front = sensors.images[channels[0]].as_numpy
            ego_state = scenario.get_ego_state_at_iteration(t_i)
            plt.imsave(os.path.join(data_root, f"{ego_state.agent.metadata.timestamp_us}.jpg"), img_front)
            image_path_list.append(os.path.join(data_root, f"{ego_state.agent.metadata.timestamp_us}.jpg"))
            timestamp_list.append(ego_state.agent.metadata.timestamp_us)
            tracked_objects = scenario.get_tracked_objects_at_iteration(curr_frame)
            tracked_objects = list(tracked_objects.tracked_objects)
            # tracked_objects = [lidar_box for lidar_box in log_db.lidar_box if lidar_box.track_token in agent_names]
            tracked_objects = [lidar_box for lidar_box in tracked_objects if lidar_box.track_token in agent_names]
            object_list.append(tracked_objects)
            ego_pose_list.append(lidar_pcs[t_i].ego_pose)
     
        print(timestamp_list)
            
        img_list = np.stack(img_list)
        ego_state = scenario.get_ego_state_at_iteration(curr_frame)
        curr_timestamp = ego_state.agent.metadata.timestamp_us

        print("-"*50)
        object_infos = defaultdict(list)
        for i, t_i in enumerate(timestamp_list):
            objects = object_list[i]
            for obj in objects:
                # the global location load from nuplan dataset has some bugs!
                c_i = agent_names.index(obj.track_token)
                agent_curr_world = agent_world_locs[c_i]
                
                # print("="*50)
                # print(curr_frame, len(agent_curr_world))
                
                agent_curr_world = agent_curr_world[-min((len(timestamp_list) - i), len(agent_curr_world))]
                agent_curr_world = agent_curr_world[:3].numpy()
                nuplan_world_loc = np.array([obj.center.x, obj.center.y])
                distance = np.linalg.norm(agent_curr_world[:2] - nuplan_world_loc)
                if distance <= 5:
                    global_x, global_y, global_z = obj.center.x, obj.center.y, curr_scene['curr_ego_pose'].z 
                else:
                    global_x, global_y, global_z = agent_curr_world[0], agent_curr_world[1], curr_scene['curr_ego_pose'].z 
                

                try:
                    corners_3d = get_3d_box_corners((global_x, global_y, global_z),
                                                    (obj.box.length, obj.box.width, obj.box.height),
                                                    obj.center.heading)
                    
        
                    cam_points, mark = get_sensor_coord(corners_3d, ego_pose_list[- len(timestamp_list) + i], curr_scene['camera'])
                    bbox_points = view_points(np.array(cam_points), curr_scene['camera'].intrinsic_np, normalize=True)
                    bbox_points = bbox_points.swapaxes(0, 1)[:, :2]

                    x_min = np.min(bbox_points[:, 0])
                    y_min = np.min(bbox_points[:, 1])
                    x_max = np.max(bbox_points[:, 0])
                    y_max = np.max(bbox_points[:, 1])
                    
                    points_2d = [x_min, y_min, x_max, y_max]
                    def clip_box(points_2d, width=1920, height=1080):
                        x_min, y_min, x_max, y_max = points_2d
                        x_min = max(0, min(width, x_min))
                        x_max = max(0, min(width, x_max))
                        y_min = max(0, min(height, y_min))
                        y_max = max(0, min(height, y_max))
                        return [x_min, y_min, x_max, y_max]
                    
                    points_2d = clip_box(points_2d)
                    
                except:
                    points_2d = []
                    
                print(points_2d)
                    
                
                relative_ego = get_relative_direction(agent_world_locs[c_i], curr_scene['curr_ego_pose'])
                
                def compute_bbox_area(bbox):
                    if  len(bbox) != 4: return 0.0
                    
                    x_min, y_min, x_max, y_max = bbox
                    width = x_max - x_min
                    height = y_max - y_min
                    return width * height

                
                if len(points_2d) == 4 and compute_bbox_area(points_2d) > 100:
                    
                    object_infos[t_i].append((
                        obj.track_token, 
                        points_2d, 
                        relative_ego,
                        curr_scene['curr_ego_pose'],
                        curr_scene['camera']
                    ))
                else:
                    object_infos[t_i] = []
                
    except:
        object_infos = defaultdict(list)
        img_list = []
        
        
    return curr_scene, img_list, image_path_list,  object_infos # [T, H, W, C]


def plot_traj_on_camera(traj, ego_pose, camera, color, ax):
    # assume the height of the trajectory is 0.5m lower than the vehicle
    traj_z = np.ones((len(traj), 1)) * (ego_pose.z - 0.5)
    traj = np.concatenate([traj[:, 0:2], traj_z], axis=1)  # [x, y, z]
    traj = traj.swapaxes(0, 1)  # [3, T]
    cam_traj, mark = get_sensor_coord(traj, ego_pose, camera)
    intrinsic_np = np.array(camera.intrinsic_np)
    points = view_points(np.array(cam_traj), intrinsic_np, normalize=True)
    points = points.swapaxes(0, 1)[:, :2]  # [T, 2]
    x_offset = IMAGE_WIDTH - 400
    s = np.linspace(0.3, 12, len(points))
    ax.scatter(points[:, 0] + x_offset, points[:, 1], c=color, s=s)
    
    return points

"""
[
    {"id": "drivelm_4a0798f849ca477ab18009c3a20b7df2_34", 
    "video": null, 
    "image": [
        "./workspace/data/vpd-sft/drivelm/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291007112404.jpg", 
        "./workspace/data/vpd-sft/drivelm/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291007612404.jpg", 
        "./workspace/data/vpd-sft/drivelm/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291008112404.jpg", 
        "./workspace/data/vpd-sft/drivelm/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291008612404.jpg", 
        "./workspace/data/vpd-sft/drivelm/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291009112404.jpg", 
        "./workspace/data/vpd-sft/drivelm/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291009612404.jpg", 
        "./workspace/data/vpd-sft/drivelm/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010112404.jpg", 
        "./workspace/data/vpd-sft/drivelm/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404.jpg"], 
        "source": "drivelm_perception", 
        "conversations": [
            {"from": "human", "value": "<image>\nWhat is the visual description of [1]?"}, 
            {"from": "gpt", "value": "Green light."}], 
        "objects": [{"0": []}, {"1": []}, {"2": []}, {"3": []}, {"4": []}, {"5": []}, {"6": []}, {"7": [{"Category": "Traffic element", "Status": null, "Visual_description": "Green light.", "2d_bbox": [676.4, 0.0, 1452.6, 171.5], "id": 1, "instance_token": null}]}]}
    
"""


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Testing Script")
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()
    return args

args = parse_args()

"""
python nuplan_dataset.py \
    --split val --start_idx 0 --end_idx 1100

python nuplan_dataset.py \
    --split train --start_idx 0 --end_idx 1100
    
python nuplan_dataset.py \
    --split train --start_idx 1100 --end_idx 2000
    
python nuplan_dataset.py \
    --split train --start_idx 2000 --end_idx 2500
    
python nuplan_dataset.py \
    --split train --start_idx 2500 --end_idx 3000

python nuplan_dataset.py \
    --split train --start_idx 3000 --end_idx 3500
    
python nuplan_dataset.py \
    --split train --start_idx 3500 --end_idx 4000
    
python nuplan_dataset.py \
    --split train --start_idx 4000 --end_idx 4500
    
python nuplan_dataset.py \
    --split train --start_idx 4500 --end_idx 5000
    
python nuplan_dataset.py \
    --split train --start_idx 5000 --end_idx 5500
"""

if __name__ == "__main__":
    scenario_list = pd.read_csv(f"./results/results_{args.split}_{args.start_idx}_{args.end_idx}.csv")
    scenario_list = scenario_list[scenario_list["Collision"] == True]
    print(len(scenario_list))
    config = load_yaml_config("./config/config.yaml")
    split = args.split
    print(split)
    dataloader, data_info = get_dataloader(args=config, split=split, shuffle=False, plot_scene=True)
    dataset_name = {"Nuscenes": "nusc_trainval", "nuPlanMini": "nuplan_mini"}[
        config.dataset
    ]

    check_folder("./results/collision")
    outputs = []
    with open(f"./workspace/data/vpd-sft/nuplan_{args.split}_{args.start_idx}_{args.end_idx}.json", "w") as f:
        for i, batch in tqdm(enumerate(dataloader)):
            if i <  args.start_idx or i >= args.end_idx: continue
            for b_i in tqdm(range(len(batch.scene_ids)), leave=False):
            
                scene_ids = batch.scene_ids[b_i]
                scene_ts = batch.scene_ts[b_i].item()
            
                # only process the scene in the list
                collision_info = scenario_list[
                    (scenario_list["Scene ID"] == scene_ids)
                    & (scenario_list["Initial Timestep"] == scene_ts)
                ]
                
                # if scene_ids != "2021.06.07.12.54.00_veh-35_01843_02314=c0d79102a9225562":
                #     continue
                
                # if  scene_ts != 19:
                #     continue
                
                if collision_info.empty:
                    continue
                
                ego_long_beh = collision_info["Ego Long Behavior"].values
                ego_lat_beh = collision_info["Ego Lat Behavior"].values
                agent_long_beh = collision_info["Agent Long Behavior"].values
                agent_lat_beh = collision_info["Agent Lat Behavior"].values
                agent_ids = collision_info["Agent ID"].values
                agent_names = collision_info["Agent Name"].values
                agent_world_locs = [[] for _ in range(len(agent_names))]
                
                # if "440a808d665c5eb1" not in agent_names:
                #     continue
                
                # for calculating behavior trajectory
                city_id = batch.extras["get_city_name"][b_i]
                city_name = ["boston", "singapore", "pittsburgh", "las_vegas"][city_id]
                ego_tf = CPU(batch.extras["agent_from_world_tf"][b_i])
                hist_world = batch.extras["get_hist_world"][b_i].float()
                hist_vehicle = batch.agent_hist[b_i]
                hist_mask = (~torch.isnan(hist_world[..., 0])).float()
                occ_model = OccupancyModel(interval=config.grid_interval)
                
        
                for a_i in range(hist_world.shape[0]):
                    agent_hist_world = hist_world[a_i]
                    agent_mask = hist_mask[a_i]
                    agent_hist_world = agent_hist_world[agent_mask == 1]
                    if torch.sum(agent_mask) == 0:  # the agent is invalid
                        continue
                    if a_i not in agent_ids or a_i == 0:
                        continue
                    
                    matching_indices = [i for i, val in enumerate(agent_ids) if val == a_i]
                    for c_i in matching_indices:
                        agent_world_locs[c_i] = agent_hist_world
                
                try:
                    # load ego_pose
                    ego_pose = load_scene_camera(
                        scene_ids, scene_ts, agent_names, agent_world_locs, config.desired_dt, DEFUALT_SENSOR=[CameraChannel.CAM_F0, CameraChannel.CAM_L0, CameraChannel.CAM_R0], only_ego_pose=True
                    )
                    remove_ids = []
                    for i in range(len(agent_world_locs)):
                        relative_ego = get_relative_direction(agent_world_locs[i], ego_pose)
                        print(relative_ego)
                        if is_visible_object(relative_ego) == False:
                            remove_ids.append(i)
                except:
                    continue
                
                   
                ego_long_beh = [item for i, item in enumerate(ego_long_beh) if i not in remove_ids]
                ego_lat_beh = [item for i, item in enumerate(ego_lat_beh) if i not in remove_ids]
                agent_long_beh = [item for i, item in enumerate(agent_long_beh) if i not in remove_ids]
                agent_lat_beh = [item for i, item in enumerate(agent_lat_beh) if i not in remove_ids]
                agent_ids = [item for i, item in enumerate(agent_ids) if i not in remove_ids]
                agent_names = [item for i, item in enumerate(agent_names) if i not in remove_ids]
                agent_world_locs = [item for i, item in enumerate(agent_world_locs) if i not in remove_ids]
                
                if (len(ego_long_beh) == 0 or len(ego_lat_beh) == 0 or \
                    len(agent_long_beh) == 0 or len(agent_lat_beh) == 0 or \
                    len(agent_ids) == 0 or len(agent_names) == 0 or len(agent_world_locs) == 0):
                    continue
                
                print(ego_long_beh, ego_lat_beh, agent_long_beh, agent_lat_beh, agent_ids, agent_names)
                
                
                # load camera info, [T, H, W, C]
                curr_scene, scene_imgs, image_path_list, object_infos = load_scene_camera(
                    scene_ids, scene_ts, agent_names, agent_world_locs, config.desired_dt, DEFUALT_SENSOR=[CameraChannel.CAM_F0, CameraChannel.CAM_L0, CameraChannel.CAM_R0]
                )
                
                if len(scene_imgs) == 0:
                    continue
                
                # plot the front camessra
                fig = plt.figure(figsize=(6, 8), dpi=300)
                gs = GridSpec(2, 1, height_ratios=[1, 4])
                ax1 = fig.add_subplot(gs[0])
                ax1.imshow(scene_imgs[0])
                IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1080
                ax1.set_xlim([0, IMAGE_WIDTH * 3 - 800])
                ax1.set_ylim([IMAGE_HEIGHT, 0])
                ax1.axis("off")

                # plot gt traj and map
                ax2 = fig.add_subplot(gs[1])
                plot_map(batch.extras["get_vector_map_local"][b_i], ax2)


                # plot the behavior trajectory of the collision agent
                for a_i in range(hist_world.shape[0]):
                    agent_hist_world = hist_world[a_i]
                    agent_mask = hist_mask[a_i]
                    agent_hist_world = agent_hist_world[agent_mask == 1]
                    if torch.sum(agent_mask) == 0:  # the agent is invalid
                        continue

                    # convert to local frame
                    agent_hist = convert_world_to_local(agent_hist_world, ego_tf)
                    agent_curr = agent_hist[-1]

                    if a_i not in agent_ids:
                        # plot ego hist and behavior
                        if a_i == 0:
                            color = "#00D0A1"
                            name = "Ego"
                            plot_agent_hist(a_i, agent_hist, color, name, ax2)
                            lon_cand = ego_long_beh
                            lat_cand = ego_lat_beh
                        # plot other agents that are not involved in the collision
                        else:
                            color = "#6491EA"
                            name = "Others (background)"
                            plot_agent_hist(a_i, agent_hist, color, name, ax2)
                            continue
                    else:
                        color = "#EB5270"
                        name = "Others (potential collision)"
                        plot_agent_hist(a_i, agent_hist, color, name, ax2)
                        c_i = agent_ids.index(a_i)
                        lon_cand = [agent_long_beh[c_i]]
                        lat_cand = [agent_lat_beh[c_i]]

                    agent_curr_world = agent_hist_world[-1]
                    # get the lane graph from the map using world frame
                    # the current point [x, y, h, v]
                    lane_graph = get_lane_graph_from_map(
                        config, agent_curr_world, dataset_name, city_name
                    )
                    # get the traj in world frame according to the target behavior
                    # lane_graph.to_local_frame(ego_tf)
                    lane_graph.to_tensor()
                    beh_traj = get_traj_from_behavior(
                        lane_graph, agent_curr_world, lon_cand, lat_cand
                    )
                

                    # plot the trajectory and the last point in local frame
                    for traj_i in range(beh_traj.shape[0]):
                        beh_traj_local = convert_world_to_local(beh_traj[traj_i], ego_tf)
                        plot_planning(beh_traj_local, color=color, ax=ax2)

                        # also plot the behavior trajectory on image
                        try:
                            plot_traj_on_camera(
                                beh_traj[traj_i],
                                curr_scene["curr_ego_pose"],
                                curr_scene["camera"],
                                color,
                                ax1,
                            )
                        except:
                            mark = False
                # rotate the BEV to match the camera view
                if ax2 is not None:
                    transform = Affine2D().rotate_deg(90) + ax2.transData
                    for child in ax2.get_children():
                        if hasattr(child, "set_transform"):
                            child.set_transform(transform)

                # plot legend without duplicate
                handles, labels = ax2.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax2.legend(by_label.values(), by_label.keys())

                plt.tight_layout()
                print(f"Save collision image to: ./workspace/data/vpd-sft/nuplan/{scene_ids}_s{scene_ts}/collision.png")
                plt.savefig(f"./workspace/data/vpd-sft/nuplan/{scene_ids}_s{scene_ts}/collision.png")
                plt.close("all")
                

                if len(object_infos.keys()) == 0:
                    continue
                
                example = {}
                token2idx = {token: i for i, token in enumerate(agent_names)}
                mark = {token: {} for i, token in enumerate(agent_names)}
                objects = defaultdict(list)
                
                for token in agent_names:
                    idx = token2idx[token]
                    for i, timestamp in enumerate(object_infos.keys()):
                        for obj in object_infos[timestamp]:
                            if obj[0] == token:
                                if i not in mark[token].keys():
                                    objects[i].append(
                                        {
                                            "id": idx + 1,
                                            "instance_token": token,
                                            "2d_bbox": obj[1],
                                            "relative_xyz": obj[2].tolist(),
                                        }
                                    )
                                    mark[token][i] = True
                    
                if len(objects.keys()) == 0:
                    continue
                        
                for l, token in enumerate(agent_names):
                    idx = token2idx[token]
                    example = {
                        "id": f"nuplan_{scene_ts}_{scene_ts}_agent{l+1}",
                        "video": None,
                        "source": "nuplan",
                        "image": image_path_list,
                        "conversations": [
                            {"from": "human", "value": f"Analyze the impact of the ego vehicle's future behavior: {ego_long_beh[idx]} and {ego_lat_beh[idx]}, while the future behavior of object [{idx+1}] is: {agent_long_beh[idx]} and {agent_lat_beh[idx]}. Consider potential risks such as collision, reduced reaction time, or merging conflicts. Provide a brief reasoning and assign a safety score (0 = extremely unsafe, 1 = completely safe)."}, 
                            {"from": "gpt", "value": None}
                        ], 
                        "ego_lat_beh": ego_lat_beh[idx],
                        "ego_long_beh": ego_long_beh[idx],
                        "agent_lat_beh": agent_lat_beh[idx],
                        "agent_long_beh": agent_long_beh[idx],
                        "objects": objects
                    }
                
                    outputs.append(example)
            
            

                f.write(f"{json.dumps(example)}\n")
                f.flush()
                
                # import sys 
                # sys.exit()

                
          
     
                    
                
                    
                
   