import os
from pathlib import Path
from functools import partial
from collections import namedtuple, defaultdict
from simplification.cutil import simplify_coords

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from trajdata import AgentType, UnifiedDataset
from trajdata.utils.arr_utils import transform_angles_np, transform_coords_np
from trajdata.utils.state_utils import convert_to_frame_state, transform_from_frame
from trajdata import MapAPI, VectorMap
from trajdata.maps.vec_map_elements import Polyline

from cbp.datasets.trajdata_dataloader import get_dataloader
from cbp.utils.train_helpers import rotate_tensor


def agent_from_world_tf(element):
    centered_agent_from_world_tf = element.centered_agent_from_world_tf  # [4, 4] matrix
    return centered_agent_from_world_tf


def get_map_name(element):
    vector_map = element.vec_map
    map_id = vector_map.map_id.split(":")[1]
    map_id_mapping = {
        "boston": 0,
        "singapore": 1,
        "pittsburgh": 2,
        "las_vegas": 3,
    }
    return map_id_mapping[map_id]


def get_dataloader(args, split, shuffle=True, incl_raster_map=False):
    if args.dataset == "Nuscenes":
        if split == "train":
            # train_val is a subset of the original train set
            desired_data = ["nusc_trainval-train", "nusc_trainval-train_val"]
        elif split == "val":
            desired_data = ["nusc_trainval-val"]
        elif split == "mini":
            desired_data = ["nusc_mini"]
        data_dirs = {
            "nusc_mini": "/home/wenhaod/data/nuScenes",
            "nusc_trainval": "/home/wenhaod/data/nuScenes",
        }
    elif args.dataset == "Waymo":
        if split == "train":
            desired_data = ["waymo_train-train"]
        elif split == "val":
            desired_data = ["waymo_val-val"]
        data_dirs = {
            "waymo_train": "/home/wenhaod/data/waymo_1.1/scenario",
            "waymo_val": "/home/wenhaod/data/waymo_1.1/scenario",
        }
    elif args.dataset == "nuPlanMini":
        if split == "train":
            desired_data = ["nuplan_mini-mini_train"]
        elif split == "val":
            desired_data = ["nuplan_mini-mini_val"]
        data_dirs = {
            "nuplan_mini": "/home/wenhaod/data/nuplan/dataset",
        }
    else:
        raise NotImplementedError("Invalid dataset name")

    extras = {
        "agent_from_world_tf": partial(agent_from_world_tf),
        "get_map_name": partial(get_map_name),
    }

    state_format = "x,y,z,xd,yd,h"
    dataset = UnifiedDataset(
        desired_data=desired_data,
        centric="scene",
        max_agent_num=args.max_agent_num,  # we will remove inactivated agents
        desired_dt=args.desired_dt,
        history_sec=(args.history_sec, args.history_sec),
        future_sec=(args.future_sec, args.future_sec),
        only_types=[
            AgentType.VEHICLE,
            AgentType.PEDESTRIAN,
            AgentType.BICYCLE,
            AgentType.MOTORCYCLE,
        ],
        state_format=state_format,
        obs_format=state_format,
        agent_interaction_distances=defaultdict(lambda: 50.0),
        incl_robot_future=False,
        incl_vector_map=True,
        incl_raster_map=incl_raster_map,
        raster_map_params={
            "px_per_m": 4,
            "map_size_px": 448,
            "offset_frac_xy": (0.0, 0.0),
        },
        vector_map_params={
            "incl_road_lanes": True,
            "incl_road_areas": True,
            "incl_ped_crosswalks": False,
            "incl_ped_walkways": False,
            "collate": False,
            # In Waymo, it should be False because maps are already partitioned geographically
            # and keeping them around significantly grows memory.
            "keep_in_memory": False if args.dataset == "Waymo" else True,
            "num_workers": args.num_workers,  # this is only avaliable for zhejun's version of trajdata
        },
        verbose=True,
        num_workers=args.num_workers,
        cache_location=args.cache_dir,
        rebuild_cache=False,
        require_map_cache=False,
        data_dirs=data_dirs,
        extras=extras,
    )
    print(f"Dataset name: {desired_data}, Num of data Samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.get_collate_fn(),
        num_workers=args.num_workers,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True,
    )
    info = {
        "dataset": dataset,
        "len_dataset": len(dataset),
        "max_agent_num": dataset.max_agent_num,
        "hist_len": int(args.history_sec / args.desired_dt) + 1,
        "fut_len": int(args.future_sec / args.desired_dt),
        "traj_attr": args.traj_attr,
        "map_attr": args.map_attr,
        "num_agent_types": args.num_agent_types,
    }
    return dataloader, info


if __name__ == "__main__":
    config_dict = {
        "dataset": "nuPlanMini",  # "Nuscenes", "Waymo", nuPlanMini
        "batch_size": 1,
        "max_agent_num": 20,
        "desired_dt": 0.5,
        "history_sec": 5.0,
        "future_sec": 4.0,
        "num_workers": 32,
        "cache_dir": "/home/wenhaod/data/waymo_unified_data_cache",
        "max_number_lane": 200,
        "num_point_per_lane": 30,
        "radius": 100,
        "traj_attr": 6,
        "map_attr": 3,
        "num_agent_types": 5,
    }
    split = "val"
    args = namedtuple("config", config_dict.keys())(*config_dict.values())
    dataloader, data_info = get_dataloader(args=args, split=split)

    cache_path = Path(args.cache_dir).expanduser()
    map_api = MapAPI(cache_path)
    env_name = "nuplan_mini"
    map_name_list = ["boston", "singapore", "pittsburgh", "las_vegas"]
    road_graph_all = {m_i: {} for m_i in map_name_list}
    for m_i in map_name_list:
        road_graph = {}
        vec_map = map_api.get_map(f"{env_name}:{m_i}")
        possible_lanes = vec_map.lanes
        for lane in possible_lanes:
            lane_id = lane.id
            next_lanes = lane.next_lanes
            prev_lanes = lane.prev_lanes
            elem_type = lane.elem_type
            adj_lanes_left = lane.adj_lanes_left
            adj_lanes_right = lane.adj_lanes_right
            centerline = lane.center.interpolate(num_pts=args.num_point_per_lane).points

            # break down the centerline into smaller segments
            # max_dist = 5
            # epsilon = 0.2
            # centerline_short = (
            #     Polyline(simplify_coords(lane.center.points[..., :2], epsilon))
            #     .interpolate(max_dist=max_dist)
            #     .points[..., :2]
            # )

            road_graph[lane_id] = {
                "next_lanes": list(next_lanes),
                "prev_lanes": list(prev_lanes),
                "left_lanes": list(adj_lanes_left),
                "right_lanes": list(adj_lanes_right),
                "centerline": centerline,
            }

        print(f"{vec_map.env_name}, {vec_map.map_name}, Lanes: {len(road_graph)}")
        road_graph_all[m_i] = road_graph

    def plot_links(center, next_lanes, road_graph, agent_from_world_tf, color):
        for lane_id in next_lanes:
            next_centerline = road_graph[lane_id]["centerline"]
            pos_xyz = transform_coords_np(next_centerline[..., :3], agent_from_world_tf)
            heading = transform_angles_np(next_centerline[..., -1], agent_from_world_tf)
            next_center = pos_xyz[len(pos_xyz) // 2]
            plt.plot(
                [center[0], next_center[0]], [center[1], next_center[1]], color=color
            )

    device = "cuda"
    count = 0
    for batch in tqdm(dataloader):
        batch_idx = 0
        map_idx = batch.extras["get_map_name"][batch_idx]
        map_name = map_name_list[map_idx]

        # plot
        plt.figure(figsize=(10, 10), dpi=300)
        road_graph = road_graph_all[map_name]
        for lane_id, lane_info in road_graph.items():
            centerline = lane_info["centerline"]
            agent_from_world_tf = batch.extras["agent_from_world_tf"][batch_idx]
            agent_from_world_tf = agent_from_world_tf.cpu().numpy()
            pos_xyz = transform_coords_np(centerline[..., :3], agent_from_world_tf)
            heading = transform_angles_np(centerline[..., -1], agent_from_world_tf)

            dist = np.linalg.norm(pos_xyz[:, 0:2], axis=-1).mean()
            if dist > 200:
                continue

            plt.plot(pos_xyz[:, 0], pos_xyz[:, 1], color="black", lw=1.0, alpha=0.1)
            # plot the center of road graph
            center = pos_xyz[len(pos_xyz) // 2]
            plt.scatter(center[0], center[1], color="black", s=20)

            # plot the link between lanes
            plot_links(
                center, lane_info["next_lanes"], road_graph, agent_from_world_tf, "blue"
            )
            plot_links(
                center, lane_info["left_lanes"], road_graph, agent_from_world_tf, "red"
            )
            plot_links(
                center, lane_info["right_lanes"], road_graph, agent_from_world_tf, "red"
            )

        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f"road_graph_{count}.png")
        plt.close()
        exit()
