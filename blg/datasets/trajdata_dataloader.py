from functools import partial
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from trajdata import AgentType, UnifiedDataset
from trajdata.utils.arr_utils import transform_angles_np, transform_coords_np
from trajdata.utils.state_utils import convert_to_frame_state, transform_from_frame
from trajdata.utils import map_utils


def agent_from_world_tf(element):
    centered_agent_from_world_tf = element.centered_agent_from_world_tf  # [4, 4] matrix
    return centered_agent_from_world_tf


def get_city_name(element):
    vector_map = element.vec_map
    city_name = vector_map.map_id.split(":")[1]
    city_id_mapping = {
        "boston": 0,
        "singapore": 1,
        "pittsburgh": 2,
        "las_vegas": 3,
    }
    return city_id_mapping[city_name]


def get_ego_hist_world(element):
    centered_agent_state_np = element.centered_agent_state_np
    agent_hist = element.agent_histories

    # need to transform back into world frame
    obs_frame = convert_to_frame_state(
        centered_agent_state_np, stationary=True, grounded=True
    )
    ego_hist_world = transform_from_frame(agent_hist[0], obs_frame)  # "x,y,z,xd,yd,h"
    xy = ego_hist_world[:, 0:2]
    velocity = np.linalg.norm(ego_hist_world[:, 3:5], axis=-1)  # velocity
    heading = ego_hist_world[:, -1]  # heading
    xyhv = np.concatenate([xy, heading[:, None], velocity[:, None]], axis=1)
    return xyhv


def get_hist_world(element, hist_len, max_agent_num):
    centered_agent_state_np = element.centered_agent_state_np
    agent_hist = element.agent_histories
    agent_type = element.agent_types_np

    # need to transform back into world frame
    obs_frame = convert_to_frame_state(
        centered_agent_state_np, stationary=True, grounded=True
    )
    agent_hist_world = []
    for a_i, hist in enumerate(agent_hist):
        if agent_type[a_i] != AgentType.VEHICLE:
            continue
        hist_world = transform_from_frame(hist, obs_frame)  # "x,y,z,xd,yd,h"
        xy = hist_world[..., 0:2]
        velocity = np.linalg.norm(hist_world[..., 3:5], axis=-1)  # velocity
        heading = hist_world[..., -1]  # heading
        xyhv = np.concatenate([xy, heading[..., None], velocity[..., None]], axis=-1)

        # padding with nan in the front
        hist_padding = np.full((hist_len - xyhv.shape[0], xyhv.shape[1]), np.nan)
        xyhv = np.concatenate([hist_padding, xyhv], axis=0)
        agent_hist_world.append(xyhv)
    agent_hist_world = np.stack(agent_hist_world, axis=0)

    # padding with the max number of agents
    # add padding or remove agents
    num_agents = len(agent_hist_world)
    if num_agents >= max_agent_num:
        # select the first M agents to ensure the reproducibility
        agent_hist_world = agent_hist_world[:max_agent_num]
    else:
        padding_num = max_agent_num - num_agents
        pad = np.full((padding_num, hist_len, agent_hist_world.shape[-1]), np.nan)
        agent_hist_world = np.concatenate((agent_hist_world, pad), axis=0)
    return agent_hist_world


# to return dict from dataloader, we need to wrap the data into a class
class LaneGraphCollateData:
    def __init__(self, curr, left, right):
        self.curr = self.to_tensor(curr)
        self.left = self.to_tensor(left)
        self.right = self.to_tensor(right)

    def to_tensor(self, x):
        # x is a list of numpy arrays
        for i in range(len(x)):
            x[i] = torch.as_tensor(x[i], dtype=torch.float32)
        return x

    @staticmethod
    def __collate__(elements: list) -> any:
        return elements


class LaneTree:
    def __init__(self):
        self.center = []
        self.left_edge = []
        self.right_edge = []

    def to_tensor(self):
        # x is a list of numpy arrays
        for i in range(len(self.center)):
            self.center[i] = torch.as_tensor(self.center[i])
            self.left_edge[i] = torch.as_tensor(self.left_edge[i])
            self.right_edge[i] = torch.as_tensor(self.right_edge[i])


class LaneGraph:
    def __init__(self, curr, left_group, right_group):
        self.curr = curr  # a LaneTree
        self.left_group = left_group  # a list of LaneTree
        self.right_group = right_group  # a list of LaneTree

    def to_tensor(self):
        self.curr.to_tensor()
        for left_tree in self.left_group:
            left_tree.to_tensor()
        for right_tree in self.right_group:
            right_tree.to_tensor()

    def to_local_frame(self, ego_tf):
        self._to_local_frame(self.curr, ego_tf)
        for left_tree in self.left_group:
            left_tree = self._to_local_frame(left_tree, ego_tf)
        for right_tree in self.right_group:
            right_tree = self._to_local_frame(right_tree, ego_tf)

    def _to_local_frame(self, lane, ego_tf):
        lane.center = self._lane_trans(lane.center, ego_tf)
        lane.left_edge = self._lane_trans(lane.left_edge, ego_tf)
        lane.right_edge = self._lane_trans(lane.right_edge, ego_tf)

    @staticmethod
    def _lane_trans(center_list, agent_from_world_tf):
        local_center_list = []
        for center in center_list:
            # transform to local coordinates
            if agent_from_world_tf.shape[0] == 4:
                pos_xyz = center[..., :3]
            else:
                pos_xyz = center[..., :2]
            heading = center[..., -1]
            pos_xyz = transform_coords_np(pos_xyz, agent_from_world_tf)
            heading = transform_angles_np(heading, agent_from_world_tf)
            # only need x, y, heading, remove z
            center = np.concatenate((pos_xyz[:, 0:2], heading[:, None]), axis=1)
            local_center_list.append(center)
        return local_center_list


def get_next_lane(
    vector_map,
    num_point,
    lane_ids,
    max_depth,
    curr_depth,
    lane_tree,
    curr_centers=None,
    curr_left_edge=None,
    curr_right_edge=None,
):
    """BFS for tree search to get the centerlines of all possible lanes

    Args:
        vector_map (VectorMap): the vector map object
        num_point (int): number of points per lane
        lane_ids (List[int]): list of lane ids
        max_depth (int): max depth to search
        curr_depth (int): current depth
        lane_tree (LaneTree): A tree class to store the centerlines, left_edges, and right_edges
        curr_centers (List, optional): current centerlines. Defaults to None.
        curr_left_edge (List, optional): current left edges. Defaults to None.
        curr_right_edge (List, optional): current right edges. Defaults to None.
    """

    # reach a leaf node or max depth
    if len(lane_ids) == 0 or curr_depth == max_depth:
        # only add the entire centerline when it is not none
        if curr_centers is not None:
            # re-interpolate the centerlines to ensure equal intervals
            # note that num_pts should be different from the original num_point, or the interpolation will be ignored
            num_pts = curr_centers.shape[0] + 1
            curr_centers = map_utils.interpolate(curr_centers, num_pts=num_pts)
            curr_left_edge = map_utils.interpolate(curr_left_edge, num_pts=num_pts)
            curr_right_edge = map_utils.interpolate(curr_right_edge, num_pts=num_pts)

            # add the centerlines to the tree
            lane_tree.center.append(curr_centers)
            lane_tree.left_edge.append(curr_left_edge)
            lane_tree.right_edge.append(curr_right_edge)
        return

    for lane_i in list(lane_ids):
        # get the lane object and centers
        lane_i = vector_map.get_road_lane(lane_i)
        lane_centers = lane_i.center.interpolate(num_pts=num_point).points
        left_edge = lane_i.left_edge.interpolate(num_pts=num_point).points
        right_edge = lane_i.right_edge.interpolate(num_pts=num_point).points

        # if this lane has ancestors, we need to add them
        if curr_centers is None:
            next_centers = lane_centers
            next_left_edges = left_edge
            next_right_edges = right_edge
        else:
            next_centers = np.concatenate((curr_centers, lane_centers), axis=0)
            next_left_edges = np.concatenate((curr_left_edge, left_edge), axis=0)
            next_right_edges = np.concatenate((curr_right_edge, right_edge), axis=0)

        get_next_lane(
            vector_map,
            num_point,
            lane_i.next_lanes,
            max_depth,
            curr_depth + 1,
            lane_tree,
            next_centers,
            next_left_edges,
            next_right_edges,
        )


def process_center(center_list, agent_from_world_tf):
    local_center_list = []
    for center in center_list:
        # transform to local coordinates
        if agent_from_world_tf.shape[0] == 4:
            pos_xyz = center[..., :3]
        else:
            pos_xyz = center[..., :2]
        heading = center[..., -1]
        pos_xyz = transform_coords_np(pos_xyz, agent_from_world_tf)
        heading = transform_angles_np(heading, agent_from_world_tf)
        # only need x, y, heading, remove z
        center = np.concatenate((pos_xyz[:, 0:2], heading[:, None]), axis=1)
        local_center_list.append(center)
    return local_center_list


def get_lane_graph(element, num_point_per_lane, radius, max_depth):
    vector_map = element.vec_map
    centered_agent_state_np = element.centered_agent_state_np
    agent_hist = element.agent_histories
    agent_from_world_tf = element.centered_agent_from_world_tf  # [4, 4] matrix

    # need to transform back into world frame
    obs_frame = convert_to_frame_state(
        centered_agent_state_np, stationary=True, grounded=True
    )
    ego_hist_world = transform_from_frame(agent_hist[0], obs_frame)  # "x,y,z,xd,yd,h"
    xyz = ego_hist_world[-1, 0:3]  # [x, y, z]
    heading = ego_hist_world[-1, -1:]  # heading
    query = np.concatenate([xyz, heading])  # [x, y, z, h]
    query[2] = 0.0  # set z to 0 for nuplan dataset

    # query the closest centerlines from vector map
    curr_lane = vector_map.get_current_lane(query, radius, np.pi / 6)
    assert len(curr_lane) > 0, "No lanes within the radius"
    closest_lane = curr_lane[0]

    # recursively search for the next lanes
    lane_ids = [closest_lane.id]
    curr_tree = LaneTree()
    get_next_lane(vector_map, num_point_per_lane, lane_ids, max_depth, 0, curr_tree)

    lane_ids = closest_lane.adj_lanes_left
    left_tree = LaneTree()
    get_next_lane(vector_map, num_point_per_lane, lane_ids, max_depth, 0, left_tree)

    right_tree = LaneTree()
    lane_ids = closest_lane.adj_lanes_right
    get_next_lane(vector_map, num_point_per_lane, lane_ids, max_depth, 0, right_tree)

    # transform to local coordinates
    curr_centers = process_center(curr_tree.center, agent_from_world_tf)
    left_centers = process_center(left_tree.center, agent_from_world_tf)
    right_centers = process_center(right_tree.center, agent_from_world_tf)

    # return the lane graph
    return LaneGraphCollateData(curr_centers, left_centers, right_centers)


def get_vector_map_local(element, max_number_lane, num_point_per_lane, radius):
    vector_map = element.vec_map
    centered_agent_state_np = element.centered_agent_state_np
    agent_hist = element.agent_histories
    centered_agent_from_world_tf = element.centered_agent_from_world_tf  # [4, 4] matrix

    # need to transform back into world frame
    obs_frame = convert_to_frame_state(
        centered_agent_state_np, stationary=True, grounded=True
    )
    # NOTE: this changes according to state_format
    ego_hist_world = transform_from_frame(agent_hist[0], obs_frame)  # [x, y, z, h, v]
    query = ego_hist_world[-1, 0:3]  # [x, y, z]
    query[2] = 0.0  # set z to 0
    possible_lanes = vector_map.get_lanes_within(query, radius)

    # collect all the lanes within the radius
    if len(possible_lanes) != 0:
        valid_centerlines = []
        for lane in possible_lanes:
            centerline = lane.center.interpolate(num_pts=num_point_per_lane).points
            valid_centerlines.append(centerline)
        valid_centerlines = np.stack(valid_centerlines, axis=0)  # [S, P, 4]

        # in newer version of trajdata, the transform matrix is [4, 4]
        if centered_agent_from_world_tf.shape[0] == 4:
            pos_xyz = valid_centerlines[..., :3]
        else:
            pos_xyz = valid_centerlines[..., :2]
        heading = valid_centerlines[..., -1]
        pos_xyz = transform_coords_np(pos_xyz, centered_agent_from_world_tf)
        heading = transform_angles_np(heading, centered_agent_from_world_tf)

        # only need x, y, heading, remove z
        valid_centerlines = np.concatenate(
            (pos_xyz[:, :, 0:2], heading[:, :, None]), axis=2
        )  # [S, P, 3]
    else:
        print(
            "No lanes within the radius. maybe due to the map issue of clipGT. still try to continue"
        )
        valid_centerlines = np.zeros((1, num_point_per_lane, 3))  # [1, P, 4]

    # add mask
    valid_centerlines = np.concatenate(
        (
            valid_centerlines,
            np.ones((valid_centerlines.shape[0], valid_centerlines.shape[1], 1)),
        ),
        axis=2,
    )

    # sort the centerlines according to the distance to the ego vehicle
    centerline_mean_point = np.mean(valid_centerlines[:, :, :2], axis=1)
    ego_point = np.array([0.0, 0.0])
    indices = np.argsort(
        np.linalg.norm(ego_point - centerline_mean_point, axis=1)
    ).tolist()
    sorted_centerlines = valid_centerlines[indices[:max_number_lane]]

    # pad zeros if not enough lanes
    close_centerlines = np.zeros((max_number_lane, num_point_per_lane, 4))
    close_centerlines[0 : len(sorted_centerlines)] = sorted_centerlines
    return close_centerlines


def get_dataloader(args, split, shuffle=True, incl_raster_map=False, plot_scene=False):
    desired_data = {
        "Nuscenes": {
            # train_val is a subset of the original train set
            "train": ["nusc_trainval-train", "nusc_trainval-train_val"],
            "val": ["nusc_trainval-val"],
            "mini": ["nusc_mini"],
        },
        "Waymo": {
            "train": ["waymo_train-train"],
            "val": ["waymo_val-val"],
        },
        "nuPlanMini": {
            "train": ["nuplan_mini-mini_train"],
            "val": ["nuplan_mini-mini_val"],
        },
        "nuPlan": {
            "train": [],
            "val": [],
        },
        "clipgt": {
            "train": ["mads_mini"],
            "val": ["mads_mini"],
        },
    }[args.dataset][split]

    # we only pass the necessary data directories to trajdata
    data_dirs_list = {
        "Nuscenes": ["nusc_trainval"],
        "Waymo": ["waymo_train", "waymo_val"],
        "nuPlanMini": ["nuplan_mini"],
    }[args.dataset]
    data_dirs = {d: args.data_dirs._asdict()[d] for d in data_dirs_list}

    extras = {
        # "get_lane_graph": partial(
        #     get_lane_graph,
        #     num_point_per_lane=args.num_point_per_lane,
        #     radius=args.radius,
        #     max_depth=3,
        # ),
        "agent_from_world_tf": agent_from_world_tf,
        "get_city_name": get_city_name,
        "get_hist_world": partial(
            get_hist_world,
            hist_len=int(args.history_sec / args.desired_dt) + 1,
            max_agent_num=args.max_agent_num,
        ),
    }

    if plot_scene:
        extras.update(
            {
                "get_vector_map_local": partial(
                    get_vector_map_local,
                    max_number_lane=args.max_number_lane,
                    num_point_per_lane=args.num_point_per_lane,
                    radius=args.radius,
                )
            }
        )

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
        pin_memory=False,
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
