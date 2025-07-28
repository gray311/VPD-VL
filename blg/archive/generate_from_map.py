from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from trajdata import MapAPI
from trajdata.utils.arr_utils import transform_angles_np, transform_coords_np

from blg.datasets.trajdata_dataloader import get_next_lane, LaneTree, LaneGraph
from blg.utils.general_helpers import (
    CPU,
    check_folder,
    load_yaml_config,
)
from blg.utils.plot_helpers import (
    plot_planning,
    plot_lane_tree,
    plot_occ_graph_beh,
    plot_occ_graph,
)
from blg.datasets.trajdata_dataloader import get_dataloader
from blg.model.behavior_model import BehaviorModel
from blg.model.occupancy_model import OccupancyModel


def get_lane_graph_from_map(
    args, ego_xyhv_world, city_name, dataset_name="nuplan_mini", max_depth=3
):
    """Obtain the lane graph given the ego pose and transformation matrix.

    Args:
        args (tuple): arguments from the config file
        ego_xyhv_world (np.array): the current ego pose containing [x, y, h, v] in world frame
        city_name (str): select from ["boston", "singapore", "pittsburgh", "las_vegas"]
        dataset_name (str, optional): Defaults to "nuplan_mini".
        max_depth (int): the maximum depth to search for the next lane
    """
    num_points = args.num_point_per_lane
    cache_path = Path(args.cache_dir).expanduser()
    map_api = MapAPI(cache_path)
    vector_map = map_api.get_map(f"{dataset_name}:{city_name}")

    # for nuplan dataset, the z coordinate is 0
    ego_world_x = ego_xyhv_world[0]
    ego_world_y = ego_xyhv_world[1]
    ego_world_z = 0
    ego_world_h = ego_xyhv_world[2]
    query = np.array([ego_world_x, ego_world_y, ego_world_z, ego_world_h])

    # query the closest centerlines from vector map
    curr_lane = vector_map.get_current_lane(query, args.radius, np.pi / 6)
    assert len(curr_lane) > 0, "No lanes within the radius"
    closest_lane = curr_lane[0]

    # points = closest_lane.center.interpolate(num_pts=10).points
    # print(points)

    # recursively search for the next lanes
    curr_ids = [closest_lane.id]
    curr_tree = LaneTree()
    get_next_lane(vector_map, num_points, curr_ids, max_depth, 0, curr_tree)

    # find the all left and right lane trees if they exist
    left_right_search_depth = 3
    left_group, right_group = [], []
    left_ids, right_ids = [closest_lane.id], [closest_lane.id]
    for i in range(left_right_search_depth):
        left_left_tree = LaneTree()
        if len(left_ids) > 0:
            left_id = list(left_ids)[0]  # only take the first one
            left_ids = vector_map.get_road_lane(left_id).adj_lanes_left
            get_next_lane(
                vector_map, num_points, left_ids, max_depth, 0, left_left_tree
            )
            left_group.append(left_left_tree)

        right_right_tree = LaneTree()
        if len(right_ids) > 0:
            right_ids = list(right_ids)[0]
            right_ids = vector_map.get_road_lane(right_ids).adj_lanes_right
            get_next_lane(
                vector_map, num_points, right_ids, max_depth, 0, right_right_tree
            )
            right_group.append(right_right_tree)

    # due to numerical problem, dont convert to float32 tensor
    return LaneGraph(curr_tree, left_group, right_group)


def get_behavior_traj(config, lane_graph, ego_xyhv, plot=True):
    bm = BehaviorModel(dt=config.dt, horizon=config.horizon)  # 3 seconds
    # ego_xyhv[3] = 4  # set v to 4 m/s

    # convert the ego point to the frenet coordinate
    # the current pose should be the same for all current lanes, so we just take the first one
    curr_lane = lane_graph.curr.center[0]

    # show all combinations of behaviors
    longitudinal_bev = {
        "keep speed": 0.0,
        "accelerate": config.acceleration,
        "decelerate": config.deceleration,
    }
    lateral_bev = {
        "keep lane": lane_graph.curr.center,
        # assume the vehicle can only change one lane at a time
        "left lane-change": lane_graph.left_group[0].center,
        "right lane-change": lane_graph.right_group[0].center,
    }
    color_list = ["deeppink", "green", "blue", "orange", "purple", "brown", "red"]

    beh_traj = []
    exist_last_pt = []
    for lon_i in longitudinal_bev.keys():
        target_acc = longitudinal_bev[lon_i]
        for l_i, lat_i in enumerate(lateral_bev.keys()):
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

                # plot the trajectory and the last point
                if plot:
                    plot_planning(new_traj_xyhv, color=color_list[l_i], label=lat_i)
                beh_traj.append(new_traj_xyhv)

    beh_traj = torch.stack(beh_traj)
    return beh_traj


def convert_world_to_local(ego_hist, ego_tf):
    pos_xy = ego_hist[..., 0:2]
    heading = ego_hist[..., 2]
    velocity = ego_hist[..., 3]
    pos_xy = transform_coords_np(pos_xy, ego_tf)
    heading = transform_angles_np(heading, ego_tf)
    # only need x, y, heading, v
    ego_hist = np.concatenate(
        (pos_xy[:, 0:2], heading[:, None], velocity[:, None]), axis=1
    )
    ego_hist = torch.as_tensor(ego_hist)
    return ego_hist


if __name__ == "__main__":
    seed = 233
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    config = load_yaml_config("./config/config.yaml")
    dataloader, data_info = get_dataloader(args=config, split="val")

    # define the occupancy model
    occ_model = OccupancyModel(interval=5.0)

    use_local_frame = True
    count = 0
    b_i = 0
    check_folder("road_graph")
    for batch in tqdm(dataloader):
        hist_world = batch.extras["get_hist_world"][b_i].float()
        ego_hist_world = hist_world[0]  # the first agent is always the ego

        city_id = batch.extras["get_city_name"][b_i]
        city_name = ["boston", "singapore", "pittsburgh", "las_vegas"][city_id]
        ego_tf = CPU(batch.extras["agent_from_world_tf"][b_i])

        # get the lane graph from the map using world frame
        ego_curr_world = ego_hist_world[-1]  # the current point [x, y, h, v]
        lane_graph = get_lane_graph_from_map(config, ego_curr_world, city_name)

        if use_local_frame:
            # transform to local coordinates
            lane_graph.to_local_frame(ego_tf)
            ego_hist_local = convert_world_to_local(ego_hist_world.clone(), ego_tf)
            ego_curr_local = ego_hist_local[-1]
            ego_curr = ego_curr_local
            ego_hist = ego_hist_local
        else:
            ego_curr = ego_curr_world
            ego_hist = ego_hist_world

        plt.figure(figsize=(10, 10), dpi=300)
        plot_planning(ego_hist, color="black", label="origin")

        # get the behavior graph
        lane_graph.to_tensor()
        beh_traj = get_behavior_traj(config, lane_graph, ego_curr)

        # generate the occupancy graph
        occ_graph = occ_model.get_occ_graph(lane_graph)

        # project the behavior trajectory to the occupancy graph
        occ_graph_beh_node = occ_graph.proj_traj_to_node(beh_traj)

        # plot entire occupancy polygon
        # plot_occ_graph(left_edge, right_edge)

        # plot the occupancy node with behavior color
        plot_occ_graph_beh(occ_graph, occ_graph_beh_node)

        # plot the lane graph
        plot_lane_tree(lane_graph.curr)
        for left_tree in lane_graph.left_group:
            plot_lane_tree(left_tree)
        for right_tree in lane_graph.right_group:
            plot_lane_tree(right_tree)

        # remove repeated labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.grid(False)
        plt.tight_layout()
        if use_local_frame:
            plt.xlim(-40, 40)
            plt.ylim(-40, 40)
        else:
            plt.axis("equal")
        plt.savefig(f"./road_graph/{count}.png")
        plt.close()
        count += 1
