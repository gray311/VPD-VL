import csv
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
    plot_occ_graph_beh_single_color,
    plot_occ_graph,
)
from blg.datasets.trajdata_dataloader import get_dataloader
from blg.model.behavior_model import BehaviorModel
from blg.model.occupancy_model import OccupancyModel

torch.multiprocessing.set_sharing_strategy("file_system")

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
python blg/generate_from_map.py \
    --split val --start_idx 0 --end_idx 1100

python blg/generate_from_map.py \
    --split train --start_idx 0 --end_idx 1100
    
python blg/generate_from_map.py \
    --split train --start_idx 1100 --end_idx 2000
    
python blg/generate_from_map.py \
    --split train --start_idx 2000 --end_idx 2500
    
python blg/generate_from_map.py \
    --split train --start_idx 2500 --end_idx 3000

python blg/generate_from_map.py \
    --split train --start_idx 3000 --end_idx 3500
    
python blg/generate_from_map.py \
    --split train --start_idx 3500 --end_idx 4000
    
python blg/generate_from_map.py \
    --split train --start_idx 4000 --end_idx 4500
    
python blg/generate_from_map.py \
    --split train --start_idx 4500 --end_idx 5000
    
python blg/generate_from_map.py \
    --split train --start_idx 5000 --end_idx 5500
"""

def check_angle_and_distance(config, ego_tf, ego_global, other_global, agent_name, timestep):
    if torch.norm(other_global - ego_global) > 15:
        return False
    
    ego_global = ego_global.numpy()
    other_global = other_global.numpy()
    ego_tf = np.array(ego_tf)
    cos_theta = ego_tf[0][0]  
    sin_theta = ego_tf[1][0]  

    dx = other_global[0] - ego_global[0]
    dy = other_global[1] - ego_global[1]

    local_x = dx * cos_theta + dy * sin_theta
    local_y = -dx * sin_theta + dy * cos_theta

    if local_x < 0:
        return False
    
    if local_x == 0 and local_y == 0:
        return True
    
    angle_rad = np.arctan2(local_y, local_x)
    angle_deg = np.degrees(angle_rad)
    

    return abs(angle_deg) <= 75


def get_lane_graph_from_map(config, xyhv_world, dataset_name, city_name, max_depth=3):
    """Obtain the lane graph given the ego pose and transformation matrix.

    Args:
        config (tuple): arguments from the config file
        xyhv_world (np.array): the current ego pose containing [x, y, h, v] in world frame
        city_name (str): select from ["boston", "singapore", "pittsburgh", "las_vegas"]
        dataset_name (str, optional): Defaults to "nuplan_mini".
        max_depth (int): the maximum depth to search for the next lane
    """
    num_points = config.num_point_per_lane
    cache_path = Path(config.cache_dir).expanduser()
    map_api = MapAPI(cache_path)
    vector_map = map_api.get_map(f"{dataset_name}:{city_name}")

    # for nuplan dataset, the z coordinate is 0
    world_x = xyhv_world[0]
    world_y = xyhv_world[1]
    world_z = 0
    world_h = xyhv_world[2]
    query = np.array([world_x, world_y, world_z, world_h])

    # query the closest centerlines from vector map
    curr_lane = vector_map.get_current_lane(query, config.radius, np.pi / 6)
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

    # due to numerical problem, dont convert to float32 tensor here
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
    
    long_beh_name = []
    lat_beh_name = []
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
                long_beh_name.append(lon_i)
                lat_beh_name.append(lat_i)

    beh_traj = torch.stack(beh_traj)
    return beh_traj, long_beh_name, lat_beh_name


def convert_world_to_local(hist, ego_tf):
    pos_xy = hist[..., 0:2].clone()
    heading = hist[..., 2].clone()
    velocity = hist[..., 3].clone()
    pos_xy = transform_coords_np(pos_xy, ego_tf)
    heading = transform_angles_np(heading, ego_tf)
    # only need x, y, heading, v
    hist = np.concatenate((pos_xy[:, 0:2], heading[:, None], velocity[:, None]), axis=1)
    hist = torch.as_tensor(hist)
    return hist


def save_results(results_per_sample, split, start_idx, end_idx):
    # save the results
    np.save(f"./results/results_{split}_{start_idx}_{end_idx}.npy", results_per_sample, allow_pickle=True)
    # write string variables to csv
    with open(f"./results/results_{split}_{start_idx}_{end_idx}.csv", mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Scene ID",
                "Initial Timestep",
                "Agent ID",
                "Agent Name",
                "Ego Long Behavior",
                "Ego Lat Behavior",
                "Agent Long Behavior",
                "Agent Lat Behavior",
                "Collision",
            ]
        )
        for r_i in range(len(results_per_sample)):
            result = results_per_sample[r_i]
            long_beh_name = result["long_beh_name"]
            lat_beh_name = result["lat_beh_name"]
            collision = result["collision"]
            agent_names = result["agent_names"]

            # we start from 1 because the first agent is the ego agent
            for a_i in range(1, len(collision)):
                for bev_i in range(collision[a_i].shape[0]):
                    for bev_j in range(collision[a_i].shape[1]):
                        # if not collision[a_i][bev_i, bev_j]:
                        #     continue
                        writer.writerow(
                            [
                                result["scene_ids"],
                                result["scene_ts"],
                                a_i,
                                agent_names[a_i],
                                long_beh_name[0][bev_i],
                                lat_beh_name[0][bev_i],
                                long_beh_name[a_i][bev_j],
                                lat_beh_name[a_i][bev_j],
                                collision[a_i][bev_i, bev_j],
                            ]
                        )


if __name__ == "__main__":
    seed = 233
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    split = args.split
    
    check_folder("./results/road_graph")
    color_list = ["deeppink", "green", "blue", "orange", "purple", "brown", "red"]

    config = load_yaml_config("./config/config.yaml")
    dataloader, data_info = get_dataloader(args=config, split=split, shuffle=False)
    dataset_name = {"Nuscenes": "nusc_trainval", "nuPlanMini": "nuplan_mini"}[
        config.dataset
    ]

    plot_scenario = False
    plot_traj = False
    use_local_frame = True
    count = 0
    results_per_sample = []
    for i, batch in tqdm(enumerate(dataloader)):
        if i <  args.start_idx or i >= args.end_idx: continue
        for b_i in tqdm(range(len(batch.scene_ids))):
            agent_names = batch.agent_names[b_i]
            scene_ids = batch.scene_ids[b_i]  # {log_id}={lidar_pc_initial_token}
            scene_ts = CPU(batch.scene_ts[b_i])
            city_id = batch.extras["get_city_name"][b_i]
            city_name = ["boston", "singapore", "pittsburgh", "las_vegas"][city_id]
            ego_tf = CPU(batch.extras["agent_from_world_tf"][b_i])
            hist_world = batch.extras["get_hist_world"][b_i].float()
            hist_mask = (~torch.isnan(hist_world[..., 0])).float()
            
     
            # define the occupancy model
            occ_model = OccupancyModel(interval=config.grid_interval)

            if plot_scenario:
                plt.figure(figsize=(10, 10), dpi=300)

            agent_names_list = []
            beh_traj_list = []
            long_beh_name_list = []
            lat_beh_name_list = []
            collision_list = []
    
            for a_i in range(config.max_agent_num):
                agent_hist_world = hist_world[a_i]
                agent_mask = hist_mask[a_i]
                
                agent_curr_xy = agent_hist_world[-1, :2]
                ego_curr_xy = hist_world[0, -1, :2]
                
                # if a_i >= len(agent_names): continue
                # if check_angle_and_distance(config, ego_tf, ego_curr_xy, agent_curr_xy, agent_names[a_i], scene_ts) == False:
                #     continue
            
                # the selected agent is invalid
                if torch.sum(agent_mask) == 0:
                    continue

                
                # remove the invalid steps
                agent_hist_world = agent_hist_world[agent_mask == 1]

                # get the lane graph from the map using world frame
                # the current point [x, y, h, v]
                agent_hist = agent_hist_world
                agent_curr = agent_hist_world[-1]
                
                lane_graph = get_lane_graph_from_map(
                    config, agent_curr, dataset_name, city_name
                )
               

                # convert to local frame and it also works
                if use_local_frame:
                    lane_graph.to_local_frame(ego_tf)
                    agent_hist = convert_world_to_local(agent_hist_world, ego_tf)
                    agent_curr = agent_hist[-1]

                if plot_scenario:
                    plot_planning(agent_hist, color="black", label="origin")

                # get the behavior graph (work for both world and local frames)
                lane_graph.to_tensor()
  
                beh_traj, long_beh_name, lat_beh_name = get_behavior_traj(
                    config, lane_graph, agent_curr, plot=plot_traj
                )
                beh_traj_list.append(beh_traj)
                long_beh_name_list.append(long_beh_name)
                lat_beh_name_list.append(lat_beh_name)

                # generate the occupancy graph
                occ_graph = occ_model.get_occ_graph(lane_graph)

                # project the behavior trajectory to the occupancy graph
                beh_node = occ_graph.proj_traj_to_node(beh_traj)

                
                # project the ego trajectory to the occupancy graph to check collision
                if a_i > 0:
                    beh_traj_ego = beh_traj_list[0]
                    beh_node_ego = occ_graph.proj_traj_to_node(beh_traj_ego)
                    # [Ego behavior, Other behavior]
                    collision = occ_graph.check_collision(beh_node_ego, beh_node)
                else:
                    collision = None
                
               
                collision_list.append(collision)
                agent_names_list.append(agent_names[a_i])

                if plot_scenario:
                    # plot the occupancy node with behavior color
                    plot_occ_graph_beh_single_color(
                        occ_graph, beh_node, color_list[a_i % len(color_list)]
                    )
                    # plot the lane graph
                    plot_lane_tree(lane_graph.curr)
                    for left_tree in lane_graph.left_group:
                        plot_lane_tree(left_tree)
                    for right_tree in lane_graph.right_group:
                        plot_lane_tree(right_tree)
                        
                # print("-"*50)
                # print(a_i)
                
                # mark = False
                # if a_i != 0:
                #     print(collision.shape)
                #     print(a_i, len(long_beh_name_list), len(lat_beh_name_list), agent_names[a_i])

                #     for i in range(collision.shape[0]):
                #         for j in range(collision.shape[1]):
                #             print(i, j, collision[i][j])
                #             if collision[i][j] == True:
                #                 mark = True
                #             print(long_beh_name_list[0][i], lat_beh_name_list[0][i])
                #             print(long_beh_name_list[-1][j], lat_beh_name_list[-1][j])
                            
   
                # if mark == True:
                #     import sys 
                #     sys.exit()    
     
            # save results to a dict
            sample_result = {
                "scene_ids": scene_ids,
                "scene_ts": scene_ts,
                "city_name": city_name,
                "agent_names": agent_names_list,
                "hist_world": hist_world,
                "hist_mask": hist_mask,
                "beh_traj": beh_traj_list,
                "long_beh_name": long_beh_name_list,
                "lat_beh_name": lat_beh_name_list,
                "collision": collision_list,
            }
            results_per_sample.append(sample_result)

            if plot_scenario:
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
                plt.savefig(f"./results/road_graph/{count}.png")
                plt.close()
                count += 1
                
            

        # save results for every batch
        save_results(results_per_sample, split, args.start_idx, args.end_idx)
        
        
