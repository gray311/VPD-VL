import multiprocessing as mp

import torch
import numpy as np
from tqdm import tqdm

from blg.utils.general_helpers import (
    CPU,
    load_yaml_config,
    namedtuple_to_dict,
    dict_to_namedtuple,
    check_folder,
)
from blg.datasets.trajdata_dataloader import get_dataloader
from blg.model.occupancy_model import OccupancyModel
from blg.generate_from_map import (
    get_lane_graph_from_map,
    convert_world_to_local,
    get_behavior_traj,
    save_results,
)

torch.multiprocessing.set_sharing_strategy("file_system")


def worker(config, batch, b_i):
    config = dict_to_namedtuple("config", config)
    agent_names = batch.agent_names[b_i]
    scene_ids = batch.scene_ids[b_i]
    scene_ts = CPU(batch.scene_ts[b_i])
    city_id = batch.extras["get_city_name"][b_i]
    city_name = ["boston", "singapore", "pittsburgh", "las_vegas"][city_id]
    ego_tf = CPU(batch.extras["agent_from_world_tf"][b_i])
    hist_world = batch.extras["get_hist_world"][b_i].float()
    hist_mask = (~torch.isnan(hist_world[..., 0])).float()

    # define the occupancy model
    occ_model = OccupancyModel(interval=config.grid_interval)
    dataset_name = {"Nuscenes": "nusc_trainval", "nuPlanMini": "nuplan_mini"}[
        config.dataset
    ]

    agent_names_list = []
    beh_traj_list = []
    long_beh_name_list = []
    lat_beh_name_list = []
    collision_list = []
    for a_i in range(config.max_agent_num):
        agent_hist_world = hist_world[a_i]
        agent_mask = hist_mask[a_i]

        # the selected agent is invalid
        if torch.sum(agent_mask) == 0:
            continue

        # remove the invalid steps
        agent_hist_world = agent_hist_world[agent_mask == 1]

        # get the lane graph from the map using world frame
        # the current point [x, y, h, v]
        agent_curr_world = agent_hist_world[-1]
        lane_graph = get_lane_graph_from_map(
            config, agent_curr_world, dataset_name, city_name
        )

        # convert the lane graph and history to the local frame
        lane_graph.to_local_frame(ego_tf)
        agent_hist = convert_world_to_local(agent_hist_world, ego_tf)
        agent_curr = agent_hist[-1]

        # get the behavior graph
        lane_graph.to_tensor()
        beh_traj, long_beh_name, lat_beh_name = get_behavior_traj(
            config, lane_graph, agent_curr, plot=False
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
    return sample_result


if __name__ == "__main__":
    seed = 233
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    split = "val"
    config = load_yaml_config("./config/config.yaml")
    dataloader, data_info = get_dataloader(args=config, split=split, shuffle=False)
    check_folder("./results")

    results_per_sample = []
    config_dict = namedtuple_to_dict(config)
    for batch in tqdm(dataloader):
        # use multiprocessing to speed up the process
        batch_size = len(batch.scene_ids)
        pool_list = []
        with mp.Pool(config.num_workers) as pool:
            for b_i in range(batch_size):
                pool_list.append(
                    pool.apply_async(worker, args=(config_dict, batch, b_i))
                )
            results_per_sample.extend([p.get() for p in pool_list])
    save_results(results_per_sample, split)
