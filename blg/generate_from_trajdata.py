import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from blg.utils.data_helpers import data_process
from blg.utils.general_helpers import CPU, check_folder, load_yaml_config
from blg.utils.plot_helpers import plot_map, plot_planning
from blg.model.behavior_model import BehaviorModel
from blg.datasets.trajdata_dataloader import get_dataloader


def get_behavior_traj(config, lane_graph, ego_xyhv, plot=True):
    bm = BehaviorModel(dt=config.dt, horizon=config.horizon)  # 3 seconds
    # ego_xyhv[3] = 4  # set v to 4 m/s

    # convert the ego point to the frenet coordinate
    # the current pose should be the same for all current lanes, so we just take the first one
    curr_lane = lane_graph.curr[0]

    # show all combinations of behaviors
    longitudinal_bev = {
        "keep speed": 0.0,
        "accelerate": config.acceleration,
        "decelerate": config.deceleration,
    }
    lateral_bev = {
        "keep lane": lane_graph.curr,
        "left lane-change": lane_graph.left,
        "right lane-change": lane_graph.right,
    }
    color_list = ["deeppink", "green", "blue", "orange", "purple", "brown", "red"]

    all_traj = []
    exist_traj = []
    for lon_i in longitudinal_bev.keys():
        target_acc = longitudinal_bev[lon_i]
        for l_i, lat_i in enumerate(lateral_bev.keys()):
            target_lanes = lateral_bev[lat_i]
            for target_lane in target_lanes:
                new_traj_xyhv, new_lane = bm.combined_behavior(
                    ego_xyhv, target_acc, curr_lane, target_lane
                )
                new_traj_xyhv = CPU(new_traj_xyhv)

                # to make the plot clear, we remove traj that are too close
                if len(exist_traj) > 0:
                    dist_to_exist_traj = np.linalg.norm(
                        new_traj_xyhv[-1, 0:2] - np.stack(exist_traj), axis=1
                    )
                    if np.min(dist_to_exist_traj) < 1:
                        continue
                exist_traj.append(new_traj_xyhv[-1, 0:2].copy())

                # plot the trajectory and the last point
                if plot:
                    plot_planning(new_traj_xyhv, color=color_list[l_i], label=lat_i)
                all_traj.append(new_traj_xyhv)
    return all_traj


if __name__ == "__main__":
    config = load_yaml_config("./config/config.yaml")
    dataloader, data_info = get_dataloader(args=config, split="val")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    count = 0
    b_i = 0
    check_folder("./results/road_graph")
    for batch in tqdm(dataloader):
        batch = data_process(batch, data_info, device)
        lane_graph = batch["extras"]["get_lane_graph"][b_i]
        ego_pose = batch["ego_hist"][b_i, -1][2:6]  # [x, y, h, v]

        plt.figure(figsize=(10, 10), dpi=300)
        plot_map(batch["centerlines"][b_i, 0])
        plot_planning(CPU(ego_pose)[None], color="black", label="origin")

        # plot ego gt future
        ego_fut_mask = batch["ego_fut"][b_i, :, -1]
        ego_gt_xy = batch["ego_fut"][b_i, ego_fut_mask == 1, 2:4]
        ego_gt_xy = torch.cat([ego_pose[0:2][None], ego_gt_xy], dim=0)
        ego_gt_xy = CPU(ego_gt_xy)
        plt.plot(ego_gt_xy[:, 0], ego_gt_xy[:, 1], "k", label="ego future", linewidth=2)

        # plot the behavior lane graph
        all_traj = get_behavior_traj(config, lane_graph, ego_pose)

        # remove repeated labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.xlim(-40, 40)
        plt.ylim(-40, 40)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f"./results/road_graph/{count}.png")
        plt.close()
        count += 1
