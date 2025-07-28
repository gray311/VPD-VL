import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Affine2D

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
from blg.datasets.nuplan_dataset import get_sensor_coord

from blg.model.occupancy_model import OccupancyModel
from blg.model.behavior_model import BehaviorModel
from blg.datasets.trajdata_dataloader import get_dataloader
from blg.utils.general_helpers import check_folder, load_yaml_config, CPU
from blg.utils.plot_helpers import plot_map, plot_planning, plot_agent_hist
from blg.generate_from_map import get_lane_graph_from_map, convert_world_to_local


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


def load_scene_camera(scene_ids, scene_ts, agent_names, desired_dt):
    NUPLAN_DATA_ROOT = "/home/wenhaod/data/nuplan/dataset"
    NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
    NUPLAN_MAPS_ROOT = "/home/wenhaod/data/nuplan/maps"
    NUPLAN_SENSOR_ROOT = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/sensor_blobs"
    NUPLAN_DB_FILES = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/mini"
    SCENARIO_DURATION = 20.0  # [s] duration of the scenario (e.g. extract 20s from when the event occurred)
    EXTRACTION_OFFSET = 0.0  # [s] offset of the scenario (e.g. start at -5s from when the event occurred)
    SUBSAMPLE_RATIO = 1.0  # ratio used sample the scenario (e.g. a 0.1 ratio means sample from 20Hz to 2Hz)
    NUPLAN_ORIGINAL_DT = 0.05

    # get the name of log file and scene token
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
    log_db = nuplandb_wrapper.get_log_db(logfile)

    # collect all camera info
    camera_dict = {}
    for item in log_db.camera:
        camera_dict[item.channel] = item

    curr_scene = None
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
    assert curr_scene is not None, f"Scene {scene_token} not found in log {logfile}"

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
    channels = [CameraChannel.CAM_F0, CameraChannel.CAM_L0, CameraChannel.CAM_R0]
    img_list = []
    for t_i in range(curr_frame, number_of_frames):
        sensors = scenario.get_sensors_at_iteration(t_i, channels)
        img_front = sensors.images[channels[0]].as_numpy
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
        img_list.append(img)
    img_list = np.stack(img_list)

    # retrieve object info
    # tracked_objects = scenario.get_tracked_objects_at_iteration(curr_frame)
    # tracked_objects = list(tracked_objects.tracked_objects)
    # for lidar_box in log_db.lidar_box:
    #     if lidar_box.track_token in agent_names:
    #         print("track", lidar_box.track_token)
    return curr_scene, img_list  # [T, H, W, C]


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


if __name__ == "__main__":
    scenario_list = pd.read_csv("./results/results_val.csv")
    scenario_list = scenario_list[scenario_list["Collision"] == True]

    config = load_yaml_config("./config/config.yaml")
    split = "val"
    dataloader, data_info = get_dataloader(args=config, split=split, plot_scene=True)
    dataset_name = {"Nuscenes": "nusc_trainval", "nuPlanMini": "nuplan_mini"}[
        config.dataset
    ]

    check_folder("./results/collision")
    for batch in tqdm(dataloader):
        for b_i in tqdm(range(len(batch.scene_ids)), leave=False):
            scene_ids = batch.scene_ids[b_i]
            scene_ts = batch.scene_ts[b_i].item()

            # only process the scene in the list
            collision_info = scenario_list[
                (scenario_list["Scene ID"] == scene_ids)
                & (scenario_list["Initial Timestep"] == scene_ts)
            ]
            if collision_info.empty:
                continue

            ego_long_beh = collision_info["Ego Long Behavior"].values
            ego_lat_beh = collision_info["Ego Lat Behavior"].values
            agent_long_beh = collision_info["Agent Long Behavior"].values
            agent_lat_beh = collision_info["Agent Lat Behavior"].values
            agent_ids = collision_info["Agent ID"].values
            agent_names = collision_info["Agent Name"].values

            # load camera info, [T, H, W, C]
            curr_scene, scene_imgs = load_scene_camera(
                scene_ids, scene_ts, agent_names, config.desired_dt
            )

            # plot the front camera
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

            # for calculating behavior trajectory
            city_id = batch.extras["get_city_name"][b_i]
            city_name = ["boston", "singapore", "pittsburgh", "las_vegas"][city_id]
            ego_tf = CPU(batch.extras["agent_from_world_tf"][b_i])
            hist_world = batch.extras["get_hist_world"][b_i].float()
            hist_mask = (~torch.isnan(hist_world[..., 0])).float()
            occ_model = OccupancyModel(interval=config.grid_interval)

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
                    c_i = agent_ids.tolist().index(a_i)
                    lon_cand = [agent_long_beh[c_i]]
                    lat_cand = [agent_lat_beh[c_i]]

                # get the lane graph from the map using world frame
                # the current point [x, y, h, v]
                agent_curr_world = agent_hist_world[-1]
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
                    plot_traj_on_camera(
                        beh_traj[traj_i],
                        curr_scene["curr_ego_pose"],
                        curr_scene["camera"],
                        color,
                        ax1,
                    )

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
            plt.savefig(f"./results/collision/{scene_ids}.png")
            plt.close("all")
