import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from nuplan.database.utils.geometry import quaternion_yaw, view_points

from blg.generate_from_map import get_lane_graph_from_map, get_behavior_traj
from blg.utils.general_helpers import load_yaml_config
from blg.utils.plot_helpers import beh_color
from blg.model.occupancy_model import OccupancyModel
from blg.datasets.nuplan_dataset import NuPlanScenarioLoader, get_sensor_coord


def get_polygon_camera_coord(polygon, ego_pose, camera):
    polygon[:, 2] = ego_pose.z - 0.5  # height of the vehicle
    polygon = polygon.swapaxes(0, 1)
    cam_intrinsic = np.array(camera.intrinsic_np)
    cam_polygon, mark = get_sensor_coord(polygon, ego_pose, camera)

    # the polygon is not in the camera view
    if len(cam_polygon) == 0:
        return None

    points = view_points(np.array(cam_polygon), cam_intrinsic, normalize=True)
    points = points.swapaxes(0, 1)[:, :2]
    # points = np.concatenate([points[:, :], points[:1, :]], axis=0)
    return points


def plot_lane_polygon(lane, color, ego_pose, camera):
    if len(lane.center) == 0:
        return

    # get the polygon of the lane
    idx = 0
    polygon = np.concatenate([lane.left_edge[idx][::-1], lane.right_edge[idx]], axis=0)
    points = get_polygon_camera_coord(polygon, ego_pose, camera)
    if points is None:
        return

    # we cannot use ax.plot but don't know why. the results are not correct
    ax.add_patch(
        patches.Polygon(
            points, closed=True, edgecolor="black", facecolor=color, alpha=0.3
        )
    )


def plot_occ_beh_node_polygon(occ_graph, occ_graph_beh, ego_pose, camera):
    # aggregate the behavior nodes across the time dimension
    # [M, T, N] -> [M, N]
    occ_graph_beh = occ_graph_beh.sum(1) > 0

    for beh_i in range(len(occ_graph_beh)):
        for n_i in range(len(occ_graph_beh[beh_i])):
            if occ_graph_beh[beh_i][n_i] >= 1:
                left_i = occ_graph.left_edge[n_i].cpu().numpy()
                right_i = occ_graph.right_edge[n_i].cpu().numpy()
                polygon = np.concatenate((left_i[::-1], right_i), axis=0)
                points = get_polygon_camera_coord(polygon, ego_pose, camera)
                if points is None:
                    continue

                ax.add_patch(
                    patches.Polygon(
                        points,
                        closed=True,
                        edgecolor="black",
                        facecolor=beh_color[beh_i],
                        alpha=0.3,
                    )
                )


def plot_beh_traj(beh_traj, ego_pose, camera):
    beh_traj = beh_traj.cpu().numpy()
    for beh_i in range(len(beh_traj)):
        traj = beh_traj[beh_i, :, :3]  # [x, y, h]
        points = get_polygon_camera_coord(traj, ego_pose, camera)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=beh_color[beh_i],
            marker="o",
            linestyle="-",
            linewidth=1,
            markersize=5,
        )


# get the scene from nuplan dataset
TEST_SCENARIO_ID = "2021.08.09.17.55.59_veh-28_00021_00307"
for i in [26]:
    scene_name = f"scene-00{str(i)}"
    nuplan_loader = NuPlanScenarioLoader(TEST_SCENARIO_ID)
    metadata = nuplan_loader.load_scenario(scene_name=scene_name)
    break

dataset_name = "nuplan_mini"
device = "cpu"
count = 0
plot_lane_polygon_flag = False
plot_beh_traj_flag = True
plot_beh_node_flag = False
for key in metadata["sample_descriptions"].keys():
    # get info from the metadata
    scene_metadata = metadata["sample_descriptions"][key]
    sample_annotations = scene_metadata["sample_annotations"]
    ego_pose = sample_annotations[0]["ego_pose"]
    camera = sample_annotations[0]["camera"]

    city_name = "las_vegas"  # scene_metadata["city"] needs to be processed
    ego_curr_world = np.array(
        [
            scene_metadata["ego_vehicle_location"].x,
            scene_metadata["ego_vehicle_location"].y,
            # different heading frames
            # note that if the vehicle stops, this heading, calculated from the velocity, will be incorrect.
            np.deg2rad(scene_metadata["ego_vehicle_heading"]) - np.pi / 2,
            scene_metadata["ego_vehicle_velocity"],
        ]
    )

    # get the lane graph from the map using world frame
    config = load_yaml_config("./config/config.yaml")
    lane_graph = get_lane_graph_from_map(
        config, ego_curr_world, dataset_name, city_name, max_depth=5
    )

    im = plt.imread(scene_metadata["filepath"])
    fig, ax = plt.subplots()
    ax.imshow(im)

    # plot polygon of all lane
    if plot_lane_polygon_flag:
        plot_lane_polygon(lane_graph.curr, "r", ego_pose, camera)
        left_color = ["g", "orange", "deeppink", "cyan"]
        for l_i in range(len(lane_graph.left_group)):
            plot_lane_polygon(
                lane_graph.left_group[l_i], left_color[l_i], ego_pose, camera
            )
        right_color = ["b", "purple", "brown", "yellow"]
        for r_i in range(len(lane_graph.right_group)):
            plot_lane_polygon(
                lane_graph.right_group[r_i], right_color[r_i], ego_pose, camera
            )

    # get and plot the behvaiors
    if plot_beh_node_flag or plot_beh_traj_flag:
        lane_graph.to_tensor()
        ego_curr_world = torch.as_tensor(ego_curr_world, device=device).float()
        beh_traj, long_beh_name, lat_beh_name = get_behavior_traj(
            config, lane_graph, ego_curr_world, plot=False
        )

        if plot_beh_traj_flag:
            plot_beh_traj(beh_traj, ego_pose, camera)

        if plot_beh_node_flag:
            occ_model = OccupancyModel(interval=5.0, threshold=1.0)
            occ_graph = occ_model.get_occ_graph(lane_graph)
            # project the behavior trajectory to the occupancy graph
            occ_graph_beh_node = occ_graph.proj_traj_to_node(beh_traj)
            plot_occ_beh_node_polygon(occ_graph, occ_graph_beh_node, ego_pose, camera)

    plt.xlim([0, 1920])
    plt.ylim([1080, 0])
    plt.axis("off")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"./results/camera_view/lane_graph_{count}.png", dpi=200)
    plt.close()
    count += 1
