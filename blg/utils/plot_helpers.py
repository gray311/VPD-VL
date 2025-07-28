from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams["axes.grid"] = False

from trajdata.maps.map_api import MapAPI
from blg.utils.general_helpers import CPU


agent_type_color = {
    "ego": "g",
    "vehicle": "b",
    "pedestrian": "orange",
    "bicycle": "k",
    "motorcycle": "c",
    "unknown": "y",
}

agent_type_mapping = {
    0: "unknown",
    1: "vehicle",
    2: "pedestrian",
    3: "bicycle",
    4: "motorcycle",
}

beh_color = [
    "deeppink",
    "green",
    "blue",
    "orange",
    "purple",
    "brown",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "gray",
    "pink",
    "lime",
    "teal",
    "lavender",
    "brown",
]


def plot_occ_graph_beh(occ_graph, beh_nodes):
    # aggregate the behavior nodes across the time dimension
    # [M, T, N] -> [M, N]
    beh_nodes = beh_nodes.sum(1) > 0

    # plot the occupancy node with behavior color
    for beh_i in range(len(beh_nodes)):
        for n_i in range(len(beh_nodes[beh_i])):
            if beh_nodes[beh_i][n_i] >= 1:
                left_i = occ_graph.left_edge[n_i].cpu().numpy()
                right_i = occ_graph.right_edge[n_i].cpu().numpy()
                edges = np.concatenate((left_i, right_i[::-1], left_i[0:1]), axis=0)
                plt.fill(
                    edges[:, 0],
                    edges[:, 1],
                    alpha=0.2,
                    color=beh_color[beh_i],
                )


def plot_occ_graph_beh_single_color(occ_graph, beh_nodes, color):
    # aggregate the behavior nodes across the time dimension
    # [M, T, N] -> [M, N]
    beh_nodes = beh_nodes.sum(1) > 0

    # plot the occupancy node with behavior color
    for beh_i in range(len(beh_nodes)):
        for n_i in range(len(beh_nodes[beh_i])):
            if beh_nodes[beh_i][n_i] >= 1:
                left_i = occ_graph.left_edge[n_i].cpu().numpy()
                right_i = occ_graph.right_edge[n_i].cpu().numpy()
                edges = np.concatenate((left_i, right_i[::-1], left_i[0:1]), axis=0)
                plt.fill(edges[:, 0], edges[:, 1], alpha=0.2, color=color)


def plot_occ_graph(occ_graph):
    for i in range(len(occ_graph.left_edge)):
        left_i = occ_graph.left_edge[i].cpu().numpy()
        right_i = occ_graph.right_edge[i].cpu().numpy()
        edges = np.concatenate((left_i, right_i[::-1], left_i[0:1]), axis=0)
        plt.fill(edges[:, 0], edges[:, 1], alpha=0.2, edgecolor="none")


def plot_lane_tree(lane_tree):
    for l_i in range(len(lane_tree.center)):
        # plot the center line
        center = CPU(lane_tree.center[l_i])
        plt.plot(center[:, 0], center[:, 1], color="gray", alpha=0.4)
        # plot the left and right edges using polygon
        left_i = CPU(lane_tree.left_edge[l_i])
        right_i = CPU(lane_tree.right_edge[l_i])
        edges = np.concatenate((left_i, right_i[::-1], left_i[0:1]), axis=0)
        plt.fill(edges[:, 0], edges[:, 1], color="gray", alpha=0.2, edgecolor="none")


def load_vec_map(map_name, cache_path="~/.unified_data_cache"):
    cache_path = Path(cache_path).expanduser()
    mapAPI = MapAPI(cache_path)
    vec_map = mapAPI.get_map(map_name, scene_cache=None)
    return vec_map


def plot_map(centerlines, ax=None):
    centerlines = CPU(centerlines)  # [S, 5] or [S, P, 5]
    # plot the map [S, P, 3]
    for s_i in range(centerlines.shape[0]):
        mask_map = centerlines[s_i, :, -1] == 1
        plt.plot(
            centerlines[s_i, mask_map, 0],
            centerlines[s_i, mask_map, 1],
            c="gray",
            alpha=0.4,
        )

    if ax is not None:
        ax.axis("off")
        ax.grid("off")
        ax.set_xlim(-70, 70)
        ax.set_ylim(-70, 70)
    else:
        # plt.legend(fontsize=7)
        plt.grid("off")
        plt.axis("off")
        plt.xlim(-70, 70)
        plt.ylim(-70, 70)


def plot_agent_hist(a_i, hist, color, name, ax=None):
    hist = CPU(hist)  # [T, 5]
    alpha = 1.0
    if ax is not None:
        ax.plot(hist[:, 0], hist[:, 1], color=color, ls="-", alpha=alpha, label=name)
    else:
        plt.plot(hist[:, 0], hist[:, 1], color=color, ls="-", alpha=alpha, label=name)

    bbox_length, bbox_width = 3.7, 1.5
    pos = hist[:, 0:2][-1]
    heading = hist[:, 2][-1]
    corners = np.array(
        [
            [-bbox_length / 2, -bbox_width / 2],
            [-bbox_length / 2, bbox_width / 2],
            [bbox_length / 2, bbox_width / 2],
            [bbox_length / 2, -bbox_width / 2],
            [-bbox_length / 2, -bbox_width / 2],
        ]
    )
    R = np.array(
        [
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)],
        ]
    )
    corners = corners @ R.T + pos
    if ax is not None:
        ax.plot(corners[:, 0], corners[:, 1], color=color, ls="-", alpha=alpha)
    else:
        plt.plot(corners[:, 0], corners[:, 1], color=color, ls="-", alpha=alpha)

    # Add text at the left bottom corner
    text_pos = corners[0]
    if ax is not None:
        ax.text(text_pos[0], text_pos[1], str(a_i), fontsize=7, ha="left", va="bottom")
    else:
        plt.text(text_pos[0], text_pos[1], str(a_i), fontsize=7, ha="left", va="bottom")


def plot_prediction(
    hist,
    fut,
    agent_types,
    pred_obs,
    mode_probs,
    plot_fut=True,
    plot_hist=True,
    plot_pred=True,
    highlight_agents=None,
):
    hist = CPU(hist)  # [M+1, T, 5]
    fut = CPU(fut)  # [M+1, T, 5]
    agent_types = CPU(agent_types)  # [A, 5]
    pred_obs = CPU(pred_obs)  # [c, M, T, 5]
    mode_probs = CPU(mode_probs)  # [c]

    # to have a better visualization, we concatenate the last point of history to the future
    # [M+1, T+1, 5]
    fut = np.concatenate([hist[:, -1:], fut], axis=1)

    global_x_axis = 0
    global_y_axis = 1
    global_heading_axis = 2

    def get_color_and_name(a_i):
        agent_name = agent_type_mapping[np.argmax(agent_types[a_i])]
        color = agent_type_color[agent_name] if a_i != 0 else "g"
        agent_name = "ego" if a_i == 0 else agent_name
        return color, agent_name

    # plot history
    for a_i in range(hist.shape[0]):
        alpha = 1.0
        mask_agent = hist[a_i, :, -1] == 1
        if sum(mask_agent) != 0:
            color, agent_name = get_color_and_name(a_i)
            if highlight_agents is not None and a_i in highlight_agents:
                color = "deeppink"
            if plot_hist:
                plt.plot(
                    hist[a_i, mask_agent, global_x_axis],
                    hist[a_i, mask_agent, global_y_axis],
                    color=color,
                    ls="-",
                    alpha=alpha,
                    label=agent_name + "-history",
                )

            # plot history last point with a rectangle
            if agent_type_mapping[np.argmax(agent_types[a_i])] in ["vehicle", "ego"]:
                bbox_length, bbox_width = 3.7, 1.5
            elif agent_type_mapping[np.argmax(agent_types[a_i])] in ["pedestrian"]:
                bbox_length, bbox_width = 0.6, 0.6
            elif agent_type_mapping[np.argmax(agent_types[a_i])] in [
                "bicycle",
                "motorcycle",
            ]:
                bbox_length, bbox_width = 1.8, 0.6
            pos = hist[a_i, mask_agent, global_x_axis : global_y_axis + 1][-1]
            heading = hist[a_i, mask_agent, global_heading_axis][-1]
            corners = np.array(
                [
                    [-bbox_length / 2, -bbox_width / 2],
                    [-bbox_length / 2, bbox_width / 2],
                    [bbox_length / 2, bbox_width / 2],
                    [bbox_length / 2, -bbox_width / 2],
                    [-bbox_length / 2, -bbox_width / 2],
                ]
            )
            R = np.array(
                [
                    [np.cos(heading), -np.sin(heading)],
                    [np.sin(heading), np.cos(heading)],
                ]
            )
            corners = corners @ R.T + pos
            plt.plot(corners[:, 0], corners[:, 1], color=color, ls="-", alpha=alpha)

            # Add text at the left bottom corner
            text_pos = corners[0]
            plt.text(
                text_pos[0], text_pos[1], str(a_i), fontsize=7, ha="left", va="bottom"
            )

    # plot future
    if plot_fut:
        for a_i in range(fut.shape[0]):
            mask_agent = fut[a_i, :, -1] == 1
            if sum(mask_agent) != 0:
                color, agent_name = get_color_and_name(a_i)
                alpha = 1.0
                plt.plot(
                    fut[a_i, mask_agent, global_x_axis],
                    fut[a_i, mask_agent, global_y_axis],
                    color=color,
                    ls="--",
                    alpha=alpha,
                    label=agent_name + "-future-gt",
                )
                # plot scatter
                # plt.scatter(
                #     fut[a_i, mask_agent, global_x_axis],
                #     fut[a_i, mask_agent, global_y_axis],
                #     facecolor=color,
                #     edgecolor="none",
                #     s=10,
                # )

    # plot multi-modal prediction
    if plot_pred:
        for c_i in range(pred_obs.shape[0]):
            pred_obs_c = pred_obs[c_i]  # [T, M, 5]
            for a_i in range(pred_obs_c.shape[0]):
                alpha = 0.4
                mask_agent = hist[a_i, :, -1] == 1  # use the mask of the history
                if sum(mask_agent) != 0:
                    color, agent_name = get_color_and_name(a_i)
                    # connect to the last point of history
                    hist_last_pred = np.concatenate(
                        [
                            hist[a_i, -1:, global_x_axis : global_y_axis + 1],
                            pred_obs_c[a_i, :, :2],
                        ],
                        axis=0,
                    )
                    plt.plot(
                        hist_last_pred[:, 0],
                        hist_last_pred[:, 1],
                        color=color,
                        ls="-",
                        alpha=alpha,
                        label=agent_name + "-future-prediction",
                    )
                    # plot the last point with a dot, size depends on the probability
                    dot_size = mode_probs[c_i] * 300
                    plt.scatter(
                        pred_obs_c[a_i, -1, 0],
                        pred_obs_c[a_i, -1, 1],
                        facecolor=color,
                        edgecolor="none",
                        s=dot_size,
                        alpha=alpha,
                    )

    # Collect handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), fontsize=12)


def plot_planning(xyhv, color="k", label=None, ax=None):
    xyhv = CPU(xyhv)  # [T, 5]
    last_xy = xyhv[-1, 0:2]
    last_h = xyhv[-1, 2]

    # trajectory
    if ax is not None:
        ax.plot(xyhv[:, 0], xyhv[:, 1], color=color, ls="-", alpha=0.5, label=label)
    else:
        plt.plot(xyhv[:, 0], xyhv[:, 1], color=color, ls="-", alpha=0.5, label=label)

    # plot bounding box
    bbox_length, bbox_width = 3.7, 1.5
    corners = np.array(
        [
            [-bbox_length / 2, -bbox_width / 2],
            [-bbox_length / 2, bbox_width / 2],
            [bbox_length / 2, bbox_width / 2],
            [bbox_length / 2, -bbox_width / 2],
            [-bbox_length / 2, -bbox_width / 2],
        ]
    )
    R = np.array(
        [
            [np.cos(last_h), -np.sin(last_h)],
            [np.sin(last_h), np.cos(last_h)],
        ]
    )
    corners = corners @ R.T + last_xy
    if ax is not None:
        ax.plot(
            corners[:, 0], corners[:, 1], color=color, ls="-", alpha=0.5, label=label
        )
    else:
        plt.plot(
            corners[:, 0], corners[:, 1], color=color, ls="-", alpha=0.5, label=label
        )

    # plot filled rectangle
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    # Plot the polygon using plt.fill() to fill the area without a border
    if ax is not None:
        ax.fill(x_coords, y_coords, color=color, edgecolor="none", alpha=0.2)
    else:
        plt.fill(x_coords, y_coords, color=color, edgecolor="none", alpha=0.2)
