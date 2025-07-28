import torch
import numpy as np
import torch.nn.functional as F


def rotate_np(x: np.ndarray, y: np.ndarray, angle: np.ndarray) -> np.ndarray:
    other_x_trans = np.cos(angle) * x - np.sin(angle) * y
    other_y_trans = np.cos(angle) * y + np.sin(angle) * x
    output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
    return output_coords


def rotate_tensor(x, y, angle):
    other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
    other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
    output_coords = torch.stack((other_x_trans, other_y_trans), dim=-1)
    return output_coords


def convert_local_to_global(ego_in, agent_in, pred_obs):
    # ego_in:   [B, T, D]    - [lx,ly,x,y,h,v,mask]
    # agent_in: [B, T, M, D] - [lx,ly,x,y,h,v,mask]
    # pred_obs: [c, T, B, M+1, 5]
    ego_agent_in = torch.cat((ego_in.unsqueeze(2), agent_in), dim=2)  # [B, T, M+1, D]
    hist_last_xy = ego_agent_in[:, -1, :, 2:4]  # [B, M+1, 2]
    hist_last_heading = ego_agent_in[:, -1, :, 4]  # [B, M+1]
    hist_last_xy = hist_last_xy[None, None, :, :, :].repeat(
        pred_obs.size(0), pred_obs.size(1), 1, 1, 1
    )  # [c, T, B, M+1, 2]
    hist_last_heading = hist_last_heading[None, None].repeat(
        pred_obs.size(0), pred_obs.size(1), 1, 1
    )  # [c, T, B, M+1]

    global_pred_xy = (
        rotate_tensor(
            pred_obs[..., 0].clone(), pred_obs[..., 1].clone(), hist_last_heading
        )
        + hist_last_xy
    )  # [c, T, B, M+1, 2]

    # add the variance back
    global_pred_xy = torch.cat(
        (global_pred_xy, pred_obs[..., 2:].clone()), dim=-1
    )  # [c, T, B, M+1, 5]
    return global_pred_xy


def data_process(batch, data_info, device):
    """Process the batch data for training.

    Args:
        batch (BatchElement): batch.agent_hist.shape / batch.agent_fut.shape - [B, M, SH, D]
        data_info (dict): information of the dataset
        device (str): cpu or cuda

    Returns:
        Dict: processed data for training
    """

    # get traj from batch
    hist = batch.agent_hist.to(device)
    fut = batch.agent_fut.to(device)
    agent_types = batch.agent_type  # (B, M)
    agent_types[agent_types == -1] = 0  # (-1 means padded agent)

    # get mask
    agent_hist_mask = (~torch.isnan(hist[:, :, :, 0])).float()
    agent_fut_mask = (~torch.isnan(fut[:, :, :, 0])).float()

    # replace nan with 0 for computation
    hist = torch.nan_to_num(hist, nan=0.0)
    fut = torch.nan_to_num(fut, nan=0.0)

    # convert [x, y, z, xd, yd, h] to [x, y, h, v]
    hist_v = (hist[..., 3] ** 2 + hist[..., 4] ** 2) ** 0.5
    fut_v = (fut[..., 3] ** 2 + fut[..., 4] ** 2) ** 0.5
    hist = torch.cat([hist[..., [0, 1, 5]], hist_v[..., None]], dim=-1)
    fut = torch.cat([fut[..., [0, 1, 5]], fut_v[..., None]], dim=-1)
    heading_dim = 2

    # process map - (B, S, P, 4) - [x, y, h, mask]
    centerlines = batch.extras["get_vector_map"].float().to(device)
    # repeat same map for all agents - [B, M, S, P, 4]
    centerlines = centerlines.unsqueeze(1).repeat(1, hist.size(1), 1, 1, 1)

    # IMPORTANT: for joint prediction, we need to add the local coordinate of agents as feature
    hist_last_xy = hist[:, :, -1:, :2].clone()  # (B, M, 1, 2)
    hist_last_heading = hist[:, :, -1:, heading_dim].clone()  # (B, M, 1)
    local_hist_xy = rotate_tensor(
        hist[..., 0] - hist_last_xy[..., 0],
        hist[..., 1] - hist_last_xy[..., 1],
        -hist_last_heading,
    )  # (B, M, SH, 2)
    local_fut_xy = rotate_tensor(
        fut[..., 0] - hist_last_xy[..., 0],
        fut[..., 1] - hist_last_xy[..., 1],
        -hist_last_heading,
    )  # (B, M, SF, 2)
    # local_hist_heading = hist[..., heading_dim] - hist_last_heading   # (B, M, SH)
    # local_fut_heading = fut[..., heading_dim] - hist_last_heading     # (B, M, SF)
    hist = torch.cat((local_hist_xy, hist), dim=-1)  # (B, M, SH, 5) - [lx,ly,x,y,h,v]
    fut = torch.cat((local_fut_xy, fut), dim=-1)  # (B, M, SF, 5) - [lx,ly,x,y,h,v]

    # IMPORTANT: for joint prediction, we need to rotate maps to local coordinate of agents
    hist_last_xy_extend = hist_last_xy[:, :, :, None, :].repeat(
        1, 1, centerlines.size(2), centerlines.size(3), 1
    )  # (B, M, S, P, 2)
    hist_last_heading_extend = hist_last_heading[:, :, None, :].repeat(
        1, 1, centerlines.size(2), centerlines.size(3)
    )  # (B, M, S, P)
    local_centerlines = rotate_tensor(
        centerlines[..., 0] - hist_last_xy_extend[..., 0],
        centerlines[..., 1] - hist_last_xy_extend[..., 1],
        -hist_last_heading_extend,
    )  # (B, M, S, P, 2)
    centerline_heading = centerlines[..., 2] - hist_last_heading_extend  # (B, M, S, P)
    centerlines = torch.cat(
        (local_centerlines, centerline_heading[..., None], centerlines[..., -1:]),
        dim=-1,
    )  # (B, M, S, P, 4) - [x, y, h, mask]

    # add mask to hist and fut
    hist = torch.cat((hist, agent_hist_mask.unsqueeze(-1)), dim=-1)
    fut = torch.cat((fut, agent_fut_mask.unsqueeze(-1)), dim=-1)
    hist = hist.permute(0, 2, 1, 3)  # (B, SH, A, D)
    fut = fut.permute(0, 2, 1, 3)  # (B, SF, A, D)

    # seperate ego and agents - (B, A, SH, D) - [lx,ly,x,y,h,v,mask]
    # (B, SH, D), (B, SH, M, D)
    ego_hist, agents_hist = hist[:, :, 0, :], hist[:, :, 1:, :]
    # (B, SF, D), (B, SF, M, D)
    ego_fut, agents_fut = fut[:, :, 0, :], fut[:, :, 1:, :]

    # agent type
    num_agent_types = data_info["num_agent_types"]
    agent_types = (
        F.one_hot(agent_types.long(), num_classes=num_agent_types).float().to(device)
    )

    return {
        "ego_hist": ego_hist,
        "agents_hist": agents_hist,
        "ego_fut": ego_fut,
        "agents_fut": agents_fut,
        "centerlines": centerlines,
        "agent_types": agent_types,
        "agent_names": batch.agent_names,
        "data_index": batch.data_idx,
        "scene_ids": batch.scene_ids,
        "scene_ts": batch.scene_ts,
        "map_names": batch.map_names,
        "extras": batch.extras,
    }
