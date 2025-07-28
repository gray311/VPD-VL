import numpy as np
import torch
from scipy.interpolate import CubicHermiteSpline

from blg.utils.frenet import (
    cartesian_to_frenet,
    frenet_to_cartesian,
    frenet_to_cartesian_batch,
)


def get_heading_velocity(trajectory):
    delta = trajectory[1:] - trajectory[:-1]
    headings = torch.atan2(delta[:, 1], delta[:, 0])
    speeds = torch.norm(delta, dim=1)

    # repeat the last heading and speed
    headings = torch.cat([headings, headings[-1:]]).unsqueeze(1)
    speeds = torch.cat([speeds, speeds[-1:]]).unsqueeze(1)
    trajectory = torch.cat([trajectory, headings, speeds], dim=1)
    return trajectory


def decompose_velocity(vehicle_vel, vehicle_h, lane_h):
    # Calculate the relative heading angle (delta theta)
    delta_theta = vehicle_h - lane_h

    # Normalize delta_theta to be within [-pi, pi]
    delta_theta = torch.atan2(torch.sin(delta_theta), torch.cos(delta_theta))

    # Decompose into longitudinal and lateral components
    long_vel = vehicle_vel * torch.cos(delta_theta)  # Longitudinal component
    lat_vel = vehicle_vel * torch.sin(delta_theta)  # Lateral component

    return long_vel, lat_vel


class BehaviorModel(object):
    def __init__(self, dt=0.5, horizon=30):
        self.dt = dt
        self.horizon = horizon
        self.min_v = 0.0
        self.max_v = 35.0
        self.max_acc = 4.0
        self.min_acc = -4.0

    def forward_straight(self, s_init, v_init, acc):
        # acc is a float type
        acc = np.clip(acc, self.min_acc, self.max_acc)

        steps = torch.arange(1, self.horizon + 1, device=s_init.device).float()
        traj_v = v_init + acc * self.dt * steps

        # Ensure velocity is not negative
        traj_v = torch.clamp(traj_v, min=self.min_v, max=self.max_v)

        # Calculate displacement only where velocity is positive
        stop = torch.tensor(0.0, device=s_init.device)
        displacement = torch.where(traj_v > 0, traj_v * self.dt, stop)

        # Calculate cumulative sum of displacements to get positions
        traj_s = s_init + torch.cumsum(displacement, dim=0)
        return traj_s, traj_v

    def combined_behavior(self, ego_xyhv, acc, curr_lane, target_lane):
        # convert current xy to frenet sd
        xy_cl = ego_xyhv[0:2]
        ego_h = ego_xyhv[2]
        ego_vel = ego_xyhv[3]
        sd_cl = cartesian_to_frenet(xy_cl, curr_lane)

        # sd_cl: sd in current lane. only use x and y
        curr_lane = curr_lane[..., 0:2].to(sd_cl.device)
        target_lane = target_lane[..., 0:2].to(sd_cl.device)

        # we need to change the reference lane to target first
        xy, seg_tan = frenet_to_cartesian(sd_cl, curr_lane)
        seg_heading = torch.atan2(seg_tan[1], seg_tan[0])
        if torch.equal(curr_lane, target_lane):
            sd_tl = sd_cl.clone()
        else:
            sd_tl = cartesian_to_frenet(xy, target_lane)

        # decompose the velocity into longitudinal and lateral velocity
        sv, dv = decompose_velocity(ego_vel, ego_h, seg_heading)

        # Longitude behavior
        # assume the vehicle use acc to change the velocity
        new_s, new_sv = self.longitude_behavior(sd_tl, sv, acc)

        # Lateral behavior
        use_dynamics = True
        start_d = sd_tl[1].clone()
        target_d = 0.0
        target_v = 0.0
        if use_dynamics:
            long_h = ego_h - seg_heading
            new_d = self.lateral_lane_change_dynamics(start_d, target_d, long_h, new_sv)
            curr_lane = target_lane.clone()
        else:
            if torch.equal(curr_lane, target_lane):
                # if curr_lane is the same as new_lane, we assume a constant velocity movement for lateral
                new_d, new_dv = self.forward_straight(start_d, dv, 0.0)
            else:
                new_d = self.lateral_lane_change(start_d, target_d, dv, target_v)
                curr_lane = target_lane.clone()

            # if the longitudinal velocity is small, the lateral distance should also be small.
            # this is an approximation of the bicycle model
            diff_d = (new_d[1:] - new_d[:-1]).clone()
            diff_d = torch.cat([diff_d, diff_d[-1:]], dim=0)
            # we roughly use the longitudinal velocity as the upper bound of lateral velocity
            # meaning that the lateral velocity should not be larger than the longitudinal velocity
            max_d = new_sv * self.dt
            # clip the diff_d to be smaller than max_d
            diff_d = torch.where(diff_d > max_d, max_d, diff_d)
            diff_d = torch.where(diff_d < -max_d, -max_d, diff_d)
            new_d = torch.cumsum(diff_d, dim=0) + new_d[0]

        # combine s and d
        new_traj_sd = torch.stack([new_s, new_d], dim=1)

        # convert the frenet coordinate to the cartesian coordinate
        new_traj_xy, _ = frenet_to_cartesian_batch(new_traj_sd, curr_lane)

        # add heading and velocity
        new_traj_xyhv = get_heading_velocity(new_traj_xy)
        return new_traj_xyhv, curr_lane

    def longitude_behavior(self, sd, velo, acc):
        new_s = sd.clone()[0]
        traj_s, traj_sv = self.forward_straight(new_s, velo, acc)
        return traj_s, traj_sv

    def lateral_lane_change(self, start_d, target_d, start_v, target_v):
        """Lateral planning using cubic hermite spline interpolation.
        There is no dynamics as constraints for the lateral velocity, so we need to post-process the trajectory.

        Args:
            start_d (tensor): shape [2], the initial position of the vehicle
            target_d (float): target lateral position of the vehicle
            start_v (float): initial lateral velocity of the vehicle
            target_v (float): target lateral velocity of the vehicle

        Returns:
            tensor: the lateral position trajectory of the vehicle
        """
        # to use CubicHermiteSpline, we need to convert the tensor to numpy
        device = start_d.device
        start_d = start_d.clone().cpu().numpy()

        # after lane change, we assume d should be 0
        traj_d = np.array([start_d, target_d])
        traj_dv = np.array([start_v, target_v])
        spline_d = CubicHermiteSpline(np.arange(len(traj_d)), traj_d, traj_dv)
        t_fine = np.linspace(0, len(traj_d) - 1, self.horizon + 1)
        traj_d = spline_d(t_fine)[1:]

        traj_d = torch.as_tensor(traj_d, dtype=torch.float32, device=device)
        return traj_d

    def lateral_lane_change_dynamics(self, start_d, target_d, theta, traj_sv, L=2.7):
        """Lateral planning using bicycle model with a PD controller.
        We assume the lateral velocity is 0 after the planning horizon.

        Args:
            start_d (tensor): shape [2], the initial position of the vehicle
            target_d (float): target vehicle lateral position
            theta (tensor): Initial yaw angle relative to the lane heading
            traj_sv (tensor): a sequence of longitudinal velocities over time
            L (float): Wheelbase of the vehicle (meters)

        Returns:
            tensor: the lateral position trajectory of the vehicle
        """

        # Initialize yaw angle (heading) and lateral position as tensors
        theta = torch.full((self.horizon + 1,), theta, device=start_d.device)
        traj_d = torch.full((self.horizon + 1,), start_d, device=start_d.device)
        theta = theta.float()
        traj_d = traj_d.float()

        # PD controller parameters
        K_p = 0.6  # Proportional gain
        K_d = 0.6  # Derivative gain for damping
        yaw_rate_max = np.pi / 6  # Maximum steering angle (30 degrees)

        # Initialize previous error (for derivative term)
        prev_d_error = target_d - traj_d[0]

        # Loop over each time step to update traj_d iteratively
        # We can't use vectorized operations here because the error depends on the previous action
        for t in range(1, self.horizon + 1):
            # Calculate dynamic lateral error at each time step
            d_error = target_d - traj_d[t - 1]

            # Compute derivative of error (rate of change of error)
            d_error_rate = (d_error - prev_d_error) / self.dt

            # Update previous error for next iteration
            prev_d_error = d_error

            # Compute control signal based on proportional and derivative gains
            delta_d = K_p * d_error + K_d * d_error_rate

            # Yaw rate based on bicycle model dynamics (vectorized)
            yaw_rate = (traj_sv[t - 1] / L) * torch.tan(delta_d)
            yaw_rate = torch.clamp(delta_d, -yaw_rate_max, yaw_rate_max)

            # Update yaw angle over time steps (cumulative sum of yaw_rate over time)
            theta[t] = theta[t - 1] + yaw_rate * self.dt

            # Update lateral velocity based on yaw angle and vehicle speed (vectorized)
            v_lateral_traj = traj_sv[t - 1] * torch.sin(theta[t])

            # Compute next lateral position based on lateral velocity trajectory
            traj_d[t] = traj_d[t - 1] + v_lateral_traj * self.dt

        return traj_d[1:]
