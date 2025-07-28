import torch


def cartesian_to_frenet(point, reference_line):
    """Convert Cartesian coordinates (x, y) to Frenet coordinates (s, d).

    Args:
        point (tensor): Tensor of shape [2] representing the Cartesian point [x, y].
        reference_line (tensor): Tensor of shape [T, 3] representing the reference line.

    Returns:
        tensor: Tensor of shape [2] representing the Frenet coordinates [s, d].
    """
    assert isinstance(point, torch.Tensor), "point must be a tensor."
    assert isinstance(reference_line, torch.Tensor), "reference_line must be a tensor."

    # only use the x and y coordinates
    reference_line = reference_line[:, 0:2].to(point.device)

    if reference_line.size(0) < 3:
        raise ValueError("Reference line must contain at least 3 points.")

    # Find the closest point on the reference line
    diffs = reference_line - point.unsqueeze(0)  # [T, 2]
    distances = torch.norm(diffs, dim=1)  # Shape [T]
    min_idx = torch.argmin(distances)

    # Calculate s as the cumulative distance along the reference line up to the closest point
    # Note that we don't check if the point is in the domain of the reference line here
    seg_len = torch.norm(reference_line[1:] - reference_line[:-1], dim=1)
    first_seg_len = torch.tensor([0.0], device=point.device)
    cumulative_lengths = torch.cat([first_seg_len, seg_len]).cumsum(dim=0)
    s = cumulative_lengths[min_idx]

    # Calculate d as the perpendicular distance to the closest segment
    closest_point = reference_line[min_idx]
    if min_idx > 0:
        prev_point = reference_line[min_idx - 1]
        seg_vector = closest_point - prev_point
    else:
        # min_idx == 0 so we use the next point to calculate the segment vector
        next_point = reference_line[min_idx + 1]
        seg_vector = next_point - closest_point
    seg_length = torch.norm(seg_vector)

    projection = torch.dot(diffs[min_idx], seg_vector) / seg_length
    d_vector = diffs[min_idx] - projection * (seg_vector / seg_length)
    d = torch.norm(d_vector)

    # Use pseudo-cross product for 2D to determine sign of d
    cross_product_value = (
        seg_vector[0] * diffs[min_idx][1] - seg_vector[1] * diffs[min_idx][0]
    )
    if cross_product_value > 0:
        d = -d

    return torch.tensor([s, d], device=point.device)


def frenet_to_cartesian(frenet_point, reference_line):
    """Convert Frenet coordinates (s, d) to Cartesian coordinates (x, y).

    Args:
        frenet_point (tensor): Tensor of shape [2] representing the Frenet coordinates [s, d].
        reference_line (tensor): Tensor of shape [T, 2] representing the reference line.

    Returns:
        tensor: Tensor of shape [2] representing the Cartesian coordinates [x, y].
        tensor: Tensor of shape [2] representing the heading vector at the closest point.
    """
    assert isinstance(frenet_point, torch.Tensor), "point must be a tensor."
    assert isinstance(reference_line, torch.Tensor), "reference_line must be a tensor."
    s, d = frenet_point

    # Calculate cumulative distances along the reference line
    seg_len = torch.norm(reference_line[1:] - reference_line[:-1], dim=1)
    cumulative_s = torch.cumsum(
        torch.cat([torch.tensor([0.0], device=frenet_point.device), seg_len]),
        dim=0,
    )

    # Find the segment where s falls into
    segment_idx = torch.searchsorted(cumulative_s, s).item() - 1

    if segment_idx >= len(reference_line) - 1:
        # If s is beyond the last point in cumulative_s
        print(f"{s} is out of longitudinal domain")
        segment_idx = len(reference_line) - 2
        s = cumulative_s[-1]

    # Calculate d as the perpendicular distance to the closest segment
    if segment_idx > 0:
        start_point = reference_line[segment_idx - 1]
        end_point = reference_line[segment_idx]
    else:
        start_point = reference_line[segment_idx]
        end_point = reference_line[segment_idx + 1]
    seg_vector = end_point - start_point
    seg_length = torch.norm(seg_vector)
    seg_tangent = seg_vector / seg_length

    # Interpolate along this segment
    t = (s - cumulative_s[segment_idx]) / seg_length
    interpolated_point = start_point + t * seg_vector

    # Calculate normal vector for perpendicular offset
    normal_vector = torch.tensor(
        [-seg_vector[1], seg_vector[0]], device=frenet_point.device
    )
    normal_vector /= torch.norm(normal_vector)
    cartesian_point = interpolated_point + d * normal_vector
    return cartesian_point, seg_tangent


def is_frenet_point_in_domain(
    frenet_point, reference_line, max_lateral_distance=float("inf")
):
    """Check if a Frenet point [s, d] is within the domain of a reference line.

    Args:
        frenet_point (tensor): Tensor of shape [2] representing the Frenet coordinates [s, d].
        reference_line (tensor): Tensor of shape [T, 2] representing the reference line.
        max_lateral_distance (tensor, optional): Maximum allowable lateral distance (d). Default is infinite.

    Returns:
        tensor: Boolean indicating whether the Frenet point is within the domain.
    """
    assert isinstance(frenet_point, torch.Tensor), "point must be a tensor."
    assert isinstance(reference_line, torch.Tensor), "reference_line must be a tensor."

    if reference_line.size(0) < 2:
        raise ValueError("Reference line must contain at least two points.")

    # Calculate cumulative lengths along the reference line
    segment_lengths = torch.norm(reference_line[1:] - reference_line[:-1], dim=1)
    cumulative_lengths = torch.cat(
        [torch.tensor([0.0], device=frenet_point.device), segment_lengths]
    ).cumsum(dim=0)

    # Get total length of the reference line
    total_length = cumulative_lengths[-1]
    s, d = frenet_point
    if s < 0 or s > total_length:
        return False

    # Check if d is within the maximum allowable lateral distance
    if abs(d) > max_lateral_distance:
        return False

    return True


def cartesian_to_frenet_batch(points, reference_line):
    """Convert a batch of Cartesian coordinates (x, y) to Frenet coordinates (s, d).

    Args:
        points (tensor): Tensor of shape [B, 2] representing the Cartesian points [x, y].
        reference_line (tensor): Tensor of shape [T, 3] representing the reference line.

    Returns:
        tensor: Tensor of shape [B, 2] representing the Frenet coordinates [s, d].
    """
    assert isinstance(points, torch.Tensor), "points must be a tensor."
    assert isinstance(reference_line, torch.Tensor), "reference_line must be a tensor."

    # only use the x and y coordinates
    reference_line = reference_line[:, 0:2].to(points.device)

    if reference_line.size(0) < 3:
        raise ValueError("Reference line must contain at least 3 points.")

    B = points.size(0)

    # Expand points to match reference line for broadcasting
    expanded_points = points.unsqueeze(1)  # Shape [B, 1, 2]

    # Calculate differences and distances
    diffs = reference_line.unsqueeze(0) - expanded_points  # Shape [B, T, 2]
    distances = torch.norm(diffs, dim=2)  # Shape [B, T]

    # Find the closest point indices on the reference line for each point
    min_indices = torch.argmin(distances, dim=1)  # Shape [B]

    # Calculate cumulative segment lengths along the reference line
    seg_len = torch.norm(reference_line[1:] - reference_line[:-1], dim=1)
    first_seg_len = torch.tensor([0.0], device=points.device)
    cumulative_lengths = torch.cat([first_seg_len, seg_len]).cumsum(dim=0)

    # Gather s values for each point in the batch
    s_values = cumulative_lengths[min_indices]  # Shape [B]

    # Calculate d values for each point in the batch
    closest_points = reference_line[min_indices]  # Shape [B, 2]

    seg_vectors = torch.where(
        min_indices.unsqueeze(1) > 0,
        closest_points - reference_line[min_indices - 1],
        reference_line[min_indices + 1] - closest_points,
    )  # Shape [B, 2]

    seg_lengths = torch.norm(seg_vectors, dim=1)  # Shape [B]

    projections = (
        torch.einsum("bij,bj->bi", diffs[range(B), min_indices], seg_vectors)
        / seg_lengths
    )
    d_vectors = diffs[range(B), min_indices] - projections.unsqueeze(1) * (
        seg_vectors / seg_lengths.unsqueeze(1)
    )

    d_values = torch.norm(d_vectors, dim=1)  # Shape [B]

    # Determine sign of d using pseudo-cross product for each point
    cross_product_values = (
        seg_vectors[:, 0] * diffs[range(B), min_indices][:, 1]
        - seg_vectors[:, 1] * diffs[range(B), min_indices][:, 0]
    )

    d_values[cross_product_values > 0] *= -1

    return torch.stack([s_values, d_values], dim=1)


def frenet_to_cartesian_batch(frenet_points, reference_line):
    """Convert a batch of Frenet coordinates (s, d) to Cartesian coordinates (x, y).

    Args:
        frenet_points (tensor): Tensor of shape [B, 2] representing the Frenet coordinates [s, d].
        reference_line (tensor): Tensor of shape [T, 2] representing the reference line.

    Returns:
        tensor: Tensor of shape [B, 2] representing the Cartesian coordinates [x, y].
        tensor: Tensor of shape [B, 2] representing the heading vectors at the closest points.
    """
    assert isinstance(frenet_points, torch.Tensor), "frenet_points must be a tensor."
    assert isinstance(reference_line, torch.Tensor), "reference_line must be a tensor."

    s_values = frenet_points[:, 0]
    d_values = frenet_points[:, 1]

    # Calculate cumulative distances along the reference line
    seg_len = torch.norm(reference_line[1:] - reference_line[:-1], dim=1)
    cumulative_s = torch.cumsum(
        torch.cat([torch.tensor([0.0], device=frenet_points.device), seg_len]),
        dim=0,
    )

    # Find segments where each s falls into
    s_values = s_values.contiguous()
    segment_indices = torch.searchsorted(cumulative_s, s_values) - 1
    segment_indices = torch.clamp(segment_indices, 0, len(reference_line) - 2)

    # Get start and end points for each segment
    start_points = reference_line[segment_indices]
    end_points = reference_line[segment_indices + 1]

    # Calculate segment vectors and lengths
    seg_vectors = end_points - start_points
    seg_lengths = torch.norm(seg_vectors, dim=1)

    # Normalize segment tangent vectors
    seg_tangents = seg_vectors / seg_lengths.unsqueeze(1)

    # Interpolate along each segment
    t_values = (s_values - cumulative_s[segment_indices]) / seg_lengths
    interpolated_points = start_points + t_values.unsqueeze(1) * seg_vectors

    # Calculate normal vectors for perpendicular offsets
    normal_vectors = torch.stack([-seg_vectors[:, 1], seg_vectors[:, 0]], dim=1)
    normal_vectors /= torch.norm(normal_vectors, dim=1).unsqueeze(1)

    # Calculate Cartesian points
    cartesian_points = interpolated_points + d_values.unsqueeze(1) * normal_vectors

    return cartesian_points, seg_tangents


def are_frenet_points_in_domain(
    frenet_points, reference_line, max_lateral_distance=float("inf")
):
    """Check if a batch of Frenet points [s, d] are within the domain of a reference line.

    Args:
        frenet_points (tensor): Tensor of shape [B, 2] representing the Frenet coordinates [s, d].
        reference_line (tensor): Tensor of shape [T, 2] representing the reference line.
        max_lateral_distance (tensor): Maximum allowable lateral distance (d). Default is infinite.

    Returns:
        tensor: Tensor of shape [B] with boolean values indicating whether each Frenet point is within the domain.
    """
    assert isinstance(frenet_points, torch.Tensor), "frenet_points must be a tensor."
    assert isinstance(reference_line, torch.Tensor), "reference_line must be a tensor."

    if reference_line.size(0) < 2:
        raise ValueError("Reference line must contain at least two points.")

    # Calculate cumulative lengths along the reference line
    segment_lengths = torch.norm(reference_line[1:] - reference_line[:-1], dim=1)
    cumulative_lengths = torch.cat(
        [torch.tensor([0.0], device=frenet_points.device), segment_lengths]
    ).cumsum(dim=0)

    # Get total length of the reference line
    total_length = cumulative_lengths[-1]

    # Extract s and d values
    s_values = frenet_points[:, 0]
    d_values = frenet_points[:, 1]

    # Check if s values are within the range
    s_in_domain = (s_values >= 0) & (s_values <= total_length)

    # Check if d values are within the maximum allowable lateral distance
    d_in_domain = torch.abs(d_values) <= max_lateral_distance

    # Combine both conditions
    in_domain = s_in_domain & d_in_domain
    return in_domain
