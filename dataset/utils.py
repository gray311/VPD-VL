import os
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_project_root():
    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get the project root
    project_root = os.path.join(script_dir, '..', '..')
    return os.path.normpath(project_root)  # Normalize t

def transform_quaternion(quaternion):
    # Transform the quaternion from the Nuscenes format to the ROS format
    return [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]

def quaternion_to_euler(rotation):
    # Convert quaternion to euler angles
    euler = rotation.as_euler('xyz', degrees=True)
    return euler

def relative_heading_angle(query_ann, reference_pose):
    # Convert quaternions to euler angles
    reference_rotation = R.from_quat(transform_quaternion(reference_pose))
    query_rotation = R.from_quat(transform_quaternion(query_ann))

    updated_query_rotation = reference_rotation.inv() * query_rotation
    euler = quaternion_to_euler(updated_query_rotation)
    
    # Compute the relative heading angle
    relative_angle = euler[2]
    
    # Normalize the relative angle to the range [-180, 180]
    relative_angle = (relative_angle + 180) % 360 - 180
    
    return relative_angle

def get_rotation_matrix(rotation):
    # Convert quaternions to euler angles
    rotation = R.from_quat(transform_quaternion(rotation))
    euler_angles = rotation.as_euler('xyz', degrees=True)
    yaw_angle = euler_angles[2]

    rotation_matrix = R.from_euler('z', -yaw_angle, degrees=True).as_matrix()
    return rotation_matrix


