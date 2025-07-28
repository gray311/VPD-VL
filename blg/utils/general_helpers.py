import yaml
import os
import shutil
from collections import namedtuple

import torch


def dict_to_namedtuple(name, d):
    # Recursively convert dictionary values to namedtuples
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namedtuple(key, value)

    return namedtuple(name, d.keys())(*d.values())


def namedtuple_to_dict(obj):
    if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
        return {k: namedtuple_to_dict(v) for k, v in obj._asdict().items()}
    elif isinstance(obj, list):
        return [namedtuple_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: namedtuple_to_dict(v) for k, v in obj.items()}
    return obj


def load_yaml_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return dict_to_namedtuple("config", config)


def save_yaml_config(config, config_path):
    config_dict = namedtuple_to_dict(config)
    with open(os.path.join(config_path, "config.yaml"), "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False)


def check_folder(path, overwrite=True):
    # If the folder exists, remove it
    if os.path.exists(path) and overwrite:
        shutil.rmtree(path)
    # create the folder
    os.makedirs(path, exist_ok=True)


def CPU(x):
    if x is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x
