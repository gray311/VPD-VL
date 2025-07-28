import os
import sys
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import supervision as sv
from collections import defaultdict
import cv2
import copy
import json
import pickle
import descartes
import random
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from transformers import AutoProcessor, AutoModelForCausalLM

from som_utils.track_utils import sample_points_from_masks
from som_utils.video_utils import create_video_from_images
from som_utils.common_utils import CommonUtils
from som_utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from som_utils.visualizer import Visualizer, bounding_box_to_mask


def generate_random_color_in_range(r_range=(50, 200), g_range=(50, 200), b_range=(50, 200)):
    import colorsys
    h = random.uniform(0, 1)       
    s = random.uniform(0.7, 1)     
    v = random.uniform(0.7, 1)     
    rgb = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(c * 255) for c in rgb)


def build_model(sam2_checkpoint, model_cfg, grounding_model_name="dino"):
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    
    if grounding_model_name == "dino":
        # init grounding dino model from huggingface
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    elif grounding_model_name == "florence2":
        FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
        grounding_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True).eval().to(device)
        processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

    return image_predictor, video_predictor, grounding_model, processor, device


def generate_realtime_data(filename):
    video_path = f"./workspace/data/drivebench/images/{filename}/masked"
    if not os.path.exists(video_path):
        os.mkdir(video_path)
    CommonUtils.creat_dirs(video_path)
    mask_data_dir = os.path.join(video_path, "mask_data")
    json_data_dir = os.path.join(video_path, "json_data")
    result_dir = os.path.join(video_path, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)
    return video_path, mask_data_dir, json_data_dir, result_dir


if __name__ == "__main__":
    sam2_checkpoint = "./workspace/checkpoint/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    grounding_model_name = "dino"
    PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
    image_predictor, video_predictor, grounding_model, processor, device = build_model(sam2_checkpoint, model_cfg, grounding_model_name=grounding_model_name)
    with open("./workspace/data/drivebench/test.json", "r") as f:
        data = json.load(f)
    
    print(len(data))
    from tqdm import tqdm
    for line in tqdm(data):
        # if "1b45a97a0e5e49fe9cd345dd4bd729c3" not in line['id']: continue
        image_list = line['image']
        object_list = line['objects']
        for i in range(len(image_list)):
            if i not in object_list.keys():
                object_list[i] = []
                
        if isinstance(object_list, dict):
            object_list = [v for k, v in object_list.items()]
        filename = image_list[-1].split("/")[-1].split(".")[0]
        media_path = f"./workspace/data/drivebench/images/{filename}/"
        if not os.path.exists(media_path):
            os.mkdir(media_path)
            
        for image_path in image_list:
            image = Image.open(image_path)
            image_name = image_path.split("/")[-1]
            image.save(os.path.join(media_path, image_name))
        
        video_path, mask_data_dir, json_data_dir, result_dir  = generate_realtime_data(filename)
        
        mark = False
        for frame_idx, img_path in enumerate(image_list):
            image_name = img_path.split("/")[-1]
            output_image_path = os.path.join(video_path, image_name)
            if os.path.exists(output_image_path):
                mark = True
        
        # if mark: continue

        instance_id2color = {}
        instance_id2type = {}
        color_list = [
            (255, 64, 64),
            (255, 159, 64),
            (255, 255, 64),
            (64, 255, 64),
            (64, 255, 255),
            (64, 64, 255),
            (159, 64, 255),
            (255, 77, 128)
        ]
        random.shuffle(color_list)
        for i, objects in enumerate(object_list):
            for j, item in enumerate(objects):
                if "global_id" in item.keys():
                    item['id'] = item['global_id']
                    object_list[i][j]['id'] = object_list[i][j]['global_id']
                if item['id'] not in instance_id2color.keys():
                    instance_id2color[item['id']] = color_list[item['id']%len(color_list)]
                    instance_id2type[item['id']] = "Car"

        init_segment = []
        for frame_idx, img_path in enumerate(image_list):
            mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{str(frame_idx).zfill(4)}.npy")

            image = Image.open(img_path)
            objects = object_list[frame_idx]
            width, height = image.size
            boxes, labels, locations = defaultdict(list), defaultdict(list), defaultdict(list)
            segments = defaultdict(dict)
            sample_token = None

            input_boxes = [item['2d_bbox'] for item in objects if "2d_bbox" in item.keys()]
            input_labels = [item['id'] for item in objects if "id" in item.keys()]
            image_predictor.set_image(np.array(image.convert("RGB")))

            if len(input_boxes) == 0:
                # print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
                continue

            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)

            if mask_dict.promote_type == "mask" and masks is not None:
                mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=input_labels)

            init_segment.append(mask_dict)

        for frame_idx in range(len(init_segment)):
            json_data, mask_image = {}, {}
    
            frame_masks_info = init_segment[frame_idx]
            mask = frame_masks_info.labels
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            for obj_id, obj_info in mask.items():
                mask_img[obj_info.mask == True] = obj_id

            mask_image = mask_img.numpy().astype(np.uint16)
            json_data = frame_masks_info.to_dict()


            mask_name = frame_masks_info.mask_name.replace(".npy", ".pkl")
            json_name = frame_masks_info.mask_name.replace(".npy", ".json")

            with open(os.path.join(mask_data_dir, mask_name), "wb") as f:
                pickle.dump(mask_image, f)

            with open(os.path.join(json_data_dir, json_name), "w") as f:
                f.write(json.dumps(json_data))

            # print(line['conversat'])

  
        CommonUtils.draw_masks_and_box_with_supervision(image_list, mask_data_dir, json_data_dir, video_path, instance_id2color, instance_id2type)
    
        # print(line['conversations'])
