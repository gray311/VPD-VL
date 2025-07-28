import os
import sys
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import supervision as sv
from tqdm import tqdm
from collections import defaultdict
from typing import List
import argparse
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

random.seed(233)

def parse_args():
    parser = argparse.ArgumentParser(description="Testing Script")
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--seg_ckpt', type=str, default=None)
    parser.add_argument('--det_ckpt', type=str, default=None)
    parser.add_argument('--prompt_type', type=str, default="mask")
    args = parser.parse_args()
    return args

args = parse_args()


def calculate_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

def find_nearest_box(boxes, agent_box):
    min_distance = float('inf')
    nearest_box = None
    
    boxes = [box for box in boxes if area(box) > 10000]

    for box in boxes:
        distance = calculate_distance(box, agent_box)
        if distance < min_distance:
            min_distance = distance
            nearest_box = box
    
    return nearest_box


def area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH

    if inter == 0:
        return 0.0
    union = area(boxA) + area(boxB) - inter
    return inter / min(area(boxA), area(boxB))

def filter_overlap_keep_smaller(boxes: List[List[float]], labels: List[str], iou_thresh=0.5):
    assert len(boxes) == len(labels)
    keep = [True] * len(boxes)

    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(boxes)):
            if not keep[j]:
                continue
            if iou(boxes[i], boxes[j]) > iou_thresh:
                if area(boxes[i]) <= area(boxes[j]):
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    filtered_boxes = [b for b, k in zip(boxes, keep) if k]
    filtered_labels = [l for l, k in zip(labels, keep) if k]
    return filtered_boxes, filtered_labels

def sam2_video_predictor(start_frame_idx, step, frame_names, mask_dict, sam2_masks, inference_state, video_predictor):
    for object_id, object_info in mask_dict.labels.items():
        frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state,
                start_frame_idx,
                object_id,
                object_info.mask,
            )
    
    video_segments = {}  # output the following {step} frames tracking masks
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx, reverse=True):
        frame_masks = MaskDictionaryModel()
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
            object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
            object_info.update_box()
            frame_masks.labels[out_obj_id] = object_info
            image_base_name = frame_names[out_frame_idx].split("/")[-1].split(".")[0]
            frame_masks.mask_name = f"mask_{image_base_name}.npy"
            frame_masks.mask_height = out_mask.shape[-2]
            frame_masks.mask_width = out_mask.shape[-1]

        video_segments[out_frame_idx] = frame_masks
        sam2_masks = copy.deepcopy(frame_masks)
    
    return video_segments, sam2_masks


def dino_detect_object(image, text, grounding_model, processor):
    inputs = processor(images=image, text=text, return_tensors="pt").to(grounding_model.device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )

    return results[0]["boxes"].cpu().numpy().tolist(), results[0]["labels"], results[0]['scores']


def generate_random_color_in_range(r_range=(50, 200), g_range=(50, 200), b_range=(50, 200)):
    import colorsys
    h = random.uniform(0, 1)       
    s = random.uniform(0.7, 1)     
    v = random.uniform(0.7, 1)     
    rgb = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(c * 255) for c in rgb)


def build_model(sam2_checkpoint, grounding_model_name="dino"):
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    model_cfg = "sam2_hiera_l.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    
    if grounding_model_name == "dino":
        # init grounding dino model from huggingface
        model_id = "IDEA-Research/grounding-dino-base"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    elif grounding_model_name == "florence2":
        FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
        grounding_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True).eval().to(device)
        processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

    return image_predictor, video_predictor, grounding_model, processor, device


def generate_realtime_data(root, filename, end_frame_idx, image_list):
    video_path = f"{root}/{filename}/{str(end_frame_idx).zfill(4)}"
    if not os.path.exists(video_path):
        os.mkdir(video_path)
        for img_path in image_list:
            image = Image.open(img_path).convert("RGB")
            image_name = img_path.split("/")[-1]
            image.save(os.path.join(video_path, image_name.replace("png", "jpg")))
            
    CommonUtils.creat_dirs(video_path)
    mask_data_dir = os.path.join(video_path, "mask_data")
    json_data_dir = os.path.join(video_path, "json_data")
    result_dir = os.path.join(video_path, "masked")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)
    CommonUtils.creat_dirs(result_dir)
    return video_path, mask_data_dir, json_data_dir, result_dir


"""

python e2e_vpd.py \
    --data_file ./workspace/data/nexar/test.json \
    --root_dir ./workspace/data/nexar/images \
    --seg_ckpt ./workspace/checkpoint/sam2_hiera_large.pt \
    --det_ckpt dino
    
python e2e_vpd.py \
    --data_file ./workspace/data/nexar/train.json \
    --root_dir ./workspace/data/nexar/images \
    --seg_ckpt ./workspace/checkpoint/sam2_hiera_large.pt \
    --det_ckpt dino
    
python e2e_vpd.py \
    --data_file ./workspace/data/vpd-sft/nuscenes/demo.json \
    --root_dir ./workspace/data/vpd-sft/nuscenes/images \
    --seg_ckpt ./workspace/checkpoint/sam2_hiera_large.pt \
    --det_ckpt dino
    
python e2e_vpd.py \
    --data_file ./workspace/data/vpd-sft/nuplan_train.json \
    --root_dir ./workspace/data/vpd-sft/nuplan \
    --seg_ckpt ./workspace/checkpoint/sam2_hiera_large.pt \
    --det_ckpt dino
"""

if __name__ == "__main__":
    PROMPT_TYPE_FOR_VIDEO = args.prompt_type
    image_predictor, video_predictor, grounding_model, processor, device = build_model(args.seg_ckpt, grounding_model_name=args.det_ckpt)

    with open(args.data_file, "r") as f:
        data = json.load(f)
        
    mark = {}
    error_list = []
    for i, line in tqdm(enumerate(data)):
        # if i <= 248: continue
        # if "2021.06.23.17.31.36_veh-16_00016_00377=eeaaa9a41b60586d_s22" not in line['image'][-1]: continue
        image_dir = "/".join(line['image'][-1].split("/")[:-1])
        if image_dir in mark.keys():
            continue
        
        mark[image_dir] = True
        image_list = line['image']
        try:
            object_list = line['objects']
        except:
            object_list = None
            
        filename = image_list[-1].split("/")[-2].strip(" ")
        print(image_list[-1])
        end_frame_idx = int(image_list[-1].split("/")[-1].rstrip(".jpg"))
        
        print(filename, end_frame_idx)
        video_path, mask_data_dir, json_data_dir, result_dir  = generate_realtime_data(args.root_dir, filename, end_frame_idx, image_list)
     
        init_segment = []
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
        
        
        use_ground_truth = True
        
     
        img_path = image_list[-1]
        image = Image.open(img_path).convert("RGB")
        start_frame_idx = int(img_path.split("/")[-1].rstrip(".jpg"))
        
        if object_list is None:
            boxes, labels = [], []
            object_name =  "a car. a bus. a train. a truck. a person.  a motorcycle. a cyclist. a bicycle  a traffic light. a stop sign.  a fire hydrant. a traffic cone. a bird."
            for obj in object_name.split("  "):
                input_boxes, input_labels, input_scores = dino_detect_object(image, obj, grounding_model, processor)
                boxes.extend(input_boxes)
                labels.extend(input_labels)
                
            remove_ids = [idx for idx, label in enumerate(labels) if label + "." not in object_name]
            boxes = [item for idx, item in enumerate(boxes) if idx not in remove_ids]
            labels = [item for idx, item in enumerate(labels) if idx not in remove_ids]
            boxes, labels = filter_overlap_keep_smaller(boxes, labels)
        else:
            
            boxes, labels = [], []
            object_name =  "a car."
            for obj in object_name.split("  "):
                input_boxes, input_labels, input_scores = dino_detect_object(image, obj, grounding_model, processor)
                boxes.extend(input_boxes)
                labels.extend(input_labels)
                
            remove_ids = [idx for idx, label in enumerate(labels) if label + "." not in object_name]
            boxes = [item for idx, item in enumerate(boxes) if idx not in remove_ids]
            labels = [item for idx, item in enumerate(labels) if idx not in remove_ids]
            boxes, labels = filter_overlap_keep_smaller(boxes, labels)
            
            agent_box = object_list["boxes"][0]
            agent_box = find_nearest_box(boxes, agent_box)
            agent_labels = ['car']
            
            boxes, labels = object_list["boxes"], object_list['labels']
            boxes[0], labels[0] = agent_box, agent_labels
            
        if agent_box is None:
            error_list.append(line['id'])
            continue
        
        sam2_masks = MaskDictionaryModel()
        objects_count, step = 0, 8
        inference_state = video_predictor.init_state(video_path=video_path, offload_video_to_cpu=True, async_loading_frames=True)
        mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{str(start_frame_idx).zfill(4)}.npy")
        width, height = image.size
        segments = {}
    
        print("shape: ", width, height)
        input_boxes = boxes
        input_labels = labels
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


        objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.80, objects_count=objects_count)
        print("objects_count", objects_count)
        video_predictor.reset_state(inference_state)
        if len(mask_dict.labels) == 0:
            print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
            continue
        print(len(image_list) - 1, step, image_list)
        video_segments, sam2_masks = sam2_video_predictor(len(image_list) - 1, step, image_list, mask_dict, sam2_masks, inference_state, video_predictor)
        segments = video_segments
        if len(list(video_segments.keys())) > 0:
            init_segment = video_segments

        use_ground_truth = False
    
       
        for frame_idx in init_segment.keys():
            json_data, mask_image = {}, {}
            frame_masks_info = init_segment[frame_idx]
            mask = frame_masks_info.labels
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            for obj_id, obj_info in mask.items():
                mask_img[obj_info.mask == True] = obj_id

            mask_image = mask_img.numpy().astype(np.uint16)
            json_data = frame_masks_info.to_dict()

            if use_ground_truth == False:
                for obj_id, item in json_data['labels'].items():
                    if item['instance_id'] not in instance_id2color.keys():
                        instance_id2color[item['instance_id']] = color_list[item['instance_id']%len(color_list)]
                        instance_id2type[item['instance_id']] = item['class_name']
                
            mask_name = frame_masks_info.mask_name.replace(".npy", ".pkl")
            json_name = frame_masks_info.mask_name.replace(".npy", ".json")

            with open(os.path.join(mask_data_dir, mask_name), "wb") as f:
                pickle.dump(mask_image, f)

            with open(os.path.join(json_data_dir, json_name), "w") as f:
                f.write(json.dumps(json_data))

        raw_image_list = os.listdir(video_path)
        raw_image_list = [os.path.join(video_path, name) for name in raw_image_list]
        CommonUtils.draw_masks_and_box_with_supervision(raw_image_list, mask_data_dir, json_data_dir, result_dir, instance_id2color, instance_id2type)

        # break
        
    print(error_list)