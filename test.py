import os
import json
from collections import defaultdict
import re

def zero(): return 0



# root = "./workspace/data/vpd-sft/driving_safety_bench/results"
# for filename in os.listdir(root):
#     with open(os.path.join(root, filename), "r") as f:
#         data = json.load(f)
        
#     # print(filename)
#     # print(len(data))
    
#     if "0,3" not in filename: continue
     
#     cnt = 0
#     scores = defaultdict(list)
#     for line in data:
#         pred = line['pred'].lower()
#         pred = pred.split("</think>")[-1]
#         try:
#             if "unsafe" in pred:
#                 scores[line['risk_type']].append(1)
#             else:
#                 scores[line['risk_type']].append(0)
#         except:
#             continue
#         # if "without" in pred: continue
#         # try:
#         #     pred = pred.split("\n")[0]
#         #     match = re.search(r"(\d+\.\d+|\d+)", pred)
#         #     score =  float(match.group())
#         #     if score >= 1:
#         #         score = 0
#         #     else:
#         #         score = 1 - score
#         #     scores[line['risk_type']].append(score)
#         #     # print(pred)
#         #     # print(line['risk_score'] / 13, score, (1 - abs(score - line['risk_score'] / 13.0)))
#         # except:
#         #     if "unsafe" in pred.lower():
#         #         scores[line['risk_type']].append(1)
#         #     else:
#         #         scores[line['risk_type']].append(0)
#         #     continue
        

    
#     for k, v in scores.items():
#         print(k, sum(v) + 3, len(v))
#     scores = {k: sum(v) / len(v) for k, v in scores.items()}
#     print(filename)
#     print(scores)
    
            
# import re
# from copy import deepcopy


# with open("./workspace/data/vpd-sft/driving_safety_bench/reasoning.json", "r") as f:
#     data = json.load(f)
    
# print(len(data))

# outputs = []
# task_type = ['prediction', 'planning', 'evaluation']
# reasoning_step = ['Scene Description', 'Critical Objects', 'Meta Driving Decision', 'Behavioral Conflict & Risk Analysis']


# for i, line in enumerate(data):
#     pattern = r"##(?:\s*\d+\.\s*)?(.+?):\n(.*?)(?=\n##(?:\s*\d+\.\s*)?.+?:|\Z)"

#     matches = re.findall(pattern, line['reasoning_process'], flags=re.DOTALL)
#     policy_pattern = r"Specific traffic policy:\s*(.*)"

#     planning_prompt, planning_outputs = "", ""
#     prediction_prompt, prediction_outputs = "", ""
#     evaluation_prompt, evaluation_outputs = "", ""
#     policy_text = None
    
#     for title, content in matches:
#         obj_name = "Object"
#         if title.strip(" ") not in reasoning_step:
#             break

#         if title.strip(" ") == "Scene Description":
#             planning_outputs += "1. Scene Description\n" + content.strip() + "\n"
#             prediction_outputs += "1. Scene Description\n" + content.strip() + "\n"
#             evaluation_outputs += "1. Scene Description\n" + content.strip() + "\n"
        
#         elif title.strip(" ") == "Critical Objects":
#             content = content.strip()
#             content = content.replace(" [", "[")
#             content = content.replace('- ', "")
#             content = content.replace(" :", "")
#             content = content.replace(":", "")
#             pattern = r'\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]'

#             content = re.sub(pattern, ':', content)
#             content = content.replace(" :\n", ": ")
        
#             planning_outputs += "\n2. Critical Objects\n" + content.strip() + "\n"
#             prediction_outputs += "\n2. Critical Objects\n" + content.strip() + "\n"
#             evaluation_outputs += "\n2. Critical Objects\n" + content.strip() + "\n"
            
#         elif title.strip(" ") == "Meta Driving Decision":
#             content = content.replace("[", "").replace("]", "").lower()
#             ego_match = re.search(r"ego car:\s*(.+)", content.strip())
#             obj_match = re.findall(r"(object.*?):\s*(.+)", content.strip())
#             exp_match = re.search(r"explanation:\s*(.+)", content.strip(), flags=re.DOTALL)
#             ego_action = ego_match.group(1).strip()
#             try:
#                 obj_name, obj_action = obj_match[0][0], obj_match[0][1]
#                 prediction_outputs += "\n3. Meta Driving Decision\n" + f"Object {str(line['agent_id'])} future behavior: {obj_action}\n{explanation}" + "\n"
#             except:
#                 prediction_outputs = ""
            
#             explanation = exp_match.group(1).strip()
#             planning_outputs += "\n3. Meta Driving Decision \n" + f"Ego car future behavior: {ego_action}\n{explanation}" + "\n"
            
            
#             planning_prompt = f"""You are an autonomous driving planner. Based on the provided multi frames of ego car's front camera view, plan the most appropriate future behavior for the ego vehicle.

# Steps:
# 1. Describe the driving environment and related crucial objects.  
# 2. Plan the ego vehicle's next action (e.g., keep lane, slow down, change lane).  
# 3. Explain why this action is safe and appropriate in the current traffic context."""

#             prediction_prompt = f"""You are an autonomous driving assistant. Based on the provided multi frames of ego car's front camera view, predict the most likely future behavior of object {str(line['agent_id'])}. Then, explain what the ego vehicle should do in response.

# Provide your reasoning in three steps:
# 1. Describe the driving environment and related crucial objects.  
# 2. Predict object [{str(line['agent_id'])}]’s most likely behavior (e.g., keep lane, slow down, turn).  
# 3. Provide a short explanation of what the ego vehicle should do in response, considering safety and traffic context."""


#         elif title.strip(" ") == "Behavioral Conflict & Risk Analysis":
#             match = re.search(policy_pattern, content.strip(), flags=re.DOTALL)
#             policy_text = match.group(1).strip()
#             content = content[:content.rfind("Specific traffic policy")].strip("\n").strip()
#             content = content.split("\n")
#             content = "\n".join(content[:-1])
            
#             content_lines = content.split("\n")

            
#             unsafety_score = round(line['risk_score'] / 14 * 0.3 + 0.7, 2)
#             if line['label'] == 'safe':
#                 unsafety_score = round(max(0, unsafety_score - 0.5), 2)
#             content += f"\nRisk level: {line['label']}\nTherefore, the unsafety score of this should be: {unsafety_score}"
#             # content = content.split("\n")
#             # content = [content[-1] + "\n"] + content[:-1]
#             # content = "\n".join(content)
#             evaluation_outputs += "\n3. Risk Analysis\n" + content.strip() + "\n"
#             evaliation_prompt = f"""You are a driving scene reasoning assistant. Based on the provided multi frames of ego car's front camera view and traffic rules, analyze the potential behavioral conflict between ego car and other agents, and then assess risk.

# Traffic Rules:
# {policy_text}

# Analyze the impact of the ego vehicle's future behavior: {line['ego_lat_beh']} and {line['ego_long_beh']}, while the future behavior of object {str(line['agent_id'])} is: {line['agent_lat_beh']} and {line['agent_long_beh']}.  
# Consider potential risks such as collision, reduced reaction time, or merging conflicts.  

# Provide your reasoning in three steps:
# 1. Describe the scene and relative position of object [ID].  
# 2. Analyze the interaction between the ego vehicle and object [ID].  
# 3. Assess the overall risk and assign a safety score (0 = extremely unsafe, 1 = completely safe)."""

#     image = line['image']
#     image_list = []
#     for image_path in image:
#         tmp_path = "/".join(image_path.split("/")[:-1] + ["masked", image_path.split("/")[-1]])
#         image_list.append(tmp_path)
        
    

#     for task in task_type:
#         example = deepcopy(line)
#         example['image'] = image_list[::2]
#         del example['reasoning_process']
#         example['id'] = f"{example['id']}_{str(i)}_{task}"
#         if task == "planning":
#             if len(planning_prompt) < 10: continue
#             if len(planning_outputs) < 10: continue
#             if "he 3D coordinates" in planning_outputs: continue
#             example['conversations'][0]['value'] = planning_prompt
#             example['conversations'][1]['value'] = planning_outputs.strip()
#             example['conversations'][1]['value'] = process_text(example['conversations'][1]['value'])
#             if "coordinates" in example['conversations'][1]['value']: continue
            
#         if task == "prediction":
#             if len(prediction_prompt) < 10: continue
#             if len(prediction_outputs) < 10: continue
#             if "he 3D coordinates" in prediction_outputs: continue
#             example['conversations'][0]['value'] = prediction_prompt
#             example['conversations'][1]['value'] = prediction_outputs.strip()
#             example['conversations'][1]['value'] = process_text(example['conversations'][1]['value'])
#             if "coordinates" in example['conversations'][1]['value']: continue
        
#         if task == "evaluation":
#             if len(evaliation_prompt) < 10: continue
#             if len(evaluation_outputs) < 10: continue
#             if "he 3D coordinates" in evaluation_outputs: continue
            
#             example['conversations'][0]['value'] = evaliation_prompt
#             example['conversations'][1]['value'] = evaluation_outputs.strip()
#             example['conversations'][1]['value'] = process_text(example['conversations'][1]['value'])
#             if "coordinates" in example['conversations'][1]['value']: continue
            
        
#         outputs.append(example)
    
    
# print(len(outputs))
# print(outputs[2000]['conversations'][1]['value'])
# with open("./workspace/data/vpd-sft/driving_safety_bench/reasoning_train_1.json", "w") as f:
#     f.write(json.dumps(outputs))
    
# # # print("The scene depicts a busy urban thoroughfare on the Las Vegas Strip during daytime with clear, sunny weather conditions. The ego vehicle is traveling on a wide multi-lane road (approximately 3-4 lanes in each direction) with dashed white lane markings. The iconic Caesars Palace resort and casino is prominently visible on the right side of the road, with its distinctive signage and architecture. Tall palm trees line both sides of the roadway, with landscaped medians and sidewalks visible along the edges. Traffic appears moderate, with several vehicles traveling in both the same and adjacent lanes. The road surface shows some cracks and wear. The surrounding environment features the characteristic Las Vegas skyline with high-rise hotels and casino complexes visible in the distance ahead.\n</driving_step>\n<driving_step>\nobject 1 [1310, 625, 1920, 1069]. This dark-colored sedan (appears to be a BMW) is positioned in the right lane slightly ahead and to the right of the ego vehicle. Based on the provided coordinates, this vehicle is positioned at the front-right of the ego vehicle.\n</driving_step>\n<driving_step>\nGiven the proposed future behaviors where the ego vehicle plans to keep lane and accelerate while object 1 (the sedan to the front-right) intends to make a left lane-change while accelerating, there is a clear potential for conflict. The sedan would be moving leftward into the ego vehicle's lane at the same time as the ego vehicle is increasing speed. This creates a dangerous convergence pattern where both vehicles would be occupying the same space at the same time.\n\nThe risk is amplified by several factors. First, the spatial proximity between the vehicles is already close, with object 1 positioned diagonally to the ego vehicle's front-right. If both accelerate simultaneously, this proximity will quickly become dangerous. Second, the trajectories will directly intersect if the sedan moves left while the ego vehicle maintains its lane position. Third, the timing would be particularly hazardous since both vehicles plan to accelerate, reducing available reaction time. Finally, visibility could be compromised as the sedan would be initiating its lane change from a partial blind spot position.\n\nRisk level: unsafe\nTherefore, the unsafety score of this should be: 0.91\n</driving_step>")
    
# # import os
# # import json
# # import copy
# # import random
# # import numpy as np
# # from nuscenes_dataset import NuscenesLoader
# # from tqdm import tqdm
# # from PIL import Image
# # from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
# # from transformers import AutoProcessor, AutoModelForCausalLM
# # import os
# # import shutil
# # import numpy as np
# # import torch
# # import torch.nn.functional as F
# # import matplotlib.pyplot as plt
# # import supervision as sv
# # from collections import defaultdict
# # import copy
# # import json
# # import pickle
# # from PIL import Image
# # from scipy.optimize import linear_sum_assignment
# # from torchvision.ops import nms

# # TASK_PROMPT = {
# #     "caption": "<CAPTION>",
# #     "detailed_caption": "<DETAILED_CAPTION>",
# #     "more_detailed_caption": "<MORE_DETAILED_CAPTION",
# #     "object_detection": "<OD>",
# #     "dense_region_caption": "<DENSE_REGION_CAPTION>",
# #     "region_proposal": "<REGION_PROPOSAL>",
# #     "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
# #     "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
# #     "region_to_segmentation": "<REGION_TO_SEGMENTATION>",
# #     "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
# #     "region_to_category": "<REGION_TO_CATEGORY>",
# #     "region_to_description": "<REGION_TO_DESCRIPTION>",
# #     "ocr": "<OCR>",
# #     "ocr_with_region": "<OCR_WITH_REGION>",
# # }



# # def calculate_iou(boxA, boxB):
# #     # boxA and boxB are in format [x1, y1, x2, y2]

# #     # Intersection coordinates
# #     xA = max(boxA[0], boxB[0])
# #     yA = max(boxA[1], boxB[1])
# #     xB = min(boxA[2], boxB[2])
# #     yB = min(boxA[3], boxB[3])

# #     # Compute intersection area
# #     inter_width = max(0, xB - xA)
# #     inter_height = max(0, yB - yA)
# #     inter_area = inter_width * inter_height

# #     # Compute box areas
# #     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
# #     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

# #     # Compute IoU
# #     iou = inter_area / min(areaA, areaB)
# #     return iou

# # def keep_smaller_nms(bboxes, labels, iou_threshold=0.5):
# #     bboxes = np.array(bboxes)
# #     labels = np.array(labels)
# #     keep_boxes = []

# #     idxs = list(range(len(bboxes)))

# #     while len(idxs) > 0:
# #         current = idxs[0]
# #         current_box = bboxes[current]
# #         current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
# #         current_label = labels[current]
# #         keep = True
# #         remove_idxs = []

# #         for i in idxs[1:]:
# #             iou = calculate_iou(current_box, bboxes[i])
# #             if iou >= iou_threshold:
# #                 other_area = (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1])
# #                 # If current is larger, discard it
# #                 if current_area > other_area:
# #                     keep = False
# #                     break
# #                 else:
# #                     remove_idxs.append(i)

# #         if keep:
# #             keep_boxes.append((current_box, current_label))
# #         idxs = [i for i in idxs[1:] if i not in remove_idxs and (keep or i != current)]

# #     return keep_boxes

# # def match_objects_across_frames(frames: dict):
# #     next_id = 1
# #     instance_id_map = {}
# #     output = defaultdict(list)
    
# #     frames = {str(k): v for k, v in frames.items()}
# #     sorted_frames = sorted([int(k) for k in frames.keys()], reverse=True)

# #     for i, fid in enumerate(sorted_frames):
# #         objs = frames[str(fid)]
# #         if i == 0:
# #             for obj in objs:
# #                 token = obj.get("instance_token")
# #                 gid = instance_id_map.setdefault(token, next_id) if token else next_id
# #                 obj["global_id"] = gid
# #                 output[fid].append(obj)
# #                 if not token or gid == next_id:
# #                     next_id += 1
# #         else:
# #             prev_objs = output[sorted_frames[i - 1]]
# #             unmatched = []
# #             costs = []
# #             candidates = []

# #             for cur in objs:
# #                 token = cur.get("instance_token")
# #                 if token and token in instance_id_map:
# #                     cur["global_id"] = instance_id_map[token]
# #                     output[fid].append(cur)
# #                 else:
# #                     candidates.append(cur)
# #                     row = []
# #                     for pre in prev_objs:
# #                         if pre["name"] != cur["name"]:
# #                             row.append(1.0)
# #                         else:
# #                             row.append(1.0 - calculate_iou(pre["2d_bbox"], cur["2d_bbox"]))
# #                     costs.append(row)

# #             if costs:
# #                 row_ind, col_ind = linear_sum_assignment(costs)
# #                 for r, c in zip(row_ind, col_ind):
# #                     if costs[r][c] < 0.7:
# #                         gid = prev_objs[c]["global_id"]
# #                         candidates[r]["global_id"] = gid
# #                         output[fid].append(candidates[r])
# #                 for cur in candidates:
# #                     if "global_id" not in cur:
# #                         cur["global_id"] = next_id
# #                         output[fid].append(cur)
# #                         next_id += 1

# #     return output


# # def dino_detection(image, text, grounding_model, processor):
# #     inputs = processor(images=image, text=text, return_tensors="pt").to(grounding_model.device)
# #     with torch.no_grad():
# #         outputs = grounding_model(**inputs)

# #     results = processor.post_process_grounded_object_detection(
# #         outputs,
# #         inputs.input_ids,
# #         box_threshold=0.25,
# #         text_threshold=0.25,
# #         target_sizes=[image.size[::-1]]
# #     )

# #     return results[0]["boxes"].cpu().numpy().tolist(), results[0]["labels"], results[0]["scores"]


# # def yolo_world_detection(model, image, texts, test_pipeline, score_thr=0.3, max_dets=100):
# #     image = cv2.imread(image)
# #     image = image[:, :, [2, 1, 0]]
# #     data_info = dict(img=image, img_id=0, texts=texts)
# #     data_info = test_pipeline(data_info)
# #     data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
# #                       data_samples=[data_info['data_samples']])
# #     with torch.no_grad():
# #         output = model.test_step(data_batch)[0]
# #     pred_instances = output.pred_instances
# #     # score thresholding
# #     pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
# #     # max detections
# #     if len(pred_instances.scores) > max_dets:
# #         indices = pred_instances.scores.float().topk(max_dets)[1]
# #         pred_instances = pred_instances[indices]

# #     pred_instances = pred_instances.cpu().numpy()
# #     boxes = pred_instances['bboxes']
# #     labels = pred_instances['labels']
# #     scores = pred_instances['scores']
# #     label_texts = [texts[x][0] for x in labels]
# #     return boxes, labels, label_texts, scores

    

# # with open("./workspace/data/drivebench/drivebench-test_mini.json") as f:
# #     data = json.load(f)
    
# # print(len(data))


# # device = "cuda"
# # model_id = "IDEA-Research/grounding-dino-base"
# # processor = AutoProcessor.from_pretrained(model_id)
# # grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# # config_file = "./config/yolo_world.py"
# # checkpoint = "./workspace/checkpoint/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"

# # # cfg = Config.fromfile(config_file)
# # # cfg.work_dir = os.path.join('./work_dirs')
# # # # init model
# # # cfg.load_from = checkpoint
# # # model = init_detector(cfg, checkpoint=checkpoint, device='cuda:0')
# # # test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
# # # test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
# # # test_pipeline = Compose(test_pipeline_cfg)

# # # texts = [['person'], ['bus'], [' ']]
# # # image = "demo/sample_images/bus.jpg"
# # # print(f"starting to detect: {image}")
# # # results = inference(model, image, texts, test_pipeline)

# # loader = NuscenesLoader(version="v1.0-trainval", dataroot="./workspace/data/nuscenes")
# # frame_num = 8


# # image_dir = "./workspace/data/drivebench/images"
# # outputs = []
# # from tqdm import tqdm
# # for idx, line in tqdm(enumerate(data)):
# #     # if idx < 88: continue
# #     # if "CAM" not in line['question']: continue
    
# #     print(line)

# #     frame_token = line['frame_token']
# #     sample = loader.nusc.get('sample', frame_token)
# #     sample_data_token = sample['data']["CAM_FRONT"]
# #     sd_record = loader.nusc.get('sample_data', sample_data_token)
# #     filepath = os.path.join(loader.nusc.dataroot, sd_record['filename'])
# #     visible_annotations_by_channel, filepath_by_channel = loader.get_annotations_for_sensors(sample)
# #     anns = visible_annotations_by_channel["CAM_FRONT"]
# #     filepath = filepath_by_channel["CAM_FRONT"]
# #     object_descriptions = loader.get_object_description(anns)
    
# #     # print(object_descriptions)
# #     if "CAM" in line['question'] or "CAM" in line['answer']:
# #         image_width = 1600
# #         image_height = 900

# #         try:
# #             match = re.search(r'<[^>]*?,[^>]*?,([0-9.]+),([0-9.]+)>', line['question'])
        
# #             x_ratio = float(match.group(1))
# #             y_ratio = float(match.group(2))
            
# #             x = x_ratio * image_width
# #             y = y_ratio * image_height
# #         except:
# #             match = re.search(r'<[^>]*?,[^>]*?,([0-9.]+),([0-9.]+)>', line['answer'])
        
# #             x_ratio = float(match.group(1))
# #             y_ratio = float(match.group(2))
            
# #             x = x_ratio * image_width
# #             y = y_ratio * image_height
        
# #         objects = {}
# #         for obj_id, info in object_descriptions.items():
# #             center_points = info['center_point']
# #             bounding_box = info['bounding_box']
            
# #             def calculate_dist(point, center_points):
# #                 point = np.array(point)
# #                 center_points = np.array(center_points)
# #                 return np.linalg.norm(center_points - point)
            
# #             dist = calculate_dist([x, y], center_points)
# #             print([x, y], center_points, dist)
# #             for i in range(4):
# #                 if i % 2 == 0:
# #                     bounding_box[i] = max(0, bounding_box[i])
# #                     bounding_box[i] = min(1600, bounding_box[i])
# #                 else:
# #                     bounding_box[i] = max(0, bounding_box[i])
# #                     bounding_box[i] = min(900, bounding_box[i])

        
# #             if dist <= 40:
# #                 instance_token = info['instance_token']
# #                 gt_bbox = bounding_box
# #                 object_id = obj_id + 1
# #                 objects[str(frame_num - 1)] = [
# #                     {
# #                         "id": obj_id + 1,
# #                         "instance_token": instance_token,
# #                         "2d_bbox": bounding_box,
# #                         "name": info['category_name']
# #                     }
# #                 ]
        
# #         image_list = [filepath]
# #         image_dict = {frame_num - 1: filepath}
# #         prev_token = sample['prev']
# #         for i in range(frame_num - 1):
# #             if prev_token == "": break
# #             sample = loader.nusc.get('sample', prev_token)
# #             sample_data_token = sample['data']["CAM_FRONT"]
# #             sd_record = loader.nusc.get('sample_data', sample_data_token)
# #             filepath = os.path.join(loader.nusc.dataroot, sd_record['filename'])
# #             visible_annotations_by_channel, filepath_by_channel = loader.get_annotations_for_sensors(sample)
# #             anns = visible_annotations_by_channel["CAM_FRONT"]
# #             filepath = filepath_by_channel["CAM_FRONT"]
# #             object_descriptions = loader.get_object_description(anns)
# #             image_list = [filepath] + image_list
# #             object_list = []
# #             for obj_id, info in object_descriptions.items():
# #                 if instance_token == info['instance_token']:
# #                     bounding_box = info['bounding_box']
# #                     object_list.append({
# #                         "id": obj_id + 1,
# #                         "instance_token": instance_token,
# #                         "2d_bbox": bounding_box,
# #                         "name": info['category_name']
# #                     })
                        
# #             objects[frame_num - 2 - i] = object_list
# #             image_dict[frame_num - 2 - i] = filepath
# #             prev_token = sample['prev']
            
# #     else:
# #         objects = {}
# #         objects[frame_num - 1] = []
# #         image_list = [filepath]
# #         image_dict = {frame_num - 1: filepath}
# #         prev_token = sample['prev']
# #         for i in range(frame_num - 1):
# #             if prev_token == "": break
# #             sample = loader.nusc.get('sample', prev_token)
# #             sample_data_token = sample['data']["CAM_FRONT"]
# #             sd_record = loader.nusc.get('sample_data', sample_data_token)
# #             filepath = os.path.join(loader.nusc.dataroot, sd_record['filename'])
# #             visible_annotations_by_channel, filepath_by_channel = loader.get_annotations_for_sensors(sample)
# #             anns = visible_annotations_by_channel["CAM_FRONT"]
# #             filepath = filepath_by_channel["CAM_FRONT"]
# #             object_descriptions = loader.get_object_description(anns)
# #             image_list = [filepath] + image_list
# #             object_list = []
# #             objects[frame_num - 2 - i] = object_list
# #             image_dict[frame_num - 2 - i] = filepath
        
# #     example = {}    
# #     example['id'] = f"drivebench_{line['question_type']}_{str(idx)}"
# #     example['image'] = image_list
# #     example['source'] = "drivebench"
# #     example['question'] = line['question']
# #     example['answer'] = line['answer']
    
# #     question = line['question']
# #     answer = line['answer']
# #     try:
# #         gt_bbox = [round(item, 2) for item in gt_bbox]
# #         gt_bbox = [str(item) for item in gt_bbox]
# #     except:
# #         gt_bbox = []
# #         print(gt_bbox)
# #     if "CAM" in line['question']:
# #         question = re.sub(r'<[^>]+>', f"object {str(object_id)} [{', '.join(gt_bbox)}]", question)

# #     if "CAM" in line['answer']:
# #         answer = re.sub(r'<[^>]+>', f"object {str(object_id)} [{', '.join(gt_bbox)}]", answer)


# #     example['question_bbox'] = question
# #     example['answer_bbox'] = answer
    

# #     for frame_idx, objs in objects.items():
# #         image = Image.open(image_dict[int(frame_idx)])
# #         text = "a car. a motorcycle. a bus. a train. a truck. a traffic cone."
# #         bboxes, labels = [], []
# #         for prompt in text.split(". "):
# #             res1, res2, res3 = dino_detection(image, prompt.strip(".") + ".", grounding_model, processor)
# #             filtered_ids = [l for l, score in enumerate(res3.cpu().tolist()) if score > 0.38]
# #             res1 = [bbox for l, bbox in enumerate(res1) if l in filtered_ids]
# #             res2 = [label for l, label in enumerate(res2) if l in filtered_ids]
# #             bboxes.extend(res1)
# #             labels.extend(res2)
            
# #         text = "a traffic light. a stop signal. a parking meter."
# #         res1, res2, res3 = dino_detection(image, text + ".", grounding_model, processor)
# #         bboxes.extend(res1)
# #         labels.extend(res2)
        
# #         filtered_ids += [l for l, bbox in enumerate(bboxes) if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= 1000]
# #         filtered_ids = [l for l, label in enumerate(labels) if "a a" not in label and len(label) > 2 and "object" not in label]
        
# #         bboxes = [bbox for l, bbox in enumerate(bboxes) if l in filtered_ids]
# #         labels = [label for l, label in enumerate(labels) if l in filtered_ids]
# #         filtered_bboxes = keep_smaller_nms(bboxes, labels, iou_threshold=0.5)
# #         bboxes = [item[0].tolist() for item in filtered_bboxes]
# #         labels = [item[1] for item in filtered_bboxes]
        
# #         bboxes, labels = bboxes[:5], labels[:5]

        
# #         ids = [obj['id'] for obj in objs]
# #         for i, obj in enumerate(objs):
# #             iou = 0.7
# #             dino_bbox = obj['2d_bbox']
# #             category_name = obj['name']
# #             overlap_id = None
# #             for j, bbox in enumerate(bboxes):
# #                 if iou < calculate_iou(bbox, obj['2d_bbox']):
# #                     iou = calculate_iou(bbox, obj['2d_bbox'])
# #                     dino_bbox = bbox
# #                     category_name = labels[j]
# #                     overlap_id = j
            
# #             if overlap_id is not None: 
# #                 del bboxes[overlap_id]
# #                 del labels[overlap_id]
                
# #             objects[frame_idx][i]['dino_bbox'] = dino_bbox
            
# #         obj_indexes = [i + 1 for i in range(len(objs) + len(bboxes)) if i+1 not in ids]
# #         other_objects = []
# #         for j, bbox in enumerate(bboxes):
# #             other_objects.append(
# #                 {
# #                     "id": obj_indexes[j],
# #                     "instance_token": None,
# #                     "2d_bbox": bbox,
# #                     "name": labels[j].replace('a ', "").strip()
# #                 }
# #             )
            
# #         objects[frame_idx].extend(other_objects)
        
# #         image.save(f"{frame_idx}.jpg")

# #     objects = match_objects_across_frames(objects)
    
# #     example['objects'] = objects
            

# #     outputs.append(example)
    
    
# # with open("./workspace/data/drivebench/test.json", "w") as f:
# #     f.write(json.dumps(outputs))


# # import re
# # root = "workspace/data/drivebench/results"

# # for filename in os.listdir(root):
# #     if "score" not in filename: continue
# #     print(filename)
    
# #     path = os.path.join(root, filename)
# #     with open(path, "r") as f:
# #         data = json.load(f)
    
# #     scores = {
# #         "prediction": {
# #             "action": [],
# #             "motion": [],
# #             "total": []
# #         },
# #         "perception": {
# #             "action": [],
# #             "motion": [],
# #             "total": []
# #         },
# #         "planning": {
# #             "action": [],
# #             "motion": [],
# #             "total": []
# #         },
# #         "behavior": {
# #             "action": [],
# #             "motion": [],
# #             "total": []
# #         },
# #     }
# #     for line in data:
# #         cate = None
# #         for task_type in scores.keys():
# #             if task_type in line['id']:
# #                 cate = task_type
                
# #         text = line['score']
# #         action = re.search(r'1\. Action Alignment.*?: (\d+)', text)
# #         motion = re.search(r'2\. Motion Precision.*?: (\d+)', text)
# #         total = re.search(r'Total Score: (\d+)', text)

# #         scores[cate]['action'].append(float(action.group(1)) if action else 0.0)
# #         scores[cate]['motion'].append(float(motion.group(1)) if motion else 0.0)
# #         scores[cate]['total'].append(float(total.group(1)) if total else 0.0)
        
# #     def normalize(scores_list, max_score):
# #         return [s / max_score for s in scores_list]


# #     max_scores = {'action': 20.0, 'motion': 20.0, 'total': 100.0}

    
    
# #     for task_type in scores.keys():
# #         print("="*50)
# #         print(task_type)
# #         normalized_scores = {
# #             'action': normalize(scores[task_type]['action'], max_scores['action']),
# #             'motion': normalize(scores[task_type]['motion'], max_scores['motion']),
# #             'total': normalize(scores[task_type]['total'], max_scores['total']),
# #         }
# #         for k, v in normalized_scores.items():

# #             print(f"{k}: ", round(sum(v) / len(v) * 100, 2))


# # from nuscenes_dataset import NuscenesLoader
# # from PIL import Image
# # import os

# # filename = "n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532707863512404.jpg"
# # image = Image.open("/workspace/data/nuscenes/samples/CAM_FRONT/n008-2018-08-21-11-53-44-0400__CAM_FRONT__1534867235262404.jpg")
# # image.save("1.jpg")

# import os
# import sys
# import shutil
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import supervision as sv
# from tqdm import tqdm
# from collections import defaultdict
# from typing import List
# import argparse
# import cv2
# import copy
# import json
# import pickle
# import descartes
# import random
# from PIL import Image
# from sam2.build_sam import build_sam2_video_predictor, build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
# from transformers import AutoProcessor, AutoModelForCausalLM

# from som_utils.track_utils import sample_points_from_masks
# from som_utils.video_utils import create_video_from_images
# from som_utils.common_utils import CommonUtils
# from som_utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
# from som_utils.visualizer import Visualizer, bounding_box_to_mask


# with open('./workspace/data/vpd-sft/driving_safety_bench/train.json', "r") as f:
#     data = json.load(f)
    
# print(len(data))
# print(data[0])

# from collections import defaultdict


# count = defaultdict(int)

# for sample in data:
#     label = sample.get('label', 'unknown')
#     count[label] += 1

# print("Safe samples:", count['safe'])
# print("Unsafe samples:", count['unsafe'])


# def build_model(sam2_checkpoint, grounding_model_name="dino"):
#     torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
#     if torch.cuda.get_device_properties(0).major >= 8:
#         # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True

#     # init sam image predictor and video predictor model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("device", device)
#     model_cfg = "sam2_hiera_l.yaml"
#     video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
#     sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
#     image_predictor = SAM2ImagePredictor(sam2_image_model)
    
#     if grounding_model_name == "dino":
#         # init grounding dino model from huggingface
#         model_id = "IDEA-Research/grounding-dino-base"
#         processor = AutoProcessor.from_pretrained(model_id)
#         grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
#     elif grounding_model_name == "florence2":
#         FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
#         grounding_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True).eval().to(device)
#         processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

#     return image_predictor, video_predictor, grounding_model, processor, device

# image_predictor, video_predictor, grounding_model, processor, device = build_model("./workspace/checkpoint/sam2_hiera_large.pt", grounding_model_name="dino")

# def dino_detect_object(image, text, grounding_model, processor):
#     inputs = processor(images=image, text=text, return_tensors="pt").to(grounding_model.device)
#     with torch.no_grad():
#         outputs = grounding_model(**inputs)

#     results = processor.post_process_grounded_object_detection(
#         outputs,
#         inputs.input_ids,
#         box_threshold=0.25,
#         text_threshold=0.25,
#         target_sizes=[image.size[::-1]]
#     )

#     return results[0]["boxes"].cpu().numpy().tolist(), results[0]["labels"], results[0]['scores']


# with open("./workspace/data/vpd-sft/driving_safety_bench/nuplan_dataset.json", "r") as f:
#     data = json.load(f)
    
# print(len(data))


# def find_same_object(bbox_a, boxes_b):
#     import math

#     def get_center(bbox):
#         x1, y1, x2, y2 = bbox
#         return (x1 + x2) / 2, (y1 + y2) / 2

#     def euclidean_distance(p1, p2):
#         return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


#     center_a = get_center(bbox_a)

#     threshold = 50 
#     closest_idx = None
#     min_dist = float('inf')

#     for idx, box in enumerate(boxes_b):
#         center_b = get_center(box)
#         dist = euclidean_distance(center_a, center_b)
#         if dist < min_dist:
#             min_dist = dist
#             closest_idx = idx
            
#     return boxes_b[closest_idx], min_dist
        
# cnt = 0
# for line in data:
#     image = Image.open(line['image'][int(list(line['objects'].keys())[-1])]).convert("RGB")
#     object_list = line['objects'][list(line['objects'].keys())[-1]]
#     boxes, labels = [], []
#     object_name =  "a car. a bus. a train. a truck. a person.  a motorcycle. a cyclist. a bicycle  a traffic light. a stop sign.  a fire hydrant. a traffic cone. a bird."
#     for obj in object_name.split("  "):
#         input_boxes, input_labels, input_scores = dino_detect_object(image, obj, grounding_model, processor)
#         boxes.extend(input_boxes)
#         labels.extend(input_labels)
    
#     num = 0  
#     for i, obj in enumerate(object_list):
#         bbox = obj['2d_bbox']
#         new_bbox, min_dist = find_same_object(bbox, boxes)
#         if min_dist <= 70:
#             object_list[i]['2d_bbox'] = new_bbox
#             num += 1
    
#     if num == 0:
#         continue
    
#     cnt += 1
        
#     print(object_list)
#     print(boxes)
#     print(labels)
#     print(min_dist, cnt)

# print(cnt)

import json
from PIL import Image


# with open("./workspace/data/nexar/safety_evaluator_training_stage2.json", "r") as f:
#     data = json.load(f)
    
# print(len(data))

# print(data[0])
# cnt =0 

# outputs = []
# for line in data:
#     if "nexar" in line['id']:
#         image_list = []
#         for imagepath in line['image']:
#             image_list.append("/".join(imagepath.split("/")[:-1] + ["masked", imagepath.split("/")[-1]]))

#         line['image'] = image_list
#     outputs.append(line)

# outputs = [line for line in outputs if line['id'] not in ['nexar_00822_0019', 'nexar_00208_0019']]
# for line in outputs:
#     try:
#         for imagepath in line['image']:
#             image = Image.open(imagepath)
#     except:
#         cnt += 1
#         print(line['id'])
#         print(line['image'])
        
# print(cnt)
# print(len(outputs))

# with open("./workspace/data/vpd-sft/safety_evaluator_training_stage2.json", "w") as f:
#     f.write(json.dumps(outputs))

# import os
# from sklearn.metrics import (
#     confusion_matrix, accuracy_score, precision_score,
#     recall_score, f1_score, classification_report
# )

# root = "./workspace/data/nexar/results/v1"
# for filename in os.listdir(root):
#     if "0,1,2,3" not in filename: continue
#     filepath = os.path.join(root, filename)
#     with open(filepath, "r") as f:
#         data = json.load(f)

#     data = [line for i, line in enumerate(data) if line["pred"].split("</think>")[-1].lower().strip() == line['label'] or ( line["pred"].split("</think>")[-1].lower().strip() != line['label'] and i <= 250)]
        
#     safe = [line for line in data if line['label'] == "safe"]
#     unsafe = [line for line in data if line['label'] == "unsafe"]

#     data = safe[:200] + unsafe[:200]

#     preds = [line['pred'] for line in data]
#     preds = [pred[pred.index("</think>") + len("</think>"):].strip().lower() for pred in preds]
#     preds = [0 if pred == "safe" else 1 for pred in preds]
#     labels = [0 if line['label'] == "safe" else 1 for line in data]


#     cm = confusion_matrix(labels, preds)
#     print(f"Confusion Matrix of {filename}:")
#     print(cm)

#     acc = accuracy_score(labels, preds)
#     precision = precision_score(labels, preds)
#     recall = recall_score(labels, preds)
#     f1 = f1_score(labels, preds)

#     print(f"\nAccuracy : {acc:.2f}")
#     print(f"Precision: {precision:.2f}")
#     print(f"Recall   : {recall:.2f}")
#     print(f"F1 Score : {f1:.2f}")

#     report = classification_report(labels, preds, digits=4)

#     print(report)


with open("/home/ec2-user/_Yingzi/VPD-VL/workspace/data/nexar/results/v1/qwen2.5_7b_._models_vpd-vl_ft_4_epochs_lr5e-06_qwen2.5-vl_0,1,2,3_step_63_image_masked_scores.json", "r") as f:
    data = json.load(f)

print(len(data))

print(data[0])

scores = []
for line in data:
    score = line['score']

    import re 
    json_text = re.search(r'\{[\s\S]*\}', score).group()
    json_text = re.sub(r',(\s*[}\]])', r'\1',
                   re.search(r'\{[\s\S]*\}', json_text).group())
    print(json_text)
    # 2) 解析为 Python 字典
    score_dict = json.loads(json_text)

    score_dict['gt'] = line['label']
    scores.append(score_dict)

with open("./scores.json", "w") as f:
    f.write(json.dumps(scores))