import os
import json
import cv2
import pickle
import numpy as np
from dataclasses import dataclass
import supervision as sv
import random

ROAD_OBJECTS = ["construction", "pedestrian", "trailer", "truck", "bus", "motorcycle", "bicycle", "car", "vehicle", "traffic light", "stop sign", "cyclist"]
OBJECTS_PRIORITY = {"construction.": 2, "pedestrian.": 0, "trailer.": 2, "truck.": 2, "bus.": 2, "motorcycle.": 2, "bicycle.": 2, "car.": 3, "vehicle":3}


def draw_number_on_bbox(image, bbox, number, font_scale=0.7, thickness=2, offset=0):
    x1, y1, x2, y2 = bbox
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    min_box_size, x_offset, y_offset= 50, 0, 0
    if box_w < min_box_size:
        x_offset = box_w // 1.5
    if box_h < min_box_size:
        y_offset = box_h // 1.5

    center_x = (x1 + x2) // 2 + offset + x_offset
    center_y = (y1 + y2) // 2 + offset + y_offset
    
    text = str(number)
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    padding = 5  
    bg_x1 = int(center_x - text_width // 2 - padding)
    bg_y1 = int(center_y - text_height // 2 - padding)
    bg_x2 = int(center_x + text_width // 2 + padding)
    bg_y2 = int(center_y + text_height // 2 + padding)
    
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1) 
    
    text_x = int(center_x - text_width // 2)
    text_y = int(center_y + text_height // 2)
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness) 
    
    return image

def compute_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def compute_iou(boxA, boxB):
    """Compute the Intersection over Union (IoU) between two bounding boxes."""
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    areaA = compute_area(boxA)
    areaB = compute_area(boxB)

    # Compute the area of the union
    union_area = min(areaA, areaB)

    # Compute the IoU, 
    iou = inter_area / union_area if union_area > 0 else 0
    return iou, areaA >= areaB


class CommonUtils:
    @staticmethod
    def creat_dirs(path):
        """
        Ensure the given path exists. If it does not exist, create it using os.makedirs.

        :param path: The directory path to check or create.
        """
        try: 
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"Path '{path}' did not exist and has been created.")
            else:
                print(f"Path '{path}' already exists.")
        except Exception as e:
            print(f"An error occurred while creating the path: {e}")

    @staticmethod
    def draw_masks_and_box_with_supervision(raw_image_list, mask_path, json_path, output_path,  colors, types, overwrite=False):
        for _ , raw_image in enumerate(raw_image_list):
            raw_image_name = raw_image.split("/")[-1]
            if ".jpg" not in raw_image_name: continue
            frame_idx = int(raw_image.split("/")[-1].rstrip(".jpg"))
            
            image_path = raw_image
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            if image is None:
                raise FileNotFoundError("Image file not found.")

            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_"+str(frame_idx).zfill(4) +".pkl")
            
            
            try:
                with open(mask_npy_path,'rb') as f:
                    mask_dict = pickle.load(f)
            except:
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                print(f"Annotated image has been saved as {output_image_path}")
                continue
            

            # load box information
            file_path = os.path.join(json_path, "mask_"+str(frame_idx).zfill(4)+".json")
            with open(file_path, "r") as file:
                json_dict = json.load(file)

            annotated_frame = image.copy()
            json_data = json_dict
            mask = np.array(mask_dict)

            if json_data is None or mask is None: continue
            # color map
            unique_ids = np.unique(mask)
            
            # get each mask from unique mask file
            all_object_masks = []
            for uid in unique_ids:
                if uid == 0: # skip background id
                    continue
                else:
                    object_mask = (mask == uid)
                    all_object_masks.append(object_mask[None])
            
            try:
                all_object_masks = np.concatenate(all_object_masks, axis=0)
                print(all_object_masks.shape)
            except:
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                print(f"Annotated image has been saved as {output_image_path}")
                continue

            
            all_object_boxes = []
            all_object_ids = []
            all_class_names = []
            all_object_colors = []
            all_object_types = []
            object_id_to_name = {}
            
            for obj_id, obj_item in json_data["labels"].items():
                # box id
                instance_id = obj_item["instance_id"]
                if instance_id not in unique_ids: # not a valid box
                    continue

                # box coordinates
                x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                if obj_item["class_name"] == "laneline":
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    all_object_boxes.append([cx-5, cy-5, cx, cy])
                else:
                    all_object_boxes.append([x1, y1, x2, y2])
                # box name
                class_name = obj_item["class_name"]
                
                # build id list and id2name mapping
                all_object_ids.append(instance_id)
                all_class_names.append(class_name)
                all_object_colors.append(colors[instance_id])
                all_object_types.append(types[instance_id])
                object_id_to_name[instance_id] = class_name
            
            # Adjust object id and boxes to ascending order
            paired_id_and_box = zip(all_object_ids, all_object_boxes)
            sorted_pair = sorted(paired_id_and_box, key=lambda pair: pair[0])
            
            # Because we get the mask data as ascending order, so we also need to ascend box and ids
            all_object_ids = [pair[0] for pair in sorted_pair]
            all_object_boxes = [pair[1] for pair in sorted_pair]
            print(all_object_boxes)
            
            # Because we need to prevent objects in the foreground from being obscured by those in the background, so 
            # we also need to achieve the drawing order where objects with larger bounding box areas are drawn on top
            paired_id_and_box = list(zip(all_object_ids, all_object_boxes, all_class_names, all_object_masks, all_object_colors, all_object_types))
            sorted_pair = sorted(paired_id_and_box, key=lambda pair: compute_area(pair[1]))
            
        
            all_object_ids = [pair[0] for pair in sorted_pair]
            all_object_boxes = [pair[1] for pair in sorted_pair]
            all_class_names = [pair[2] for pair in sorted_pair]
            all_object_masks = [pair[3] for pair in sorted_pair]
            all_object_colors = [pair[4] for pair in sorted_pair]
            all_object_types = [pair[5] for pair in sorted_pair]


                            
            detections = sv.Detections(
                xyxy=np.array(all_object_boxes),
                mask=np.array(all_object_masks),
                class_id=np.array(all_object_ids, dtype=np.int32),
            )
            
            # custom label to show both id and class name
            labels = [
                f"{instance_id}: {class_name}" for instance_id, class_name in zip(all_object_ids, all_class_names)
            ]
            
         
            for i in range(len(all_object_masks)):
                binary_mask = all_object_masks[i].astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_frame, contours, -1, all_object_colors[i], 2) 
                offset = 0
                if "light" in all_object_types[i]:
                    offset = 10
                annotated_frame = draw_number_on_bbox(annotated_frame, all_object_boxes[i], all_object_ids[i], offset=offset)

            output_image_path = os.path.join(output_path, raw_image_name)
            cv2.imwrite(output_image_path, annotated_frame)
            print(f"Annotated image saved as {output_image_path}")

    @staticmethod
    def draw_masks_and_box(raw_image_path, mask_path, json_path, output_path, colors):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_"+raw_image_name.split(".")[0]+".npy")
            mask = np.load(mask_npy_path)
            # color map
            unique_ids = np.unique(mask)
            colors = {uid: CommonUtils.random_color() for uid in unique_ids}
            colors[0] = (0, 0, 0)  # background color

            # apply mask to image in RBG channels
            colored_mask = np.zeros_like(image)
            for uid in unique_ids:
                colored_mask[mask == uid] = colors[uid]
            alpha = 0.5  # 调整 alpha 值以改变透明度
            output_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)


            file_path = os.path.join(json_path, "mask_"+raw_image_name.split(".")[0]+".json")
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                # Draw bounding boxes and labels
                for obj_id, obj_item in json_data["labels"].items():
                    # Extract data from JSON
                    x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                    instance_id = obj_item["instance_id"]
                    class_name = obj_item["class_name"]

                    # Draw rectangle
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put text
                    label = f"{instance_id}: {class_name}"
                    cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Save the modified image
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, output_image)

                print(f"Annotated image saved as {output_image_path}")

    @staticmethod
    def random_color():
        """random color generator"""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))