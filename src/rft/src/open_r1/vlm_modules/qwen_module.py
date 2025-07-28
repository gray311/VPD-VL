from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union, List, Tuple, Iterable
from trl.data_utils import maybe_apply_chat_template
import torch
from copy import deepcopy
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from PIL import Image
import re

LONGI_TAGS = {
    "slow_down":        [r"slow\s*down", r"decelerate", r"brake"],
    "maintain_speed":   [r"maintain\s*speed", r"keep\s*speed", r"follow vehicle"],
    "accelerate":       [r"accelerate", r"speed\s*up"],
    "stop":             [r"\bstop\b", r"remain stopped", r"stopped"],
    "yield":            [r"\byield\b"],
    "walk":             [r"\bwalk\b", r"continue walking", r"cross street|cross road|cross roadway"],
    "stationary":       [r"stationary", r"stay (in place|static)", r"no movement",
                         r"maintain position", r"remain parked"],
    "na":               [r"not applicable", r"N/A", r"no other dynamic objects",
                         r"static (object|barrier|traffic signal|road marking)"]
}

LAT_TAGS = {
    "keep_lane":          [r"keep lane", r"keep path", r"keep direction",
                           r"keep lane \(parked\)", r"keep lane \(stationary\)"],
    "lane_change_right":  [r"change lane right", r"merge right", r"merge into traffic( lane)?",
                           r"right lane-change"],
    "lane_change_left":   [r"change lane left", r"merge left", r"left lane-change", r"merge lane left"],
    "turn_right":         [r"turn right", r"prepare to turn right"],
    "turn_left":          [r"turn left"],
    "yield":              [r"\byield\b(?!.*signal)"],  # avoid matching 'yield (stationary object)'
    "avoid":              [r"avoid object"],
    "cross_walk":         [r"cross (street|road|roadway|intersection)"],
    "park_pull_over":     [r"park(ed)?", r"pull over", r"remain parked"],
    "na":                 [r"not applicable", r"N/A", r"static (object|barrier|traffic signal|road marking)",
                           r"no other vehicles ahead"]
}

def clean_dict(gt: dict) -> dict:
    """Remove keys containing '()' or '/' and any keys whose value is None."""
    cleaned = {}
    for k, v in gt.items():
        if any(sym in k for sym in ('(', ')', '/')):
            continue          # skip keys with '(' or ')' or '/'
        if v is None:
            continue          # skip keys whose value is None
        cleaned[k] = v
    return cleaned

def parse_objects(section: str):
    """
    Parse 'Key Dynamic Objects' section and return only position & type.

    Example line:
      object 1 — ahead-left to ahead, black pickup truck. It appears ...
    -> {"object 1": {"position": "ahead-left to ahead", "type": "black pickup truck"}}
    """
    import re
    TYPE_MAP = {
        r"\b(pickup|pickup\s*truck)\b": "truck",
        r"\bsuv|crossover\b": "car",
        r"\bvan|minivan\b": "van",
        r"\bbus|coach\b": "bus",
        r"\bmotorcycle|bike\b": "motorcycle",
        r"\bbicycle|cyclist\b": "bicycle",
        r"\btruck|lorry|semi|tractor[-\s]?trailer\b": "truck",
        r"\bcar|sedan|hatchback|coupe\b": "car",
        r"\bpedestrian|person\b": "pedestrian",
    }

    POS_BUCKETS = {
        "ahead-left": ["ahead-left", "front-left", "left-front"],
        "ahead-right": ["ahead-right", "front-right", "right-front"],
        "left": ["left"],
        "right": ["right"],
        "far": ["far"],
        "near": ["near", "close"],
        "ahead": ["ahead", "front"],
        "behind": ["behind", "rear", "back"],
    }


    OBJ_BLOCK = re.compile(
        r"(object\s+\d+)\s*—\s*(.+?)(?=(?:\nobject\s+\d+\s*—)|\Z)",
        flags=re.IGNORECASE | re.DOTALL
    )
    
    def normalize_type(raw_type: str):
        raw_type_l = raw_type.lower()
        coarse = None
        for pat, cls in TYPE_MAP.items():
            if re.search(pat, raw_type_l):
                coarse = cls
                break
        # fallback
        if coarse is None:
            coarse = "vehicle" if "vehicle" in raw_type_l else "unknown"
        # also return a list of candidate labels (coarse + super class "vehicle")
        candidates = list({coarse, "vehicle"}) if coarse != "pedestrian" else ["pedestrian"]
        return coarse, candidates

    def normalize_position(raw_pos: str):
        raw_pos_l = raw_pos.lower()
        buckets = []
        for canon, kws in POS_BUCKETS.items():
            if any(k in raw_pos_l for k in kws):
                buckets.append(canon)
        # if nothing matched, fall back to raw
        if not buckets:
            buckets = [raw_pos.strip()]
        return buckets
    
    results = {}
    text = section.replace("–", "—").replace("--", "—")

    for label, content in OBJ_BLOCK.findall(text):
        label = label.lower().strip()

        first_sentence = re.split(r"\.\s+", content.strip(), maxsplit=1)[0]

        if "," in first_sentence:
            raw_pos, raw_type = [s.strip() for s in first_sentence.split(",", 1)]
        else:
            toks = first_sentence.split()
            raw_pos = toks[0] if toks else ""
            raw_type = " ".join(toks[1:]) if len(toks) > 1 else ""

        pos_tags = normalize_position(raw_pos)
        _, type_tags = normalize_type(raw_type)

        results[label] = {
            "raw_position": raw_pos,
            "raw_type": raw_type,
            "position_tags": pos_tags,
            "type_tags": type_tags
        }

    return results


def _match_tag(raw: str, tag_dict: Dict[str, Iterable[str]], default="unknown") -> str:
    low = raw.lower()
    for tag, patterns in tag_dict.items():
        for p in patterns:
            if re.search(p, low):
                return tag
    return default

def normalize_behaviors(long_raw: str, lat_raw: str) -> Tuple[str, str]:
    """
    Map raw longitudinal & lateral strings to canonical tags.
    """
    long_tag = _match_tag(long_raw, LONGI_TAGS)
    lat_tag  = _match_tag(lat_raw,  LAT_TAGS)
    return long_tag, lat_tag


def parse_behaviors(block: str) -> Dict[str, Dict[str, str]]:
    """
    Input block:
        ego: slow down, keep lane
        object 1: maintain speed, change lane right
        ...

    Output:
        {'ego': {'longitudinal': 'slow_down', 'lateral': 'keep_lane',
                 'raw_longitudinal': 'slow down', 'raw_lateral': 'keep lane'}, ...}
    """
    result = {}
    for line in block.strip().splitlines():
        if ":" not in line:
            continue
        agent, actions = line.split(":", 1)
        agent = agent.strip().lower()

        parts = [p.strip() for p in actions.split(",")]
        longi_raw = parts[0] if parts else ""
        lat_raw   = parts[1] if len(parts) > 1 else ""

        longi, lat = normalize_behaviors(longi_raw, lat_raw)

        result[agent] = {
            "long_beh": longi,
            "lat_beh": lat,
            "raw_long_beh": longi_raw,
            "raw_lat_beh": lat_raw
        }
    return result


class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        additional_output = None
        if len(images) > 0:
            
            prompt_inputs = processing_class(
                text=prompts_text,
                videos=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
      
            # additional_output = [{"image_grid_thw": image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']] 
    
            # print(processing_class, len(prompts_text), len(prompt_inputs), len(additional_output))
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs, additional_output
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case "vpd":
                return "{Question} \n\nFirst output the thinking process in <think>1. Scene Understanding\n...\n\n2.  Key Dynamic Objects\n...\n\n3. Driving Decision Predictio\n...\n\n4.  Risk Assessment & Justificatio\n...</think> tags and then output the safe/unsafe (only one word) for collision risk prediction."
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    
    
    
    @staticmethod
    def format_reward_vpd(completions, **kwargs):
        """
        Check if each completion matches the 4-step reasoning format:
        1. Scene Understanding
        2. Key Dynamic Objects
        3. Driving Decision Prediction
        4. Risk Assessment & Justification

        Returns 1.0 if exactly four properly headed steps are present, else 0.0.
        """
        import re
        import os
        from datetime import datetime

        HEADERS = [
            r"Scene\s+Understanding",
            r"Key\s+Dynamic\s+Objects",
            r"Driving\s+Decision\s+Prediction",
            r"Risk\s+Assessment\s*&\s*Justification",
        ]
        headers_re = "|".join(HEADERS)
        step_pattern = re.compile(
            rf"(?m)(^\d+\.\s(?:{headers_re}).*?)(?=^\d+\.\s(?:{headers_re})|\Z)",
            flags=re.IGNORECASE | re.DOTALL
        )

        completion_contents = [comp[0].get("content", "") for comp in completions]
        matches = []
        

        for content in completion_contents:
            content = content.strip()
            reason_format_score = 0.0
            if "<think>" in content and "</think>" in content:
                if content.index("<think>") == 0 and content.count("<think>") == 1 and content.count("</think>") == 1:
                    reason_format_score = 1.0
        
            result = content[content.rfind("</think>") + len("</think>"):].strip()
            content = content.replace("<think>", "").replace("</think>", "")
            steps = [m.group(0).rstrip() for m in step_pattern.finditer(content)]
       
            process_format_score = float(len(steps) == 4)
            result_format_score = float(result.lower() in ["safe", "unsafe"])
            
            matches.append(process_format_score * 1.0 + result_format_score * 0.5 + reason_format_score * 0.5)

        # Optional debug logging
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "format_reasoning.log")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8") as f:
                f.write(f"--- {current_time} format_reward_reasoning ---\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")

        # Return 1.0 for valid format, else 0.0
        return matches

    
    
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        def resize_bbox(bbox, input_height, input_width, image_height, image_width):
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
            bbox[2] = bbox[2] / input_width * image_width
            bbox[3] = bbox[3] / input_height * image_height
            return bbox
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

        for i, (content, sol) in enumerate(zip(contents, solution)):
            image_grid_thw = kwargs.get("image_grid_thw")[i]
            image_path = kwargs.get("image_path")[i][0]
            image = Image.open(image_path)
            image_width, image_height = image.size
            input_height = int(image_grid_thw[1]*14)
            input_width = int(image_grid_thw[2]*14)
            
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[i]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards
    
    @staticmethod
    def beh_reward(completions, **kwargs):
        import re
        import os
        from datetime import datetime
        
        HEADERS = [
            r"Scene\s+Understanding",
            r"Key\s+Dynamic\s+Objects",
            r"Driving\s+Decision\s+Prediction",
            r"Risk\s+Assessment\s*&\s*Justification",
        ]
        headers_re = "|".join(HEADERS)
        step_pattern = re.compile(
            rf"(?m)(^\d+\.\s(?:{headers_re}).*?)(?=^\d+\.\s(?:{headers_re})|\Z)",
            flags=re.IGNORECASE | re.DOTALL
        )

        
        def compute_reward(pred, gt):
            pred, gt = clean_dict(pred), clean_dict(gt)
            agents = set(gt.keys()) & set(pred.keys())
            per_agent = {}
            total, weight_sum, ego_weight = 0.0, 0.0, 1.5
            
            for a in agents:
                g = gt.get(a, {})
                p = pred.get(a, {})
                score = 0.0
                if g.get("long_beh") and p.get("long_beh"):
                    if g["long_beh"] == p["long_beh"]:
                        score += 0.5
                    elif g["long_beh"] != "unkown":
                        score -= 0.5
                    else:
                        score += 0.0
  
                if g.get("lat_beh") and p.get("lat_beh"):
                    if g["lat_beh"] == p["lat_beh"]:
                        score += 0.5
                    elif g["lat_beh"] != "unkown":
                        score -= 0.5
                    else:
                        score += 0.0
                
                w = ego_weight if a == "ego" else 1.0
                per_agent[a] = score
                total += score * w
                weight_sum += w
            
            overall = total / weight_sum if weight_sum else 0.0
            return {"per_agent": per_agent, "overall": overall}
            
        completion_contents = [comp[0].get("content", "") for comp in completions]
        rewards = []
        for i, content in enumerate(completion_contents):
            result = content[content.rfind("</think>") + len("</think>"):].strip()
            content = content.replace("<think>", "").replace("</think>", "")
            steps = [m.group(0).rstrip() for m in step_pattern.finditer(content)]
            
            if len(steps) != 4:
                rewards.append(0.0)
                continue
            
            example_type = kwargs.get("type")[i]
            if example_type == "counterfactual":
                rewards.append(0.0)
                continue
            

            obj_step, beh_step = steps[1], steps[2]
            behavior_info = kwargs.get("behavior")[i]
            
            pred_beh_info = parse_behaviors(beh_step)
            reward = compute_reward(pred_beh_info, behavior_info)

            rewards.append(reward['overall'])
            
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                if reward['overall'] <= 1.0:  # this condition can be changed for debug
                    with open(log_path.replace(".txt", "_beh.txt"), "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"Pred Behavior: {pred_beh_info}\n")
                        f.write(f"GT Behavior: {behavior_info}\n") 
                        
        return rewards
    
    
    @staticmethod
    def safety_reward(completions, **kwargs):
        import re
        import os
        from datetime import datetime
        
        
        completion_contents = [comp[0].get("content", "") for comp in completions]
        rewards = []
        for i, content in enumerate(completion_contents):
            pred = content[content.rfind("</think>") + len("</think>"):].strip()
            safety_label = kwargs.get("label")[i]
            
            if safety_label.lower() == pred.lower():
                reward = 2.0
                rewards.append(2.0)
            else:
                reward = -2.0
                rewards.append(-2.0)
            
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path.replace(".txt", "_safety.txt"), "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"Pred Safety: {pred}\n")
                        f.write(f"GT Safety: {safety_label}\n") 
                        
        return rewards
            
            
    
    @staticmethod
    def obj_reward(completions, **kwargs):
        import re
        import os
        from datetime import datetime
        
        HEADERS = [
            r"Scene\s+Understanding",
            r"Key\s+Dynamic\s+Objects",
            r"Driving\s+Decision\s+Prediction",
            r"Risk\s+Assessment\s*&\s*Justification",
        ]
        headers_re = "|".join(HEADERS)
        step_pattern = re.compile(
            rf"(?m)(^\d+\.\s(?:{headers_re}).*?)(?=^\d+\.\s(?:{headers_re})|\Z)",
            flags=re.IGNORECASE | re.DOTALL
        )
        
        def compute_reward(pred, gt, decay=0.7):
            pred, gt = clean_dict(pred), clean_dict(gt)
            agents = set(gt.keys()) & set(pred.keys())
            per_agent = {}
            total, weight_sum = 0.0, 0.0
            
            for a in agents:
                g = gt.get(a, {})
                p = pred.get(a, {})
                score = 0.0
                
                if g.get("raw_position") and g.get("position_tags") and p.get("raw_position") and p.get("position_tags"):
                    if g['raw_position'] == p['raw_position']:
                        score += 0.5
                    else:
                        position_tags = [x for x in g['position_tags'] if x != "unknown"]
                        num_tags = len(position_tags)
                        weights = [decay**i for i in range(num_tags)]
                        total_w = sum(weights)
                        matched_w = 0.0
                        for tag, w in zip(position_tags, weights):
                            if tag.lower() in p['raw_position']:
                                matched_w += w

                        score += 0.5 * matched_w / total_w
                        
                if g.get("raw_type") and g.get("type_tags") and p.get("raw_type") and p.get("type_tags"):
                    if g['raw_type'] == p['raw_type']:
                        score += 0.5
                    else:
                        type_tags = [x for x in g['type_tags'] if x != "unknown"]
                        num_tags = len(type_tags)
                        mark = False
                        for tag in type_tags:
                            if tag.lower() in p['raw_type']:
                                mark = True
                
                        score += 0.5 if mark else 0.0
                         
                w = 1.0   
                per_agent[a] = score
                total += score * w
                weight_sum += w
            
            overall = total / weight_sum if weight_sum else 0.0
            return {"per_agent": per_agent, "overall": overall}
            
        completion_contents = [comp[0].get("content", "") for comp in completions]
        rewards = []
        for i, content in enumerate(completion_contents):
            result = content[content.rfind("</think>") + len("</think>"):].strip()
            content = content.replace("<think>", "").replace("</think>", "")
            steps = [m.group(0).rstrip() for m in step_pattern.finditer(content)]
            
            if len(steps) != 4:
                rewards.append(0.0)
                continue
            
            obj_step, beh_step = steps[1], steps[2]
            object_info = kwargs.get("object")[i]
            
            pred_obj_info = parse_objects(obj_step)
            reward = compute_reward(pred_obj_info, object_info)
            rewards.append(reward['overall'])
            
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                if reward['overall'] <= 1.0:  # this condition can be changed for debug
                    with open(log_path.replace(".txt", "_obj.txt"), "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"Pred Object: {pred_obj_info}\n")
                        f.write(f"GT Object: {object_info}\n") 
                        
        return rewards
            
        
            
            

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return Qwen2VLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case "vpd":
                    return Qwen2VLModule.format_reward_vpd
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "beh":
            match task_type:
                case "vpd":
                    return Qwen2VLModule.beh_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "obj":
            match task_type:
                case "vpd":
                    return Qwen2VLModule.obj_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "safety":
            match task_type:
                case "vpd":
                    return Qwen2VLModule.safety_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")
