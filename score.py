import os
import re
import json
import copy
import torch
import random
import transformers
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from rouge_score import rouge_scorer
import argparse
from api.api_models import APIModel
from collections import defaultdict

scores = {
    "rougel": [],
    "llm-eval": []
}

evaluation_prompt = prompt = """You are an autonomous‑driving safety auditor.
Given:
  • VISUAL_EVIDENCE   – one image or a short frame sequence
  • REASONING_TRACE   – four numbered steps from another model
  • GROUND_TRUTH_LABEL – “safe” or “unsafe” for the clip
Your tasks:

1. Score each reasoning step on factual correctness.
   Use a 0‑1 scale:
     0   = incorrect / contradicts evidence
     0.5 = partially correct / ambiguous
     1   = correct and complete

2. Provide one concise justification per step (≤ 25 words).

3. Identify the PRIMARY_ERROR step that most contributed to a
   wrong final verdict. Use “None” if the trace matches evidence
   and label.

Return results strictly in the JSON schema below—no extra text.

JSON SCHEMA:
{
  "scores": {
    "Scene":   <float 0‑1>,
    "Objects": <float 0‑1>,
    "Action":  <float 0‑1>,
    "Risk":    <float 0‑1>
  },
  "justification": {
    "Scene":   "<short sentence>",
    "Objects": "<short sentence>",
    "Action":  "<short sentence>",
    "Risk":    "<short sentence>"
  },
  "PRIMARY_ERROR": "Scene | Objects | Action | Risk | None"
}"""

user_prompt = """REASONING_TRACE:
<pred>

GROUND_TRUTH_LABEL: 
<label>

Please evaluate according to the JSON schema."""



random.seed(233)

def parse_args():
    parser = argparse.ArgumentParser(description="Testing Script")
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--result_file', type=str, default=None)
    args = parser.parse_args()
    return args

args = parse_args()

import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
"""
python score.py \
    --data_file ./workspace/data/drivebench/results/qwen2.5_7b_image_masked.json \
    --result_file ./workspace/data/drivebench/results/qwen2.5_7b_image_masked_scores_llama.json 

python score.py \
    --data_file ./workspace/data/drivebench/results/internvl3_8b_image_masked.json \
    --result_file ./workspace/data/drivebench/results/internvl3_8b_image_masked_scores.json 

python score.py \
    --data_file ./workspace/data/drivebench/results/gpt-4o_image_masked.json \
    --result_file ./workspace/data/drivebench/results/gpt-4o_image_masked_scores.json 

python score.py \
    --data_file ./workspace/data/drivebench/results/llamavision_image_masked.json \
    --result_file ./workspace/data/drivebench/results/llamavision_image_masked_scores.json 

python score.py \
    --data_file ./workspace/data/drivebench/results/qwen2.5_7b_image_orignal.json \
    --result_file ./workspace/data/drivebench/results/qwen2.5_7b_image_orignal_scores.json 

python score.py \
    --data_file ./workspace/data/nexar/results/v1/qwen2.5_7b_._models_vpd-vl_ft_4_epochs_lr5e-06_qwen2.5-vl_0,1,2,3_step_63_image_masked.json \
    --result_file ./workspace/data/nexar/results/v1/qwen2.5_7b_._models_vpd-vl_ft_4_epochs_lr5e-06_qwen2.5-vl_0,1,2,3_step_63_image_masked_scores.json


    
"""

if __name__ == "__main__":
    with open(args.data_file, "r") as f:
        data = json.load(f)

    print(len(data))
    data = [line for line in data if line["pred"].split("</think>")[-1].lower().strip() == line['label'] or ()]
        
    safe = [line for line in data if line['label'] == "safe"]
    unsafe = [line for line in data if line['label'] == "unsafe"]

    data = safe[:200] + unsafe[:200]

    
    print(len(data))
    eval_model = APIModel("claude3.7sonnet")
    
    for line in tqdm(data):

        image_list = []
        filename = line['image'][-1].split("/")[-1].split(".")[0]
        media_path = f"./workspace/data/nexar/images/{line['image'][-1].split('/')[-2]}/{filename}/"
        for image_path in line['image']:
            image_name = image_path.split("/")[-1]
            image_list.append(os.path.join(media_path, image_name))

        content = []
        for image_path in image_list[-4:]:
            tmp_path = "/".join(image_path.split("/")[:-1] + ["masked", image_path.split("/")[-1]])
            base64_image = encode_image(tmp_path)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                }
            )

        content.append({
            "type": "text",
            "text": user_prompt.replace("<pred>", line['pred']).replace("<label>", line['label'])
            }
        )
        conversation = [
            {
                "role": "user",
                "content": content
            }
        ]
     
        system = evaluation_prompt
        pred = eval_model.generate(system, conversation)
        print(line['id'])
        print(pred)
        example = {k: v for k, v in line.items() if "image" not in k and "objects" not in k}
        example['score'] = pred
        scores['llm-eval'].append(example)
        with open(args.result_file, "w") as f:
            f.write(json.dumps(scores['llm-eval']))
            

  