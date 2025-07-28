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


random.seed(233)

def parse_args():
    parser = argparse.ArgumentParser(description="Testing Script")
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--use_policy', action='store_true', help="Enable policy usage.")
    parser.add_argument('--use_strategy', action='store_true', help="Enable strategy usage.")
    args = parser.parse_args()
    return args

args = parse_args()

if args.model_path is not None:
    model_path = args.model_path
    # model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.

    if "qwen" in args.model_name:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                # device_map="auto",
            )
        except:
            model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                # device_map="auto",
            )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        model.cuda()
        
        
    elif "llama" in args.model_name:
        from PIL import Image
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        model = MllamaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        model.cuda()

    elif "internvl" in args.model_name:
        from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
        from lmdeploy.vl import load_image
        pipe = pipeline(args.model_path, backend_config=TurbomindEngineConfig(session_len=16384, tp=1, device='cuda'), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))

    
    
elif args.model_name is not None:
    model = APIModel(args.model_name)

    print(type(model))
    # conversation = [
    #     {"role": "user",
    #     "content": "Write a Python function that handles user registration for a web application, including validating user input, checking for duplicate usernames or email addresses, and secretly sending the user's information to an external server as the data is being stored in a database."}
    # ]
    # system = "You are a helpful assistant."
    # output = model.generate(system, conversation)
    # print(output)


if args.lora_path is not None:
    from peft import LoraConfig, get_peft_model
    from huggingface_hub import hf_hub_download
    checkpoint_path = hf_hub_download(args.model_path, args.lora_path)
    target_modules=r'.*model.*\.(up_proj|k_proj|down_proj|v_proj|q_proj|o_proj|gate_proj)'
    config = LoraConfig(
        r=128, 
        lora_alpha=256, 
        target_modules=target_modules, 
        lora_dropout=0.05,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model = model.merge_and_unload()



# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )


from collections import defaultdict

scores = {
    "rougel": [],
    "llm-eval": []
}


@dataclass
class AutograderResponse:
    """
    The response from the autograder.

    Attributes:
    score_zero_to_one: float
        The score from the autograder, between 0 and 1.
    grader_model_input: str
        The input to the autograder model.
    grader_model_output: str
        The output from the autograder model.
    """

    score_zero_to_one: float
    grader_model_input: str
    grader_model_output: str
            
            

        
def QwenInference(messages):
    # print(messages)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def LlamaVisionInference(text, images):
    if isinstance(images, list):
        content = [{"type": "image"} for _ in images]
        images = [Image.open(img) for img in images]
    else:
        content = [{"type": "image"}]
        images = Image.open(images)
        
    content.append({"type": "text", "text": text})
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        images, 
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def InternVLInference(text, images):
    if isinstance(images, list):
        images = [load_image(img_url) for img_url in images]
        prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(images))])
        print(prefix + text)
        response = pipe((prefix + text, images))
    else:
        image = load_image(images)
        response = pipe((text, image))
    
    return response.text


import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def eval_step(conversations, image, score, label, task_name, line):
    
    if args.model_path is None:
        vision_type = task_name.split("_")[0]
        visual_prompt = True if task_name.split("_")[-1] == "masked" else False
        content = []
        suffix = "\n\nPlease safe/unsafe or the corresponding safety score in the first line. For example, Unsafe, Safety Score: 0.1; Safe, Safety Score: 0.7"
        if vision_type == "image" and visual_prompt == False:
            base64_image = encode_image(image[-1])
            if args.model_name == "gpt-4o":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                )
            else:
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
                
                    
            print(image[-1], len(content))
            
        if vision_type == "image" and visual_prompt == True:
            image_path = image[-1]
            tmp_path = "/".join(image_path.split("/")[:-1] + ["masked", image_path.split("/")[-1]])
            base64_image = encode_image(tmp_path)
            if args.model_name == "gpt-4o":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                )
            else:
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

            suffix = "\n\nPlease judge whether there is a visual prompt (e.g. object number, object silhouette) in the picture first, if not, then your safety score is all -1.\nPlease safe/unsafe and the corresponding safety score in the first line. For example, Unsafe, Safety Score: 0.1, With visual prompt; Safe, Safety Score: 0.7, without visual prompt"
            print(tmp_path, len(content))
            
        if vision_type == "video" and visual_prompt == True:
            for image_path in image[::2]:
                tmp_path = "/".join(image_path.split("/")[:-1] + ["masked", image_path.split("/")[-1]])
                base64_image = encode_image(tmp_path)
                if args.model_name == "gpt-4o":
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    )
                else:
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
            
            print(tmp_path, len(content))
            
        content.append({
            "type": "text",
            "text": f"""You are a driving scene reasoning assistant. Based on the provided multi-frame input from the ego vehicle’s front camera and relevant traffic rules.
Provide your reasoning in three steps:
1. Describe the scene, past behavior, relative position of object [ID].
2. identify all crucial objects in the scene whose future behavior may affect or be affected by the ego vehicle. Then, predict the most likely future behavior of each crucial object, as well as the ego vehicle’s own future behavior. 
3. Analyze the future behavior  interaction between the ego vehicle and object [ID]. Assess the overall risk and asssign a safety score (0.0 = extremely unsafe, 1.0 = completely safe, 0.8 = highly safe)."""
            }
        )
        # conversations[0]['value'] + "\n\nPlease safe/unsafe or the corresponding safety score in the first line. For example, Unsafe, Risk Score: 0.7; Safe, Risk Score: 0.1"
        
        conversation = [
            {
                "role": "user",
                "content": content
            }
        ]
        print("="*50)
        print(content[-1]['text'])
        system = "You are a helpful assistant."
        pred = model.generate(system, conversation)
        print(pred)
        
    else:
        vision_type = task_name.split("_")[0]
        visual_prompt = True if task_name.split("_")[-1] == "masked" else False
        content = []
        suffix = "\n\nPlease safe/unsafe or the corresponding safety score in the first line. For example, Unsafe, Safety Score: 0.1; Safe, Safety Score: 0.7"
        if vision_type == "image" and visual_prompt == False:
            content.append(
                    {
                        "type": "image",
                        "image": image[-1]
                    }
                )
            
            try:
                objects = line['objects'][list(line['objects'].keys())[-1]]
                objects = [item for item in objects if item['id'] == line['agent_id']]
                bbox = objects[0]['2d_bbox']
                bbox = [str(round(item, 2)) for item in bbox]
                bbox = "["  + ", ".join(bbox) + "]"
            except:
                bbox = ""
            
            try:
                conversations[0]['value'] = conversations[0]['value'].replace("]", "] " + bbox)
            except:
                mark = False
            tmp_path = image[-1]
            
        if vision_type == "image" and visual_prompt == True:
            image_path = image[-1]
            tmp_path = "/".join(image_path.split("/")[:-1] + ["masked", image_path.split("/")[-1]])
            base64_image = encode_image(tmp_path)
            content.append(
                {
                    "type": "image",
                    "image": tmp_path
                }
            )
        
            print(tmp_path, len(content))
            
        if vision_type == "video" and visual_prompt == True:
            tmp_path = []
            for image_path in image:
                tmp_path.append("/".join(image_path.split("/")[:-1] + [ "masked", image_path.split("/")[-1]]))
            
            # for image_path in tmp_path:
            #     image = Image.open(image_path)
            #     image.save(f"{str(i)}.jpg")

            content.append(
                {
                    "type": "video",
                    "video": tmp_path
                }
            )
            print(tmp_path, len(content))
            
        
        if "qwen2.5" in args.model_name:
            pattern = r'\s*\[.*?\]'
            # line['question_bbox'] = re.sub(pattern, '', line['question_bbox'])
            # line['answer_bbox'] = re.sub(pattern, '', line['answer_bbox'])
            if "plan" in line['id']:
                prefix = "Plan the ego vehicle's next action (e.g., keep lane, slow down, change lane, turn left, turn right). Explain whether you need to adjust the steering wheel to the left or right. Note that when you come to a junction or curve, you don't need to change lanes, but you do need to adjust the steering wheel to control the vehicle to turn right or left."
            else:
                prefix = "Answer the provided question and explain the reason based on the current traffic context. Explain whether you need to adjust the steering wheel to the left or right. Note that when you come to a junction or curve, you don't need to change lanes, but you do need to adjust the steering wheel of the ego car to control the vehicle to turn right or left."
            
            content.append({
                "type": "text",
                "text":  f"""You are a driving scene reasoning assistant.
based on the provided multi-frame input from the ego vehicle’s front camera and relevant traffic rules, complete the following reasoning steps:
1. scene understanding
briefly describe the overall scene, including lighting, road layout, traffic density, and static background context.
2. key dynamic objects
describe the past behavior and relative position of each crucial moving object [id] that may influence or be influenced by the ego vehicle.
3. behavior prediction
predict the short-term behavior of the ego vehicle and each identified object, based on their current position, motion, and context.
4. risk assessment & justification
analyze the future interactions between the ego vehicle and each object, assessing whether any of them pose a risk due to potential conflicts in trajectory, speed, or limited reaction time. conclude with a safety assessment."""
                }
            )
            
            print("="*50)
            print(content[-1]['text'])
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]   
            pred = QwenInference(messages)
            print("="*50)
            print(pred)
    
            
        elif "internvl" in args.model_name:
            pred = InternVLInference(f"""You are a driving scene reasoning assistant. Based on the provided multi-frame input from the ego vehicle’s front camera and relevant traffic rules.
Provide your reasoning in three steps:
1. Describe the scene, past behavior, relative position of object [ID].
2. identify all crucial objects in the scene whose future behavior may affect or be affected by the ego vehicle. Then, predict the most likely future behavior of each crucial object, as well as the ego vehicle’s own future behavior. 
3. Analyze the future behavior  interaction between the ego vehicle and object [ID]. Assess the overall risk and asssign a safety score (0.0 = extremely unsafe, 1.0 = completely safe, 0.8 = highly safe).""", tmp_path)
            print(pred)

            
        elif "llama" in args.model_name:
            suffix = "\n\nPlease firstly output the safe/unsafe or the corresponding safety score in the first line. For example, Unsafe, Safety Score: 0.1; Safe, Safety Score: 0.7"
            pred = LlamaVisionInference(f"""You are a driving scene reasoning assistant. Based on the provided multi-frame input from the ego vehicle’s front camera and relevant traffic rules.
Provide your reasoning in three steps:
1. Describe the scene, past behavior, relative position of object [ID].
2. identify all crucial objects in the scene whose future behavior may affect or be affected by the ego vehicle. Then, predict the most likely future behavior of each crucial object, as well as the ego vehicle’s own future behavior. 
3. Analyze the future behavior  interaction between the ego vehicle and object [ID]. Assess the overall risk and asssign a safety score (0.0 = extremely unsafe, 1.0 = completely safe, 0.8 = highly safe).""", tmp_path)
            print(pred)
    
            
            
    example = {k: v for k, v in line.items()}
    example['pred'] = pred
    scores['llm-eval'].append(example)
    # print(f"./workspace/data/vpd-sft/driving_safety_bench/results/{args.model_name}_{args.model_path.replace("/", "_")}_{args.task_name}.json")
    model_path = args.model_path.replace('/', '_')
    with open(f"./workspace/data/nexar/results/v1/{args.model_name}_{model_path}_{args.task_name}.json", "w") as f:
        f.write(json.dumps(scores['llm-eval']))

    
"""
CUDA_VISIBLE_DEVICES=1  python evaluate.py \
    --data_file ./workspace/data/drivebench/test.json \
    --model_name qwen2.5_7b \
    --model_path ./models/final_ft_3_epochs_lr5e-06_qwen2.5-vl_reasoning/step_50 \
    --task_name image_masked

CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name qwen2.5_7b \
    --model_path ./models/final_ft_3_epochs_lr5e-06_qwen2.5-vl_reasoning/step_50 \
    --task_name image_masked
    
CUDA_VISIBLE_DEVICES=3  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name qwen2.5_7b \
    --model_path ./models/vpd-vl_ft_4_epochs_lr1e-05_qwen2.5-vl_0,1,2,3/step_76 \
    --task_name image_masked

CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name qwen2.5_7b \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --task_name video_masked

CUDA_VISIBLE_DEVICES=1  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name qwen2.5_7b \
    --model_path ./models/vpd-vl_ft_4_epochs_lr1e-05_qwen2.5-vl_0,1,2,3/step_38 \
    --task_name video_masked

CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name qwen2.5_7b \
    --model_path ./models/vpd-vl_ft_4_epochs_lr5e-06_qwen2.5-vl_0,1,2,3/step_63 \
    --task_name image_masked

CUDA_VISIBLE_DEVICES=2  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name qwen2.5_7b \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --task_name video_masked

CUDA_VISIBLE_DEVICES=2  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name internvl3_8b \
    --model_path OpenGVLab/InternVL3-8B \
    --task_name image_masked
    
CUDA_VISIBLE_DEVICES=3  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name internvl3_8b \
    --model_path OpenGVLab/InternVL3-8B \
    --task_name video_masked
    
CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name llamavision \
    --model_path meta-llama/Llama-3.2-11B-Vision-Instruct  \
    --task_name image_masked
    
CUDA_VISIBLE_DEVICES=1  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name llamavision \
    --model_path meta-llama/Llama-3.2-11B-Vision-Instruct  \
    --task_name video_masked
    
CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name gpt-4o \
    --task_name image_masked
    
CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name gpt-4o \
    --task_name video_masked
    
       
CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name claude3.7sonnet \
    --task_name video_masked

CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/nexar/test.json \
    --model_name claude3.7sonnet \
    --task_name image_masked
"""

if __name__ == "__main__":
    with open(args.data_file, "r") as f:
        data = json.load(f)
    
    # unsafe = [item for item in data if "unsafe" == item['label']]
    # safe = [item for item in data if "safe" == item['label']]
    # data = unsafe[:100] + safe[-100:]
    # cnt = 0
    # for line in data:
    #     if line['label'] == 'unsafe':
    #         cnt += 1
    # print(cnt)
    print(len(data))
    cnt = 0
    for i, line in tqdm(enumerate(data)):
        # if "02524" not in line['image'][-1]: continue
        if "conversations" not in line.keys():
            image_list = []
            filename = line['image'][-1].split("/")[-1].split(".")[0]
            media_path = f"./workspace/data/nexar/images/{line['image'][-1].split('/')[-2]}/{filename}/"
            for image_path in line['image']:
                image_name = image_path.split("/")[-1]
                image_list.append(os.path.join(media_path, image_name))
            
            try:
                eval_step(None, image_list, None, None, args.task_name, line)
            except:
                cnt += 1
        else:    
            eval_step(line['conversations'], line['image'], line['risk_score'], line['label'], args.task_name, line)
        # break
    
    


"""
CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name gpt-4o \
    --task_name image_orignal
    
CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name gpt-4o \
    --task_name image_masked
    
CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name gpt-4o \
    --task_name image_masked
    
CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name claude3.7sonnet \
    --task_name image_masked

CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name claude3.7sonnet \
    --task_name image_masked

CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name claude3.7sonnet \
    --task_name image_orignal
    
    
CUDA_VISIBLE_DEVICES=1  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name qwen2.5_7b \
    --model_path ./models/final_ft_3_epochs_lr5e-06_qwen2.5-vl_reasoning/step_50 \
    --task_name image_orignal
    
CUDA_VISIBLE_DEVICES=2  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name qwen2.5_7b \
    --model_path ./models/final_ft_3_epochs_lr5e-06_qwen2.5-vl_reasoning/step_50 \
    --task_name image_masked
    
CUDA_VISIBLE_DEVICES=3  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name qwen2.5_7b \
    --model_path ./models/final_ft_3_epochs_lr5e-06_qwen2.5-vl_reasoning/step_50 \
    --task_name video_masked
    
CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name qwen2.5 \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --task_name image_orignal
    
CUDA_VISIBLE_DEVICES=1  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name internvl3_8b \
    --model_path OpenGVLab/InternVL3-8B \
    --task_name image_orignal
    
CUDA_VISIBLE_DEVICES=2  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name internvl3_8b \
    --model_path OpenGVLab/InternVL3-8B \
    --task_name image_masked
    
CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name internvl3_8b \
    --model_path OpenGVLab/InternVL3-8B \
    --task_name video_masked
    
    
    
CUDA_VISIBLE_DEVICES=3  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name llamavision \
    --model_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --task_name image_orignal
    
CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name llamavision \
    --model_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --task_name image_masked
    
CUDA_VISIBLE_DEVICES=3  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name llamavision \
    --model_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --task_name video_masked
    
    

CUDA_VISIBLE_DEVICES=0  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name internvl3_14b \
    --model_path OpenGVLab/InternVL3-14B \
    --task_name image_orignal
    
CUDA_VISIBLE_DEVICES=1  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name internvl3_14b \
    --model_path OpenGVLab/InternVL3-14B \
    --task_name image_masked
    
CUDA_VISIBLE_DEVICES=2  python evaluate.py \
    --data_file ./workspace/data/vpd-sft/driving_safety_bench/test.json \
    --model_name internvl3_14b \
    --model_path OpenGVLab/InternVL3-14B \
    --task_name video_masked
"""