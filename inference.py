from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from huggingface_hub import hf_hub_download
import json
import os
from PIL import Image

model_path = "dadsdasdsa/Qwen2.5-VL_VPD-SFT"
media_type = "image"

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    # device_map="auto",
)
model.cuda()

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

with open("./workspace/data/vpd-sft/waymo/test.json", "r") as f:
    data = json.load(f)
    
# you can set your own prompt for testing
prompt = """You are a driving scene reasoning assistant. Based on the provided multi frames of ego car's front camera view and traffic rules, analyze the potential behavioral conflict between ego car and other agents, and then assess risk.

Analyze the impact of the ego vehicle's future behavior and other agents' future behaviors.  
Consider potential risks such as collision, reduced reaction time, or merging conflicts.  

Provide your reasoning in three steps:
1. Describe the scene, past behavior and relative position of object [ID].  
2. Analyze the interaction between the ego vehicle and object [ID].  
3. Assess the overall risk and asssign a safety score (0 = extremely unsafe, 1 = completely safe)."""


scenes = {}
for line in data:

    image_list = line['image']
    if "605a964cc30b61fedf8a41bdf130f505-15" not in line['id']: continue
    # print(line)
    scene_id = image_list[-1].split("/")[-2]
    image_path = os.path.join("/".join(image_list[-1].split("/")[:-1]), image_list[-1].split("/")[-1].split(".")[0], "masked")
    video_path = [os.path.join(image_path, item.split("/")[-1].replace("png", "jpg")) for item in image_list]
    # output_dir = os.path.join("./workspace/data/examples/waymo", scene_id)
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    # else:
    #     continue
            
    if scene_id not in scenes.keys():
        scenes[scene_id] = True
    else:
        continue
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {
                    "type": "text", 
                    "text": prompt
                },
            ],
        },
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    

    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])

    for img in video_path:
        image = Image.open(img)
        output_dir = os.path.join("./workspace/data/examples/waymo", scene_id)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_path = os.path.join(output_dir, img.split("/")[-1])
        image.save(output_path)
        
    with open(os.path.join(output_dir, "vpd-lm.txt"), "w") as f:
        f.write(output_text[0])

    break
        



