import json
import os
from tqdm import tqdm
from api.api_models import APIModel
import base64



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def describe_relative_position(x, y):
    if abs(x) < 1 and abs(y) < 1:
        return "directly ahead"
    position = ""
    if x >= 1:
        position += "front"
    elif x <= -1:
        position += "rear"
    
    if y >= 1:
        position += "-left" if position else "left"
    elif y <= -1:
        position += "-right" if position else "right"
    
    return position if position else "nearby"

# system_prompt = """
# You are a driving policy reasoning assistant. Given frames from the ego car's front-facing camera view and metadata, your task is to:

# 1. Scene Description:
# Write one paragraph to describe the overall driving scenario shown in the given frame. The frame represents the front-facing camera view from the ego vehicle. Your description should include observable factors such as weather, time of day, traffic conditions, road structure (e.g., number of lanes, presence of dividers or crosswalks), and surrounding environment (e.g., buildings, parked vehicles, pedestrians). Avoid bullet points or vague phrases. Focus on visible details that help characterize the scene from the driver’s perspective.

# 2. Critical Objects:
# Identify all dynamic agents in the scene that could influence the behavior of the ego vehicle. For each object, provide its ID and 2D bounding box in the format: object <id> [x1, y1, x2, y2]. Then, based on the provided relative 3D coordinates, describe the object’s position relative to the ego vehicle (e.g., directly in front, rear-right, to the left, etc.). Only include objects that are on the road and have the potential to interact with the ego vehicle’s future motion.


# 4. Meta Driving Decision:
# Based solely on the visual scene shown in the given front-facing frame (ignore any future behavior information provided earlier in Metadata), predict:
# - The most reasonable high-level driving action the ego vehicle should take next
# - The likely future behavior of each critical object in the scene
# - Speed-related decision: Choose the most appropriate action based on the ego vehicle’s distance from surrounding traffic, traffic flow, and any potential need to slow down or stop. Select one from:
# [accelerate, slow down, stop, follow vehicle]
# - Lane-related decision: Choose the most appropriate action based on lane layout, surrounding vehicles, obstacles, and road features. Select one from:
# [keep lane, change lane left, change lane right, yield, overtake, turn left, turn right, avoid object]

# Output Format:
# ego car: <future behavior, including speed and lane decision>
# object [id]: <future behavior, including speed and lane decision>
# explanation: For each object, describe its anticipated short-term behavior (e.g., maintain lane, prepare to merge, slow down, accelerate, change lane). Then, write a paragraph justifying the ego vehicle’s decision based on the scene context, traffic structure, and safe driving practices. Use visible factors (e.g., gaps, congestion, intersection proximity) to support your predictions.

# 4. Behavioral Conflict & Risk Analysis:
# You will be given the future behavior of both the ego vehicle and one object (e.g., accelerate and keep lane vs. accelerate and left lane-change), as well as a label indicating whether the scenario is considered safe or unsafe. Analyze whether there is a potential behavioral conflict between the two agents. Consider spatial proximity, motion trajectories, timing of maneuvers, and visibility (e.g., blind spots). Clearly explain how these factors contribute to the risk level.
# - Conclude the analysis with a paragraph starting with: “Safe/Unsafe:”
# - Conclude the analysis with a paragraph starting with: “Specific traffic policy:”
# In this paragraph, describe relevant traffic laws, defensive driving principles, or official guidance **without referring to any specific vehicles or behaviors in this scenario**. The description should reflect what is generally required or prohibited in the given region (e.g., U.S., EU, China) when it comes to actions such as lane changes, yielding, maintaining following distance, or handling merging conflicts. This allows the reasoning to remain modular and adaptable across different jurisdictions.
# """


system_prompt = """You are a driving-policy reasoning assistant.

Input:
  • 8 consecutive front-camera frames from the ego vehicle, sampled every 0.5 s or a single frame:
      <IMAGE_t0> … <IMAGE_t7>  (t0 is earliest, t7 is latest)
  • risk_label = <RISK_LABEL>  ← either "collision_risky" or "safe"
    (do NOT reveal this label in your answer)

Analyse the temporal sequence and complete the four steps below **in order**.
Write concise, professional English; do not refer to individual frame numbers or timestamps in your answer.

────────────────────────────────────────
1. Scene Understanding
Summarise, in one short paragraph (4–5 sentences), how the scene changes from t0 to t7: weather/light, road layout, traffic density, and any static context. Focus on what is visible; do not speculate beyond the frames.

2. Key Dynamic Objects
List every moving agent that could influence the ego vehicle.
For each agent, give:
  object <n> — relative position (use terms like “directly ahead”, “ahead-left”, “rear-right”, etc.) and a short descriptor (car, truck, pedestrian, cyclist, etc.), and its past behavior (last 4s) based on the given frames.
Example: object 1 — directly ahead, white sedan. It was slowing down in the last 4s.

3. Driving Decision Prediction  
Predict each agent’s most likely behaviour over the next ~1.5 s (≈3 frames ahead).  
Choose for the ego vehicle and dynamic objects:
  • Speed action ∈ {accelerate | slow down | stop | follow vehicle}
  • Lateral action ∈ {keep lane | change lane left | change lane right | yield | overtake | turn left | turn right | avoid object}

Predict each listed object’s short-term behaviour (e.g., keep lane, merge left, brake).

Output format:
ego: <speed action>, <lateral action>
object <n>: <speed action>, <lateral action>

4. Risk Assessment & Justification  
Select **the single car** most likely to create a collision risk with the ego vehicle. (based on Steps 2–3).  
In 4–6 sentences, analyse the future interaction between the ego and that object, citing relative positions, trajectory convergence, speed differences, and visibility constraints observed across the frames.  
Conclude with exactly one of the following lines (matching risk_label but **without mentioning or revealing it**):  
Assessment: Safe.  
Assessment: Unsafe.

────────────────────────────────────────
Return the four steps separated by blank lines, nothing else."""


cf_system_prompt = """You are a driving-policy reasoning assistant.

Input:
  • 8 consecutive front-camera frames from the ego vehicle, sampled every 0.5 s or a single frame:  
      <IMAGE_t0> … <IMAGE_t7>  (t0 is earliest, t7 is latest)  
  • counterfactual_behavior = {                       ← provided by the user  
        "ego":       "<ego_long_beh>, <ego_lat_beh>",  
        "object 1":  "<agent_long_beh>, <agent_lat_beh>"  
    }  
  • risk_label = <RISK_LABEL>   ← “collision_risky” or “safe”  
    (do NOT reveal this label in your answer)

Analyse the temporal sequence and complete the four steps below **in order**.  
Write concise, professional English; do not mention frame numbers or timestamps.

────────────────────────────────────────
1. Scene Understanding  
Summarise, in one short paragraph (4-5 sentences), how the scene evolves from t0 to t7: weather/light, road layout, traffic density, and notable static context. Describe only what is visible; do not speculate beyond the frames.

2. Key Dynamic Objects  
List every moving agent that could influence the ego vehicle.  
For each agent, provide:  
  object <n> — relative position (e.g. “directly ahead”, “ahead-left”, “rear-right”), a brief descriptor (car, truck, pedestrian, cyclist, …), and its observed behaviour over the last 4 s.  
Example: object 1 — directly ahead, white sedan. It was slowing down over the last 4 s.

3. Driving Decision Prediction  
For **ego** and **object 1**, output **exactly** the actions given in *counterfactual_behavior* (do not alter them).  
For every other listed agent, predict its most likely behaviour for the next ≈ 1.5 s (≈ 3 frames) choosing from:  
  • Speed action ∈ {accelerate | slow down | stop | follow vehicle}  
  • Lateral action ∈ {keep lane | change lane left | change lane right | yield | overtake | turn left | turn right | avoid object}

Output format:  
ego: <speed action>, <lateral action>           ← must match counterfactual_behavior  
object 1: <speed action>, <lateral action>      ← must match counterfactual_behavior  
object <n>: <speed action>, <lateral action>    ← predicted for others

4. Risk Assessment & Justification  
Using the counterfactual actions of the ego and object 1 from Step 3, select **one vehicle** most likely to pose a collision risk to the ego vehicle.  
In 4-6 sentences, analyse the future interaction between the ego and that object, referencing relative positions, trajectory convergence, speed differences, and visibility constraints observed across the frames.  
End with exactly one of the following lines (matching risk_label but **without revealing it**):  
Assessment: Safe.  
Assessment: Unsafe.

────────────────────────────────────────
Return the four steps separated by blank lines, and nothing else."""


if __name__ == "__main__":
        
    with open("./workspace/data/vpd-sft/nuplan_train_v1.json", "r") as f:
        data = json.load(f)
        
    with open("./workspace/data/vpd-sft/nuplan_reasoning.json", "r") as f:
        reasoning_data = json.load(f)
    
    remove_ids = [500, 541, 550, 720, 855, 995, 1053]
    for i in sorted(remove_ids, reverse=True):
        reasoning_data.pop(i)
            
            
    reasoning_data = [line['id'] + line['ego_lat_beh'] + line['ego_long_beh'] + line['agent_lat_beh'] + line['agent_long_beh'] for line in reasoning_data]
    print(len(data))
    
    model = APIModel("claude3.7sonnet")
    
    for i, line in tqdm(enumerate(data)):
        # if "2021.06.23.17.31.36_veh-16_00016_00377=f9cb22e728e15da2_s10" not in line['image'][-1]: continue
        unique_id = line['id'] + line['ego_lat_beh'] + line['ego_long_beh'] + line['agent_lat_beh'] + line['agent_long_beh']
        if unique_id in reasoning_data: continue
        if line['label'] == "unsafe":
            meta_info = """risk_label = collision_risky"""
        else:
            meta_info = """risk_label = safe"""

     
        image = line['image']
        content = []
        
        if line['type'] == "counterfactual":
            prompt = cf_system_prompt.replace("<ego_lat_beh>", line['ego_lat_beh']).replace("<ego_long_beh>", line['ego_long_beh'])
            prompt = prompt.replace("<agent_lat_beh>", line['agent_lat_beh']).replace("<agent_long_beh>", line['agent_long_beh'])
            image = [image[-1]]
        else:
            prompt = system_prompt
        
        for image_path in image:
            image = image[-4:]
            filename = image[-1].split("/")[-1].rstrip(".jpg")
            tmp_path = "/".join(image_path.split("/")[:-1] + [str(filename).zfill(4), "masked", image_path.split("/")[-1]])
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
            
            # content.append(
            #     {
            #         "type": "image_url",
            #         "image_url": {
            #             "url": f"data:image/jpeg;base64,{base64_image}"
            #         }
            #     }
            # )
        
        content.append({
            "type": "text",
            "text": meta_info
            }
        )
  
        conversation = [
            {
                "role": "user",
                "content": content
            }
        ]

        print(prompt)
        pred = model.generate(prompt, conversation)
        print(pred)
        
        with open("./workspace/data/vpd-sft/nuplan_reasoning.json", "r") as f:
            outputs = json.load(f)
        
        for i in sorted(remove_ids, reverse=True):
            outputs.pop(i)
    
        line['reasoning_process'] = pred
        outputs.append(line)
        
        with open("./workspace/data/vpd-sft/nuplan_reasoning.json", "w") as f:
            f.write(json.dumps(outputs))
