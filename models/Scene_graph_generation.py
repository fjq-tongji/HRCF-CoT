from PIL import Image
import torch
import time
import re
from typing import Dict
import json
import ast



prompts = '''
Task:
Given an image, you need to only identify 3 critical objects involved in the image and extract a partial scene graph in JSON 
format of these objects. 
This graph should include the following elements:
1. 1-3 Critical objects ahead in the image.
2. Position and status of these objects.
3. Relationships between these objects.
Do not repeatedly extracting the same object. Please only return valid json format, use double quotation marks, and do not add explanatory language.
The structure of the output JSON scene graph should be the following format:

Example:
{
"objects": [
    {"id": 1, "type": "traffic_light", "position": "above the intersection", "status": "red"},
    {"id": 2, "type": "streetlight", "position": "along the street", "status": "on"},
    {"id": 3, "type": "vehicle", "position": "in the distance on the road", "status": "stationary"}
  ],
"relationships": [
{"source": 1, "target": 2, "relation": "controlling"},
{"source": 2, "target": 3, "relation": "next_to"},
  ]
}


Input:
Please construct a scene image for the given image.
'''





def scene_graph_generation(model, image):
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    max_partition = 9
    #########################################################
    images = [image]
    text_ = prompts
    query = f'<image>\n{text_}'
    # format conversation
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]

    # generate output
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=256,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        text_critic_ovis_dict_ = text_tokenizer.decode(output_ids, skip_special_tokens=True)

    return text_critic_ovis_dict_




##############################################################################################################################
##############################################################################################################################
class Scene_graph_generation:
    def __init__(self, args):
        self.args = args

    def scene_graph_generation_(self, model_ovis_2, sample: Dict):
        time_1 = time.time()

        image_path_ = sample['image']
        image_path_ = '/share/home/MLLMS/CODA2022/images_total/' + image_path_.split('/')[-1]
        image = Image.open(image_path_)


        # 1. 生成场景图
        scene_graph_str = scene_graph_generation(model_ovis_2, image)
        print(scene_graph_str)
        print('---------------------------------------')

        # 2. 字符串转化为字典格式保存
        scene_graph_dict = json.loads(scene_graph_str)  # 关键转换步骤


        sample['Scene_graph'] = scene_graph_dict

        time_2 = time.time()
        print('SG模块的推理时间为：')
        print(time_2 - time_1)

        return sample











