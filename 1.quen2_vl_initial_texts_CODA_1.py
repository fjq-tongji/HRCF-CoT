from PIL import Image
import torch
import time
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from torch.utils.data import Dataset, DataLoader


def Qwen2_VL_eval_model(model, processor, image, prompt):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    # Inference: Generate the output
    output_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    return output_text



#############################################################################################################################
#############################################################################################################################
if __name__ == '__main__':
    coco_imgs_file = '/share/home/u21012/fjq/MLLMS/CODA2022/images_total/'
    coco_random_json_file = '/share/home/u21012/fjq/MLLMS/POPE/CODA2022_pope_random.json'
    output_json_file = './CODA_Qwen2_VL_initial_texts.json'

    ##所有的coco_json文件，存到列表中
    coco_random_questions = []
    with open(coco_random_json_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                coco_random_questions.append(item)

    ##所有的图片和问题，分别存到两个列表中
    prompts = []
    image_list = []
    for i in range(len(coco_random_questions)):
        if i % 6 == 0:
            prompt = 'Generate a paragraph describing the current driving scenario.'
            img = coco_imgs_file + coco_random_questions[i]['image']
            prompts.append(prompt)
            image_list.append(img)

    # image_list = image_list[2500:2502]     ###2500:5000
    # prompts = prompts[2500:2502]


    ############### 加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "/share/home/u21012/fjq/MLLMS/0_MLLM_weights/Qwen2-VL-7B-Instruct", torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        "/share/home/u21012/fjq/MLLMS/0_MLLM_weights/Qwen2-VL-7B-Instruct")

    ############### 生成文本描述
    answers = []
    for k in range(len(image_list)):   #### k=0,1,2,3,...
        time_1 = time.time()
        each_img = image_list[k]
        image = Image.open(each_img)
        each_prompt = prompts[k]
        answer = Qwen2_VL_eval_model(model, processor, image, each_prompt)
        answers.append(answer)
        # print(each_img)
        print(answer)
        time_2 = time.time()
        print(time_2 - time_1)

    ##将所有的问题和答案存入一个字典中，再写入一个json中
    final_lst = []
    for j in range(len(answers)):     #### j=0,1,2,3,...
        dict = {}
        dict['image'] = image_list[j]
        dict['initial_texts'] = answers[j]
        final_lst.append(dict)

    #####################################################################
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(final_lst, f, indent=4, ensure_ascii=False)







