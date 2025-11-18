from PIL import Image
import torch
import time
import re
from typing import Dict
from sentence_transformers import util


def traffic_element_extraction(refined_texts):
    possible_traffic_elements_lst = [                     ### 定义13类需要从initial_texts中提取的实体
        ["traffic light", "traffic lights"],
        ["traffic cone", "traffic cones"],
        ["sidewalk", "sidewalks", "pavement", "pavements"],
        ["dog", "dogs", "animal", "animals"],
        ["cat", "cats", "animal", "animals"],
        ["person", "persons", "adult", "adults", "child", "children"],      ###"pedestrian", "pedestrians"
        ["cyclist", "cyclists", "bicyclist", "bicyclists", "motorcyclist", "motorcyclists", "biker", "bikers"],       ###骑车的人，不区分具体骑的是什么车
        # ["construction worker", "construction workers", "construction_worker", "worker", "workers", "police_officer", "police officer", "police officers"],
        ["wheelchair", "wheelchairs", "stroller", "strollers"],    ####其他类型的人，占位符
        ["car", "cars"],
        ["truck", "trucks", "trailer", "trailers", "construction vehicle", "construction vehicles"],
        ["bus", "buses"],
        ["bicycle", "bicycles", "bike", "bikes", "motorcycle", "motorcycles", "motorbike", "motorbikes"],     ####低速度的车，不区分自行车还是摩托车
        ["emergency vehicle", "emergency vehicles", "emergency car", "emergency cars", "ambulance", "ambulances", "police vehicle", "police vehicles"],
    ]
    traffic_element_extraction_results_lst = [0] * len(possible_traffic_elements_lst)       #####对应13类实体类别

    return traffic_element_extraction_results_lst



def Qwen2_vl_VQAs(model, processor, image, question):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
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
    output_text_qwen2_vl = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return output_text_qwen2_vl



def GLM_4V_VQAs(model, tokenizer, image, question):
    question_ = 'Based on strict judgement, ' + question.split('? ')[0].lower() + '? ' + question.split('? ')[1]
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": question}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True).to("cuda")
    gen_kwargs = {"max_length": 256,
                  "do_sample": False,
                  "top_k": 1}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        output_text_GLM_4V = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text_GLM_4V





def traffic_element_VQAs(model, processor, model_glm_4v, tokenizer_glm_4v, image, traffic_element_extraction_results_lst):
    questions = [
        "In this image, do you see any traffic lights? Please respond with 'Red', 'Green', 'Yellow', or 'None'.",
        "In this image, do you see any traffic cones? Please respond with 'Yes' with a detailed position description or 'None'. Avoid only answering a 'Yes'.",
        "In this image, do you see any pavement? Please respond with 'Yes' with a detailed position description or 'None'. Avoid only answering a 'Yes'.",
        "In this image, do you see any dogs? Please respond with 'Yes' with a detailed position description, or 'None'. Avoid only answering a 'Yes'.",
        "In this image, do you see any cats? Please respond with 'Yes' with a detailed position description, or 'None'. Avoid only answering a 'Yes'.",
        "In this image, do you see any persons? The 'persons' here do not include cyclists. Please respond 'Yes' with a detailed description or 'None'. Avoid only answering a 'Yes'.",
        "In this image, do you see any cyclists including bicyclists and motorcyclists? Please respond 'Yes' with a detailed description or 'None'. Avoid only answering a 'Yes'.",
        "In this image, do you see any wheelchairs and strollers? Please respond 'Yes' with a detailed description or 'None'. Avoid only answering a 'Yes'.",
        "In this image, do you see any cars? Please respond 'Yes' with a detailed description or 'None'. Avoid only answering a 'Yes'.",
        "In this image, do you see any trucks, trailers or construction vehicles? Please respond 'Yes' with a detailed description or 'None'. Avoid only answering a 'Yes'.",
        "In this image, do you see any buses? Please respond 'Yes' with a detailed description or 'None'. Avoid only answering a 'Yes'.",
        "In this image, do you see any bicycles or bikes or motorcycles or motorbikes? Please respond 'Yes' with a detailed description or 'None'. Avoid only answering a 'Yes'.",
        "In this image, do you see any emergency vehicles, such as ambulance and police vehicle? Please respond 'Yes' with a detailed description or 'None'. Avoid only answering a 'Yes'.",
    ]

    ##################################################################################### Qwen2-VL的判断结果
    final_answers_lst = []
    for k in range(len(traffic_element_extraction_results_lst)):
        if not traffic_element_extraction_results_lst[k]:   #############全都走这个
            #####################################################################################
            ################################################ Qwen2-VL的结果
            output_text_qwen2_vl = Qwen2_vl_VQAs(model, processor, image, questions[k])

            if 'Yes' in output_text_qwen2_vl or 'yes' in output_text_qwen2_vl:
                final_answers_lst.append(output_text_qwen2_vl)
            else:
                output_text_glm_4v = GLM_4V_VQAs(model_glm_4v, tokenizer_glm_4v, image, questions[k])
                final_answers_lst.append(output_text_glm_4v)

        else:
            final_answers_lst.append('Previous extracted!')


    return final_answers_lst




def VQAs_results_zhenghe(final_answers_lst):
    ########################################### 将答案为yes的整合成一句话，去掉答案为none的
    VQA_Yes_lst = []
    for k in range(len(final_answers_lst)):
        each_VQA_result = final_answers_lst[k]
        if k == 0:
            if 'Red' in each_VQA_result or 'red' in each_VQA_result:
                traffic_light_sent = 'There is a red traffic light in the driving scene.'
                VQA_Yes_lst.append(traffic_light_sent)
            elif 'Yellow' in each_VQA_result or 'yellow' in each_VQA_result:
                traffic_light_sent = 'There is a yellow traffic light in the driving scene.'
                VQA_Yes_lst.append(traffic_light_sent)
            elif 'Green' in each_VQA_result or 'green' in each_VQA_result:
                traffic_light_sent = 'There is a green traffic light in the driving scene.'
                VQA_Yes_lst.append(traffic_light_sent)
        else:
            if 'Yes' in each_VQA_result or 'yes' in each_VQA_result:
                VQA_Yes_lst.append(each_VQA_result)

    return VQA_Yes_lst




def VQAs_results_remove_duplicate(model_sentence_transformer, VQA_Yes_lst, refined_texts):
    thre_ = 0.65
    ### 将refined_texts一段话拆分成若干个独立的句子
    refined_texts_sentence_split_lst = refined_texts.split('. ')
    ### 分别计算VQA_Yes_lst中的每一个句子，和refined_texts中的每一个句子之间的相似度
    final_lst = []
    if len(VQA_Yes_lst) > 0:
        for k in range(len(VQA_Yes_lst)):
            each_VQA_Yes_str_similarity_scores = []
            VQA_Yes_str_ = VQA_Yes_lst[k]
            final_lst.append(VQA_Yes_str_)


    return final_lst




prompts = '''
Each_VQA_ans:
{Each_VQA_ans}

Task:
Please determine if Each_VQA_ans is correct based on the given image, following the three rules:
1. If Each_VQA_ans is correct, directly return Each_VQA_ans. Avoid only return a 'Yes'.
2. If Each_VQA_ans has some errors, please revise and return the revised text.
3. Avoid adding new contents to Each_VQA_ans.

Output_ours:
'''



def VQA_ans_correction(model, tokenizer, image, initial_VQA_ans_lst):
    if len(initial_VQA_ans_lst) > 0:
        corrected_VQA_ans_lst = []
        for k in range(len(initial_VQA_ans_lst)):
            each_VQA_ans = initial_VQA_ans_lst[k]
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "image": image, "content": prompts.format(Each_VQA_ans=each_VQA_ans)}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True).to("cuda")

            gen_kwargs = {"max_length": 256,
                          "do_sample": False,
                          "top_k": 1}

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                each_VQA_ans_correction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print('//////////////////////////////////////////////')
                print(each_VQA_ans_correction)

                #### 后处理，如果GLM出错，只回答一个yes，则替换为原来的文本
                if each_VQA_ans_correction == 'Yes' or each_VQA_ans_correction == 'Yes.' or each_VQA_ans_correction == 'yes' or each_VQA_ans_correction == 'yes.':
                    corrected_VQA_ans_lst.append(each_VQA_ans)
                elif each_VQA_ans_correction == 'none' or each_VQA_ans_correction == 'none.' or each_VQA_ans_correction == 'None' or each_VQA_ans_correction == 'None.' or each_VQA_ans_correction == 'No' or each_VQA_ans_correction == 'No.' or each_VQA_ans_correction == 'no' or each_VQA_ans_correction == 'no.':
                    continue
                else:
                    corrected_VQA_ans_lst.append(each_VQA_ans_correction)

        return corrected_VQA_ans_lst

    else:
        return initial_VQA_ans_lst



##############################################################################################################################
##############################################################################################################################
class Traffic_element_extraction_VQAs:
    def __init__(self, args):
        self.args = args

    def traffic_element_extraction_VQAs_(self, model_qwen2_vl, processor_qwen2_vl, model_sentence_transformer, model_glm_4v, tokenizer_glm_4v, sample: Dict):
        time_1 = time.time()

        image_path_ = sample['image']
        image = Image.open(image_path_)
        refined_texts = sample['initial_texts']

        # 1. 提取refined_texts中存在的交通元素相关的实体列别
        traffic_element_extraction_results_lst = traffic_element_extraction(refined_texts)
        print(traffic_element_extraction_results_lst)
        print('---------------------------------------')

        # 2. 对于refined_texts中没有的实体，进行询问。有17个答案，Yes, xxxxx或者None或者Previous extracted!占个位置
        traffic_element_VQAs_results_lst = traffic_element_VQAs(model_qwen2_vl, processor_qwen2_vl, model_glm_4v, tokenizer_glm_4v, image, traffic_element_extraction_results_lst)
        print(traffic_element_VQAs_results_lst)
        print('---------------------------------------')

        # 3. 对于VQA的结果进行整理，把句子放到列表中
        VQA_Yes_lst = VQAs_results_zhenghe(traffic_element_VQAs_results_lst)
        print(VQA_Yes_lst)
        print('---------------------------------------')

        # 4. 对于VQA_Yes_lst中的每一句话，计算和refined_texts中每句话之间的相似性，如果相似性超过0.5，则去掉这句话
        VQA_Yes_lst_remove_duplicate_lst = VQAs_results_remove_duplicate(model_sentence_transformer, VQA_Yes_lst, refined_texts)
        print(VQA_Yes_lst_remove_duplicate_lst)
        print('---------------------------------------')

        sample['VQA_ans'] = VQA_Yes_lst_remove_duplicate_lst

        time_2 = time.time()
        print(time_2 - time_1)

        return sample











