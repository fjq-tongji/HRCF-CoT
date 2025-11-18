from PIL import Image
import torch
import time
from typing import Dict
import copy



prompts_pos = '''
There is a {Target} in the image, and the position of the object is {Position}.

Task:
Based on the given image, please determine if the above statement is correct. 
If it is correct, directly return 'yes' without any additional explanations. 
If it is not correct, please revise the position of the object and only return the revised position without any additional explanations.

Output_ours:
'''



prompts_sta = '''
There is a {Target} in the image, and the status of the object is {Status}.

Task:
Based on the given image, please determine if the above statement is correct. 
If it is correct, directly return 'yes' without any additional explanations. 
If it is not correct, please revise the status of the object and only return the revised status without any additional explanations.

Output_ours:
'''



prompts_rel = '''
There is a {Source} and {Target} in the image, and the first object is {Relationships} the second object.

Task:
Based on the given image, please determine if the above statement is correct. 
If it is correct, directly return 'yes' without any additional explanations. 
If it is not correct, please revise the relationship of two objects and only return the revised relationship without any additional explanations.

Output_ours:
'''




prompts_boxes = '''
Task:
There is a {Type} in the image and its position is {Position}. 
Its bounding box coordinate is {Boxes}, with four values representing the top left horizontal axis, top left vertical axis, bottom right horizontal axis, and bottom right vertical axis of the detection box. 
The coordinate origin is located at the top left vertex of the entire image, with a width value of 1 and a height value of 1.

Based on the given image, please determine if the coordinates and position information match.
If matched, directly return 'yes' without any additional explanations. 
If not matched, directly return 'no' without any additional explanations. 

Output_ours: 
'''





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
    output_ids = model.generate(**inputs, max_new_tokens=10)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text_qwen2_vl = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return output_text_qwen2_vl





def groundingdino_detect_obj(model, processor, image, text):
    inputs = processor(images=image, text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=None
    )[0]
    return results


def filter_top_score_per_category(detections):
    """
    保留每个类别分数最高的检测框
    :param detections: 包含 'scores', 'labels', 和 'boxes' 的字典
    :return: 筛选后的检测结果字典
    """
    scores = detections["scores"]
    labels = detections["labels"]
    boxes = detections["boxes"]

    # 转换标签为集合，获取唯一类别
    unique_labels = set(labels)
    if len(unique_labels) == 0:
        return {"scores": [], "labels": [], "boxes": []}

    else:
        filtered_results = {"scores": [], "labels": [], "boxes": []}
        filtered_results['scores'] = scores.tolist()
        filtered_results['labels'] = labels
        filtered_results['boxes'] = boxes.tolist()


    round_boxes_final = []
    for each_box_ in filtered_results["boxes"]:
        rounded_box = [round(coord, 2) for coord in each_box_]
        round_boxes_final.append(rounded_box)

    filtered_results["boxes"] = round_boxes_final

    return filtered_results




def replace_id_to_obj(objects, relationships):
    relationships_new = copy.deepcopy(relationships)
    objects_new = copy.deepcopy(objects)
    id_to_type = {obj["id"]: obj["type"] for obj in objects_new}

    # 替换 relationships 中的 id 为 type
    for rel in relationships_new:
        rel["source"] = id_to_type[rel["source"]]
        rel["target"] = id_to_type[rel["target"]]

    return relationships_new





def boxes_filter_(model, processor, image, obj_infor):
    obj_type = obj_infor[0]
    position = obj_infor[1]
    boxes = obj_infor[2]

    question = prompts_boxes.format(Type=obj_type, Position=position, Boxes=boxes)
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
    output_ids = model.generate(**inputs, max_new_tokens=10)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text_qwen2_vl = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    return output_text_qwen2_vl






def boxes_filter_rules_(obj_infor):
    position = obj_infor[1]
    boxes = obj_infor[2]
    ans = 'no'

    x_center = (boxes[0] + boxes[2]) / 2
    y_center = (boxes[1] + boxes[3]) / 2
    if 'left' in position:
        if x_center < 0.5:
            ans = 'yes'
    elif 'right' in position:
        if x_center > 0.5:
            ans = 'yes'
    elif 'middle' in position:
        if x_center > 0.3 and x_center < 0.7:
            ans = 'yes'
    else:
        ans = 'yes'

    #################################
    if ans == 'yes':
        return boxes
    else:
        return []




def graph_to_texts(normed_object_dict):
    obj_type = normed_object_dict['type']
    obj_position = normed_object_dict['position']

    text = f"There is a {obj_type} {obj_position}."

    return text




def graph_to_texts_relation(each_relation_dict):
    source = each_relation_dict['source']
    target = each_relation_dict['target']
    relation = each_relation_dict['relation']

    text = f"The {source} is {relation} the {target}."

    return text




def texts_integrate(sentences):
    """
    将任意数量的对象描述和关系描述合并为一段连贯文字
    参数:
        *sentences: 可变数量的句子，最后一句是关系描述，前面都是对象描述
    返回:
        合并后的段落字符串
    """
    if len(sentences) == 1:
        return sentences[0]     # 如果只有一个句子直接返回

    cleaned_sentences = []
    for s in range(len(sentences)):
        if s == 0:
            cleaned_sentences.append(sentences[s][:-1])
        else:
            cleaned_sentences.append(sentences[s][:-1].lower())

    # 智能连接句子
    if len(cleaned_sentences) == 2:
        combined = f"{cleaned_sentences[0]} and {cleaned_sentences[1]}."
    else:
        combined = ", ".join(cleaned_sentences[:-1]) + ", and " + cleaned_sentences[-1] + "."

    return combined




def remove_duplicate_boxes(scene_graph_obj_lst):
    seen_boxes = set()
    new_objects = []
    not_removed_types = []

    for obj in scene_graph_obj_lst:
        unique_boxes = []
        for box in obj["boxes"]:
            box_tuple = tuple(box)
            if box_tuple not in seen_boxes:
                seen_boxes.add(box_tuple)
                unique_boxes.append(box)

        if unique_boxes:
            new_obj = obj.copy()
            new_obj["boxes"] = unique_boxes
            new_objects.append(new_obj)
            not_removed_types.append(obj["type"])

    return {
        "objects": new_objects,
        "retained_types": not_removed_types
    }




##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
class Scene_graph_validation:
    def __init__(self, args):
        self.args = args

    def scene_graph_groundingdino_validation_(self, model, processor, sample):
        image_path_ = sample['image']
        image = Image.open(image_path_)
        initial_scene_grapth = sample['Scene_graph']
        objects = initial_scene_grapth['objects']
        relationships_yuan = initial_scene_grapth['relationships']
        revised_scene_graph = {'objects': [], 'relationships': []}

        exist_objs = []
        non_exist_objs = []
        for k in range(len(objects)):
            object_dict = objects[k]
            obj_id = int(object_dict['id'])
            text_obj = (object_dict['type'] + '.').lower()

            #################### 1. 判断场景图中的物体类别 对不对
            obj_judge_ans_ = groundingdino_detect_obj(model, processor, image, text_obj)
            print(obj_judge_ans_)

            #################### 2. 对GroundingDINO判断的结果，进行过滤
            obj_judge_ans_new = filter_top_score_per_category(obj_judge_ans_)
            print(obj_judge_ans_new)

            #################### 3. 如果GroundingDINO判断的最高分都低于指定的阈值，则去掉这个物体
            if len(obj_judge_ans_new['scores']) != 0:
                object_dict_new = copy.deepcopy(object_dict)
                object_dict_new['boxes'] = obj_judge_ans_new['boxes']
                revised_scene_graph['objects'].append(object_dict_new)
                exist_objs.append(obj_id)
            else:
                non_exist_objs.append(obj_id)

        # ----------------------------------------------------#
        if len(non_exist_objs) == 0:
            relationships_replace = replace_id_to_obj(objects, relationships_yuan)
            revised_scene_graph['relationships'] = relationships_replace
        else:
            filtered_relationships = [rel for rel in relationships_yuan if
                                      rel["source"] in exist_objs and rel["target"] in exist_objs]
            relationships_replace = replace_id_to_obj(objects, filtered_relationships)
            revised_scene_graph['relationships'] = relationships_replace
        # ----------------------------------------------------#

        sample['revised_scene_graph_objs'] = revised_scene_graph

        return sample

    def scene_graph_others_validation_(self, model_qwen2_vl, processor_qwen2_vl, sample: Dict):
        image_path_ = sample['image']
        image_path_ = '/share/home/MLLMS/CODA2022/images_total/' + image_path_.split('/')[-1]
        image = Image.open(image_path_)
        initial_scene_grapth = sample['revised_scene_graph_objs']
        objects = initial_scene_grapth['objects']
        relationships_yuan = initial_scene_grapth['relationships']
        revised_scene_graph_2 = {'objects': [], 'relationships': []}

        for k in range(len(objects)):
            object_dict = objects[k]
            obj_type = object_dict['type']
            obj_position = object_dict['position']
            obj_status = object_dict['status']

            ################################### 1.物体位置验证
            type_position_moban = prompts_pos.format(Target=obj_type, Position=obj_position)
            output_position = Qwen2_vl_VQAs(model_qwen2_vl, processor_qwen2_vl, image, type_position_moban)
            print(output_position)
            if 'yes' in output_position.lower():
                final_position = obj_position
            else:
                final_position = output_position

            ################################### 2.物体状态验证
            type_position_moban = prompts_sta.format(Target=obj_type, Status=obj_status)
            output_status = Qwen2_vl_VQAs(model_qwen2_vl, processor_qwen2_vl, image, type_position_moban)
            print(output_status)
            if 'yes' in output_status.lower():
                final_status = obj_status
            else:
                final_status = output_status

            object_dict_new = copy.deepcopy(object_dict)
            object_dict_new['position'] = final_position
            object_dict_new['status'] = final_status
            revised_scene_graph_2['objects'].append(object_dict_new)


        ################################### 3.物体之间的关系验证
        for j in range(len(relationships_yuan)):
            relationship_dict = relationships_yuan[j]
            obj_source = relationship_dict['source']
            obj_target = relationship_dict['target']
            obj_relation = relationship_dict['relation']
            type_relation_moban = prompts_rel.format(Source=obj_source, Target=obj_target, Relationships=obj_relation)
            output_relationship = Qwen2_vl_VQAs(model_qwen2_vl, processor_qwen2_vl, image, type_relation_moban)
            if 'yes' in output_relationship.lower():
                final_relation = obj_relation
            else:
                final_relation = output_relationship

            relation_dict_new = copy.deepcopy(relationship_dict)
            relation_dict_new['relation'] = final_relation
            revised_scene_graph_2['relationships'].append(relation_dict_new)


        sample['revised_scene_graph_others'] = revised_scene_graph_2

        return sample




    def scene_graph_boxes_filter_rules_(self, sample):
        ####################### 1.将每个boxes值归一化
        image_path_ = sample['image']
        image_path_ = '/share/home/MLLMS/CODA2022/images_total/' + image_path_.split('/')[-1]
        image = Image.open(image_path_)
        revised_scene_graph_3 = {'objects': [], 'relationships': []}
        scene_grapth_2 = sample['revised_scene_graph_others']
        objects = scene_grapth_2['objects']
        relationships_yuan = scene_grapth_2['relationships']

        for k in range(len(objects)):
            object_dict = objects[k]
            bbox_norm = object_dict['boxes']
            object_dict_new = copy.deepcopy(object_dict)
            if len(bbox_norm) > 1:
                obj_type = object_dict_new['type']
                position = object_dict_new['position']
                boxes = object_dict_new['boxes']
                filtered_boxes = []
                for i in range(len(boxes)):
                    each_box_infor = [obj_type, position, boxes[i]]
                    boxes_filter_result = boxes_filter_rules_(each_box_infor)
                    if len(boxes_filter_result) > 0:
                        filtered_boxes.append(boxes[i])
                object_dict_new['boxes'] = filtered_boxes

            revised_scene_graph_3['objects'].append(object_dict_new)

        revised_scene_graph_3['relationships'] = copy.deepcopy(relationships_yuan)
        sample['revised_scene_graph_boxes'] = revised_scene_graph_3

        return sample




    def scene_graph_boxes_filter_remove_duplicate_(self, sample: Dict):
        scene_graph = sample['revised_scene_graph_boxes']
        objects_yuan = scene_graph['objects']
        relationships_yuan = scene_graph['relationships']
        revised_scene_graph_4 = {'objects': [], 'relationships': []}

        ######################## 1.得到去重之后的物体列表
        remove_duplicate_boxes_results_dict = remove_duplicate_boxes(objects_yuan)
        revised_scene_graph_4['objects'] = remove_duplicate_boxes_results_dict['objects']

        ######################## 2.根据去重之后的物体列表，对关系列表进行去重
        exist_objs = remove_duplicate_boxes_results_dict['retained_types']
        filtered_relationships = [rel for rel in relationships_yuan if
                                  rel["source"] in exist_objs and rel["target"] in exist_objs]
        revised_scene_graph_4['relationships'] = filtered_relationships


        sample['revised_scene_graph_remove_duplicate_boxes'] = revised_scene_graph_4

        return sample





    def scene_graph_to_texts_(self, sample: Dict):
        scene_grapth = sample['Scene_graph']
        normed_objects = scene_grapth['objects']
        normed_relations = scene_grapth['relationships']

        ##################################### 1.将场景图转化为文本保存[每一个物体文本] [每一个关系文本]
        normed_texts_all_objs = []
        normed_texts_all_relations = []
        for p in range(len(normed_objects)):
            normed_object_dict = normed_objects[p]
            normed_object_texts = graph_to_texts(normed_object_dict)   ## There is 1 person standing on the pavement ([x1,y1,x2,y2]).
            normed_texts_all_objs.append(normed_object_texts)
        for m in range(len(normed_relations)):
            each_relation_dict = normed_relations[m]
            each_relation_texts = graph_to_texts_relation(each_relation_dict)
            normed_texts_all_relations.append(each_relation_texts)

        ##################################### 2.整理每一个物体文本，和每一个关系文本，分别形成 一段话，再加到一块
        if len(normed_texts_all_objs) > 0:
            final_integrated_texts = texts_integrate(normed_texts_all_objs)
            final_integrated_texts += ' '
        else:
            final_integrated_texts = ''

        if len(normed_texts_all_relations) > 0:
            normed_texts_all_relations_ = texts_integrate(normed_texts_all_relations)
            final_integrated_texts += normed_texts_all_relations_

        sample['scene_graph_texts'] = final_integrated_texts

        return sample




    def scene_graph_validation_all(self, model_groundingdino, processor_groundingdino, model_qwen2_vl, processor_qwen2_vl, sample):
        sample_1 = self.scene_graph_groundingdino_validation_(model_groundingdino, processor_groundingdino, sample)
        print('--------------groundingdino_validation--------------')
        sample_2 = self.scene_graph_others_validation_(model_qwen2_vl, processor_qwen2_vl, sample_1)
        print('--------------others_validation--------------')
        sample_3 = self.scene_graph_boxes_filter_rules_(sample_2)
        print('--------------box_filter_validation--------------')
        sample_4 = self.scene_graph_boxes_filter_remove_duplicate_(sample_3)
        print('--------------box_remove_duplicate_boxes--------------')
        sample_5 = self.scene_graph_to_texts_(sample_4)

        return sample_5
















