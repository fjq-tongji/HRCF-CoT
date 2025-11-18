from PIL import Image
import torch
import time
from typing import Dict
import copy
import cv2






prompts = '''
Task:
Please organize the input sentences into a structured paragraph and ensure that there are no grammar errors.

Example:
Input:
There is 1 person standing on the pavement ([x1,y1,x2,y2]).
There is 2 cars parked on the left side ([x1,y1,x2,y2], [x1,y1,x2,y2]).
The person is near the car.
Output:
There is 1 person standing on the pavement ([x1,y1,x2,y2]) and 2 cars parked on the left side ([x1,y1,x2,y2], [x1,y1,x2,y2]). The person is near the car.

Input:{Input_sentences}
'''




def normalize_bbox(image_path, bbox):
    """
    读取图片并将bbox坐标归一化到[0,1]区间
    :param image_path: 图片路径
    :param bbox: GroundingDINO输出的bbox列表 [x_min, y_min, x_max, y_max]（绝对坐标）
    :return: 归一化后的bbox [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
    """
    # 读取图片并获取宽高
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"图片 {image_path} 未找到或无法读取")
    height, width = image.shape[:2]

    # 将bbox坐标归一化
    x_min, y_min, x_max, y_max = bbox
    bbox_norm = [
        x_min / width,  # x_min_norm
        y_min / height,  # y_min_norm
        x_max / width,  # x_max_norm
        y_max / height  # y_max_norm
    ]
    formatted_bbox = [round(x, 2) for x in bbox_norm]

    return formatted_bbox





def graph_to_texts(normed_object_dict):   #### There is 1 person standing on the pavement ([x1,y1,x2,y2]),
    obj_nums = len(normed_object_dict['boxes'])
    obj_type = normed_object_dict['type']
    obj_position = normed_object_dict['position']
    obj_boxes = normed_object_dict['boxes']

    # 处理单复数
    verb = "is" if obj_nums == 1 else "are"
    noun = obj_type if obj_nums == 1 else obj_type + "s"  # 简单复数形式

    # 格式化boxes
    boxes_str = ", ".join([str(box) for box in obj_boxes])

    text = f"There {verb} {obj_nums} {noun} {boxes_str} {obj_position}."

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
        return sentences[0]

    # 去掉所有句子末尾的句号['aaaaaa', 'bbbbbb']
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
        # 牛津逗号格式：A, B, and C
        combined = ", ".join(cleaned_sentences[:-1]) + ", and " + cleaned_sentences[-1] + "."

    return combined




##############################################################################################################################
##############################################################################################################################
class Scene_graph_to_texts:
    def __init__(self, args):
        self.args = args

    def scene_graph_to_texts_(self, sample: Dict):
        image_path_ = sample['image']
        image_path = '/share/home/MLLMS/CODA2022/images_total/' + image_path_.split('/')[-1]
        scene_grapth = sample['revised_scene_graph_boxes_revise']
        normed_objects = scene_grapth['objects']
        normed_relations = scene_grapth['relationships']
        norm_scene_graph = {'objects': [], 'relationships': []}


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


        ##################################### 2.整理每一个物体文本，和每一个关系文本，形成 一段话
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















