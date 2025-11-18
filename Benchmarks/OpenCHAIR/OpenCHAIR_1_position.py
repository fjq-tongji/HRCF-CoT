from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
import torch
import time
import json
import spacy
import random
import openai
import base64
import os
import io
from filelock import FileLock
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor


######################################################################################################## 读取json文件
input_file = './Each_stage_texts_CODA/Final_descriptions.json'
output_file = './OpenCHAIR_final_ans_LLaVA_1_5_7B_w_o_ours.json'
# 使用文件锁来确保每个进程单独写入
lock = FileLock('./OpenCHAIR_final_ans_LLaVA_1_5_7B_w_o_ours.json.lock')


####################################################################################################################################
####################################################################################################################################
################################################################################################# 加载一个excel，里面存的是具体度列表
# 加载 spaCy 模型
nlp = spacy.load("en_core_web_sm")

################################################################################################# 加载Qwen2-VL-72B模型
model_qwen2_vl = Qwen2VLForConditionalGeneration.from_pretrained(
    "/share/home/MLLMS/0_MLLM_weights/Qwen2-VL-72B-Instruct-GPTQ-Int4", torch_dtype="auto", device_map="auto")
processor_qwen2_vl = AutoProcessor.from_pretrained("/share/home/MLLMS/0_MLLM_weights/Qwen2-VL-72B-Instruct-GPTQ-Int4")



openai.api_key = 'sk-ncfk9rjuC9ji5MMAE8qVShAnvrYjuXQzhxofzT4WTFYcmln3'
openai.api_base = "https://api.moonshot.cn/v1"


#######################################################################################################################################
PROMPT_kimi_position = '''
The following paragraph describes a traffic scene. Please analyze this description and find out phrases about relative positional relationships?
Only return phrases of relative positional relationships, separated by a comma between two elements. For example, the traffic light directly ahead. 
Please strictly following the output format: element1, element2. 
{Texts}
'''

PROMPTS_qwen = '''
According to the given image, is the following information correct: {Entity_lst}? Only answer yes/no.
'''



#######################################################################################################################################
def Qwen2_VL_query(image_path, text_):
    image = Image.open(image_path)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_},
            ],
        }
    ]
    # Preprocess the inputs
    text_prompt = processor_qwen2_vl.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

    inputs = processor_qwen2_vl(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    output_ids = model_qwen2_vl.generate(**inputs, max_new_tokens=3)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor_qwen2_vl.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    return output_text





####################################################################################################
def calculate_API_score(entity_lst, image_path):
    final_entity_judge_results_lst = []
    for each_entity_ in entity_lst:
        each_prompt_ = PROMPTS_qwen.format(Entity_lst=each_entity_)
        each_entity_results_str_ = Qwen2_VL_query(image_path, each_prompt_).lower()
        if 'yes' in each_entity_results_str_:
            final_entity_judge_results_lst.append('yes')
        elif 'no' in each_entity_results_str_:
            final_entity_judge_results_lst.append('no')

    return final_entity_judge_results_lst





#######################################################################################################################################
#######################################################################################################################################
################################################################################################# 将段落拆成句子存到列表中，从句子中提取实体
def Kimi_extract_entity(texts):
    texts_extract_entity_prompts_ = PROMPT_kimi_position.format(Texts=texts)
    completion = openai.ChatCompletion.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system", "content": "You are Kimi."},
            {"role": "user", "content": texts_extract_entity_prompts_}
        ],
        temperature=0.1,
    )
    print("Token用量情况：", "输入总Token：", completion.usage["total_tokens"])  ###### 一张图大约1200个tokens

    # 输出回复
    return completion.choices[0].message["content"]




def split_into_sentences(caption):
    # 处理输入文本
    doc = nlp(caption)
    # 提取句子并存储到列表
    sentences = [sent.text for sent in doc.sents]

    return sentences



def texts_information_extract(texts):
    ############## 1.初始描述的，段落拆成一个一个的子句子，提取每个句子中的实体名词
    final_texts_lst = split_into_sentences(texts)  ###### 将一段话拆开成若干个子句子: [句子1，句子2，句子3]

    ############## 2.利用kimi提取这段话中所包含的交通元素
    traffic_entity_str = Kimi_extract_entity(texts)
    final_entity_lst_quchong = traffic_entity_str.split(', ')        ######## [ent1, ent2, ent3, ent4, ent5]

    ############# 3.根据提取到的实体结果，进行整理，得到每句话中包含的实体词：[[句子1的实体词], [句子2的实体词], [句子3的实体词]]
    final_objs_lst_each_sent = []
    for _ in range(len(final_texts_lst)):
        final_objs_lst_each_sent.append([])             ########## [[],[],[]]
    for each_entity_ in final_entity_lst_quchong:
        for k in range(len(final_texts_lst)):
            if each_entity_ in final_texts_lst[k]:
                final_objs_lst_each_sent[k].append(each_entity_)
                break

    return final_entity_lst_quchong, final_objs_lst_each_sent, final_texts_lst



######################################################################################################################## 计算sentence_level的代码
def compute_sentence_level_score(ans_lst, initial_texts_lst, obj_entity_lst_quchong, objs_lst_each_sent):
    sentence_level_CHAIR = []

    no_num_each_sentence = ans_lst.count('no')
    if no_num_each_sentence == 0:             ###########################这个图像没有句子级别的幻觉
        sentence_level_CHAIR.append(0)
    elif no_num_each_sentence == 1:          ###########################这个图像只有1个句子存在幻觉
        sentence_level_CHAIR.append(1 / len(initial_texts_lst))
    else:
        hallucinated_words_lst = []
        for k in range(len(ans_lst)):
            if ans_lst[k] == 'no' or ans_lst[k] == 'no.':
                hallucinated_words_lst.append(obj_entity_lst_quchong[k])

        contain_hallucination_sent_index = []
        for each_hallucinated_word_str in hallucinated_words_lst:
            for j in range(len(objs_lst_each_sent)):  ##### objs_lst_each_sent=[[obj1,obj2],[obj3],[obj4,obj5]]
                each_sent_zhong_de_entities_lst = objs_lst_each_sent[j]
                if each_hallucinated_word_str in each_sent_zhong_de_entities_lst:
                    contain_hallucination_sent_index.append(j)
                    break
        hallucinated_sent_shumu = len(list(set(contain_hallucination_sent_index)))  #### 有幻觉的句子有几个，具体数目
        sentence_level_CHAIR.append(hallucinated_sent_shumu / len(initial_texts_lst))

    return sentence_level_CHAIR




######################################################################################################################## 主函数
########################################################################################################################
########################################################################################################################
######################################################################################################## 读取json文件
with open(input_file, 'r', encoding='utf-8') as f:
    input_file_ = json.load(f)


initial_instance_level_CHAIR = []
initial_sentence_level_CHAIR = []
final_instance_level_CHAIR = []
final_sentence_level_CHAIR = []
num = 0
for each_img_dict in input_file_:
    time_1 = time.time()

    image_path_ = each_img_dict['image']
    image_path = '/share/home/MLLMS/CODA2022/images_total/' + image_path_.split('/')[-1]
    initial_texts = each_img_dict['initial_texts']
    final_texts = each_img_dict['integrated_texts']     ####################################################################

    image = Image.open(image_path)

    ####################### 1.初始文本和最终文本的：拆分结果、去重的实体列表、句子等
    initial_entity_lst_quchong, initial_objs_lst_each_sent, initial_texts_lst = texts_information_extract(initial_texts)
    final_entity_lst_quchong, final_objs_lst_each_sent, final_texts_lst = texts_information_extract(final_texts)
    print(initial_entity_lst_quchong)
    print(final_entity_lst_quchong)
    print('--------------------------------------实体词提取完成！------------------------------------------')

    ###################### 2.分两种情况
    if all(item in initial_entity_lst_quchong for item in final_entity_lst_quchong):   ######################## 两个实体列表不完全相同/完全相同，但最终描述中的每一个实体都在初始描述中出现过
        initial_ans_lst = calculate_API_score(initial_entity_lst_quchong, image_path)
        print(initial_ans_lst)
        initial_sentence_level_CHAIR_ = compute_sentence_level_score(initial_ans_lst, initial_texts_lst,
                                                                     initial_entity_lst_quchong,
                                                                     initial_objs_lst_each_sent)
        final_ans_lst = []
        for each_final_entity in final_entity_lst_quchong:
            final_ans_lst.append(initial_ans_lst[initial_entity_lst_quchong.index(each_final_entity)])
        print(final_ans_lst)
        final_sentence_level_CHAIR_ = compute_sentence_level_score(final_ans_lst, final_texts_lst,
                                                                   final_entity_lst_quchong,
                                                                   final_objs_lst_each_sent)
        if len(initial_ans_lst) == len(initial_entity_lst_quchong) and len(final_ans_lst) == len(final_entity_lst_quchong):
            initial_instance_level_CHAIR.extend(initial_ans_lst)
            initial_sentence_level_CHAIR.extend(initial_sentence_level_CHAIR_)
            final_instance_level_CHAIR.extend(final_ans_lst)
            final_sentence_level_CHAIR.extend(final_sentence_level_CHAIR_)
            print(initial_sentence_level_CHAIR)
            print(final_sentence_level_CHAIR)
        else:
            print('Error！！！这张图片的推理结果存在异常！！！')


    else:
        ############################################### 最终描述中有很多实体，没有在初始描述中出现过
        addition_entity_lst = [each_entity_ for each_entity_ in final_entity_lst_quchong if each_entity_ not in initial_entity_lst_quchong]
        hebing_entity_lst = []
        hebing_entity_lst.extend(initial_entity_lst_quchong)
        hebing_entity_lst.extend(addition_entity_lst)
        print(hebing_entity_lst)
        hebing_ans_lst = calculate_API_score(hebing_entity_lst, image_path)     ##### 合并后的实体，问答结果
        print(hebing_ans_lst)

        if len(hebing_ans_lst) == len(hebing_entity_lst):
            ##########################################################  得到初始实体的问答结果
            initial_ans_lst = hebing_ans_lst[0: len(initial_entity_lst_quchong)]
            print(initial_ans_lst)
            initial_sentence_level_CHAIR_ = compute_sentence_level_score(initial_ans_lst, initial_texts_lst,
                                                                         initial_entity_lst_quchong,
                                                                         initial_objs_lst_each_sent)
            ##########################################################  得到最终实体的问答结果
            addition_ans_lst = hebing_ans_lst[len(initial_entity_lst_quchong):]
            final_ans_lst = []
            for k in range(len(final_entity_lst_quchong)):
                each_final_ent_ = final_entity_lst_quchong[k]
                if each_final_ent_ in initial_entity_lst_quchong:
                    ans_each_ = initial_ans_lst[initial_entity_lst_quchong.index(each_final_ent_)]
                    final_ans_lst.append(ans_each_)
                elif each_final_ent_ in addition_entity_lst:
                    ans_each_ = addition_ans_lst[addition_entity_lst.index(each_final_ent_)]
                    final_ans_lst.append(ans_each_)
            print(final_ans_lst)
            final_sentence_level_CHAIR_ = compute_sentence_level_score(final_ans_lst, final_texts_lst,
                                                                       final_entity_lst_quchong,
                                                                       final_objs_lst_each_sent)
            initial_instance_level_CHAIR.extend(initial_ans_lst)
            initial_sentence_level_CHAIR.extend(initial_sentence_level_CHAIR_)
            final_instance_level_CHAIR.extend(final_ans_lst)
            final_sentence_level_CHAIR.extend(final_sentence_level_CHAIR_)
            print(initial_sentence_level_CHAIR)
            print(final_sentence_level_CHAIR)
        else:
            print('Error！！！这张图片的推理结果存在异常！！！')



    time_2 = time.time()
    print(time_2 - time_1)
    num += 1
    print(num)
    print('--------------------------------------------------------------------------------------------------------')






########################################################################################################################
########################################################################################################################
############################################################################################### 统计列表中的yes和no的比率, instance-level
############################################################################################### 统计列表中的yes和no的比率, sentence-level
no_num = 0
for each_ans_ in initial_instance_level_CHAIR:
    if 'no' in each_ans_:
        no_num += 1

print('Initial_Instance_level的幻觉比例为：%.4f' %(no_num / len(initial_instance_level_CHAIR)))
print('Initial_Sentence_level的幻觉比例为：%.4f' %(sum(initial_sentence_level_CHAIR) / len(initial_sentence_level_CHAIR)))

ans_lst_initial = [len(initial_sentence_level_CHAIR), 'Initial_Instance_level: %.4f' %(no_num / len(initial_instance_level_CHAIR)), 'Initial_Sentence_level: %.4f' %(sum(initial_sentence_level_CHAIR) / len(initial_sentence_level_CHAIR))]


########################################################################################################################
no_num_2 = 0
for each_ans_ in final_instance_level_CHAIR:
    if 'no' in each_ans_:
        no_num_2 += 1

print('Final_Instance_level的幻觉比例为：%.4f' %(no_num_2 / len(final_instance_level_CHAIR)))
print('Final_Sentence_level的幻觉比例为：%.4f' %(sum(final_sentence_level_CHAIR) / len(final_sentence_level_CHAIR)))

ans_lst_final = [len(final_sentence_level_CHAIR), 'Final_Instance_level: %.4f' %(no_num_2 / len(final_instance_level_CHAIR)), 'Final_Sentence_level: %.4f' %(sum(final_sentence_level_CHAIR) / len(final_sentence_level_CHAIR))]





########################################################################################################################
########################################################################################################################
with lock:
    with open(output_file, 'a', encoding='utf-8') as f:
        # 将数据写入文件时追加（防止覆盖）
        json.dump(ans_lst_initial, f, ensure_ascii=False, indent=4)
        f.write('\n')
        json.dump(ans_lst_final, f, ensure_ascii=False, indent=4)
        f.write('\n')
        f.write('----------------------------------------------')
        f.write('\n')









