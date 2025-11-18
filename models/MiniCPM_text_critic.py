from PIL import Image
import torch
import time
import re
from typing import Dict
import re
import transformers
import spacy


PROMPT_ = '''
Sentence:
{Sent}

Is the above description consistent with the given image? 
If it is consistent, please only return 'yes' without other words. 
If there are inconsistencies, please directly return the modified sentence without analysis. Avoid returning the words 'Modified sentence'.
Avoiding only return a 'no'. Just remove the description errors, do no need to add new descriptions to the original sentence.
'''



def extract_sentence_lst_(nlp, text):
    # 处理输入文本
    doc = nlp(text)
    # 提取句子并存储到列表
    sentences = [sent.text for sent in doc.sents]

    return sentences



############################################################################## 余弦相似度选择文本
def text_critic(model_minicpm, tokenizer_minicpm, image, split_sents):
    text_critic_minicpm_results_lst = []
    for each_sent in split_sents:
        time_1 = time.time()
        text_ = PROMPT_.format(Sent=each_sent)
        msgs = [{'role': 'user', 'content': [image, text_]}]
        res = model_minicpm.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer_minicpm
        )
        ################################################################# 去掉不合适的文本
        if 'Modified sentence:' in res:
            res = res.split('Modified sentence:')[1].strip()

        ################################################################# 返回
        if res.lower() == 'yes' or res.lower() == 'yes.':
            text_critic_minicpm_results_lst.append('yes')
        else:
            text_critic_minicpm_results_lst.append(res)


        time_2 = time.time()
        print(time_2 - time_1)
        print(res)
        print('----------------------------MiniCPM_o_2_6判断完成了--------------------------------')

    return text_critic_minicpm_results_lst




##############################################################################################################################
##############################################################################################################################
class Text_critic_minicpm:
    def __init__(self, args):
        self.args = args

    def text_critic_minicpm_process(self, model_minicpm_o_26, tokenizer_minicpm_o_26, sample: Dict):
        time_1 = time.time()

        image_path_ = sample['image']
        image_path_ = '/share/home/MLLMS/CODA2022/images_total/' + image_path_.split('/')[-1]
        image = Image.open(image_path_)
        initial_texts = sample['initial_texts']

        ##### 1. 拆分初始描述，加载 spaCy 模型
        nlp = spacy.load("en_core_web_sm")
        split_sents = extract_sentence_lst_(nlp, initial_texts)

        ##### 2. 初始描述，利用MiniCPM校正
        text_critic_minicpm_o_26_results_lst = text_critic(model_minicpm_o_26, tokenizer_minicpm_o_26, image, split_sents)
        print('---------------------------------------')

        ##### 3. 结果保存
        sample['initial_text_split_sents'] = split_sents
        sample['initial_text_minicpm_critic'] = text_critic_minicpm_o_26_results_lst

        time_2 = time.time()
        print('每个图片的MiniCPM-o模型校正时间为：')
        print(time_2 - time_1)

        return sample




