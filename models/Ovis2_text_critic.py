from PIL import Image
import torch
import time
import re
from typing import Dict
import re
import transformers


PROMPT_1 = '''
Sentence:
{Sent}

Is the above description consistent with the given image? 
If it is consistent, please only return 'yes' without any analytical or meaningless words.
If there are inconsistencies, please remove the description errors and directly return the modified sentence without any analytical or meaningless words.
Avoiding only return a 'no'. Just remove the description errors, do no need to add new descriptions to the original sentence.

Output_ours:
'''



PROMPT_2 = '''
Sent_lst:
{Sent_lst}

In the above sentences, please choose only one sentence that you think is most consistent with the given image, and avoid adding any new contents in the sentence.
If you think these sentences are not very accurate, please give your own revised sentence.
Please directly return the sentence without any analytical or meaningless words.

Output_ours:
'''



def text_critic_ovis_2_func(model, image, text_):
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    max_partition = 9
    #########################################################
    images = [image]
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
            max_new_tokens=100,
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
        output_text_ovis = text_tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text_ovis








############################################################# 余弦相似度选择文本
def text_choose_text(model_ovis_16, image, split_sents, initial_text_minicpm_critic_lst, initial_text_internvl_25_critic_lst):
    text_choose_results_lst = []
    for k in range(len(split_sents)):
        time_1 = time.time()
        internvl_25_sent_ = initial_text_internvl_25_critic_lst[k]
        text_ = PROMPT_1.format(Sent=internvl_25_sent_)
        text_critic_ovis_results_ = text_critic_ovis_2_func(model_ovis_16, image, text_)
        if text_critic_ovis_results_.lower() == 'yes' or text_critic_ovis_results_.lower() == 'yes.':
            text_choose_results_lst.append(internvl_25_sent_)
        else:
            text_choose_results_lst.append(text_critic_ovis_results_)

        time_2 = time.time()
        print(time_2 - time_1)
        print(text_choose_results_lst[-1])
        print('---------------------------------Ovis2_34B判断完成了--------------------------------------')


    return text_choose_results_lst




##############################################################################################################################
##############################################################################################################################
class Text_critic_ovis_2_:
    def __init__(self, args):
        self.args = args

    def text_critic_ovis_2_(self, model_ovis_16, sample: Dict):
        time_1 = time.time()

        image_path_ = sample['image']
        image_path_ = '/share/home/MLLMS/CODA2022/images_total/' + image_path_.split('/')[-1]
        image = Image.open(image_path_)
        initial_text_split_sents = sample['initial_text_split_sents']
        initial_text_minicpm_critic_lst = sample['initial_text_minicpm_critic']
        initial_text_internvl_25_critic_lst = sample['initial_text_internvl_25_critic']

        # 1. 对MiniCPM和Internvl模型的评论进行选择---初始描述的
        initial_text_refined_lst_ = text_choose_text(model_ovis_16, image, initial_text_split_sents, initial_text_minicpm_critic_lst, initial_text_internvl_25_critic_lst)

        # 2. 保存结果
        sample['refined_texts'] = ' '.join(initial_text_refined_lst_)


        time_2 = time.time()
        print('每个图片的Ovis2模型校正时间为：')
        print(time_2 - time_1)

        return sample











