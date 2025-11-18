import transformers
import torch
from typing import Dict
import time
from PIL import Image


PROMPT_TEMPLATE='''
Refined_texts:
{Refined_texts}

VQA_texts:
{VQA_texts}

Task:
Please supplement the content of Refined_texts according to the VQA_texts, following the six rules.
1. For the content included in VQA_texts but not included in Refined_texts, please add it to Refined_texts in concise text.
2. For the content included in VQA_texts and also included in Refined_texts, please supplement Refined_texts based on the content in VQA_texts and merge similar descriptions.
3. For objects that do not exist in the image, there is no need to describe them specifically.
4. Avoid directly concatenating the contents of Refined_texts and VQA_texts.
5. Avoid duplicate descriptions of the same content strictly. 
6. Remove sentences or words that have no practical meaning.
7. Return texts, ending with '######################'.

Output_ours: 
'''


def cross_check_text_correction(pipeline, refined_texts, VQA_texts_lst, max_tokens=768):
    if len(VQA_texts_lst) > 0:
        VQA_texts_str = ' '.join(VQA_texts_lst)
        content = PROMPT_TEMPLATE.format(Refined_texts=refined_texts, VQA_texts=VQA_texts_str)
        generated_text = pipeline(content, pad_token_id=pipeline.tokenizer.eos_token_id,
                                  max_new_tokens=max_tokens)[0]['generated_text']
        output_start = generated_text.find("Output_ours:") + len("Output_ours:")
        generated_text_1 = generated_text[output_start:].strip()
        output_end = generated_text_1.find("######################")
        generated_text_1 = generated_text_1[:output_end].strip()

        return generated_text_1
    else:
        return refined_texts


##############################################################################################################################
##############################################################################################################################
class Text_integration:
    def __init__(self, args):
        self.args = args

    def text_integration_(self, pipeline, sample: Dict):
        time_1 = time.time()
        refined_texts = sample['refined_texts']
        VQA_texts_lst = sample['VQA_ans']
        final_sent = cross_check_text_correction(pipeline, refined_texts, VQA_texts_lst)
        print(final_sent)
        print('---------------------------')

        sample['integrated_texts'] = final_sent


        time_2 = time.time()
        print(time_2 - time_1)

        return sample






