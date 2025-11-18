from vis_corrector_recap_w import Corrector
from types import SimpleNamespace
import argparse
import math
import numpy as np
import spacy
import json
import nltk
import transformers, torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer



def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code for 'ReCaption'.")
    parser.add_argument('--stage-1', default='./Each_stage_texts_CODA/CODA_text_critic_ovis2_calculate_inference_time_minicpm_o.json')
    parser.add_argument('--query', default='Describe this image.', type=str, help="text query for MLLM")
    parser.add_argument('--cache-dir', type=str, default='./cache_dir')
    args = parser.parse_args()
    args_dict = {
        'cache_dir': args.cache_dir}

    model_args = SimpleNamespace(**args_dict)


    ##########################################################################################################################
    ##########################################################################################################################
    # ################################################## 0. SentenceTransformer，计算句子之间的余弦相似性
    # model_sentence_transformer = SentenceTransformer('/share/home/tj21012/fjq/MLLMS/0_MLLM_weights/sentence-transformer')


    # # ################################################## 1. Spacy，加载 NLP 模型
    # path = "/share/home/tj21012/fjq/MLLMS/0_MLLM_weights/Qwen2-VL-72B-Instruct-GPTQ-Int4"
    # model_qwen2_vl = Qwen2VLForConditionalGeneration.from_pretrained(
    #     path,
    #     torch_dtype="auto",
    #     device_map="auto"
    # )
    # processor_qwen2_vl = AutoProcessor.from_pretrained(path)

    # ################################################## 1. MiniCPM-V 2.6
    # model_minicpm_v_2_6 = AutoModel.from_pretrained('/share/home/tj21012/fjq/MLLMS/0_MLLM_weights/MiniCPM-V-2_6_weights',
    #                                   trust_remote_code=True, attn_implementation='sdpa',
    #                                   torch_dtype=torch.float16)  # sdpa or flash_attention_2, no eager
    # model_minicpm_v_2_6 = model_minicpm_v_2_6.eval().cuda()
    # tokenizer_minicpm_v_2_6 = AutoTokenizer.from_pretrained('/share/home/tj21012/fjq/MLLMS/0_MLLM_weights/MiniCPM-V-2_6_weights',
    #                                           trust_remote_code=True)

    ################################################## 1. InstructBLIP-vicuna-13b
    # model_instructblip = InstructBlipForConditionalGeneration.from_pretrained(
    #     '/share/home/tj21012/fjq/MLLMS/0_MLLM_weights/Instructblip-vicuna-13b',
    #     torch_dtype=torch.float32)
    # model_instructblip.to("cuda")
    # processor_instructblip = InstructBlipProcessor.from_pretrained(
    #     '/share/home/tj21012/fjq/MLLMS/0_MLLM_weights/Instructblip-vicuna-13b')

    ################################################## 2. LLama-3.1-8B
    # pipeline_llama_3_8B = transformers.pipeline(
    #     "text-generation", model='/share/home/tj21012/fjq/LLAMA3/Meta-Llama-3-8B-Instruct',
    #     model_kwargs={"torch_dtype": torch.float32},
    #     device_map={"": 0})

    # ################################################## 3. GLM-4V-9B
    # model_glm_4V_9B = AutoModelForCausalLM.from_pretrained(
    #     "/share/home/tj21012/fjq/MLLMS/0_MLLM_weights/glm-4v-9b_weights",
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True
    # ).to("cuda").eval()
    # tokenizer_glm_4V_9B = AutoTokenizer.from_pretrained(
    #     "/share/home/tj21012/fjq/MLLMS/0_MLLM_weights/glm-4v-9b_weights",
    #     trust_remote_code=True)

    # ################################################## 4. GroundingDINO
    # model_path = "/share/home/tj21012/fjq/MLLMS/0_MLLM_weights/groundingdino"
    # model_groundingdino = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to("cuda")
    # processor_groundingdino = AutoProcessor.from_pretrained(model_path)


    # ################################################## 5. InternVL-2.5-38B
    path = '/share/home/u21012/fjq/MLLMS/0_MLLM_weights/InternVL2_5-38B'
    device_map = split_model('InternVL2_5-38B')
    model_internvl_2_5 = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer_internvl_2_5 = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


    ##########################################################################################################################
    ##########################################################################################################################
    ##########################################################################################################################
    ##########################################################################################################################
    ##########################################################################################################################
    corrector = Corrector(model_args)
    final_text = []

    ##所有的coda图片的详细描述，存到列表中
    with open(args.stage_1, 'r', encoding='utf-8') as f:
        coda_detail_captions = json.load(f)

    coda_detail_captions_correct = []
    for sample in coda_detail_captions[:5]:
        output = corrector.correct(model_sentence_transformer=1,
                                   model_instructblip=1, processor_instructblip=1,
                                   pipeline_llama_3_8B=1,
                                   model_glm_4V_9B=1, tokenizer_glm_4V_9B=1,
                                   model_qwen2_vl=1, processor_qwen2_vl=1,
                                   model_internvl_2_5=model_internvl_2_5, tokenizer_internvl_2_5=tokenizer_internvl_2_5,
                                   model_groundingdino=1, processor_groundingdino=1,
                                   sample=sample)
        print(output)
        coda_detail_captions_correct.append(output)
        print(len(coda_detail_captions_correct))


    with open('./Each_stage_texts_CODA/CODA_text_critic_ovis2_calculate_inference_time_internvl.json', 'w', encoding='utf-8') as f:
        json.dump(coda_detail_captions_correct, f, indent=4, ensure_ascii=False)



















