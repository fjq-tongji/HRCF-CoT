from vis_corrector_recap_w import Corrector
from types import SimpleNamespace
import argparse
import spacy
import json
import time
import nltk
import transformers, torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
# from sentence_transformers import SentenceTransformer



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code for 'ReCaption'.")
    parser.add_argument('--stage-1', default='./Each_stage_texts_CODA/COCO_initial_texts_1.json')
    parser.add_argument('--query', default='Describe this image.', type=str, help="text query for MLLM")
    parser.add_argument('--cache-dir', type=str, default='./cache_dir')
    args = parser.parse_args()
    args_dict = {
        'cache_dir': args.cache_dir}

    model_args = SimpleNamespace(**args_dict)


    ##########################################################################################################################
    ##########################################################################################################################
    # ################################################## 0. SentenceTransformer，计算句子之间的余弦相似性
    # model_sentence_transformer = SentenceTransformer('/share/home/u21012/fjq/MLLMS/0_MLLM_weights/sentence-transformer')


    # # ################################################## 1. Spacy，加载 NLP 模型
    model_qwen2_vl = Qwen2VLForConditionalGeneration.from_pretrained(
        "/share/home/u21012/fjq/MLLMS/0_MLLM_weights/Qwen2-VL-7B-Instruct", torch_dtype="auto",
        device_map="auto"
    )
    processor_qwen2_vl = AutoProcessor.from_pretrained(
        "/share/home/u21012/fjq/MLLMS/0_MLLM_weights/Qwen2-VL-7B-Instruct")

    # ################################################## 1. MiniCPM-V 2.6
    # model_minicpm_llama3_v_2_5 = AutoModel.from_pretrained('/share/home/tj21012/fjq/MLLMS/0_MLLM_weights/MiniCPM-Llama3-V-2_5_weights',
    #                                   trust_remote_code=True,
    #                                   torch_dtype=torch.float16)  # sdpa or flash_attention_2, no eager
    # model_minicpm_llama3_v_2_5 = model_minicpm_llama3_v_2_5.to("cuda")
    # model_minicpm_llama3_v_2_5.eval()
    # tokenizer_minicpm_llama3_v_2_5 = AutoTokenizer.from_pretrained('/share/home/tj21012/fjq/MLLMS/0_MLLM_weights/MiniCPM-Llama3-V-2_5_weights',
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
    model_glm_4V_9B = AutoModelForCausalLM.from_pretrained(
        "/share/home/u21012/fjq/MLLMS/0_MLLM_weights/glm-4v-9b_weights",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda").eval()
    tokenizer_glm_4V_9B = AutoTokenizer.from_pretrained(
        "/share/home/u21012/fjq/MLLMS/0_MLLM_weights/glm-4v-9b_weights",
        trust_remote_code=True)

    # ################################################## 4. GroundingDINO
    # model_path = "/share/home/tj21012/fjq/MLLMS/0_MLLM_weights/groundingdino"
    # model_groundingdino = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to("cuda")
    # processor_groundingdino = AutoProcessor.from_pretrained(model_path)




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
    for sample in coda_detail_captions:
        time_1 = time.time()
        output = corrector.correct(model_sentence_transformer=1,
                                   model_instructblip=1, processor_instructblip=1,
                                   pipeline_llama_3_8B=1,
                                   model_glm_4V_9B=model_glm_4V_9B, tokenizer_glm_4V_9B=tokenizer_glm_4V_9B,
                                   model_qwen2_vl=model_qwen2_vl, processor_qwen2_vl=processor_qwen2_vl,
                                   model_internvl_2_5=1, tokenizer_internvl_2_5=1,
                                   model_groundingdino=1, processor_groundingdino=1,
                                   sample=sample)
        print(output)
        coda_detail_captions_correct.append(output)
        print(len(coda_detail_captions_correct))
        time_2 = time.time()
        print(time_2 - time_1)


    with open('./Each_stage_texts_CODA/COCO_traffic_element_VQAs_1.json', 'w', encoding='utf-8') as f:
        json.dump(coda_detail_captions_correct, f, indent=4, ensure_ascii=False)














