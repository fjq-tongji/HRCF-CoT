from vis_corrector_recap_w import Corrector
from types import SimpleNamespace
import argparse
import spacy
import json
import nltk
import transformers, torch
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForZeroShotObjectDetection, InstructBlipProcessor, InstructBlipForConditionalGeneration, Qwen2VLForConditionalGeneration
from sentence_transformers import SentenceTransformer



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code for 'ReCaption'.")
    parser.add_argument('--stage-1', default='./Each_stage_texts_CODA/CODA_text_critic_ovis2_calculate_inference_time_internvl.json')
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


    # ################################################## 5. InternVL-2.5-26B
    path = '/share/home/u21012/fjq/MLLMS/0_MLLM_weights/Ovis2-34B'
    model_ovis_2 = AutoModelForCausalLM.from_pretrained(path,
                                                         torch_dtype=torch.bfloat16,
                                                         multimodal_max_length=32768,
                                                         trust_remote_code=True).cuda()


    # ################################################## 6. CLIP模型
    # clip_path = '/share/home/tj21012/fjq/MLLMS/0_MLLM_weights/clip-vit-large-patch14'
    # model_clip = CLIPModel.from_pretrained(clip_path)
    # processor_clip = CLIPProcessor.from_pretrained(clip_path)




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
                                   model_internvl_2_5=model_ovis_2, tokenizer_internvl_2_5=1,
                                   model_groundingdino=1, processor_groundingdino=1,
                                   sample=sample)
        print(output)
        coda_detail_captions_correct.append(output)
        print(len(coda_detail_captions_correct))


    with open('./Each_stage_texts_CODA/CODA_text_critic_ovis2_calculate_inference_time_ovis.json', 'w', encoding='utf-8') as f:
        json.dump(coda_detail_captions_correct, f, indent=4, ensure_ascii=False)



















