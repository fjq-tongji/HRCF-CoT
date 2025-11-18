from vis_corrector_recap_w import Corrector
from types import SimpleNamespace
import argparse
import json
import transformers, torch
from transformers import AutoModel, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoModelForZeroShotObjectDetection
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
    parser.add_argument('--stage-1', default='./Each_stage_texts_CODA/Initial_descriptions.json')
    parser.add_argument('--query', default='Describe this image.', type=str, help="text query for MLLM")
    parser.add_argument('--cache-dir', type=str, default='./cache_dir')
    args = parser.parse_args()
    args_dict = {'cache_dir': args.cache_dir}

    model_args = SimpleNamespace(**args_dict)


    ##########################################################################################################################
    ##########################################################################################################################
    path_1 = '/share/home/MLLMS/0_MLLM_weights/sentence-transformer'
    model_sentence_transformer = SentenceTransformer(path_1)

    # ##################################################
    path_2 = "/share/home/MLLMS/0_MLLM_weights/Qwen2-VL-72B-Instruct-GPTQ-Int4"
    model_qwen2_vl = Qwen2VLForConditionalGeneration.from_pretrained(
        path_2,
        torch_dtype="auto",
        device_map="auto"
    )
    processor_qwen2_vl = AutoProcessor.from_pretrained(path_2)

    ##################################################
    path_3 = "/share/home/MLLMS/0_MLLM_weights/glm-4v-9b_weights"
    model_glm_4V_9B = AutoModelForCausalLM.from_pretrained(
        path_3,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda").eval()
    tokenizer_glm_4V_9B = AutoTokenizer.from_pretrained(path_3, trust_remote_code=True)

    ##################################################
    path_4 = "/share/home/MLLMS/0_MLLM_weights/MiniCPM-o-2_6"
    model_minicpm_o_26 = AutoModel.from_pretrained(
        path_4,
        trust_remote_code=True,
        attn_implementation='sdpa',  # sdpa or flash_attention_2
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=False,
        init_tts=False
    )
    model_minicpm_o_26 = model_minicpm_o_26.eval().cuda()
    tokenizer_minicpm_o_26 = AutoTokenizer.from_pretrained(path_4, trust_remote_code=True)

    ##################################################
    path_5 = '/share/home/MLLMS/0_MLLM_weights/InternVL2_5-38B'
    device_map = split_model('InternVL2_5-38B')
    model_internvl_2_5 = AutoModel.from_pretrained(
        path_5,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer_internvl_2_5 = AutoTokenizer.from_pretrained(path_5, trust_remote_code=True, use_fast=False)

    ##################################################
    path_6 = '/share/home/MLLMS/0_MLLM_weights/Ovis2-34B'
    model_ovis_2 = AutoModelForCausalLM.from_pretrained(path_6,
                                                        torch_dtype=torch.bfloat16,
                                                        multimodal_max_length=32768,
                                                        trust_remote_code=True).cuda()

    ##################################################
    path_7 = '/share/home/LLAMA3/Meta-Llama-3-8B-Instruct'
    pipeline_llama_3_8B = transformers.pipeline(
        "text-generation",
        model=path_7,
        model_kwargs={"torch_dtype": torch.float32},
        device_map={"": 0})

    ##################################################
    path_8 = "/share/home/MLLMS/0_MLLM_weights/groundingdino"
    model_groundingdino = AutoModelForZeroShotObjectDetection.from_pretrained(path_8, device_map="auto")
    processor_groundingdino = AutoProcessor.from_pretrained(path_8)


    ###########################################################################################################################################################
    ###########################################################################################################################################################
    corrector = Corrector(model_args)

    ## Open file
    with open(args.stage_1, 'r', encoding='utf-8') as f:
        coda_detail_captions = json.load(f)

    coda_detail_captions_correct = []
    for sample in coda_detail_captions:
        output = corrector.correct(model_sentence_transformer=model_sentence_transformer,
                                   model_qwen2_vl=model_qwen2_vl, processor_qwen2_vl=processor_qwen2_vl,
                                   model_glm_4V_9B=model_glm_4V_9B, tokenizer_glm_4V_9B=model_glm_4V_9B,
                                   model_minicpm_o_26=model_minicpm_o_26, tokenizer_minicpm_o_26=tokenizer_minicpm_o_26,
                                   model_internvl_2_5=model_internvl_2_5, tokenizer_internvl_2_5=tokenizer_internvl_2_5,
                                   model_ovis_2=model_ovis_2,
                                   pipeline_llama_3_8B=pipeline_llama_3_8B,
                                   model_groundingdino=model_groundingdino, processor_groundingdino=processor_groundingdino,
                                   sample=sample)
        print(output)
        coda_detail_captions_correct.append(output)
        print(len(coda_detail_captions_correct))


    with open('./Each_stage_texts_CODA/Final_descriptions.json', 'w', encoding='utf-8') as f:
        json.dump(coda_detail_captions_correct, f, indent=4, ensure_ascii=False)





