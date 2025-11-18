from vis_corrector_recap_w import Corrector
from types import SimpleNamespace
import argparse
import json
import time
import transformers, torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor  #, AutoModelForZeroShotObjectDetection



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code for 'ReCaption'.")
    parser.add_argument('--stage-1', default='./Each_stage_texts_CODA/CODA_scene_graph_validation_initial.json')
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
    # alignscore = AlignScore(model='roberta-large', batch_size=32, device='cuda:0',
    #                         ckpt_path='/share/home/u21012/fjq/MLLMS/0_MLLM_weights/AlignScore/AlignScore-large.ckpt',
    #                         evaluation_mode='nli_sp')

    # # ################################################## 1. Spacy，加载 NLP 模型
    model_qwen2_vl = Qwen2VLForConditionalGeneration.from_pretrained(
        "/share/home/u21012/fjq/MLLMS/0_MLLM_weights/Qwen2-VL-7B-Instruct", torch_dtype="auto",
        device_map="auto"
    )
    processor_qwen2_vl = AutoProcessor.from_pretrained(
        "/share/home/u21012/fjq/MLLMS/0_MLLM_weights/Qwen2-VL-7B-Instruct")


    ################################################### Ovis2模型
    # path = '/share/home/u21012/fjq/MLLMS/0_MLLM_weights/Ovis2-34B'
    # model_ovis_2 = AutoModelForCausalLM.from_pretrained(path,
    #                                                     torch_dtype=torch.bfloat16,
    #                                                     multimodal_max_length=32768,
    #                                                     trust_remote_code=True).cuda()




    # ################################################## 4. GroundingDINO
    # model_path = "/share/home/u21012/fjq/MLLMS/0_MLLM_weights/groundingdino"
    # model_groundingdino = AutoModelForZeroShotObjectDetection.from_pretrained(model_path,
    #                                                                           device_map="auto")
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
    for sample in coda_detail_captions[0:5]:
        time_1 = time.time()
        output = corrector.correct(model_sentence_transformer=1,
                                   model_instructblip=1, processor_instructblip=1,
                                   pipeline_llama_3_8B=1,
                                   model_glm_4V_9B=1, tokenizer_glm_4V_9B=1,
                                   model_qwen2_vl=model_qwen2_vl, processor_qwen2_vl=processor_qwen2_vl,
                                   model_internvl_2_5=1, tokenizer_internvl_2_5=1,
                                   model_groundingdino=1, processor_groundingdino=1,
                                   sample=sample)
        print(output)
        coda_detail_captions_correct.append(output)
        print(len(coda_detail_captions_correct))
        time_2 = time.time()
        print(time_2 - time_1)


    with open('./Each_stage_texts_CODA/CODA_scene_graph_validation_inference_time.json', 'w', encoding='utf-8') as f:
        json.dump(coda_detail_captions_correct, f, indent=4, ensure_ascii=False)














