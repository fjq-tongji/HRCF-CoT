import torch
import json

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import re


def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image


class ImageTextDataset(Dataset):
    def __init__(self, questions, captions, image_folder):
        self.questions = questions
        self.captions = captions
        self.image_folder = image_folder

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        k = idx // 6
        prompt = self.captions[k]['final_texts_'] + '\n' + question['text'] + '\n' + 'Please answer the question only with yes or no.'
        image_file = self.image_folder + question['image']
        image = load_image(image_file)
        return prompt, image


def collate_fn(batch):
    prompts, images = zip(*batch)
    return list(prompts), list(images)


def eval_model_batch(model, tokenizer, image_processor, images, queries, conv_mode, args):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    processed_prompts = []
    for query in queries:
        if IMAGE_PLACEHOLDER in query:
            if model.config.mm_use_im_start_end:
                query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
            else:
                query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
        else:
            if model.config.mm_use_im_start_end:
                query = image_token_se + "\n" + query
            else:
                query = DEFAULT_IMAGE_TOKEN + "\n" + query

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        processed_prompts.append(conv.get_prompt())

    image_sizes = [image.size for image in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_tensors = [
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        for prompt in processed_prompts
    ]
    max_length = max(tensor.shape[1] for tensor in input_tensors)
    padded_tensors = [
        torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[1]))
        for tensor in input_tensors
    ]
    input_ids = torch.cat(padded_tensors).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #print(sentences)

    return sentences



if __name__ == "__main__":
    # Define the parameters
    coco_imgs_file = '/share/home/MLLMS/CODA2022/images_total/'
    coco_random_json_file = '/share/home/MLLMS/POPE/CODA2022_pope_adversarial.json'  ##问题
    description_file = './Each_stage_texts_CODA/Final_descriptions.json'  ##详细的描述
    output_json_file = './Each_stage_texts_CODA/CODA_llava_7b_adversarial_answers.json'  ##输出的yes/no结果
    model_path = "/share/home/MLLMS/0_MLLM_weights/LLaVA-v1.5-7b_weights"
    model_base = None
    conv_mode = None
    sep = ","
    temperature = 0.2
    top_p = None
    num_beams = 1
    batch_size_ = 1
    max_new_tokens = 5

    ##所有的coco_json文件，存到列表中
    coco_random_questions = []
    with open(coco_random_json_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                coco_random_questions.append(item)


    ##所有的详细描述结果，存到列表中
    with open(description_file, 'r', encoding='utf-8') as f:
        description_captions = json.load(f)



    dataset = ImageTextDataset(coco_random_questions, description_captions, coco_imgs_file)
    dataloader = DataLoader(dataset, batch_size=batch_size_, shuffle=False, collate_fn=collate_fn)

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )

    if "llama-2" in model_name.lower():
        inferred_conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        inferred_conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        inferred_conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        inferred_conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        inferred_conv_mode = "mpt"
    else:
        inferred_conv_mode = "llava_v0"

    if conv_mode is not None and inferred_conv_mode != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while specified conv-mode is {}, using {}".format(
                inferred_conv_mode, conv_mode, conv_mode
            )
        )
    else:
        conv_mode = inferred_conv_mode

    class Args:
        def __init__(self, temperature, top_p, num_beams, max_new_tokens):
            self.temperature = temperature
            self.top_p = top_p
            self.num_beams = num_beams
            self.max_new_tokens = max_new_tokens

    args = Args(temperature, top_p, num_beams, max_new_tokens)

    answers = []
    for batch in dataloader:
        batch_queries, batch_images = batch
        batch_answers = eval_model_batch(model, tokenizer, image_processor, batch_images, batch_queries, conv_mode,
                                         args)
        answers.extend(batch_answers)
        print(batch_answers)
        print(len(answers))


    ############################################################################################################
    final_lst = []
    for j in range(len(answers)):
        dict = {
            'image': coco_random_questions[j]['image'],
            'question': coco_random_questions[j]['text'],
            'answer': answers[j]
        }
        final_lst.append(dict)

    with open(output_json_file, 'w', encoding='utf-8') as f:
        for d in final_lst:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')






