import time
import os
import base64
import re
import openai
import json
from filelock import FileLock


#################################################################### GPT-4的API--文字+图像版本
# 设置 API Key
openai.api_key = "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"     ##### 课题组的API key
openai.api_key = "sk-proj-OiklAoCznCELNxEydFOmT3BlbkFJo5OTkrC0QCrCKCpIlOE1"     ##### 自己申请的API key


# 使用文件锁来确保每个进程单独写入
lock = FileLock('./GPT_dafen_5LVLMs_accuracy.json.lock')
# 保存输出结果的文档
output_file = './GPT4-scores/GPT_CODA_llava_1.6_7b_richness.json'               ########### 8个文件的答案存到一个json里面

initial_refined_text_path = './GPT4-scores/CODA_all_three_branch_infor_final_integrate_total.json'



gpt4_evaluation_score_ = '''
You are required to score the performance of two AI assistants strictly in describing a given image. Please return two different scores.

Please rate the responses of the assistants on a scale of 1 to 5, where a higher score indicates better performance, according to the following criteria:
Richness: whether the object categories, attribute, and position described in the texts are rich enough. 

Please strictly follow the output format of the following example, without any explanations.
Please output a single line, containing only two different values indicating the scores for Assistant 1 and 2, respectively. 
Please avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

[Assistant 1]
{Initial_response_1}

[Assistant 2]
{Final_response_2}

An output example format:
Richness scores of the two answers:
2 4
'''



# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")





##############################################################################################
def GPT_query(image_path, initial_response_1, final_response_2):
    base64_image = encode_image(image_path)

    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": gpt4_evaluation_score_.format(Initial_response_1=initial_response_1,
                                                                                   Final_response_2=final_response_2)},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=50
            )
            print("Token用量情况：", "输入总Token：", response.usage["total_tokens"])

            return response['choices'][0]['message']['content']

        except openai.error.RateLimitError as e:
            print(f"Rate limit reached: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 等待 2 秒再重试



####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
######## 提取每一行的数字并存入列表
initial_response_accuracy = []
final_response_accuracy = []
initial_response_completeness = []
final_response_completeness = []


with open(initial_refined_text_path, 'r', encoding='utf-8') as f:
    input_file_ = json.load(f)



for k in range(500):
    print(k)
    time_1 = time.time()

    image_path = input_file_[k]['image']
    image_path_new = '/data/NuScenes_CAM_FRONT/' + input_file_[k]['image'].split('/')[-1]
    initial_response = input_file_[k]['initial_texts']
    final_response = input_file_[k]['final_texts_']

    text = GPT_query(image_path_new, initial_response, final_response)
    print(text)

    ##########################################################################################
    # 使用正则表达式匹配两行数字
    numbers = re.findall(r'\d+\s+\d+', text)
    print(numbers)

    for i, line in enumerate(numbers):
        nums = list(map(int, line.split()))  # 拆分并转换为整数
        if i == 0:
            initial_response_accuracy.append(nums[0])
            final_response_accuracy.append(nums[1])

    time_2 = time.time()
    print(time_2 - time_1)



initial_response_accuracy_avg = sum(initial_response_accuracy) / len(initial_response_accuracy)
final_response_accuracy_avg = sum(final_response_accuracy) / len(final_response_accuracy)


print(initial_response_accuracy_avg)
print(final_response_accuracy_avg)






with lock:
    with open(output_file, 'a', encoding='utf-8') as f:
        # 将数据写入文件时追加（防止覆盖）
        json.dump(initial_response_accuracy_avg, f, ensure_ascii=False, indent=4)
        f.write('\n')
        json.dump(final_response_accuracy_avg, f, ensure_ascii=False, indent=4)
        f.write('\n')
        f.write('----------------------------------------------')
        f.write('\n')












