<div style="text-align: center;">
  <h1>HRCF-CoT: Hierarchical Reasoning and Cascaded Feedback Framework for Mitigating Hallucination and Enriching Understanding in Traffic Scenarios</h1>
</div>


> With the rapid development of Vision-Language Models (VLMs), large-scale, high-quality annotated data have become increasingly critical for effective model training. While leveraging VLMs for automatic annotation provides a scalable alternative to labor-intensive manual labeling, this approach is still hindered by two major issues: hallucination and omission. 
To tackle these challenges, we propose a novel HRCF-CoT framework that integrates hierarchical reasoning with cascaded feedback, guiding the model toward producing accurate and semantically rich descriptions. 
Specifically, we first introduce a residual-guided hallucination elimination method that corrects texts and yields context-aware descriptions. 
Then, we design an element-aware question-answering module to extract diverse elements, significantly improving the completeness of scene understanding.
After removing redundant contents, we leverage VLMs to construct graph-based scene representations, further deriving attribute-aware information for key objects.
Extensive experiments on seven hallucination benchmarks demonstrate the effectiveness of our framework: hallucination rates of GPT-4o and DeepSeek-VL2 are reduced by 1.58% on POPE and 5.60% on Object HalBench, respectively. Furthermore, LLaVA-1.6 model achieves a 48.26% increase in detailness and a 53.25% improvement in richness score.



## :fire: News
- Video demo [https://github.com/fjq-tongji/HRCF-CoT/releases/tag/video]
- Created datasets [https://github.com/fjq-tongji/HRCF-CoT/tree/main/Annotation_results]


## :book: Model
<p align="center">
  <img src="images/overall-new8.jpg" alt="Logo" width="1000">
<p align="center">



## :pill: Installation
1. LLaVA: https://github.com/haotian-liu/LLaVA
2. mPLUG-Owl: https://github.com/X-PLUG/mPLUG-Owl
3. MiniGPT-4: https://github.com/Vision-CAIR/MiniGPT-4
4. InternVL: https://github.com/OpenGVLab/InternVL
5. BLIP-2: https://huggingface.co/Salesforce/blip2-flan-t5-xxl
6. InstructBLIP: https://huggingface.co/Salesforce/instructblip-flan-t5-xxl
7. RAM: https://github.com/xinyu1205/recognize-anything
8. GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
9. Qwen2-VL: https://github.com/QwenLM/Qwen3-VL
10. MiniCPM-o: https://github.com/OpenBMB/MiniCPM
11. InternVL2.5: https://github.com/OpenGVLab/InternVL
12. Ovis2: https://github.com/AIDC-AI/Ovis
13. GLM-4V: https://huggingface.co/zai-org/glm-4v-9b


## :star: Inference
Generate annotations using HRCF-CoT framework: 
```
$ python 1.quen2_vl_initial_texts_CODA_1.py
$ python 2.inference_text_critic_minicpm_o_2_6.py
$ python 2.inference_text_critic_internvl_25.py
$ python 2.inference_text_critic_ovis2.py
$ python 3.inference_traffic_element_VQAs.py
$ python 4.inference_text_integrate.py
$ python 5.inference_scene_graph_generation.py
$ python 6.inference_scene_graph_validation.py
```
The specific reasoning code is in:
```
$ python vis_corrector_recap_w.py
```
The detailed code for each module is shown in folder models. 


## :trophy: Experimental Results
### Results in Traffic Scenarios
#### POPE Benchmark:
<p align="center">
  <img src="images/POPE.png" alt="Logo" width="800">  
<p align="center">
  
#### OpenCHAIR Benchmark:
<p align="center">
  <img src="images/OpenCHAIR.png" alt="Logo" width="800">  
<p align="center">
  
#### GPT-4 Evaluation Benchmark:
<p align="center">
  <img src="images/GPT-4.png" alt="Logo" width="450">  
<p align="center">

### Results in General Scenarios
#### POPE Benchmark:
<p align="center">
  <img src="images/POPE_COCO.png" alt="Logo" width="450">  
<p align="center">
  
#### Object HalBench & MMHal-Bench Benchmark:
<p align="center">
  <img src="images/MMHal-Bench.png" alt="Logo" width="450">  
<p align="center">
  
#### AMBER Benchmark:
<p align="center">
  <img src="images/AMBER.png" alt="Logo" width="800">  
<p align="center">

### Qualitative results  
#### Results of Hallucination Correction and Element Compensation:
<p align="center">
  <img src="images/fig-6.png" alt="Logo" width="450"> 
<p align="center">

#### Results of Hallucination Correction and Scene Graph Generation in Corner-Case Senarios:
<p align="center">
  <img src="images/fig-8.png" alt="Logo" width="450"> 
<p align="center">



## :date: Prompt
### Module ②：
<p align="center">
  <img src="images/p2.jpg" alt="Logo" width="450">
<p align="center">
  
### Module ④：
<p align="center">
  <img src="images/p4.jpg" alt="Logo" width="450">
<p align="center">
  
### Module ⑤：
<p align="center">
  <img src="images/p5.jpg" alt="Logo" width="450">
<p align="center">
  
### Module ⑥：
<p align="center">
  <img src="images/p6.jpg" alt="Logo" width="450">
<p align="center">


## :sunflower: Acknowledgement
This repository benefits from the following codes. Thanks for their awesome works.
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl)
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [InternVL](https://github.com/OpenGVLab/InternVL)
- [BLIP-2](https://huggingface.co/Salesforce/blip2-flan-t5-xxl)
- [InstructBLIP](https://huggingface.co/Salesforce/instructblip-flan-t5-xxl)(https://huggingface.co/Salesforce/instructblip-vicuna-13b)
- [Woodpecker](https://github.com/BradyFU/Woodpecker)
- [POPE](https://github.com/AoiDragon/POPE)
- [RAM](https://github.com/xinyu1205/recognize-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

## :scroll: Citation

