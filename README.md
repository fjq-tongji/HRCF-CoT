<div style="text-align: center;">
  <h1>HRCF-CoT: Hierarchical Reasoning and Cascaded Feedback Framework for Mitigating Hallucination and Enriching Understanding in Traffic Scenarios</h1>
</div>


> With the rapid development of Vision-Language Models (VLMs), large-scale, high-quality annotated data have become increasingly critical for effective model training. While leveraging VLMs for automatic annotation provides a scalable alternative to labor-intensive manual labeling, this approach is still hindered by two major issues: hallucination and omission. 
To tackle these challenges, we propose a novel HRCF-CoT framework that integrates hierarchical reasoning with cascaded feedback, guiding the model toward producing accurate and semantically rich descriptions. 
Specifically, we first introduce a residual-guided hallucination elimination method that corrects texts and yields context-aware descriptions. 
Then, we design an element-aware question-answering module to extract diverse elements, significantly improving the completeness of scene understanding.
After removing redundant contents, we leverage VLMs to construct graph-based scene representations, further deriving attribute-aware information for key objects.
Extensive experiments on seven hallucination benchmarks demonstrate the effectiveness of our framework: hallucination rates of GPT-4o and DeepSeek-VL2 are reduced by 1.58% on POPE and 5.60% on Object HalBench, respectively. Furthermore, LLaVA-1.6 model achieves a 48.26% increase in detailness and a 53.25% improvement in richness score.

If you have any question, please feel free to email fanjq@tongji.edu.cn.

## :fire: News
- Video demo [https://github.com/fjq-tongji/HRCF-CoT/releases/tag/video]
- Created datasets []


