from typing import Dict
from models.Qwen2_VL_traffic_element_VQAs import Traffic_element_extraction_VQAs
from models.MiniCPM_text_critic import Text_critic_minicpm
from models.Intern25_text_critic import Text_critic_internvl_25_
from models.Ovis2_text_critic import Text_critic_ovis_2_
from models.LLama_3_integrate_text import Text_integration
from models.Scene_graph_generation import Scene_graph_generation
from models.Scene_graph_validation import Scene_graph_validation
from models.Scene_graph_to_text import Scene_graph_to_texts



class Corrector:
    def __init__(self, args):
        self.qwen2_vl_traffic_element_VQAs = Traffic_element_extraction_VQAs(args)
        self.text_critic_minicpm_ = Text_critic_minicpm(args)
        self.text_critic_internvl_25_ = Text_critic_internvl_25_(args)
        self.text_critic_ovis_2_ = Text_critic_ovis_2_(args)
        self.llama3_text_integration = Text_integration(args)
        self.scene_graph_generation_ = Scene_graph_generation(args)
        self.scene_graph_validation_ = Scene_graph_validation(args)
        self.scene_graph_to_texts_ = Scene_graph_to_texts(args)

        print("Finish loading models.")



    def correct(self, model_sentence_transformer, model_qwen2_vl, processor_qwen2_vl, model_glm_4V_9B, tokenizer_glm_4V_9B, model_minicpm_o_26, tokenizer_minicpm_o_26,
                model_internvl_2_5, tokenizer_internvl_2_5, model_ovis_2, pipeline_llama_3_8B, model_groundingdino, processor_groundingdino, sample: Dict):

        sample = self.qwen2_vl_traffic_element_VQAs.traffic_element_extraction_VQAs_(model_qwen2_vl, processor_qwen2_vl, model_sentence_transformer, model_glm_4V_9B, tokenizer_glm_4V_9B, sample)
        sample = self.text_critic_minicpm_.text_critic_minicpm_process(model_minicpm_o_26, tokenizer_minicpm_o_26, sample)
        sample = self.text_critic_internvl_25_.text_critic_internvl_25_(model_internvl_2_5, tokenizer_internvl_2_5, sample)
        sample = self.text_critic_ovis_2_.text_critic_ovis_2_(model_ovis_2, sample)
        sample = self.llama3_text_integration.text_integration_(pipeline_llama_3_8B, sample)
        sample = self.scene_graph_generation_.scene_graph_generation_(model_ovis_2, sample)
        sample = self.scene_graph_validation_.scene_graph_validation_all(model_groundingdino, processor_groundingdino, model_qwen2_vl, processor_qwen2_vl, sample)
        sample = self.scene_graph_to_texts_.scene_graph_to_texts_(sample)

        return sample








