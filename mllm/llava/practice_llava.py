"""学生闭环：只复用视觉/语言 toy 外围；论文关键路径调用 lesson。"""
import torch
import reference_llava as ref
import lesson1_vision_and_projector as l1
import lesson2_image_token_packing as l2
import lesson3_sft_loss as l3
from lesson4_training_and_generation import train_one_step

class PracticeLlava(ref.LlavaForConditionalGeneration):
    def encode_images(self, pixel_values):
        return l1.vision_to_llm_tokens(self.vision_encoder, self.projector, pixel_values)
    def pack_multimodal_inputs(self, input_ids, image_features, attention_mask=None, labels=None):
        return l2.pack_one_image(input_ids, image_features, self.llm.embed_tokens, attention_mask, labels)

def build_toy_llava(vocab_size=64):
    return PracticeLlava(ref.MockVisionEncoder(), ref.MockDecoderLM(vocab_size=vocab_size, max_positions=64), ref.LlavaProjector(24,32))

def build_sft_example(system_ids, user_ids, answer_ids, *, bos_token_id=1, eos_token_id=2):
    ids=torch.tensor([bos_token_id,*system_ids,ref.IMAGE_TOKEN_INDEX,*user_ids,*answer_ids,eos_token_id])
    start=2+len(system_ids)+len(user_ids)
    assistant_mask=torch.arange(ids.numel())>=start
    labels=l3.assistant_only_labels(ids.unsqueeze(0), assistant_mask=assistant_mask.unsqueeze(0))[0]
    return ids,labels
