"""Optional component integration; keeps the learner's projector and packing mechanisms."""
from edu_core.backbones.components import VisionWithCLS, CausalLanguage
from reference_llava import LlavaForConditionalGeneration, LlavaProjector


def build_with_backbone(backbone, implementation="reference"):
    if implementation == "practice":
        from practice_llava import PracticeLlava
        cls = PracticeLlava
    elif implementation == "reference":
        cls = LlavaForConditionalGeneration
    else:
        raise ValueError("choose reference or practice")
    return cls(VisionWithCLS(backbone), CausalLanguage(backbone),
               LlavaProjector(backbone.vision_dim, backbone.spec.hidden_size),
               pad_token_id=backbone.tokenizer.pad_token_id)
