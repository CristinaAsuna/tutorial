"""关卡 3：Q-Former instruction tokenizer 与 LLM prompt tokenizer 不能混用。"""


def build_llm_prefix(visual_queries, llm_prompt_embeds):
    # TODO: visual query 先过投影，再放在有效 LLM prompt 前；prefix label 全为 -100。
    raise NotImplementedError("参考 InstructBlipForConditionalGeneration.forward")
