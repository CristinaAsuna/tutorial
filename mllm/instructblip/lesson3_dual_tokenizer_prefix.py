"""关卡 3：投影视觉 query，构造 LLM prefix/prompt/answer。
前置：Linear、cat、loss mask；两套 id 显式分开，指令只送 Q-Former。
形状：queries(B,M,Dq) -> prefix(B,M,Dlm)；每行拼接有效 prompt/answer。
手算：M=2,prompt=[9,0] mask=[1,0],answer=[14,0] mask=[1,0] -> labels=[-100,-100,-100,14]。
常见错：padding 留在回答前；监督视觉 prefix；混用两套词表。
接入：PracticeInstructBlip.forward/generate；生成的 answer 是 (B,0)。
外层循环、去 padding、embedding、右补齐已给出，只实现三个小 helper。
"""
import torch
from reference_instructblip import IGNORE_INDEX


def project_visual_queries(visual_queries, llm_proj):
    """(B,M,Dq) -> (B,M,Dlm)，投影必须保留参数梯度。"""
    # TODO 3.1：调用 llm_proj；不要 detach/no_grad。
    raise NotImplementedError("InstructBLIP lesson 3.1: project_visual_queries 尚未完成；见 HINTS.md")


def concatenate_prefix_row(visual_prefix, prompt_embeds, answer_embeds):
    """单样本：(M,D),(P,D),(A,D) -> (M+P+A,D)。A 可以为 0。"""
    # TODO 3.2：按 prefix -> prompt -> answer 沿 dim=0 拼接。
    raise NotImplementedError("InstructBLIP lesson 3.2: concatenate_prefix_row 尚未完成；见 HINTS.md")


def prefix_answer_labels(prefix_length: int, prompt_length: int, valid_answer_ids):
    """单样本：(A,) -> (M+P+A,)，只有 answer 是监督目标。"""
    # TODO 3.3：创建 M+P 个 IGNORE_INDEX，device/dtype 同 answer_ids，再 cat answer_ids。
    raise NotImplementedError("InstructBLIP lesson 3.3: prefix_answer_labels 尚未完成；见 HINTS.md")


def build_llm_prefix(visual_queries, llm_prompt_embeds=None, *, llm_proj=None, embed_tokens=None,
                     prompt_ids=None, answer_ids=None, prompt_mask=None, answer_mask=None):
    """提供的外层工程骨架；完整接口返回 embeds/mask/labels 三元组。"""
    if llm_proj is None:
        if llm_prompt_embeds is None:
            raise ValueError("legacy interface requires prompt embeddings")
        # 旧二参数接口只有 embedding 拼接；仍调用学生拼接 helper。
        rows = [concatenate_prefix_row(visual_queries[r], llm_prompt_embeds[r], llm_prompt_embeds[r, :0]) for r in range(visual_queries.shape[0])]
        return torch.stack(rows)
    if prompt_ids is None or answer_ids is None or embed_tokens is None:
        raise ValueError("full interface requires embedding, prompt ids and answer ids")
    if prompt_ids.ndim != 2 or answer_ids.ndim != 2 or prompt_ids.shape[0] != visual_queries.shape[0] or answer_ids.shape[0] != visual_queries.shape[0]:
        raise ValueError("matching visual/prompt/answer batches required")
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool) if prompt_mask is None else prompt_mask.bool()
    answer_mask = torch.ones_like(answer_ids, dtype=torch.bool) if answer_mask is None else answer_mask.bool()
    if prompt_mask.shape != prompt_ids.shape or answer_mask.shape != answer_ids.shape:
        raise ValueError("prompt/answer masks must match ids")
    prefix = project_visual_queries(visual_queries, llm_proj)
    rows, targets, masks = [], [], []
    for row in range(visual_queries.shape[0]):
        valid_prompt_ids = prompt_ids[row, prompt_mask[row]]
        valid_answer_ids = answer_ids[row, answer_mask[row]]
        prompt = embed_tokens(valid_prompt_ids)
        answer = embed_tokens(valid_answer_ids)
        packed = concatenate_prefix_row(prefix[row], prompt, answer)
        rows.append(packed)
        targets.append(prefix_answer_labels(prefix.shape[1], prompt.shape[0], valid_answer_ids))
        masks.append(torch.ones(packed.shape[0], dtype=torch.bool, device=packed.device))
    embeds = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
    valid = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)
    labels = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=IGNORE_INDEX)
    return embeds, valid, labels
