"""关卡 2：用图像 patch 替换 sentinel，同时展开 mask/labels。
目标：负 sentinel 不进入 embedding；前置：cat、embedding、右 padding。
形状：ids/mask/labels (B,L)，image (B,N,D) -> embeds (B,L+N-1,D)。
手算：ids=[3,-200,7], labels=[-100,-100,7], N=2 -> [-100,-100,-100,7]。
常见错：直接 embedding(-200)；图像标签未设 -100；丢失原文本 padding。
接入：PracticeLlava.pack_multimodal_inputs 调用已提供的批量骨架。
先实现两个短 helper，工程验证/循环/补齐已给出。
"""
import torch
from reference_llava import IMAGE_TOKEN_INDEX, IGNORE_INDEX


def replace_image_sentinel(row_ids, image_features, embed_tokens, image_position: int):
    """单样本：(L,), (N,D) -> (L+N-1,D)。pos 由骨架验证提供。"""
    # TODO 2.1：分别 embedding row_ids[:pos] 与 row_ids[pos+1:]，负 sentinel 被排除。
    # TODO 2.2：沿 token 轴 cat(before,image_features,after)，返回 embedding 序列。
    raise NotImplementedError("LLaVA lesson 2.1: replace_image_sentinel 尚未完成；见 HINTS.md")


def expand_image_supervision(row_mask, row_labels, image_position: int, num_patches: int):
    """单样本：(L,), labels 或 None -> expanded_mask, expanded_labels 或 None。"""
    # TODO 2.3：mask 的 pos 替换为 N 个 True，cat 两侧原 mask。
    # TODO 2.4：有 labels 时同位置替换为 N 个 -100，无 labels 时返回 None。
    # 插入值必须在原张量 device 上，保持 dtype；长度应为 L+N-1。
    raise NotImplementedError("LLaVA lesson 2.2: expand_image_supervision 尚未完成；见 HINTS.md")


def pack_one_image(input_ids: torch.Tensor, image_features: torch.Tensor, embed_tokens,
                   attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None):
    """已提供的批量工程骨架；只调用上面两个学生 helper，不回退答案。"""
    if input_ids.ndim != 2 or image_features.ndim != 3 or input_ids.shape[0] != image_features.shape[0]:
        raise ValueError("ids (B,L) and image features (B,N,D) require matching batches")
    if input_ids.shape[0] == 0 or image_features.shape[1] == 0:
        raise ValueError("batch and patch count must be positive")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask.shape != input_ids.shape or not ((attention_mask == 0) | (attention_mask == 1)).all():
        raise ValueError("attention_mask must match ids and contain only 0/1")
    if labels is not None and labels.shape != input_ids.shape:
        raise ValueError("labels must match ids")
    if not (input_ids == IMAGE_TOKEN_INDEX).sum(1).eq(1).all():
        raise ValueError("each row must contain exactly one image sentinel")
    text_ids = input_ids[input_ids != IMAGE_TOKEN_INDEX]
    if (text_ids < 0).any() or (text_ids >= embed_tokens.num_embeddings).any():
        raise ValueError("text token id outside embedding vocabulary")
    embeddings, masks, targets = [], [], []
    for row in range(input_ids.shape[0]):
        pos = (input_ids[row] == IMAGE_TOKEN_INDEX).nonzero().item()
        embeddings.append(replace_image_sentinel(input_ids[row], image_features[row], embed_tokens, pos))
        mask, target = expand_image_supervision(attention_mask[row], None if labels is None else labels[row], pos, image_features.shape[1])
        masks.append(mask)
        if labels is not None:
            targets.append(target)
    # 每个 helper 返回单样本结果，pad_sequence 只承担批量工程。
    packed = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    valid = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)
    output_labels = None if labels is None else torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=IGNORE_INDEX)
    return packed, valid, output_labels
