"""Sequence batching primitives shared by language and multimodal tutorials."""
import torch

IGNORE_INDEX = -100


def right_pad(sequences: list[torch.Tensor], *, pad_value: int, dtype=None):
    if not sequences or any(x.ndim != 1 for x in sequences):
        raise ValueError("sequences must be a non-empty list of 1-D tensors")
    length = max(x.numel() for x in sequences)
    dtype = dtype or sequences[0].dtype
    output = torch.full((len(sequences), length), pad_value, dtype=dtype, device=sequences[0].device)
    valid = torch.zeros((len(sequences), length), dtype=torch.bool, device=output.device)
    for row, sequence in enumerate(sequences):
        output[row, :sequence.numel()] = sequence
        valid[row, :sequence.numel()] = True
    return output, valid


def expand_single_image_token(input_ids: torch.Tensor, image_features: torch.Tensor, embed_tokens, *, image_token_index: int, attention_mask=None, labels=None):
    """Replace exactly one image sentinel in each row with projected image features.

    Returns right-padded `(inputs_embeds, attention_mask, labels_or_none)`.
    """
    if input_ids.ndim != 2 or image_features.ndim != 3 or input_ids.shape[0] != image_features.shape[0]:
        raise ValueError("input_ids (B,L) and image_features (B,N,D) need matching batches")
    if (input_ids == image_token_index).sum(1).ne(1).any():
        raise ValueError("each sample must contain exactly one image sentinel")
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    if attention_mask.shape != input_ids.shape or (labels is not None and labels.shape != input_ids.shape):
        raise ValueError("attention_mask and labels must match input_ids")
    rows, masks, target_rows, patches = [], [], [], image_features.shape[1]
    for row in range(input_ids.shape[0]):
        pos = (input_ids[row] == image_token_index).nonzero().item()
        text = torch.cat((embed_tokens(input_ids[row, :pos]), image_features[row], embed_tokens(input_ids[row, pos + 1:])))
        rows.append(text)
        masks.append(torch.cat((attention_mask[row, :pos], torch.ones(patches, device=input_ids.device, dtype=torch.bool), attention_mask[row, pos + 1:])))
        if labels is not None:
            target_rows.append(torch.cat((labels[row, :pos], torch.full((patches,), IGNORE_INDEX, dtype=labels.dtype, device=labels.device), labels[row, pos + 1:])))
    max_length, dim = max(x.shape[0] for x in rows), image_features.shape[-1]
    embeds = image_features.new_zeros(len(rows), max_length, dim)
    mask = torch.zeros(len(rows), max_length, dtype=torch.bool, device=input_ids.device)
    targets = torch.full((len(rows), max_length), IGNORE_INDEX, dtype=labels.dtype, device=labels.device) if labels is not None else None
    for row, (tokens, valid) in enumerate(zip(rows, masks)):
        embeds[row, :tokens.shape[0]], mask[row, :tokens.shape[0]] = tokens, valid
        if targets is not None:
            targets[row, :tokens.shape[0]] = target_rows[row]
    return embeds, mask, targets
