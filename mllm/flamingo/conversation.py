"""Prompt and loss construction for interleaved Flamingo toy examples."""
import torch

from reference_flamingo import IGNORE_INDEX, IMAGE_TOKEN_INDEX


def build_interleaved_sft_example(segments: list[tuple[str, list[int]]], answer_ids: list[int], *, bos_token_id: int = 1, eos_token_id: int = 2):
    """Build a prompt from ``text`` and ``image`` segments; supervise answer only."""
    ids, labels = [bos_token_id], [IGNORE_INDEX]
    for kind, tokens in segments:
        if kind == "text":
            ids.extend(tokens); labels.extend([IGNORE_INDEX] * len(tokens))
        elif kind == "image":
            if tokens:
                raise ValueError("an image segment takes an empty token list")
            ids.append(IMAGE_TOKEN_INDEX); labels.append(IGNORE_INDEX)
        else:
            raise ValueError("segment kind must be 'text' or 'image'")
    ids.extend(answer_ids + [eos_token_id])
    labels.extend(answer_ids + [eos_token_id])
    return torch.tensor(ids), torch.tensor(labels)


def right_pad_examples(examples: list[tuple[torch.Tensor, torch.Tensor]], pad_token_id: int = 0):
    max_len = max(ids.numel() for ids, _ in examples)
    ids, labels, masks = [], [], []
    for row_ids, row_labels in examples:
        pad = max_len - row_ids.numel()
        ids.append(torch.cat((row_ids, torch.full((pad,), pad_token_id, dtype=torch.long))))
        labels.append(torch.cat((row_labels, torch.full((pad,), IGNORE_INDEX, dtype=torch.long))))
        masks.append(torch.cat((torch.ones(row_ids.numel(), dtype=torch.bool), torch.zeros(pad, dtype=torch.bool))))
    return torch.stack(ids), torch.stack(labels), torch.stack(masks)
