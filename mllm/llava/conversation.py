"""Prompt construction for the toy LLaVA SFT examples."""
import torch

from reference_llava import IGNORE_INDEX, IMAGE_TOKEN_INDEX


def build_sft_example(system_ids, user_ids, answer_ids, *, bos_token_id=1, eos_token_id=2):
    """Build ``BOS system <image> user answer EOS`` and loss labels.

    Only answer tokens and EOS are supervised.  The image sentinel is an input
    control value, never a vocabulary id and never a loss target.
    """
    ids = [bos_token_id, *system_ids, IMAGE_TOKEN_INDEX, *user_ids, *answer_ids, eos_token_id]
    labels = [IGNORE_INDEX] * (1 + len(system_ids) + 1 + len(user_ids)) + [*answer_ids, eos_token_id]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def right_pad_examples(examples, pad_token_id=0):
    """Right-pad variable-length ``(ids, labels)`` pairs for a batch."""
    max_len = max(ids.numel() for ids, _ in examples)
    ids_batch, labels_batch, masks = [], [], []
    for ids, labels in examples:
        pad = max_len - ids.numel()
        ids_batch.append(torch.cat((ids, torch.full((pad,), pad_token_id, dtype=torch.long))))
        labels_batch.append(torch.cat((labels, torch.full((pad,), IGNORE_INDEX, dtype=torch.long))))
        masks.append(torch.cat((torch.ones(ids.numel(), dtype=torch.bool), torch.zeros(pad, dtype=torch.bool))))
    return torch.stack(ids_batch), torch.stack(labels_batch), torch.stack(masks)
