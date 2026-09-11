"""Attention-mask conventions: bool ``True`` always means an allowed key."""
import torch


def key_padding_allow_mask(valid_tokens: torch.Tensor) -> torch.Tensor:
    """Validate and return a `(B, K)` bool mask where True is a real token."""
    if valid_tokens.ndim != 2:
        raise ValueError("valid_tokens must have shape (batch, sequence)")
    if not torch.all((valid_tokens == 0) | (valid_tokens == 1)):
        raise ValueError("valid_tokens values must be 0/1 or bool")
    return valid_tokens.bool()


def causal_allow_mask(query_length: int, key_length: int | None = None, *, device=None) -> torch.Tensor:
    """Return `(Q, K)` True-for-allowed causal attention mask."""
    if query_length < 1:
        raise ValueError("query_length must be positive")
    key_length = query_length if key_length is None else key_length
    if key_length < 1:
        raise ValueError("key_length must be positive")
    # For decoding with a cache, the final Q positions attend within a K prefix.
    offset = key_length - query_length
    q = torch.arange(query_length, device=device)[:, None] + offset
    k = torch.arange(key_length, device=device)[None, :]
    return k <= q


def combine_allow_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    """Broadcast-compatible logical AND; None means no restriction."""
    result = None
    for mask in masks:
        if mask is None:
            continue
        mask = mask.bool()
        result = mask if result is None else result & mask
    return result
