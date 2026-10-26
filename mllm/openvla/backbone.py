"""Pretrained language integration; the dual visual toy branches remain explicit."""
from types import SimpleNamespace
import torch
from edu_core.backbones.components import EmbeddingLanguage
from scaffolding import OpenVLAPolicy


class ActionVocabulary:
    """256 descending non-special IDs. Bin order is unchanged; reserved tokens are untouched."""
    def __init__(self, tokenizer, vocab_size):
        excluded = set(tokenizer.all_special_ids)
        ids = [i for i in range(vocab_size-1, -1, -1) if i not in excluded][:256]
        if len(ids) != 256:
            raise ValueError("not enough ordinary vocabulary tokens for 256 action codes")
        self.ids = torch.tensor(ids)

    def to_model(self, canonical, vocab_size):
        index = vocab_size-canonical-1
        return self.ids.to(canonical.device)[index]

    def to_canonical(self, ids, vocab_size):
        table = self.ids.to(ids.device)
        match = ids[..., None] == table
        index = match.long().argmax(-1)
        # As with the official decoder's out-of-action-range IDs, clip to an endpoint.
        return torch.where(match.any(-1), vocab_size-index-1, torch.zeros_like(ids))

    def wrap_ops(self, ops):
        def encode(actions, vocab_size=320, n_bins=256):
            if n_bins != 256:
                raise ValueError("pretrained action vocabulary uses 256 codes")
            return self.to_model(ops.encode_actions(actions, vocab_size, n_bins), vocab_size)
        def decode(ids, vocab_size=320, n_bins=256):
            return ops.decode_actions(self.to_canonical(ids, vocab_size), vocab_size, n_bins)
        return SimpleNamespace(**{**vars(ops), "encode_actions": encode, "decode_actions": decode})


def build_with_backbone(backbone, implementation="reference"):
    if implementation == "practice":
        from practice_openvla import practice_ops
        ops = practice_ops()
    elif implementation == "reference":
        from reference_openvla import reference_ops
        ops = reference_ops()
    else:
        raise ValueError("choose reference or practice")
    vocabulary = ActionVocabulary(backbone.tokenizer, backbone.spec.vocab_size)
    model = OpenVLAPolicy(ops=vocabulary.wrap_ops(ops), language=EmbeddingLanguage(backbone))
    model.action_vocabulary = vocabulary
    return model
