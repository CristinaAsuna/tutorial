"""Explicit lesson composition. Reference classes supply initialization/outer routing only."""
from types import SimpleNamespace
from scaffolding import OpenVLAPolicy
import lesson1_action_normalization
import lesson2_action_tokenizer
import lesson3_fused_vision
import lesson4_multimodal_sequence
import lesson5_training
import lesson6_autoregressive_control


def practice_ops():
    return SimpleNamespace(
        fit_action_stats=lesson1_action_normalization.fit_action_stats,
        normalize_actions=lesson1_action_normalization.normalize_actions,
        unnormalize_actions=lesson1_action_normalization.unnormalize_actions,
        encode_actions=lesson2_action_tokenizer.encode_actions,
        decode_actions=lesson2_action_tokenizer.decode_actions,
        fuse_features=lesson3_fused_vision.fuse_features,
        pack_single=lesson4_multimodal_sequence.pack_single,
        action_labels=lesson4_multimodal_sequence.action_labels,
        shifted_loss=lesson5_training.shifted_loss,
        training_step=lesson5_training.training_step,
        generate_action_ids=lesson6_autoregressive_control.generate_action_ids,
        predict_action=lesson6_autoregressive_control.predict_action,
    )


def build_toy_openvla() -> OpenVLAPolicy:
    return OpenVLAPolicy(ops=practice_ops())
