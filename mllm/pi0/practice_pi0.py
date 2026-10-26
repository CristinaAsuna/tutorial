"""All defining operations come from lessons; no reference fallback."""
from types import SimpleNamespace
from scaffolding import Pi0Policy
import lesson1_observations
import lesson2_prefix_suffix
import lesson3_block_attention
import lesson4_dual_stream_transformer
import lesson5_flow_matching
import lesson6_training
import lesson7_cached_control


def practice_ops():
    return SimpleNamespace(
        action_chunk=lesson1_observations.action_chunk,
        normalize_values=lesson1_observations.normalize_values,
        unnormalize_values=lesson1_observations.unnormalize_values,
        embed_prefix=lesson2_prefix_suffix.embed_prefix,
        time_embedding=lesson2_prefix_suffix.time_embedding,
        embed_suffix=lesson2_prefix_suffix.embed_suffix,
        block_mask=lesson3_block_attention.block_mask,
        position_ids=lesson3_block_attention.position_ids,
        joint_attention=lesson4_dual_stream_transformer.joint_attention,
        joint_forward=lesson4_dual_stream_transformer.joint_forward,
        suffix_attention=lesson4_dual_stream_transformer.suffix_attention,
        cached_forward=lesson4_dual_stream_transformer.cached_forward,
        sample_time=lesson5_flow_matching.sample_time,
        flow_pair=lesson5_flow_matching.flow_pair,
        flow_loss=lesson5_flow_matching.flow_loss,
        training_step=lesson6_training.training_step,
        cache_prefix=lesson7_cached_control.cache_prefix,
        euler_step=lesson7_cached_control.euler_step,
        sample_actions=lesson7_cached_control.sample_actions,
        execution_chunk=lesson7_cached_control.execution_chunk,
        receding_control=lesson7_cached_control.receding_control,
    )


def build_toy_pi0() -> Pi0Policy:
    return Pi0Policy(ops=practice_ops())
