"""Exercise routing registry, no implementations."""
GROUPS = [('lesson1_observations', ['action_chunk', 'normalize_values', 'unnormalize_values']),
 ('lesson2_prefix_suffix', ['embed_prefix', 'time_embedding', 'embed_suffix']),
 ('lesson3_block_attention', ['block_mask', 'position_ids']),
 ('lesson4_dual_stream_transformer',
  ['joint_attention', 'joint_forward', 'suffix_attention', 'cached_forward']),
 ('lesson5_flow_matching', ['sample_time', 'flow_pair', 'flow_loss']),
 ('lesson6_training', ['training_step']),
 ('lesson7_cached_control',
  ['cache_prefix', 'euler_step', 'sample_actions', 'execution_chunk', 'receding_control'])]
