"""组合学生机制；只借参考模型的参数初始化与 predict_chunk 外围。无答案回退。"""
from reference_act import ACTPolicy
from lesson1_action_chunks import sample_future_chunks as sample_action_chunks
from lesson2_cvae_style import StyleEncoder
from lesson3_transformer_policy import encode_observation, decode_action_chunk
from lesson4_temporal_ensemble import ensemble_current_action
from lesson5_training_and_inference import policy_forward, training_step

class PracticeACTPolicy(ACTPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.style_encoder=StyleEncoder(self.qpos_dim,self.action_dim,self.chunk_size,
            self.qpos_proj.out_features,self.z_proj.in_features,self.action_decoder.layers[0].self_attn.num_heads)
    def _encode_observation(self, images, qpos, z):
        return encode_observation(self,images,qpos,z)
    def _decode(self, images, qpos, z):
        return decode_action_chunk(self._encode_observation(images,qpos,z),self.action_queries,self.action_decoder,self.action_head)
    def forward(self, images, qpos, action_chunks=None, action_mask=None, *, beta=10.):
        return policy_forward(self,images,qpos,action_chunks,action_mask,beta=beta)

class TemporalEnsembler:
    def __init__(self, chunk_size, action_dim, decay=.01):
        self.chunk_size,self.action_dim,self.decay=chunk_size,action_dim,decay
        self._history=[]
    def add(self,chunk):
        if chunk.shape!=(self.chunk_size,self.action_dim): raise ValueError("chunk shape mismatch")
        self._history.append(chunk)
        self._history=self._history[-self.chunk_size:]
        return ensemble_current_action(self._history,self.decay)

def build_toy_act():
    return PracticeACTPolicy()
