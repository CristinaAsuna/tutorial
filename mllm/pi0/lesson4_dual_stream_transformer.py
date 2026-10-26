"""第4关 dual_stream_transformer：先读 README 对应数据流。
前置：基础 Tensor/attention；组合函数依赖前序关，单关检查提供外围参考实现。
已初始化模块与属性在 scaffolding.py，网络骨架无需学生重新设计。
局部命令（仓库根）：PYTHONPATH=general_utils/edu_core python3 mllm/pi0/check_lessons.py --lesson 4
常见错误：混淆环境时间与flow时间、反转allow mask、使prefix看见actions、缓存被不断追加。
"""
import torch
from edu_core.backbones import PrefixEncoding
from edu_core.backbones.layers import projected_attention


def joint_attention(prefix_layer, expert_layer, prefix, suffix, positions, allow):
    """prefix=(B,P,Dv)，suffix=(B,1+H,De)，Dv可以不同于De，但attention head几何相同。
    TODO 4.1: 分别用两套 layer.project 得到含RoPE的Q/K和V。
    TODO 4.2: 沿token轴拼Q/K/V，用shared projected_attention和allow。
    TODO 4.3: 按P切分attention结果，再调用各自layer.finish。
    """
    raise NotImplementedError("lesson4.joint_attention: 完成本函数编号任务")


def joint_forward(model, prefix, prefix_valid, suffix):
    """多层双流：prefix在VLM层，suffix在expert层。输出只取动作位 (B,H,7)，不对state算flow loss。
    TODO 4.1: 连接prefix与suffix有效位，调用ops.block_mask/position_ids。
    TODO 4.2: 逐层调用ops.joint_attention。
    TODO 4.3: 去掉suffix首state位，expert_norm/action_out。
    """
    raise NotImplementedError("lesson4.joint_forward: 完成本函数编号任务")


def suffix_attention(expert_layer, suffix, positions, prefix_kv, allow):
    """缓存 prefix K/V=(B,Hkv,P,Dh) 已含RoPE。suffix只计算自己的Q/K/V。
    TODO 4.1: 对suffix调用expert_layer.project。
    TODO 4.2: 拼prefix和suffix的K/V，Q只有suffix。
    TODO 4.3: attention结果送expert_layer.finish，不写回prefix缓存。
    """
    raise NotImplementedError("lesson4.suffix_attention: 完成本函数编号任务")


def cached_forward(model, cache, suffix):
    """每个expert层读同层的prefix缓存，state位置为有效prefix长度，后面动作连续编号。
    TODO 4.1: 构造suffix内state/actions分组mask与prefix有效mask。
    TODO 4.2: suffix位置加cache.valid_mask.sum。
    TODO 4.3: 每层调用ops.suffix_attention。
    TODO 4.4: 去掉state后投影动作速度。
    """
    raise NotImplementedError("lesson4.cached_forward: 完成本函数编号任务")


