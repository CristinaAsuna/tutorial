"""第1关：同一干净图像的两个空间对应路径。
前置：foundations 1/4、requires_grad 与 no_grad。
输入 images=(B,3,16,16)，输出 clean=(B,4,8,8)、targets=(B,16,24)。
教师patch4覆盖RGB4x4；codec stride2后student patch2也覆盖RGB4x4。
手算：第(row=1,col=2)格按宽优先flatten为1*4+2=6，两路必须指同一区域。
模块已由scaffolding冻结并在train时保持eval；这里负责截断目标计算图。
常见错：teacher读带噪latent；把两路结果在feature轴拼接；误称随机teacher为预训练。
检查：python3 -m mllm.repa.check_lessons --lesson 1（先配置edu_core路径）。
"""
import torch


def clean_targets(system, images):
    # TODO 1.1：确认images为(B,3,16,16)，错误尺寸抛ValueError。
    # TODO 1.2：no_grad中分别调用system.codec.encode(images)、system.teacher(images)。
    # TODO 1.3：返回两个结果，不加入noise，不用detach student hidden。
    raise NotImplementedError('REPA lesson1.clean_targets TODO 1.1-1.3')
