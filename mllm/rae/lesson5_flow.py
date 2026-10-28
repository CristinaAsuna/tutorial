"""第5关：flow。
前置：foundations patchify/attention/mask/EMA；各函数目的、形状和手算见下方。
接入：practice_ops把本关函数注入RAECourse；单关检查提供其他关参考外围。
检查（仓库根）：PYTHONPATH=general_utils/edu_core python3 -m mllm.rae.check_lessons --lesson 5
常见错误：对错轴归约、错时间方向、切断可训练梯度、把eval误当冻结。
"""
import torch
from torch.nn import functional as F
from edu_core.vision import unpatchify
from edu_core.training import update_ema

def shift_time(time, effective_dim, reference_dim=4096):
    """目的：按latent总元素数调整噪声时间；不是只看通道数。
    形状：time=(B,)，m=N*D，alpha=sqrt(m/n)，输出(B,)。
    手算：m/n=4,t=.5 → 2/3；m/n=1 → .5。toy m=1024<n=4096，因此.5→1/3。
    """
    # TODO 5.1.1: 验证effective_dim/reference_dim>0、time在[0,1]。
    # TODO 5.1.2: alpha=sqrt(effective_dim/reference_dim)，计算alpha*time/(1+(alpha-1)*time)。
    # TODO 5.1.3: 保持device/dtype与(B,)形状；不能截断成整数或只看通道维度。
    raise NotImplementedError("RAE lesson5.shift_time TODO 5.1")


def flow_pair(latent, time, noise):
    """目的：时间0是数据，时间1是噪声；监督是插值对时间的导数。
    形状：latent/noise=(B,N,D)，time=(B,)需扩成(B,1,1)。
    手算：z=2,noise=0,t=.25 → x=1.5，velocity=-2。
    """
    # TODO 5.2.1: 验证latent/noise形状相同且time为(B,)；time[:,None,None]用于广播。
    # TODO 5.2.2: noisy=(1-time)*latent+time*noise。
    # TODO 5.2.3: target=noise-latent；返回(noisy,target)，不是预测干净latent。
    raise NotImplementedError("RAE lesson5.flow_pair TODO 5.2")


def velocity_loss(prediction, target):
    """目的：对速度场回归，不是让网络直接预测干净图像。
    形状：prediction/target=(B,N,D) → scalar。
    手算：prediction=[1,3],target=[0,1] → (1+4)/2=2.5。
    """
    # TODO 5.3.1: target.detach()防止teacher方向反传。
    # TODO 5.3.2: 对prediction-target的逐元素平方在所有轴mean；不detach prediction。
    raise NotImplementedError("RAE lesson5.velocity_loss TODO 5.3")


