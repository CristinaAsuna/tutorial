"""关卡 4：时间对齐的重叠窗口融合。
目标：每时刻重新预测，并只融合指向当前时刻的动作。
前置：Python list、stack、指数权重；history 按 oldest->newest 存放。
形状：n 个 (K,A) chunk -> 当前 action(A)；只保留最近 K 个窗口。
手算：旧[1,2,3]、新[10,20,30]，decay=ln2 -> (2+0.5*10)/1.5=14/3。
常见错：混合所有 offset=0；按值是否为0判断有效；颠倒权重顺序。
接入：practice TemporalEnsembler.add 管理历史，调用 ensemble_current_action。
"""
import torch

def ensemble_current_action(history, decay):
    # TODO 4.1：n=len(history)>0，取 chunk_i[n-1-i]，这些动作才对应同一时刻。
    # TODO 4.2：weights=exp(-decay*arange(n))，oldest 权重最高；保持 dtype/device。
    # TODO 4.3：加权求和/权重总和，输出(A)。
    raise NotImplementedError("ACT TODO 4.1-4.3: ensemble_current_action")
