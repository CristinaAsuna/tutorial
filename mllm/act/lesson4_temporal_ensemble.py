"""关卡 4：融合重叠 chunk 中针对同一当前时刻的 action 预测。"""


def ensemble_current_action(history, decay):
    # TODO: history 从 oldest 到 newest；各 chunk 选择它对当前时刻的 offset，再按 exp(-decay*i) 加权。
    raise NotImplementedError("参考 TemporalEnsembler.add")
