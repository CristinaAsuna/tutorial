"""关卡 4：冻结视觉塔/LLM，只训练 instruction-aware Q-Former 与投影层。"""


def training_step(model, optimizer, batch):
    # TODO: forward、answer-only next-token loss、backward、step；再验证 greedy generation。
    raise NotImplementedError("参考 run_instructblip_demo.py")
