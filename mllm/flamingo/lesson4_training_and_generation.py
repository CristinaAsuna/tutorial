"""关卡 4：仅训练 Resampler 与 gated connectors 的自回归步骤。"""


def connector_training_step(model, optimizer, input_ids, pixel_values, attention_mask, labels):
    # TODO: forward -> masked next-token loss -> backward -> optimizer.step。
    raise NotImplementedError("参考 run_flamingo_demo.py")
