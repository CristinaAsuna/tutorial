"""关卡 3：构造只监督 assistant 回答的 SFT labels。"""
from reference_llava import IGNORE_INDEX


def assistant_only_labels(input_ids, assistant_start: int):
    # TODO: system/user/<image>/padding all receive IGNORE_INDEX.
    # TODO: assistant answer tokens are their vocabulary ids.
    raise NotImplementedError("参考 conversation.build_sft_example")
