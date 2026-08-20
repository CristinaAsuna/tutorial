"""DDPM model architecture: a Time-U-Net that predicts epsilon noise."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.basic_function import TimeUNet


class DDPMUNet(TimeUNet):
    """The output has the same shape as x_t and represents epsilon noise."""

    def __init__(self, in_channels=3, base_channels=64, time_dim=256, context_dim=768):
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            time_dim=time_dim,
            context_dim=context_dim,
        )
        self.model_config = {
            "in_channels": in_channels,
            "base_channels": base_channels,
            "time_dim": time_dim,
            "context_dim": context_dim,
        }

    def get_config(self):
        return dict(self.model_config)
