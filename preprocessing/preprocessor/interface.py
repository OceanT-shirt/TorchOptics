import pandas as pd

import torch
from typing import NamedTuple


class DecoderOutput(NamedTuple):
    stop_idx: int
    sequence: str
    hfov: torch.tensor  # shape: (1,)
    epd: torch.tensor  # shape: (1,)
    c_wo_last: torch.tensor
    t: torch.tensor
    nd: torch.tensor
    v: torch.tensor


class PreprocessorInterface:
    def encode(self, df_to_conv) -> pd.DataFrame:
        raise NotImplementedError

    def decode(self, input_tensor) -> DecoderOutput:
        raise NotImplementedError
