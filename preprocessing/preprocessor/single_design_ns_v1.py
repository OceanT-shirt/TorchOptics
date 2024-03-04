import torch

from preprocessing.preprocessor.interface import PreprocessorInterface, DecoderOutput
from torchlens.lens_modeling import n_v_from_g
from preprocessing.glass_material import g_from_nv
from preprocessing.utils import str2list, int2str
import pandas as pd
import numpy as np


class PreprocessorSingleDesignNSV1(PreprocessorInterface):
    def __init__(self, sequence: str):
        self.SEQUENCE = sequence
        self.N = len(sequence)
        self.G = sequence.count("G")

    def encode(self, df_to_conv) -> pd.DataFrame:
        invalid_sequences = df_to_conv[df_to_conv["sequence"].str.contains(r"A{2,}")]
        if not invalid_sequences.empty:
            raise ValueError(f"Invalid sequence (two or more consecutive As) found: {invalid_sequences}. Please check "
                             f"the input file.")
        # sequenceが全てSEQUENCEに一致していることを確認
        if not df_to_conv["sequence"].eq(self.SEQUENCE).all():
            raise ValueError("Invalid sequence found. Please check the input file.")
        # Listデータがstr型として読み込まれている列の値をList[float]に変換
        df_to_conv["c"] = df_to_conv["c"].apply(str2list)
        df_to_conv["nd"] = df_to_conv["nd"].apply(str2list)
        df_to_conv["t_all"] = df_to_conv["t_all"].apply(str2list)
        df_to_conv["v"] = df_to_conv["v"].apply(str2list)

        # 出力ファイルの作成
        df_converted = pd.DataFrame(columns=["epd", "hfov"] + [f"c{int2str(i)}" for i in range(1, self.N)]
                                            + [f"t{int2str(i)}" for i in range(1, self.N + 1)]
                                            + [f"g{int2str(i)}_1" for i in range(1, self.G + 1)]
                                            + [f"g{int2str(i)}_2" for i in range(1, self.G + 1)])
        df_converted["epd"] = df_to_conv["epd"]
        df_converted["hfov"] = df_to_conv["hfov"].abs()
        df_converted[[f"c{int2str(i)}" for i in range(1, self.N)]] = df_to_conv["c"].apply(lambda x: pd.Series(x[:-1]))
        df_converted[[f"t{int2str(i)}" for i in range(1, self.N + 1)]] = df_to_conv["t_all"].apply(pd.Series)
        for i in range(self.G):
            ni = df_to_conv["nd"].apply(lambda x: x[i])
            vi = df_to_conv["v"].apply(lambda x: x[i])
            n_array = np.array(ni)
            v_array = np.array(vi)
            g_array = g_from_nv(n_array, v_array)
            df_converted["g" + int2str(i + 1) + "_1"] = g_array[:, 0]
            df_converted["g" + int2str(i + 1) + "_2"] = g_array[:, 1]
        return df_converted

    def decode(self, input_tensor, default_stop_idx: int = 1) -> DecoderOutput:
        """
        input_tensor: torch.tensor (shape: (1, 4N + 1))
        """
        if input_tensor.shape[0] != 1:
            raise ValueError(f"Invalid input tensor shape: {input_tensor.shape}.")
        input_tensor = input_tensor.squeeze()
        epd_tensor, hfov_tensor, c_tensor_wo_last, t_tensor, g1_tensor, g2_tensor = self.split_input_tensor(
            input_tensor)
        stop_idx = default_stop_idx
        nd, v = n_v_from_g(torch.stack([g1_tensor, g2_tensor], dim=1))
        hfov = hfov_tensor.squeeze()
        epd = epd_tensor.squeeze()

        return DecoderOutput(stop_idx=stop_idx, sequence=self.SEQUENCE, hfov=hfov, epd=epd, t=t_tensor,
                             c_wo_last=c_tensor_wo_last, nd=nd, v=v)

    def split_input_tensor(self, input_tensor):
        N = self.N
        G = self.G
        if input_tensor.shape[0] != 2*(N+G)+1:
            raise ValueError(f"Invalid input tensor shape: {input_tensor.shape}.")
        epd = input_tensor[0:1]
        hfov = input_tensor[1:2]
        c = input_tensor[2:1 + N]
        t = input_tensor[1 + N:1 + 2 * N]
        g1 = input_tensor[1 + 2 * N:1 + 2 * N + G]
        g2 = input_tensor[1 + 2 * N + G:]
        return epd, hfov, c, t, g1, g2

