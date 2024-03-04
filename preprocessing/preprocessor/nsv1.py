import torch

from preprocessing.preprocessor.interface import PreprocessorInterface, DecoderOutput
from preprocessing.lens_sequence import calculate_sequence, merge_air, extract_nd_and_v
from torchlens.lens_modeling import n_v_from_g
from preprocessing.glass_material import g_from_nv
from preprocessing.utils import calc_t_len, str2list, int2str
import pandas as pd
import numpy as np


class PreprocessorNSV1(PreprocessorInterface):
    def encode(self, df_to_conv) -> pd.DataFrame:
        invalid_sequences = df_to_conv[df_to_conv["sequence"].str.contains(r"A{2,}")]
        if not invalid_sequences.empty:
            raise ValueError(f"Invalid sequence (two or more consecutive As) found: {invalid_sequences}. Please check "
                             f"the input file.")
        # Listデータがstr型として読み込まれている列の値をList[float]に変換
        df_to_conv["c"] = df_to_conv["c"].apply(str2list)
        df_to_conv["nd"] = df_to_conv["nd"].apply(str2list)
        df_to_conv["t_all"] = df_to_conv["t_all"].apply(str2list)
        df_to_conv["v"] = df_to_conv["v"].apply(str2list)
        # sequence lengthの計算
        t_len = calc_t_len(df_to_conv["t_all"])
        # 最長のsequence lengthの表示
        print("N:", t_len)
        # c, tをそれぞれの列に配置
        df_to_conv["c"] = df_to_conv["c"].apply(lambda x: x + [0.] * (t_len - len(x)))  # gとの整合性のため
        df_to_conv["t_all"] = df_to_conv["t_all"].apply(lambda x: [0.] * (t_len - len(x)) + x)
        df_to_conv["sequence"] = df_to_conv["sequence"].apply(lambda x: x + "A" * (t_len - len(x)))
        df_to_conv["nd"] = df_to_conv.apply(
            lambda row: [0. if seq == "A" else row["nd"].pop(0) for seq in row["sequence"]],
            axis=1)
        df_to_conv["v"] = df_to_conv.apply(
            lambda row: [0. if seq == "A" else row["v"].pop(0) for seq in row["sequence"]],
            axis=1)

        # 出力ファイルの作成
        df_converted = pd.DataFrame(columns=["epd", "hfov"] + [f"c{int2str(i)}" for i in range(1, t_len)]
                                            + [f"t{int2str(i)}" for i in range(1, t_len + 1)]
                                            + [f"g{int2str(i)}_1" for i in range(1, t_len + 1)]
                                            + [f"g{int2str(i)}_2" for i in range(1, t_len + 1)])
        df_converted["epd"] = df_to_conv["epd"]
        df_converted["hfov"] = df_to_conv["hfov"].abs()
        df_converted[[f"c{int2str(i)}" for i in range(1, t_len)]] = df_to_conv["c"].apply(lambda x: pd.Series(x[:-1]))
        df_converted[[f"t{int2str(i)}" for i in range(1, t_len + 1)]] = df_to_conv["t_all"].apply(pd.Series)
        for i in range(t_len):
            ni = df_to_conv["nd"].apply(lambda x: x[i])
            vi = df_to_conv["v"].apply(lambda x: x[i])
            n_array = np.array(ni)
            v_array = np.array(vi)
            g_array = g_from_nv(n_array, v_array)
            df_converted["g" + int2str(i + 1) + "_1"] = g_array[:, 0]
            df_converted["g" + int2str(i + 1) + "_2"] = g_array[:, 1]
        return df_converted

    @staticmethod
    def decode(input_tensor, default_stop_idx: int = 1) -> DecoderOutput:
        """
        input_tensor: torch.tensor (shape: (1, 4N + 1))
        """
        if input_tensor.shape[0] != 1:
            raise ValueError(f"Invalid input tensor shape: {input_tensor.shape}.")
        input_tensor = input_tensor.squeeze()
        epd_tensor, hfov_tensor, c_tensor_wo_last_w_air, t_tensor, g1_tensor_w_air, g2_tensor_w_air = PreprocessorNSV1.split_input_tensor(input_tensor)
        stop_idx = default_stop_idx
        nd_tensor_w_air, v_tensor_w_air = n_v_from_g(torch.stack([g1_tensor_w_air, g2_tensor_w_air], dim=1))
        sequence_w_sequential_air = calculate_sequence(nd_tensor_w_air)  # GAAAGAのように、連続するAを含む (GAGAのように圧縮される必要がある)
        nd, v = extract_nd_and_v(nd_tensor_w_air, v_tensor_w_air, sequence_w_sequential_air)
        hfov = hfov_tensor.squeeze()
        epd = epd_tensor.squeeze()
        sequence, t_tensor, c_tensor_wo_last = merge_air(sequence_w_sequential_air, t_tensor, c_tensor_wo_last_w_air)

        return DecoderOutput(stop_idx=stop_idx, sequence=sequence, hfov=hfov, epd=epd, t=t_tensor,
                             c_wo_last=c_tensor_wo_last, nd=nd, v=v)

    @staticmethod
    def split_input_tensor(input_tensor):
        # レンズシーケンスの長さNを算出 Nは整数 input_tensor.shape[0] = 4N + 1
        n = (input_tensor.shape[0] - 1) // 4
        if input_tensor.shape[0] != 4 * n + 1:
            raise ValueError(f"Invalid input tensor shape: {input_tensor.shape}.")
        epd = input_tensor[0:1]
        hfov = input_tensor[1:2]
        c = input_tensor[2:1+n]
        t = input_tensor[1+n:1+2*n]
        g1 = input_tensor[1+2*n:1+3*n]
        g2 = input_tensor[1+3*n:]
        return epd, hfov, c, t, g1, g2

