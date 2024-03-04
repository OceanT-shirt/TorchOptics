from preprocessing.preprocessor.interface import PreprocessorInterface
from preprocessing.lens_sequence import sequence_encoder
from preprocessing.glass_material import g_from_nv
from preprocessing.utils import calc_t_len, str2list, int2str
import pandas as pd
import numpy as np


class PreprocessorSV1(PreprocessorInterface):
    def encode(self, df_to_conv) -> pd.DataFrame:
        # Listデータがstr型として読み込まれている列の値をList[float]に変換
        df_to_conv["c"] = df_to_conv["c"].apply(str2list)
        df_to_conv["nd"] = df_to_conv["nd"].apply(str2list)
        df_to_conv["t_all"] = df_to_conv["t_all"].apply(str2list)
        df_to_conv["v"] = df_to_conv["v"].apply(str2list)
        # sequence lengthの計算
        t_len = calc_t_len(df_to_conv["t_all"])
        # 最長のsequence lengthの表示
        print("N:", t_len)
        # sequenceのエンコード
        df_to_conv[["t", "as_d"]] = df_to_conv.apply(
            lambda row: sequence_encoder(row['t_all'], row['independent_as'], row['stop_idx'], t_len),
            axis=1, result_type='expand'
        )
        # glass variableのエンコード
        # df_to_conv["g"] = df_to_conv.apply(lambda row: g_from_nv(row['nd'], row['v']), axis=1)
        # c, tをそれぞれの列に配置
        df_to_conv["c"] = df_to_conv["c"].apply(lambda x: x + [0.] * (t_len - len(x)))  # gとの整合性のため
        df_to_conv["t"] = df_to_conv["t"].apply(lambda x: [0.] * (t_len - len(x)) + x)
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
                                            + [f"g{int2str(i)}_2" for i in range(1, t_len + 1)] + ["as_d"])
        df_converted["epd"] = df_to_conv["epd"]
        df_converted["hfov"] = df_to_conv["hfov"].abs()
        df_converted[[f"c{int2str(i)}" for i in range(1, t_len)]] = df_to_conv["c"].apply(lambda x: pd.Series(x[:-1]))
        df_converted[[f"t{int2str(i)}" for i in range(1, t_len + 1)]] = df_to_conv["t"].apply(pd.Series)
        df_converted["as_d"] = df_to_conv["as_d"]
        for i in range(t_len):
            ni = df_to_conv["nd"].apply(lambda x: x[i])
            vi = df_to_conv["v"].apply(lambda x: x[i])
            n_array = np.array(ni)
            v_array = np.array(vi)
            g_array = g_from_nv(n_array, v_array)
            df_converted["g" + int2str(i + 1) + "_1"] = g_array[:, 0]
            df_converted["g" + int2str(i + 1) + "_2"] = g_array[:, 1]
        return df_converted

    def decode(self, input_tensor):
        raise NotImplementedError
