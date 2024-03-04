from typing import List, Tuple
import torch

ERROR = 1e-10


def calc_as_d(t_all: List[float], stop_idx: int, is_independent_as: bool):
    """
    is_independent_as: Whether the aperture stop is the same location with any
    lens surfaces or not.
    """
    if len(t_all) < stop_idx:
        raise ValueError("stop_idx must be less than or equal to the length of t_all.")
    if stop_idx == 1 and is_independent_as:
        d = -1 * t_all[0]
    else:
        d = sum(t_all[:stop_idx - 1])
    return d


def sequence_encoder(t_all: List[float], is_independent_as: bool, stop_idx: int, t_length: int) -> Tuple[
    List[float], float]:
    # Aperture Stopまでの距離を計算
    as_d = calc_as_d(t_all, stop_idx, is_independent_as)

    if stop_idx == 1 and is_independent_as:
        t = t_all[1:]
    elif is_independent_as:
        t = []
        i = 0
        while len(t_all) > 0:
            if i == stop_idx - 2:
                # インデックスがstop_idxのとき、次の要素と結合
                t_before_as = t_all.pop(0)
                t_aftr_as = t_all.pop(0)
                t.append(t_before_as + t_aftr_as)
            else:
                t.append(t_all.pop(0))
            i += 1
    else:
        t = t_all

    while len(t) < t_length:
        t.append(0.)

    return t, as_d


GLASS_REFRACTIVE_IDX_THRESHOLD = 1.2


def merge_air(sequence: str, t_tensor: torch.Tensor, c_tensor_wo_last: torch.Tensor) \
        -> Tuple[str, torch.Tensor, torch.Tensor]:
    assert len(sequence) == t_tensor.shape[0]
    assert len(sequence) >= 2
    assert sequence[-1] == 'A'
    # 全てのシーケンスがAの場合、エラーを吐く
    assert not all(char == 'A' for char in sequence)
    result_seq = ""
    result_t = []
    result_c = []
    start = None  # 'A' の連続が開始するインデックス
    already_glass = False

    for i, char in enumerate(sequence):
        if char == 'A':
            if not already_glass:
                continue
            if start is None:
                start = i  # 'A' の連続の開始
                if i != len(sequence) -1:
                    result_c.append(c_tensor_wo_last[i])
        else:
            # char == 'G'
            if start is not None:
                # 'A' の連続が終了したので、その部分の和を計算
                if already_glass:
                    result_seq += 'A'
                    result_t.append(torch.sum(t_tensor[start:i]))
                start = None  # 連続カウントをリセット
            result_seq += "G"
            result_t.append(t_tensor[i])
            result_c.append(c_tensor_wo_last[i])
            already_glass = True

    # 文字列の最後が 'A' の連続で終わっている場合
    if start is not None:
        result_seq += 'A'
        result_t.append(torch.sum(t_tensor[start:]))

    # 結果をテンソルに変換
    return result_seq, torch.stack(result_t), torch.stack(result_c) if sequence[-2:] == 'GA' \
        else torch.stack(result_c[:-1])


def calculate_sequence(nd: torch.Tensor) -> str:
    is_glass = nd > GLASS_REFRACTIVE_IDX_THRESHOLD  # tensor([True, True, False, False])
    sequence = ''.join(['G' if x else 'A' for x in is_glass])  # 'GGAA'
    return sequence


def validate_input(c_wo_last: torch.Tensor, t: torch.Tensor, nd: torch.Tensor, v: torch.Tensor, as_d: torch.Tensor):
    try:
        assert t.shape == nd.shape == v.shape
        assert len(c_wo_last) == len(t) - 1
        assert len(c_wo_last.shape) == 1
        assert len(t.shape) == 1
        assert len(nd.shape) == 1
        assert len(v.shape) == 1
        assert len(as_d.shape) == 0
        assert torch.all(t >= 0)
    except AssertionError:
        raise ValueError("Invalid input_object.")


def extract_nd_and_v(nd: torch.Tensor, v: torch.Tensor, sequence: str) -> Tuple[torch.Tensor, torch.Tensor]:
    g_indices = [i for i, x in enumerate(sequence) if x == 'G']  # [0, 1]
    nd_tensor_converted = nd[g_indices]
    v_tensor_converted = v[g_indices]
    return nd_tensor_converted, v_tensor_converted


def sequence_decoder(c_tensor_wo_last: torch.Tensor, t_tensor: torch.Tensor, nd_tensor: torch.Tensor,
                     v_tensor: torch.Tensor, as_d_tensor: torch.Tensor, debug_mode: bool = False) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, bool]:
    """
    - 屈折率を用いたテンソルの変換
    - n, vのテンソルのトリミング: レンズに対応する値のみを抽出
    - 入力テンソルのtは全て正でなければならない
    返り値
    - c
    - t
    - nd
    - v
    - sequence
    - stop index
    - is_independent_as
    """
    validate_input(c_tensor_wo_last, t_tensor, nd_tensor, v_tensor, as_d_tensor)
    sequence = calculate_sequence(nd_tensor)
    if debug_mode:
        print("sequence:", sequence)
    if sequence[-1] != 'A':
        raise ValueError("The last element of the sequence must be 'A'.")
    # レンズ表面の個数（N_curv）の制約から、最後の表面はAに固定される
    # nd, vに対する処理
    nd_tensor_converted, v_tensor_converted = extract_nd_and_v(nd_tensor, v_tensor, sequence)
    # lens_sequenceにおいて、連続するAの部分をまとめる
    # これらの値は、Aperture Stopを含まない
    seq_wo_as, t_wo_as, c_wo_as = merge_air(sequence, t_tensor, c_tensor_wo_last)
    if debug_mode:
        print("seq_wo_as:", seq_wo_as)
        print("t_wo_as:", t_wo_as)
        print("c_wo_as:", c_wo_as)

    # Aperture Stopに対する処理
    # TODO ここ以下の責任分離
    t_cumsum = torch.cumsum(t_wo_as, dim=0)
    stop_idx = 0
    seq_all = None
    t_all = None
    c_all = None
    is_independent_as = None

    if as_d_tensor < 0:
        # Aperture Stopが最初のレンズ面の前にある場合
        stop_idx = 1
        seq_all = "A" + seq_wo_as
        t_all = torch.cat((-1*as_d_tensor.unsqueeze(dim=0), t_wo_as), dim=0)
        c_all = torch.cat((c_wo_as.new_tensor([0]), c_wo_as), dim=0)
        is_independent_as = True
    elif as_d_tensor == 0:
        # Aperture Stopが最初のレンズ面の上にある場合
        stop_idx = 1
        seq_all = seq_wo_as
        t_all = t_wo_as
        c_all = c_wo_as
        is_independent_as = False
    else:
        # G...S...の場合
        for n in range(len(t_wo_as)):
            if as_d_tensor > t_cumsum[n]:
                if n == len(t_wo_as) - 1:
                    # Aperture Stopが最後のレンズ面の後ろにある場合
                    stop_idx = n + 1
                    seq_all = seq_wo_as + "A"
                    t_all = torch.cat((t_wo_as, as_d_tensor - t_cumsum[n]), dim=0)
                    c_all = torch.cat((c_wo_as, c_wo_as.new_tensor([0])), dim=0)
                    is_independent_as = True
                    break
                elif as_d_tensor < t_cumsum[n + 1]:
                    # Aperture Stopがn番目のレンズ面とn+1番目のレンズ面の間にある場合
                    if seq_wo_as[n+1] == 'G':
                        raise ValueError("Aperture Stop is in the middle of a glass surface.")
                    stop_idx = n + 3
                    seq_all = seq_wo_as[:n + 1] + "A" + seq_wo_as[n + 1:]
                    t_all = torch.cat((t_wo_as[:n + 1], torch.unsqueeze(as_d_tensor - t_cumsum[n], dim=0),
                                       torch.unsqueeze(t_cumsum[n+1] - as_d_tensor, dim=0), t_wo_as[n + 2:]), dim=0)
                    c_all = torch.cat((c_wo_as[:n+2], c_wo_as.new_tensor([0]), c_wo_as[n+2:]), dim=0)
                    is_independent_as = True
                    break
                elif as_d_tensor == t_cumsum[n + 1]:
                    # Aperture Stopがn+1番目のレンズ面の位置にある場合
                    stop_idx = n + 3
                    seq_all = seq_wo_as
                    t_all = t_wo_as
                    c_all = c_wo_as
                    is_independent_as = False
                    break
                else:
                    """
                    ...ASA or ...AS
                    """
                    # ASがAirの中にある場合

                    # ASがtの外にある場合：エラー
                    raise ValueError("Could not determine the stop index based on the given as_d.")
            elif as_d_tensor <= t_cumsum[0]:
                # stop_idx = 1
                # as_dが最初の平面と次の平面との間にあるということは、Gと重なっているということである
                raise ValueError("The aperture stop is overlapping with the first surface.")
            else:
                continue

    return c_all, t_all, nd_tensor_converted, v_tensor_converted, seq_all, stop_idx, is_independent_as
