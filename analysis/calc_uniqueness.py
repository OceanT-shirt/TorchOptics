import torch
import math


def calculate_uniqueness(c_tensor, t_tensor, dataset_df):
    """
    新規設計のc, tの値が、既存の設計からどれだけ離れているのか計算する指標。
    新規設計のシーケンスは既存の設計と一致している必要がある。
    返り値: average(|c_i - c_closest|^2 + |t_i - t_closest|^2) (iはシーケンスの要素のインデックス, c_closest, t_closestは既存の設計のc, tの値の中で最も新規設計に近いものの値
    """
    # c_tensor, t_tensorがtorch.tensorであること
    # c_tensor, t_tensorのshapeが(1,)であること
    if not isinstance(c_tensor, torch.Tensor):
        raise ValueError("c_tensor must be torch.Tensor")
    if len(c_tensor.shape) != 1 or len(t_tensor.shape) != 1:
        raise ValueError("c_tensor must have shape (1,)")
    if len(t_tensor) != dataset_df["sequence"].str.len().max():
        raise ValueError("t_tensor must have the same length as the sequence in dataset_df")
    # "c", "t"カラムがstrではなくlistであること
    if dataset_df["c"].apply(type).eq(str).any() or dataset_df["t_all"].apply(type).eq(str).any():
        raise ValueError("c column in dataset_df must be list")

    # それぞれのシーケンスに対し、最も近いc, tの値を見つける
    # 全ての組み合わせに対し、差の二乗を計算し、その最小値を返り値とする
    diff_smallest = float("inf")
    c = c_tensor.detach().cpu().numpy()
    t = t_tensor.detach().cpu().numpy()
    for i in range(len(dataset_df)):
        c_ = dataset_df["c"].iloc[i]
        t_ = dataset_df["t_all"].iloc[i]

        # 各cに対して、二乗距離を求める
        c_avg = sum([math.log(abs(c[i]/c_[i])+1) for i in range(len(c_))])/len(c_)
        t_avg = sum([math.log(abs(t[i]/t_[i])+1) for i in range(len(t_))])/len(t_)
        if c_avg + t_avg < diff_smallest:
            diff_smallest = c_avg + t_avg
    return diff_smallest
