from typing import List
import ast
import pandas as pd


def calc_t_len(series: pd.Series) -> int:
    return series.apply(len).max()


def str2list(s: str) -> List[float]:
    return ast.literal_eval(s)


def int2str(n: int):
    """
    Create the string of the column name for the dataframe.
    """
    l = len(str(n))
    assert (l == 1 or l == 2)
    return "0" + str(n) if l == 1 else str(n)
