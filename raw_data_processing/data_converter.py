import pandas as pd
from typing import List
from raw_data_processing.yaml_loader import LensDesign
import os


def lens_design_to_df(ldl: List[LensDesign]) -> pd.DataFrame:
    return pd.DataFrame([ld.model_dump() for ld in ldl])


"""
TODO: Check if this function is used anywhere
"""
# def seq2lens_type(seq: str, stop_idx: int) -> str:
#     if seq[stop_idx - 1] == "A":
#         return seq[0:stop_idx - 1] + seq[stop_idx:]
#     else:
#         # Aperture stop is the same point as the previous surface
#         return seq


def load_yml_from_dir(dir_name: str):
    # sometimes default_lens_air_count can be +1 because of the aperture stop
    files = os.listdir(dir_name)

    lens_design_list = []
    error_log = []  # [file_name, error_message]

    for file in files:
        if file.endswith('.yml'):
            file_path = os.path.join(dir_name, file)
            try:
                ld = LensDesign.from_yaml(file_path)
                lens_design_list.append(ld)
            except Exception as e:
                print("Error in converting", file_path)
                print(e)
                error_log.append([file_path, e])
                continue
    return lens_design_list, error_log


def yml2df_from_dir(dir_path: str):
    lens_design_list, error_log = load_yml_from_dir(dir_path)
    return lens_design_to_df(lens_design_list), error_log
