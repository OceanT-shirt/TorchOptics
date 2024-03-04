import sys
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from preprocessing.preprocessor.utils import ProcessorFactory, ProcessorCommand

if __name__ == "__main__":
    # processor_product = ProcessorFactory.create_from_user_input()
    # cmd = ProcessorCommand(processor_product)
    # impfile = input("Input file path of the raw csv file / 入力するファイルのパスを選択: ")
    # expfile = input("Input file path of the output csv file / 出力するファイルのパスを選択:")
    # base_df = pd.read_csv(impfile)
    # converted_df = cmd.execute(base_df)
    # converted_df.to_csv(expfile, index=False)
    # print(f"Conversion completed. File saved at {expfile}")
    
    # import csv table
    df_zebase = pd.read_csv("C:/Users/KU/Desktop/AI/design/Zebase/Zebase.csv")
    df_lensview = pd.read_csv("C:/Users/KU/Desktop/AI/design/Lensview/LensView_all.csv")
    # Merge them depending on sequence GA, GGA, GAGA
    df = pd.concat([df_zebase, df_lensview], ignore_index=True, axis=0)
    print(df.shape)
    # delete duplicates
    col_search = ['epd', 'hfov', 'hfov_all', 'wave', 'c', 'as_c', 'c_all', 'd_all', 't_all',
                    'nd', 'v', 'efl', 'sequence', 'stop_idx', 'independent_as']
    df = df.drop_duplicates(subset=col_search)
    print(df.shape)