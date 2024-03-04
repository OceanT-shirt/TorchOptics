import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from raw_data_processing.data_converter import yml2df_from_dir
from raw_data_processing.zmx2yml_all import zex2yml_all

def main():
    mode = input("Select Convert Mode: 'zmx2yml' or 'yml2csv'")
    if mode == "yml2csv":
        impdir = input("Input directory path of the .yml files / 変換対象が入っているディレクトリを選択: ")
        expdir = input("Output directory path of the export .csv file / 出力を入れるディレクトリを選択: ")

        # 出力ディレクトリが存在しない場合は作成する
        os.makedirs(expdir, exist_ok=True)

        ldf, e = yml2df_from_dir(impdir)
        if e:
            print("Error Log:", e)
        ldf.to_csv(os.path.join(expdir, "lens_design.csv"), index=False)
    elif mode == "zmx2yml":
        zex2yml_all()
    else:
        print("Invalid mode")

def main_auto(impdir, expcsv, mode='yml2csv'):
    if mode == "yml2csv":
        expdir = os.path.dirname(expcsv)
        # 出力ディレクトリが存在しない場合は作成する
        os.makedirs(expdir, exist_ok=True)

        ldf, e = yml2df_from_dir(impdir)
        if e:
            print("Error Log:", e)
        ldf.to_csv(expcsv, index=False)
    elif mode == "zmx2yml":
        zex2yml_all()
    else:
        print("Invalid mode")

if __name__ == "__main__":
    # zmx to yaml
    impdir = ''
    expcsv = ''
    main_auto(impdir, expcsv, mode='zmx2yml')

    # yaml to csv
    impdir = 'C:/Users/KU/Desktop/Zebase_T/yaml'
    expcsv = 'C:/Users/KU/Desktop/Zebase_T/Zebase_T.csv'
    main_auto(impdir, expcsv, mode='yml2csv')
    
    # from raw_data_processing.zmx2yml import zmx2yml
    # inpath = 'C:/Users/KU/Desktop/AI/design/Zebase/zmx/T_001.ZMX'
    # outpath = 'C:/Users/KU/Desktop/dummy10.yml'
    # zmx2yml(inpath, outpath)



    # import numpy as np
    # import pandas as pd
    # import ast
    # from raw_data_processing.zmx2yml import ComputeEFL

    # Delete multiple space from d_all column
    # df = pd.read_csv("C:/Users/KU/Desktop/AI/design/Lensview/LensView_all.csv")
    # df['d_all'] = df['d_all'].str.replace('     ', ' ')
    # df['d_all'] = df['d_all'].str.replace('    ', ' ')
    # df['d_all'] = df['d_all'].str.replace('   ', ' ')
    # df['d_all'] = df['d_all'].str.replace('  ', ' ')
    
    # df['d_all'] = df['d_all'].str.replace('[ ', '[')
    # df['d_all'] = df['d_all'].str.replace(' ]', ']')

    # df['d_all'] = df['d_all'].str.replace(' ', ',')

    # # export to csv
    # df.to_csv('C:/Users/KU/Desktop/AI/design/Lensview/LensView_all2.csv', index=False)


    # # import csv table
    # df_zebase = pd.read_csv("C:/Users/KU/Desktop/AI/design/Zebase/Zebase.csv")
    # # df_lensview = pd.read_csv("C:/Users/KU/Desktop/AI/design/Lensview/LensView_all.csv")
    # # # Merge them depending on sequence GA, GGA, GAGA
    # # df = pd.concat([df_zebase, df_lensview], ignore_index=True, axis=0)

    # # select data contains 'conf'
    # df = df_zebase
    # df_conf = df[df['file_name'].str.contains('conf')]
    # index = np.where(df['file_name'].str.contains('conf')==True)[0]
    
    # # # convert string list to list
    # # df_conf['c'] = df_conf['c'].apply(lambda x: ast.literal_eval(x))
    # # df_conf['c_all'] = df_conf['c_all'].apply(lambda x: ast.literal_eval(x))
    # # df_conf['d_all'] = df_conf['d_all'].apply(lambda x: ast.literal_eval(x))
    # # df_conf['t_all'] = df_conf['t_all'].apply(lambda x: ast.literal_eval(x))
    # # df_conf['nd'] = df['nd'].apply(lambda x: ast.literal_eval(x))

    # for i in range(len(index)):
    #     df_e = df_conf.iloc[i]

    #     # convert string list to list
    #     c = ast.literal_eval(df_e['c'])
    #     c_all = ast.literal_eval(df_e['c_all'])
    #     d_all = ast.literal_eval(df_e['d_all'])
    #     t_all = ast.literal_eval(df_e['t_all'])
    #     nd = ast.literal_eval(df_e['nd'])

    #     # Denormalise epd, c, c_all, d_all and t_all
    #     df_e['epd'] = df_e['epd'] * df_e['efl']
    #     c = np.array(c) / df_e['efl']
    #     c_all = np.array(c_all) * df_e['efl']
    #     d_all = np.array(d_all) * df_e['efl']
    #     t_all = np.array(t_all) * df_e['efl']
    #     nd = np.array(nd)
        
    #     # Calculate efl
    #     df_e['efl'] = ComputeEFL(df_e['sequence'], c_all.tolist(), t_all.tolist(), nd.tolist())

    #     # Normalise epd, c, c_all, d_all and t_all
    #     df_e['epd'] = df_e['epd'] / df_e['efl']
    #     c = np.array(df_e['c']) * df_e['efl']
    #     c_all = np.array(df_e['c_all']) / df_e['efl']
    #     d_all = np.array(df_e['d_all']) / df_e['efl']
    #     t_all = np.array(df_e['t_all']) / df_e['efl']

    #     # update data
    #     df.iloc[index[i]] = df_e

    # # export to csv
    # df.to_csv('C:/Users/KU/Desktop/AI/design/Lensview/LensView_all2.csv', index=False)