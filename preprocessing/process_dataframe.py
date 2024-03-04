import sys
sys.path.append('C:/Users/KU/Desktop/AI/src')

import numpy as np
import pandas as pd
import random
import ast
import torch
from torchlens.lens_modeling import g_from_n_v

class DataframeProcessor():
  def __init__(self, lens_type):
    self.lens_type = lens_type
    self.code_lenstype = sequence_encoder(self.lens_type)
    self.numsurf = len(str(self.code_lenstype))
    self.numglass = sum(map(int, str(self.code_lenstype)))
    self.ratio_DT = [3.2, 8.5]
    self.rate_t = 1/4

  #################################################
  def process_dataframe(self, df):
    # Step1. Arrange column names
    df = self.arrange_frame_column(df)

    # Step2. Convert n_refr and Abbe into g and add column
    for i in range(self.numglass):
      n_values = torch.tensor(df['n'+str(i+1)].values, dtype=torch.float32)
      v_values = torch.tensor(df['v'+str(i+1)].values, dtype=torch.float32)
      g_values = g_from_n_v(n_values, v_values)
      df['g'+str(i+1)+'1'] = g_values[:, 0]
      df['g'+str(i+1)+'2'] = g_values[:, 1] 

    # Step3. Generate t_min and t_range
    df = self.thickness_min_range(df)

    # Step5. Convert the dataframe into torch.tensor
    # split the dataset into the train data and the test data
    colX_list = ['epd', 'hfov']
    for i in range(self.numsurf):
      colX_list.append('t'+str(i+1)+'_min')
      colX_list.append('t'+str(i+1)+'_range')
    for i in range(self.numsurf):
      colX_list.append('d'+str(i+1))
    colX_list = colX_list + ['sequence_encoded', 'stop_idx', 'as_c', 'as_t']

    coly_list = []
    for i in range(self.numglass):
      coly_list.append('g'+str(i+1)+'1')
      coly_list.append('g'+str(i+1)+'2')
    for i in range(self.numsurf):
      if i+1 < self.numsurf:  # c_last is calculated using epd
        coly_list.append('c'+str(i+1))  
    for i in range(self.numsurf):
        coly_list.append('t'+str(i+1))

    array_X = df[colX_list].values
    array_y = df[coly_list].values  
    Tensor_X = torch.tensor(array_X, dtype=torch.float32)
    Tensor_y = torch.tensor(array_y, dtype=torch.float32)

    return df, Tensor_X, Tensor_y

  #################################################
  def arrange_frame_column(self, df):
    # convert string list to list
    df['c'] = df['c'].apply(lambda x: ast.literal_eval(x))
    df['d_all'] = df['d_all'].apply(lambda x: ast.literal_eval(x))
    df['t_all'] = df['t_all'].apply(lambda x: ast.literal_eval(x))
    df['nd'] = df['nd'].apply(lambda x: ast.literal_eval(x))
    df['v'] = df['v'].apply(lambda x: ast.literal_eval(x))

    # converted frame
    df_converted = pd.DataFrame()
    for i in range(self.numsurf):
      df_converted["c"+str(i+1)] = df["c"].apply(lambda x: x[i])
    for i in range(self.numsurf):
      df_converted["d"+str(i+1)] = df["d_all"].apply(lambda x: x[i])
    for i in range(self.numsurf):
      df_converted["t"+str(i+1)] = df["t_all"].apply(lambda x: x[i])
    for i in range(self.numglass):
      df_converted["n"+str(i+1)] = df["nd"].apply(lambda x: x[i])
    for i in range(self.numglass):
      df_converted["v"+str(i+1)] = df["v"].apply(lambda x: x[i])
    df_converted["epd"] = df["epd"]
    df_converted["hfov"] = df["hfov"]
    df_converted["stop_idx"] = df["stop_idx"]
    df_converted["sequence_encoded"] = df["sequence"].apply(sequence_encoder)
    df_converted["as_c"] = df["as_c"]
    df_converted["as_t"] = df["as_c"]

    return df_converted

  #################################################
  def thickness_min_range(self, df):
    for i in range(self.numsurf):
      istr = str(i+1)

      intv = (1/self.ratio_DT[0]-1/self.ratio_DT[1]) * df['d'+istr]
      t_range = intv * self.rate_t
      t_mean = (1/self.ratio_DT[0]+1/self.ratio_DT[1])/2 * df['d'+istr]
      t_min = t_mean - t_range/2

      df["t"+istr+"_min"] = t_min
      df["t"+istr+"_range"] = t_range
    return df
  
#################################################
def sequence_encoder(s: str):
  binary_sequence = ""
  for ss in s:
    assert(ss == "G" or ss == "A")
    if ss == "G":
      binary_sequence += "1"
    elif ss == "A":
      binary_sequence += "0"
  return int(binary_sequence)

def sequence_decoder(n: int):
  binary_sequence = str(n)
  sequence = ""
  for nn in binary_sequence:
    assert(nn == "0" or nn == "1")
    if nn == "1":
      sequence += "G"
    elif nn == "0":
      sequence += "A"
  return sequence
  
#################################################
def range_EPDHFOV(df):
    epd_max = df['epd'].max()
    epd_mean = df['epd'].mean()
    epd_min = df['epd'].min()
    epd_range = [epd_max, epd_mean, epd_min]

    hfov_max = df['hfov'].max()
    hfov_mean = df['hfov'].mean()
    hfov_min = df['hfov'].min()
    hfov_range = [hfov_max, hfov_mean, hfov_min]
    return epd_range, hfov_range

########################################################################
# test
if __name__ == '__main__':
  import matplotlib.pyplot as plt

  # Import zebase and lensview database
  df_zebase = pd.read_csv("C:/Users/KU/Desktop/AI/design/Zebase/Zebase.csv")
  df_lensview = pd.read_csv("C:/Users/KU/Desktop/AI/design/Lensview/LensView_all.csv")
  # Merge them depending on sequence GA, GGA, GAGA
  df = pd.concat([df_zebase, df_lensview], ignore_index=True, axis=0)

  # Exclude zero EFL
  df = df[df["efl"] != 0.]
  # Exclude config more than 2
  df = df[~df['file_name'].str.contains('conf') | df['file_name'].str.contains('conf1')]
  print(df.shape)

  # database for sequence 
  df_ga = df[df['sequence']=='GA']
  df_gga = df[df['sequence']=='GGA']
  df_gaga = df[df['sequence']=='GAGA']
  # reset row indexing starting from zero
  df_ga = df_ga.reset_index()
  df_gga = df_gga.reset_index()
  df_gaga = df_gaga.reset_index()

  print('num_GA: {0}'.format(df_ga.shape))
  print('num_GGA: {0}'.format(df_gga.shape))
  print('num_GAGA: {0}'.format(df_gaga.shape))
  
  # convert dataframe into tensor
  DP = DataframeProcessor('GA')
  df_seq = df_ga
  df_seq, tensor_X, tensor_y = DP.process_dataframe(df_seq)
  print(df_seq)


  # # df = pd.read_csv("../design/GAGA_NS_1/gaga_ns_1_raw.csv")
  # # df = df[df["hfov"] <= 5]
  # # df = df.reset_index()

  # # DP = DataframeProcessor('GAGA')
  # # df, ggaX, ggay = DP.process_dataframe(df, 1024)
  # # pd.set_option('display.max_columns', None)
  # # pd.set_option('display.max_rows', None)
  # # print(df)
  # # epd_range, hfov_range = range_EPDHFOV(df)

  # df_ga = pd.read_csv("../design/lensview/ga_ns_1_raw.csv")
  # df_gga = pd.read_csv("../design/lensview/gga_ns_1_raw.csv")
  # df_ggga = pd.read_csv("../design/lensview/ggga_ns_1_raw.csv")
  # df_gaga = pd.read_csv("../design/lensview/gaga_ns_1_raw.csv")
  # df_gagaga = pd.read_csv("../design/lensview/gagaga_ns_1_raw.csv")

  # # compute quadtile of HFOV
  # df = df_gagaga
  # # quar = df['hfov'].quantile([0.25,0.5,0.75])
  # # quar = quar.iloc
  # # iqr = quar[2]-quar[0]
  # # up = quar[2] + 1.5*iqr
  # # low = quar[0] - 1.5*iqr
  # # print('up: {0}, low: {1}'.format(up, low))
  # # # plot dataset histgram
  # # fig, ax = plt.subplots(1, 2)
  # # df['epd'].hist(bins = 25, ax=ax[0])
  # # df['hfov'].hist(bins = 25, ax=ax[1])
  # # plt.suptitle('gagaga_ns_1 - ' + str(df.shape[0])+' designs')
  # # ax[0].set_title('EPD')
  # # ax[1].set_title('HFOV')
  # # plt.show()
  # # EPD-HFOV scattered diagram
  # fig, ax = plt.subplots()
  # ax.scatter(df['epd'].to_numpy(), df['hfov'].to_numpy(), c='b')
  # plt.suptitle('gagaga_ns_1 - ' + str(df.shape[0])+' designs')
  # ax.set_xlabel('EPD [mm]')
  # ax.set_ylabel('HFOV [deg]')
  # plt.show()