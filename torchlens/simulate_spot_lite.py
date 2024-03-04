import sys
sys.path.append('C:/Users/KU/Desktop/AI/src')

import numpy as np
import torch
from torchlens import optics_simulator_lite

def SimulateSpotLite(LensDict={'c': [0.0, -0.242432341, -0.424975232],
                           'epd': [0.7],
                           'hfov': [0.0, 17.5, 25.0],
                           'nd': [1.5224147149313454],
                           'sequence': ['AGA'],
                           'stop_idx': [1],
                           't': [1.21071062, 0.25, 9.86362667],
                           'v': [59.450346241693694]},
                 NField=8, 
                 NPuple=8, 
                 Wave=[459, 520, 640], 
                 PupleSampling='circular'):
    
    model = optics_simulator_lite.RaytracedOptics(
            initial_lens_path="",
            stop_index=np.array(LensDict['stop_idx']),
            sequence=np.array(LensDict['sequence']),
            hfov=torch.tensor(LensDict['hfov']),
            epd=torch.tensor(LensDict['epd']),
            curvature=torch.tensor(LensDict['c'], requires_grad=True),
            thickness=torch.tensor(LensDict['t'], requires_grad=True),
            n_refractive=torch.tensor(LensDict['nd'], requires_grad=True),
            abbe_number=torch.tensor(LensDict['v'], requires_grad=True),
            n_sampled_fields=NField,  # Down from 21 (default) to fit into memory
            n_pupil_rings=NPuple,  # Down from 32 (default) to fit into memory
            n_ray_aiming_iter=128,
            wavelengths = torch.tensor(Wave),
            penalty_rate = 0.2,
            pupil_sampling=PupleSampling,
            simulated_res_factor=1,
            apply_distortion=True,
            apply_relative_illumination=True,
            lazy_init=True,
            glass_catalog_path='../glass/selected_glass.csv',
            device = 'cpu'
        )
    return model


###########################################################
if __name__ == '__main__':
    import sys
    import os
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(test_dir)
    sys.path.append(project_dir)

    import yaml
    from raw_data_processing.zmx2yml import zmx2yml

    # import matplotlib.pyplot as plt
    # import pandas as pd
    # from analysis.yml2zmx import yml2zmx
    # from pytictoc import TicToc

    importpath = 'C:/Users/KU/Desktop/Hyperion/TestFiles/dummy.ZMX'; exportpath = 'C:/Users/KU/Desktop/Hyperion/TestFiles/dummy.yml'
    # importpath = 'C:/Users/KU\Desktop/dummy2.ZMX'; exportpath = 'C:/Users/KU\Desktop/dummy.yml'
    # importpath = 'C:/Users/KU/Desktop/AI/design/zebase/F_006.ZMX'; exportpath = 'C:/Users/KU/Desktop/AI/design/zebase/yaml/F_006.yaml'
    # importpath = 'C:/Users/KU/Desktop/16428.ZMX'; exportpath = 'C:/Users/KU/Desktop/dummy.yaml'
    # importpath = 'C:/Users/KU/Desktop/AI/TestData/A_003.ZMX'; exportpath = 'C:/Users/KU/Desktop/AI/TestData/A_003.yml'
    # importpath = 'C:/Users/KU/Desktop/AI/TestData/4179183F.ZMX'; exportpath = 'C:/Users/KU/Desktop/AI/TestData/4179183F.yml'
    
    # zemax into yaml file
    value = zmx2yml(importpath, exportpath)

    # yml into zemax file
    # yml2zmx(exportpath, 'C:/Users/KU/Desktop/A_009R.ZMX')

    if value==0:
        print('No file found.')
    elif value==1:
        # read yaml file
        with open(exportpath, 'r') as f:
            data = f.read()
    elif value>1:
        # read yaml file
        exportconfpath = exportpath[0:-4] + '_conf1' + '.yml'
        with open(exportconfpath, 'r') as f:
            data = f.read()
    MyDict = yaml.safe_load(data)

    # # HFOV = [0, 10, 30, 60, 70, 80]
    # # Loss = []
    # # for f in HFOV:
    # #     MyDict['hfov'] = [f]

    #     #t = TicToc()
    #     #t.tic()
    model = SimulateSpotLite(
                LensDict=MyDict,
                NField=1, NPuple=8, 
                Wave=[520.], 
                PupleSampling='circular')
    
    # Compute image on screens
    x, y, ray_ok = model.do_ray_tracing(model.lensR)
    loss_unsup = model.loss_dict["loss_unsup"]
    # Show spot diagram
    model.ShowTraceResult(x, y, ray_ok, loss_unsup)

    # # Compute entrance pupil rate
    # yp_scale, yp_offset = model.do_ray_aiming(model.lensR)
    # print('Y_scale: {0}'.format(yp_scale))
    # print('Y_offset: {0}'.format(yp_offset))


    # Loss.append(loss_unsup.detach().float())
    #     #t.toc()

    # h = plt.figure()
    # ax = h.add_subplot()
    # ax.plot(HFOV, Loss, color='b', marker='.')
    # plt.show()

    # # loss function value
    # print(float(loss_unsup))
    # # # export to csv
    # # # convert array into dataframe
    # # xy = torch.vstack((x[0,0,:,0], y[0,0,:,0]))
    # # xy = torch.transpose(xy, 0, 1)
    # # DF = pd.DataFrame(xy.detach().numpy())
    # # # save the dataframe as a csv file
    # # DF.to_csv("./PYResult.csv")