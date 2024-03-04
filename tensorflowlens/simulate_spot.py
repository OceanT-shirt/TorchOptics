
"""
    Spot computation via ray tracing
"""

# import os
import optics_simulator
import numpy as np
import tensorflow as tf
import yaml
from zmx2yml import zmx2yml

def simulate_spot(LensDict={'c': [0.0, -0.242432341, -0.424975232],
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
                 PupleSampling='circular', isShow=True):
    
    model = optics_simulator.RaytracedOptics(
        StopIndex=np.array(LensDict['stop_idx']),
        Sequence=np.array(LensDict['sequence']),
        HFOV=np.array(LensDict['hfov']),
        EPD=np.array(LensDict['epd']),
        Curv=tf.constant(LensDict['c']),
        Thickness=tf.constant(LensDict['t']),
        Nref=tf.constant(LensDict['nd']),
        Abbe=tf.constant(LensDict['v']),
        n_sampled_fields=NField,  # Down from 21 (default) to fit into memory
        n_pupil_rings=NPuple,  # Down from 32 (default) to fit into memory
        wavelengths = Wave,
        pupil_sampling=PupleSampling,
        simulated_res_factor=1,
        apply_distortion=True,
        apply_relative_illumination=True,
        lazy_init=True
    )

    xd, yd, ray_ok, lossParams = model.GetTraceResult()
    if isShow:
        model.ShowTraceResult(xd, yd, ray_ok, lossParams)

    return xd, yd, ray_ok, lossParams


if __name__ == '__main__':
    # importpath = './Database/GA/A_001.ZMX'; exportpath = './Database/GA/A_001.yml'
    # importpath = './Database/raw-images/N_080.ZMX'; exportpath = './Database/raw-images/N_080.yml'
    importpath = './Database/LensView/4634237A.ZMX'; exportpath = './Database/LensView/4634237A.yml'
    # zemax into yaml file
    value = zmx2yml(importpath, exportpath)

    if value==0:
        print('no file found')
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

    [xd, yd, ray_ok, lossParams] = simulate_spot(LensDict=MyDict,
                                                NField=8, NPuple=8,
                                                Wave=[459, 520, 640],
                                                PupleSampling='circular', isShow=True)