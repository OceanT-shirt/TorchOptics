import sys
sys.path.append('C:/Users/KU/Desktop/AI/src')

import os
import numpy as np
import matplotlib.pyplot as plt

from analysis import design2system
from analysis.analysis_mod import Analysis_Mod
from analysis.geometric_trace_mod import GeometricTrace_Mod

def show_analysis(filepath, print_system=False):
    # len or zemax?
    split_tup = os.path.splitext(filepath)
    ext = split_tup[1]
    if ext in ['.len', '.LEN']:
        # import system from oslo
        with open(filepath, 'r') as file:
            sys = design2system.len_to_system(file)
        
    elif ext in ['.zmx', '.ZMX']:
        # import system from zemax
        with open(filepath, 'r') as file:
            data = file.read()
            sys = design2system.zmx_to_system(data)

    # modify variables
    # half FOV (rad)
    sys.object.angle = np.deg2rad(max(sys.fieldY)) 
    # angle rate
    fieldY_abs = [abs(fy) for fy in sys.fieldY]
    if max(fieldY_abs) == 0.:
        sys.fields = [1]
    else:
        sys.fields = [fy / max(fieldY_abs) for fy in sys.fieldY]
    sys.update()
    
    # show analysis report
    if print_system:
        print(sys)

    Analysis_Mod(sys)
    plt.show()

def rms_aberr_analysis(zmxpath):
    # import system from zemax
    with open(zmxpath, 'r') as file:
        data = file.read()
        sys = design2system.zmx_to_system(data)

    # Compute RMS spot size
    num_rays = 128
    RMS = np.zeros(3)
    for i, w in enumerate([459e-9, 520e-9, 640e-9]):
        t = GeometricTrace_Mod(sys)
        t.rays_point((0, 1), w, nrays=num_rays,
                distribution="hexapolar", clip=True)
        xy = t.y[-1][:,0:2]
        # exclude nan
        xy = xy[np.isnan(xy[:,0])==False, :]

        centre = np.array([np.mean(xy[:,0]), np.mean(xy[:,1])])
        dist = np.sqrt(np.sum((xy-centre)**2,1))
        RMS[i] = np.sqrt(np.mean(dist**2))
        rms = np.mean(RMS)

    # Compute abberations
    # third order - Spherical, Coma, Astigma, Peztval, Distor, Ax-color, Lat-color
    if sys.object.angle == 0.:
        sys.object.angle = 1e-6
    sys.update()

    PT = sys.paraxial
    aberrs = PT.transverse3
    sum_aberr = np.sum(aberrs, 0)

    return rms, sum_aberr

if __name__ == '__main__':
    # filepath = "C:/Users/KU/Desktop/CatalogMicroscope.zmx"
    # filepath = "C:/Users/KU/Desktop/dummy2.zmx"
    # filepath = "C:/Users/KU/Desktop/AI/src/analysis/temp.zmx"
    # filepath = "C:/Users/KU/Desktop/AI/src/analysis/20240112 - analysis.zmx"
    # filepath = "C:/Users/KU/Desktop/AI/src/analysis/20240115 - analysis.zmx"
    # filepath = "C:/Users/KU/Desktop/Hyperion/TestFiles/dummy.zmx"
    filepath = "./analysis/augment.zmx"
    
    show_analysis(filepath, print_system=True)
