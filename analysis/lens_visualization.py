import sys
sys.path.append('C:/Users/KU/Desktop/AI/src')

from typing import NamedTuple
import torch
from torchlens.lens_modeling import n_v_from_g
from analysis.LensReport import show_analysis, rms_aberr_analysis
from analysis.tensor2zmx import tensor2zemax

import numpy as np
from torchlens.lens_modeling import Structure
from torchlens.ray_tracing_lite import compute_last_curvature

# NamedTuple index
class LensVisualizationInput(NamedTuple):
    stop_idx: int
    sequence: str
    hfov: float  # shape: (1,)
    epd: float  # shape: (1,)
    c: np.ndarray
    t: np.ndarray
    nd: np.ndarray
    v: np.ndarray    

# Show optical report
def display_lens_analysis(design, print_system=False):
    zmxpath = 'analysis/temp.zmx'
    tensor2zemax(design, zmxpath)
    # produce lens report
    show_analysis(zmxpath, print_system)

# Show optical report
def evaluate_system(design):
    zmxpath = 'analysis/temp.zmx'
    tensor2zemax(design, zmxpath)
    # compute rms spot size and third order aberrations
    RMS, Aberrations = rms_aberr_analysis(zmxpath)
    return RMS, Aberrations

# Evaluate model and visualize output system
def evaluate_model(input, model, lens_type="GGA", device="cuda"):
    """
    GA
    Example input tensor
    "epd", "hfov", "t1_min", "t1_range", "t2_min", "t2_range", "sequence_encoded", "stop_idx", "as_c", "as_t"
    Example output tensor
    "g11", "g12", "c1", "t1", "t2"
    GGA
    Example input tensor
    "epd", "hfov", "t1_min", "t1_range", "t2_min", "t2_range", "t3_min", "t3_range", "sequence_encoded", "stop_idx", "as_c", "as_t"
    Example output tensor
    "g11", "g12", "g21", "g22", "c1", "c2", "t1", "t2", "t3"
    GAGA
    Example input tensor
    "epd", "hfov", "t1_min", "t1_range", "t2_min", "t2_range", "t3_min", "t3_range", "sequence_encoded", "stop_idx", "as_c", "as_t"
    Example output tensor
    "g11", "g12", "g21", "g22", "c1", "c2", "t1", "t2", "t3"
    """
    
    if torch.cuda.is_available():
        input = input.cuda()

    code_lenstype = int(input[-4].item())
    numsurf = len(str(code_lenstype))
    numin = 2 + 2*numsurf

    # evaluate model
    inputX = input[:numin]
    output = model(inputX)

    design = Tensor2DesignTuple(input, output, lens_type, device)
    return design

def Tensor2DesignTuple(input, output, lens_type, device='cuda'):
    code_lenstype = int(input[-4].item())
    numsurf = len(str(code_lenstype))
    numglass = sum(map(int, str(code_lenstype)))
    numout = 2*numglass + 2*numsurf-1

    # epd, hfov, thickness
    epd = input[[0]]
    hfov = input[1]
    t = output[2*numglass+numsurf-1:numout]  
    # curvature without last surface
    c_wo_last = output[2*numglass:2*numglass+numsurf-1]  

    # convert g in n_refr and Abbe
    # Convert g into n and v
    n = torch.zeros(numglass, device=device)
    v = torch.zeros(numglass, device=device)
    # iterate through glasses
    for i in range(numglass):
        gi = output[2*i:2*i+2].unsqueeze(dim=0)
        # Make sure nv_converter is the same one as the first convert
        ni, vi = n_v_from_g(gi)
        n[i] = ni
        v[i] = vi

    # stop index
    stop_idx = input[-3].detach().cpu().unsqueeze(dim=0).numpy()
    stop_idx = stop_idx.astype("int64")

    # extract lens property from tensor
    STOP = stop_idx[0]
    Seq = lens_type
    EPD = epd.item()
    HFOV = hfov.item()
    T = t.detach().cpu().numpy()
    N = n.detach().cpu().numpy()
    V = v.detach().cpu().numpy()
    # c_without_last to c
    seq_np = np.array([Seq])
    stop_idx_np = np.array([STOP])
    structure = Structure(stop_idx=stop_idx_np, sequence=seq_np, default_device=device)
    C = compute_last_curvature(structure, c_wo_last, t, n)
    C = C.detach().cpu().numpy()

    design = LensVisualizationInput(STOP, Seq, HFOV, EPD, C, T, N, V)
    return design

def Reference2DesignTuple(df_single, lens_type):
    STOP = df_single.stop_idx
    Seq = lens_type
    HFOV = df_single.hfov
    EPD = df_single.epd
    # delete parentheses
    c_str = df_single.c_all.replace('[', '').replace(']','')
    t_str = df_single.t_all.replace('[', '').replace(']','')
    nd_str = df_single.nd.replace('[', '').replace(']','')
    v_str = df_single.v.replace('[', '').replace(']','')
    C = np.fromstring(c_str, sep=',')
    T = np.fromstring(t_str, sep=',')
    N = np.fromstring(nd_str, sep=',')
    V = np.fromstring(v_str, sep=',')
    design = LensVisualizationInput(STOP, Seq, HFOV, EPD, C, T, N, V)
    return design



########################################################################
# test
if __name__ == '__main__':
    # design = LensVisualizationInput(1, 'GA',
    #                                 0, 1.4554,
    #                                 np.array([0.040, -0.04]), np.array([3.0, 21.0]),
    #                                 np.array([1.527]), torch.tensor([64.17]))
    design = LensVisualizationInput(1, 'GA',
                                    1e-6, 0.1,
                                    np.array([2.7736,-2.0]), np.array([0.1211, 0.7111]),
                                    np.array([1.527]), np.array([64.17]))
    display_lens_analysis(design)
    print('**********')
    print('Hello!!!')
    print('**********')
    # RMS, Aberr = evaluate_system(design)
