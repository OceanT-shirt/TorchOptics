import sys
sys.path.append('C:/Users/KU/Desktop/AI/src')
import traceback
sys.tracebacklimit = 8
import matplotlib.pyplot as plt

import torch

from training.lensdesign_model import LensDesignCreator

from analysis.lens_visualization import evaluate_model, evaluate_system

from preprocessing.augmentation_EPD_HFOV import maximum_diameter
from preprocessing.process_dataframe import sequence_encoder

def generate_design(xx, EFL, path_WB, Seq):
    try:
        xx[-4] = sequence_encoder(Seq)

        # Define the DNN model
        model = LensDesignCreator(Seq)
        # Load the weights
        model.load_state_dict(torch.load(path_WB, map_location=torch.device('cpu')))
        # Set the model to evaluation mode
        model.eval()

        # Generate design
        design = evaluate_model(torch.Tensor(xx), model, Seq, device='cpu')
        # compute maximum diameter
        Ds = maximum_diameter(Seq, design.c, design.t)

        # Denormalise with EFL
        design = design._replace(epd=design.epd*EFL)
        design = design._replace(c=design.c/EFL)
        design = design._replace(t=design.t*EFL)
        Ds = Ds * EFL

        # Compute RMS
        RMS, Aberr = evaluate_system(design)
    except:
        plt.figure()
        plt.text(0, 0.1, traceback.format_exc())
        plt.show()
        RMS = 0
        Aberr = 0
    return design, Ds, RMS, Aberr

############################
if __name__ == '__main__':
    from analysis.lens_visualization import display_lens_analysis

    # xx = [0.227, 1.274, 0, 2.066, 0, 2.112, 0.277, 2.117, 0, 1, -1, -1]
    # Seq = 'GGA'
    # path_WB = 'C:/Users/KU/Desktop/Hyperion/Modules/AI/lensview_gga_SR0.3PR2e-3.pth'
    # EFL = 10

    xx = [0.628970902033151, 0.0, 0.0, 1.0, 0.0, 1.0, 0, 1, -1, -1]
    Seq = 'GA'
    path_WB = 'C:/Users/KU/Desktop/Hyperion/Modules/AI/lensview_GA_SR0.02_PR0.0002.pth'
    EFL = 10

    design, Ds, RMS, Aberr = generate_design(xx, EFL, path_WB, Seq)
    print(RMS)
    print(Aberr)

    # display layout
    # display_lens_analysis(design)