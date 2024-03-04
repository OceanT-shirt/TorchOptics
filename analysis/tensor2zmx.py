import numpy as np
# from torchlens.lens_modeling import Structure
# from torchlens.ray_tracing_lite import compute_last_curvature
from preprocessing.augmentation_EPD_HFOV import maximum_diameter

def tensor2zemax(design, exportpath):
    rate_scr = 1.5

    # extract lens property from NamedTuple
    STOP = design.stop_idx
    Seq = design.sequence
    # convert tensor to double/array
    EPD = design.epd
    HFOV = design.hfov
    C = design.c
    T = design.t
    N = design.nd
    V = design.v
    # compute maximum diameter
    D = maximum_diameter(Seq, C, T)

    lines = []

    # append EPD, HFOV, WAVL
    lines.append('ENPD ' + str(EPD) + '\n')
    lines.append('YFLD 0 ' + str(HFOV) + '\n')
    lines.append('WAVL 4.590000000000E-001 5.200000000000E-001 6.400000000000E-001\n')

    # append SURF 0
    lines.append('SURF 0\n')
    lines.append('CURV 0.000000000000000000E+000 0 0.000000000000E+000 0.000000000000E+000 0')
    lines.append('HIDE 0 0 0 0 0 0 0 0 0 0\n')
    lines.append('MIRR 2 0.000000000E+000\n')
    lines.append('DISZ ' + str(np.sum(T)*0.2) + '\n')
    lines.append('DIAM 0.05 0 0 0 1.000000000000E+000\n')
    lines.append('POPS 0 0 0 0 0 0 0 0 1 1 1.000000000000E+000 1.000000000000E+000 0 0.000000000000E+000 0.000000000000E+000\n')

    numSurf = len(Seq)
    n = 0
    # iterate through surfaces
    for s in range(numSurf):
        SurfType = Seq[s]

        # Surface number
        lines.append('SURF ' + str(s+1) + '\n')

        # stop index
        if s+1 == STOP:
            lines.append('STOP\n')

        # Curvature
        lines.append('CURV ' + str(C[s]) + '\n')

        # Thickness
        lines.append('DISZ ' + str(T[s]) + '\n')

        # Refractive index & Abbe number
        if SurfType == 'G':
            lines.append('GLAS XXX 1 0 ' + str(N[n]) + ' ' + str(V[n]) + '\n')
            n += 1

        # semi-diameter 
        lines.append('DIAM ' + str(D[s]/2) + ' 1 0\n')
        

    # append last SURF (screen)
    lines.append('SURF ' + str(numSurf+1) + '\n')
    lines.append('CURV 1.0e-10\n')
    lines.append('DISZ 0.000000000000000000E+000\n')
    lines.append('DIAM ' + str(D.max()/2*rate_scr) + ' 0 0\n')

    # export zmx file
    with open(exportpath, "w") as file:
        file.writelines(lines)