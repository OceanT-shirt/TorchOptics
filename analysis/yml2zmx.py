
import yaml
import re

def yml2zmx(importpath, exportpath):
    # read yml data
    with open(importpath, 'r') as f:
        data = f.read()

    Dict = yaml.safe_load(data)
    Seq = Dict['sequence'][0]
    HFOV = Dict['hfov']
    EPD = Dict['epd'][0]
    STOP = Dict['stop_idx'][0]
    C = Dict['c']
    T = Dict['t']
    N = Dict['nd']
    V = Dict['v']

    lines = []

    # append EPD, HFOV, WAVL
    String = str(HFOV)
    removed = [',', '[', ']']
    replace = ''
    pattern = "[" + re.escape("".join(removed)) + "]"
    String = re.sub(pattern, replace, String)
    lines.append('ENPD ' + str(EPD) + '\n')
    lines.append('YFLD ' + String + '\n')
    lines.append('WAVL 4.590000000000E-001 5.200000000000E-001 6.400000000000E-001\n')

    # append SURF 0
    lines.append('SURF 0\n')
    lines.append('CURV 0.000000000000000000E+000 0 0.000000000000E+000 0.000000000000E+000 0')
    lines.append('HIDE 0 0 0 0 0 0 0 0 0 0\n')
    lines.append('MIRR 2 0.000000000E+000\n')
    lines.append('DISZ INFINITY\n')
    lines.append('DIAM 0.000000000000E+000 0 0 0 1.000000000000E+000\n')
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

    # append last SURF
    lines.append('SURF ' + str(numSurf+1) + '\n')

    # export zmx file
    with open(exportpath, "w") as file:
        file.writelines(lines)