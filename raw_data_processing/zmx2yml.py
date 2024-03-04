
from rayopt.material import Material
import numpy as np
import yaml
import torch
import torchlens.ray_tracing_lite as rt

# Convert zemax file into yaml file
def zmx2yml(importpath, exportpath):
    # read ZMX file
    with open(importpath, 'r') as f:
        data = f.read()
    
    Sequence = ''
    Curv = []
    Diam = []
    Thick = []
    Nd = []
    Abbe = []
    
    # how many "SURF" in data?
    for line in data.splitlines():
        if not line.strip():
            continue
        line = line.strip().split(" ", 1)
        cmd = line[0]
        args = len(line) == 2 and line[1] or ""
        if cmd == "SURF":
            NSurf = int(args)

    f_skip = False # skip Surf0
    f_glass = False # medium type
    f_curv = False  # curvature
    f_ENPD = False  # entrance puple size
    f_exclude = False # exclude aspheric and mirror file
    f_MultiConf = False # multiple layout configuration
    for line in data.splitlines():

        # skip data of initial SURF and last SURF
        if line == "SURF 0":
            f_skip = True
        if line == "SURF 1":
            f_skip = False
        if line == "SURF " + str(NSurf):
            f_skip = True

            # process previous surface's medium and curvature
            if f_glass:
                Sequence = Sequence + 'G'
            else:
                Sequence = Sequence + 'A'
            if f_curv == False:
                Curv.append(0)
        if line.find("MNUM") != -1:
            f_skip = False
        
        # read data
        if not f_skip:
            if not line.strip():
                continue
            line = line.strip().split(" ", 1)
            cmd = line[0]
            args = len(line) == 2 and line[1] or ""
            if cmd == "SURF":
                # Surface type
                if line[1] != "1":
                    if f_glass:
                        Sequence = Sequence + 'G'
                    else:
                        Sequence = Sequence + 'A'
                    # reset flag for next SURF
                    f_glass = False
                    # default curvature is zero
                    if f_curv == False:
                        Curv.append(0)
                    # reset flag for next SURF
                    f_curv = False

                SurfIdx = int(args)
            elif cmd == "FTYP":
                SourceType = int(args.split()[0])
                if SourceType != 0:
                    f_exclude = True
                    break
            elif cmd == "STOP":
                StopIdx = SurfIdx
            elif cmd == "ENPD":
                Epd = float(args)
                f_ENPD = True
            elif cmd == "PUPD":
                if not f_ENPD:
                    Epd = float(args.split()[1])
            elif cmd == "YFLD":
                HFov = [float(i) for i in args.split()]
            elif cmd == "WAVL":
                Wave = [float(i)*1e3 for i in args.split() if i]
            elif cmd == "CURV":
                Curv.append(float(args.split()[0]))
                f_curv = True
            elif cmd == "DISZ":
                thick = float(args)
                if thick < 0:
                    # negative thickness
                    f_exclude = True
                    break
                else:
                    Thick.append(float(args))
            elif cmd == "GLAS":
                f_glass = True
                # reference refractive index
                args = args.split()
                name = args[0]
                if name == "XXX" or name == "___BLANK":
                    nd = float(args[3])
                    v = float(args[4])
                else:
                    Mat = Material.make(name)
                    nc = float(Mat.refractive_index(656.3*1e-9))
                    nd = float(Mat.refractive_index(589.2*1e-9))
                    nf = float(Mat.refractive_index(486.1*1e-9))
                    # calculate Abbe number
                    if nf-nc == 0:
                        v = 9e9
                    else:
                        v = (nd-1)/(nf-nc)
                # add to list
                Nd.append(nd)
                Abbe.append(v)
            elif cmd == "DIAM":
                Diam.append(float(args.split()[0])*2)
            elif cmd == "MNUM":
                NumConf = int(args.split()[0])
                if NumConf > 1:
                    f_MultiConf = True
                    AptList = []
                    YFldList = [[] for i in range(NumConf)]
                    ThickIList = [[] for i in range(NumConf)]
                    ThickVList = [[] for i in range(NumConf)]
            elif cmd == "YFIE":
                if NumConf >= 2:
                    IdxConf = int(args.split()[1])-1
                    value = float(args.split()[2])
                    YFldList[IdxConf].append(value)
            elif cmd == "APER":
                if NumConf >= 2:
                    value = float(args.split()[2])
                    AptList.append(value)
            elif cmd == "THIC":
                if NumConf >= 2:
                    IdxSurf = int(args.split()[0])-1
                    IdxConf = int(args.split()[1])-1
                    value = float(args.split()[2])
                    if value >= 0:
                        ThickIList[IdxConf].append(IdxSurf)
                        ThickVList[IdxConf].append(value)
                    else:
                        # negative thickness
                        f_exclude = True
                        break
            elif cmd == "PARM":
                # exclude file contains aspheric lens
                f_exclude = True
                break
            elif cmd == "MIRR":
                # exclude file contains mirror elements
                if float(args.split()[1])!=0.0:
                    f_exclude = True
                    break
            elif cmd in ("GCAT",  # glass catalog names
                        "OPDX",  # opd
                        "RAIM",  # ray aiming
                        "CONF",  # configurations
                        "PUPD",  # pupil
                        "EFFL",  # focal lengths
                        "VERS",  # version
                        "MODE",  # mode
                        "NOTE",  # note
                        "TYPE",  # surface type
                        "HIDE",  # surface hide
                        "SQAP",  # square aperture?
                        "XDAT", "YDAT",  # xy toroidal data
                        "OBNA",  # object na
                        "PKUP",  # pickup
                        "MAZH", "CLAP", "PPAR", "VPAR", "EDGE", "VCON",
                        "UDAD", "USAP", "TOLE", "PFIL", "TCED", "FNUM",
                        "TOL", "MOFF", "FTYP", "SDMA", "GFAC",
                        "PUSH", "PICB", "ROPD", "PWAV", "POLS", "GLRS",
                        "BLNK", "COFN", "NSCD", "GSTD", "DMFS", "ISNA",
                        "VDSZ", "ENVD", "ZVDX", "ZVDY", "ZVCX", "ZVCY",
                        "ZVAN", "XFLN", "YFLN", "VDXN", "VDYN", "VCXN",
                        "VCYN", "VANN", "FWGT", "FWGN", "WWGT", "WWGN",
                        "WAVN", "WAVM", "XFLD", "MNCA", "MNEA", "UNIT",
                        "MNCG", "MNEG", "MXCA", "MXCG", "RGLA", "TRAC",
                        "FLAP", "TCMM", "FLOA", "PMAG", "TOTR", "SLAB",
                        "POPS", "COMM", "PZUP", "LANG", "FIMP", "NAME",
                        "DIAM", "WAVL", "COAT", "CONI"
                        ):
                pass
            else:
                print(cmd, "not handled", args)
                continue


    # Export result
    if not f_exclude:
        # check empty in config list
        if f_MultiConf:
            if not AptList:
                AptList = [Epd] * NumConf
            if not YFldList[0]:
                YFldList = [HFov] * NumConf

        if not f_MultiConf:
            # normalise by EFL
            EFL = ComputeEFL(Sequence, Curv, Thick, Nd)
            Epd = Epd/abs(EFL)
            Curv = [v*abs(EFL) for v in Curv]
            Diam = [v/abs(EFL) for v in Diam]
            Thick = [v/abs(EFL) for v in Thick]

            # create dist
            LensDict = {'stop_idx': [StopIdx], 'sequence': [Sequence], 'epd': [Epd], 'hfov': HFov,
                     'efl': [EFL], 'wave': Wave, 'c': Curv, 'd': Diam, 't': Thick, 'nd': Nd, 'v': Abbe}
            # convert dist into YML file and export
            with open(exportpath, 'w') as file:
                yaml.dump(LensDict, file)

            return 1
        else:
            for i in range(0,NumConf):
                Epd_conf = AptList[i]
                HFov_conf = YFldList[i]
                NumSurf = len(ThickIList[i])
                Thick_conf = Thick.copy()
                if ThickIList[0]:
                    for s in range(0,NumSurf):
                        IdxSurf = ThickIList[i][s]
                        Thick_conf[IdxSurf] = ThickVList[i][s]

                # normalise by EFL
                EFL_conf = ComputeEFL(Sequence, Curv, Thick_conf, Nd)
                Epd_conf = Epd_conf/abs(EFL_conf)
                Curv = [v*abs(EFL_conf) for v in Curv]
                Diam = [v/abs(EFL_conf) for v in Diam]
                Thick_conf = [v/abs(EFL_conf) for v in Thick_conf]

                # create dict.
                LensDict = {'stop_idx': [StopIdx], 'sequence': [Sequence], 'epd': [Epd_conf], 'hfov': HFov_conf,
                         'efl': [EFL_conf], 'wave': Wave, 'c': Curv, 'd': Diam, 't': Thick_conf, 'nd': Nd, 'v': Abbe}
                
                exportconfpath = exportpath[0:-4] + '_conf' + str(i+1) + '.yml'
                # convert dist into YML file and export
                with open(exportconfpath, 'w') as file:
                    yaml.dump(LensDict, file)

            return NumConf
    else:
        # export empty test file
        exportpath = exportpath[0:-3] + 'txt'
        message = 'System is excluded in following reasons: Negative thickness, Aspheric surface, Mirror surface, and Object type source'
        with open(exportpath, 'w') as file:
            file.write(message)

        return 0
    
def ComputeEFL(Sequence, Curv, Thick, Nd):
    numSequence = len(Sequence)
    CurvT = torch.tensor(np.array([Curv]).reshape([1,numSequence]))
    ThickT = torch.tensor(np.array([Thick]).reshape([1,numSequence]))
    Nd2 = np.zeros((1,numSequence))
    iglass = 0
    for i in range(numSequence):
        if Sequence[i] == 'A':
            Nd2[0,i] = 1.0
        elif Sequence[i] == 'G':
            Nd2[0,i] = Nd[iglass]
            iglass = iglass + 1
    NdT = torch.tensor(Nd2)
    NdT = torch.cat((torch.ones_like(NdT[:, 0:1]), NdT), axis=1)
    abcd = rt.interface_propagation_abcd(CurvT, ThickT, NdT)
    abcd = rt.reduce_abcd(abcd)
    EFL = float(-1 / abcd[:, 1, 0])
    return EFL