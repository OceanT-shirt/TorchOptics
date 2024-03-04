from rayopt.material import air
from rayopt.elements import Spheroid
from analysis.system_mod import System_Mod
import numpy as np

from analysis.material_mod import Material_Mod

def zmx_to_system(data, item=None):
    s = System_Mod()
    next_pos = 0.
    count_surf = 0
    DiamList = []
    rate_epd = 0.94
    s.stop = 1
    for line in data.splitlines():
        if count_surf >= 1:
            e = s[-1]
        if not line.strip():
            continue
        line = line.strip().split(" ", 1)
        cmd = line[0]
        args = len(line) == 2 and line[1] or ""
        if cmd == "UNIT":
            s.scale = {
                    "MM": 1e-3,
                    "INCH": 25.4e-3,
                    "IN": 25.4e-3,
                    }[args.split()[0]]
        elif cmd == "NAME":
            s.description = args.strip("\"")
        elif cmd == "SURF":
            s.append(Spheroid(distance=next_pos, material=air))
            count_surf += 1
        elif cmd == "CURV":
            e.curvature = float(args.split()[0])
        elif cmd == "DISZ":
            next_pos = float(args)
            if np.isinf(next_pos):
                next_pos = 10
        elif cmd == "GLAS":
            args = args.split()
            name = args[0]
            try:
                e.material = Material_Mod.make(name)
            except KeyError:
                try:
                    nd = float(args[3])
                    Abbe = float(args[4])
                    if Abbe < 20.:
                        Abbe = 80.
                    e.material = Material_Mod.make((nd, Abbe))
                except Exception as e:
                    print("material not found", name, e)
        elif cmd == "DIAM":
            e.radius = float(args.split()[0])
            DiamList.append(e.radius*2)
        elif cmd == "STOP":
            s.stop = count_surf-1
        elif cmd == "ENPD":
            s.epd = float(args.split()[0])
        elif cmd == "YFLD":
            s.fieldY = [float(i) for i in args.split() if i]
        elif cmd == "WAVL":
            s.wavelengths = [float(i)*1e-6 for i in args.split() if i]
        elif cmd == "COAT":
            e.coating = args.split()[0]
        elif cmd == "CONI":
            e.conic = float(args.split()[0])
        elif cmd == "PARM":
            i, j = args.split()
            i = int(i) - 1
            j = float(j)
            if i < 0:
                if j:
                    print("aspheric 0 degree not supported", cmd, args)
                continue
            if e.aspherics is None:
                e.aspherics = []
            while len(e.aspherics) <= i:
                e.aspherics.append(0.)
            e.aspherics[i] = j
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
                     "MIRR",  # surface is mirror
                     "PARM",  # aspheric parameters
                     "SQAP",  # square aperture?
                     "XDAT", "YDAT",  # xy toroidal data
                     "OBNA",  # object na
                     "PKUP",  # pickup
                     "MAZH", "CLAP", "PPAR", "VPAR", "EDGE", "VCON",
                     "UDAD", "USAP", "TOLE", "PFIL", "TCED", "FNUM",
                     "TOL", "MNUM", "MOFF", "FTYP", "SDMA", "GFAC",
                     "PUSH", "PICB", "ROPD", "PWAV", "POLS", "GLRS",
                     "BLNK", "COFN", "NSCD", "GSTD", "DMFS", "ISNA",
                     "VDSZ", "ENVD", "ZVDX", "ZVDY", "ZVCX", "ZVCY",
                     "ZVAN", "XFLN", "YFLN", "VDXN", "VDYN", "VCXN",
                     "VCYN", "VANN", "FWGT", "FWGN", "WWGT", "WWGN",
                     "WAVN", "WAVM", "XFLD", "MNCA", "MNEA",
                     "MNCG", "MNEG", "MXCA", "MXCG", "RGLA", "TRAC",
                     "FLAP", "TCMM", "FLOA", "PMAG", "TOTR", "SLAB",
                     "POPS", "COMM", "PZUP", "LANG", "FIMP",
                     ):
            pass
        else:
            print(cmd, "not handled", args)
            continue

    # check ENPD is smaller than diameter of lens
    if s.epd > DiamList[s.stop]:
        s.epd = DiamList[s.stop] * rate_epd
    return s


def len_to_system(fil, item=None):
    s = System_Mod()
    e = Spheroid()
    th = 0.
    count_surf = 0
    for line in fil.readlines():
        p = line.split()
        if not p:
            continue
        cmd, args = p[0], p[1:]
        if cmd == "LEN":
            s.description = " ".join(args[1:-2]).strip("\"")
        elif cmd == "UNI":
            s.scale = float(args[0])*1e-3
        elif cmd == "AIR":
            e.material = air
        elif cmd == "TH":
            th = float(args[0])
            if th > 1e2:
                th = np.inf
        elif cmd == "AP":
            if args[0] == "CHK":
                del args[0]
            e.radius = float(args[0])
        elif cmd == "GLA":
            e.material = Material_Mod.make(args[0])
        elif cmd == "AST":
            s.stop = count_surf
        elif cmd == "EBR":
            s.epd = float(args.split()[0])
        elif cmd == "ANG":
            s.fieldY = [float(i) for i in args.split() if i]
        elif cmd == "WV":
            s.wavelengths = [float(i)*1e-6 for i in args.split() if i]
        elif cmd == "RD":
            e.curvature = 1/float(args[0])
        elif cmd in ("NXT", "END"):
            s.append(e)
            e = Spheroid()
            e.distance = th
            count_surf += 1
        elif cmd in ("//", "DES", "GIH", "DLRS", "WW"):
            pass
        else:
            print(cmd, "not handled", args)
    return s