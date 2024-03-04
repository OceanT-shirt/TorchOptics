import numpy as np
import scipy
from rayopt.system import System
from fastcache import clru_cache

from analysis.geometric_trace_mod import GeometricTrace_Mod

class System_Mod(System):
    def plot(self, ax, axis=1, npoints=31, adjust=True, **kwargs):
        # compute display range
        # resy = 16
        # resx = 16
        ratey = 1.1
        buffrate = 0.1
        radi_max = 0
        for i in range(len(self)):
            if self[i].radius > radi_max:
                radi_max = self[i].radius
        Xmin = self.track[0]
        Xmax = self.track[-1]
        Ymin = -radi_max*ratey
        Ymax = radi_max*ratey
        Xbuff = (Xmax-Xmin)*buffrate
        Ybuff = (Ymax-Ymin)*buffrate

        kwargs.setdefault("color", "black")
        ax.set_aspect("equal")
        if adjust:
            ax.set_aspect("equal")
            for s in ax.spines.values():
                s.set_visible(False)
        for x, z in self.surfaces_cut(axis, npoints):
            ax.plot(z, x, **kwargs)
        o = np.cumsum([e.offset for e in self], axis=0)
        ax.plot(o[:, 2], o[:, axis], ":", **kwargs)

        # set display range
        ax.yaxis.tick_right()
        # ax.set_xticks(np.linspace(Xmin-Xbuff, Xmax+Xbuff, resx))
        # ax.set_yticks(np.linspace(Ymin-Ybuff, Ymax+Ybuff, resy))
        ax.tick_params(axis='both', labelsize=6)
        ax.set_xlim(Xmin-Xbuff, Xmax+Xbuff) 
        ax.set_ylim(Ymin-Ybuff, Ymax+Ybuff)

    def base_text(self):
        print("EPD[mm]: {0:.3f}, HFOV[deg]: {1:.3f}, Stop: {2}".format(self.epd, self.fieldY[-1], self.stop))
        yield "Elements:"
        yield "%2s %1s %10s %10s %10s %17s %7s %7s %7s" % (
                "#", "T", "Distance", "Rad Curv", "Diameter",
                "Material", "n", "nd", "Vd")
        for i, e in enumerate(self):
            curv = getattr(e, "curvature", 0)
            roc = curv == 0 and np.inf or 1./curv
            rad = e.radius
            mat = getattr(e, "material", "")
            nd = getattr(mat, "nd", np.nan)
            vd = getattr(mat, "vd", np.nan)
            n = nd
            if mat:
                n = self.refractive_index(self.wavelengths[0], i)
            yield "%2i %1s %10.5g %10.4g %10.5g %17s %7.3f %7.3f %7.2f" % (
                    i, e.typeletter, e.distance, roc, rad*2, mat, n, nd, vd)
            
    # def aim_marginal(self, yo, yp, z, p, l=None, stop=None, **kwargs):
    #     assert p
    #     rim = stop == -1
    #     if not self.object.pupil.aim and not rim:
    #         return p
    #     if l is None:
    #         l = self.wavelengths[0]
    #     n = self.refractive_index(l, 0)
    #     stop = self.stop+1
    #     rate = 0.94

    #     @clru_cache(maxsize=1024)
    #     def dist(a):
    #         y, u = self.aim(yo, yp, z, a*p, filter=False)
    #         ys = [y]
    #         for yunit in self.propagate(y, u, n, l, stop=stop):
    #             ys.append(yunit[0])
    #         d = np.square(ys)[1:, 0, :2].sum(1)[-1]/np.square(rate*self.epd/2)-1
    #         return d
        
    #     try:
    #         a = self.solve_brentq(dist, **kwargs)
    #     except:
    #         a = np.nan
    #     assert a
    #     return a*p
            
    # def RayAiming_custom(self):
    #     shrinkrate = 0.94
    #     FirstZ = self[1].distance
    #     AptZ = self[self.stop].distance
    #     AptRadi = self.epd/2

    #     # Target Aperture points (center and upper/lower edges)
    #     AptYs = np.zeros(3)
    #     AptYs[1] = AptYs[1] + AptRadi*shrinkrate
    #     AptYs[2] = AptYs[2] - AptRadi*shrinkrate

    #     # L = AptZ - FirstZ
    #     # # InitialYs2 = AptYs - IncRay[1]*L
    #     # InitialYs = np.zeros(3)

    #     # trace object
    #     t = GeometricTrace_Mod(self)
    #     IncRay = np.array([0, np.sin(self.object.angle), np.cos(self.object.angle)])

    #     # compute initial guess on the first surface
    #     # trace at HFOV is 0.
    #     arguments = (t, AptYs[1], np.array([0., 0., 1.]))
    #     boundaries = [[-self.epd/2, self.epd/2]]
    #     opt = scipy.optimize.minimize(self.SearchRay, 0., method='SLSQP',
    #                                     args=arguments, bounds=boundaries)
    #     y_opt = opt.x[0]
    #     InitialYs = np.array([0., y_opt, -y_opt])
    #     # InitialYs = np.zeros(3)

    #     # Compute RayAiming to aperture targets
    #     IncPoints = np.zeros([3, 3])
    #     # iterate through aperture targets: center, upper and lower edges
    #     for i in range(3):
    #         arguments = (t, AptYs[i], IncRay)
    #         opt = scipy.optimize.minimize(self.SearchRay, InitialYs[i], method='L-BFGS-B',
    #                                        args=arguments, bounds=boundaries)
    #         AimedY = opt.x[0]
    #         # spherical sag
    #         AimedZ = System_Mod.SphericalSag_2D(self[1].curvature, AimedY)
    #         # add z offset
    #         AimedZ = AimedZ + self[1].distance
    #         IncPoints[i] = [0, AimedY, AimedZ] - IncRay*AimedZ/IncRay[2]

    #     IncRays = np.tile(IncRay, [3,1])
    #     return IncPoints, IncRays

    # def SearchRay(self, y, T, AptY, IncRay):
    #     y = y[0]    # !?

    #     # spherical sag
    #     FirstZ = System_Mod.SphericalSag_2D(self[1].curvature, y)
    #     # add z offset
    #     FirstZ = FirstZ + self[1].distance
    #     FirstPoint = [0, y, FirstZ]
        
    #     # trace ray
    #     IncPoint = FirstPoint - IncRay*FirstZ/IncRay[2]
    #     T.rays_given(IncPoint, IncRay, np.mean(self.wavelengths))
    #     T.propagate()
    #     # Y deviation
    #     dev = np.abs(T.y[self.stop][0][1]-AptY)
    #     if np.isinf(dev) or np.isnan(dev):
    #         dev = 9e4

    #     return dev
