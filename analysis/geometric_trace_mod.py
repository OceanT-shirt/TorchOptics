import numpy as np
import torch

# from rayopt.utils import pupil_distribution
from rayopt.geometric_trace import GeometricTrace

from torchlens.simulate_spot_lite import SimulateSpotLite
from raw_data_processing.zmx2yml import ComputeEFL

class GeometricTrace_Mod(GeometricTrace):
    
    def allocate(self, nrays):
        super().allocate(nrays)
        self.fail = np.zeros((self.length, nrays), dtype=bool)

    def rays(self, yo, yp, wavelength, stop=None, filter=None,
             clip=False, weight=None, ref=0):
        rate = 0.94
        num = yp.shape[0]

        # Compute light source position
        # creat y data points in diameter of 2.0
        # x_rel = torch.tensor(yp[:,0])
        # y_rel = torch.tensor(yp[:,1])
        x_rel = torch.tensor([0., 0., 0.])
        y_rel = torch.tensor([0., 1., -1.])
        # create sepecification
        MyDict = self.System2Dict()
        # do ray aiming of y coords on "first surface"
        model = SimulateSpotLite(
                LensDict=MyDict,
                NField=1, NPuple=1, 
                Wave=[wavelength*1e+9], 
                PupleSampling='circular')
        x, y = model.do_ray_aiming(x_rel, y_rel, lens=model.lensR)
        x = x.detach().numpy()
        y = y.detach().numpy()
        # Denormalise from EFL scale
        x *= MyDict['efl']
        y *= MyDict['efl']

        # Incident rays vector
        field_Y = self.system.object.angle * yo[1]
        u = np.array([0, np.sin(field_Y), np.cos(field_Y)])
        u = np.tile(u, [num,1])

        # xyz coordinates on light source plan 
        # spherical sag
        r = np.sqrt(x**2+y**2)
        distZ = GeometricTrace_Mod.SphericalSag_2D(self.system[1].curvature, r)
        # add z offset
        distZ = distZ + self.system[1].distance
        xyz = np.hstack([(x-u[0,0]*distZ/u[0,2]).reshape([num,1]), (y-u[0,1]*distZ/u[0,2]).reshape([num,1]), np.zeros([num,1])])

        # trace rays
        self.rays_given(xyz, u, wavelength, weight, ref)
        self.propagate(clip=clip)

    def rays_clipping(self, yo, wavelength=None, axis=1):
        yp = np.zeros((3, 2))
        self.rays(yo, yp, wavelength, stop=-1, filter=False)

    def System2Dict(self):
        # Creat lens spec. in efl-normalised form
        # Without d, efl, wave

        c_list = []
        t_list = []
        nd_list = []
        v_list = []
        Seq = ''
        num_surf = len(self.system)-2
        for i in range(num_surf):
            curv = getattr(self.system[i+1], "curvature", 0)
            thick = self.system[i+2].distance
            c_list.append(curv)
            t_list.append(thick)
            mat = getattr(self.system[i+1], "material", "")
            if mat.name == 'air':
                Seq += 'A'
            else:
                Seq += 'G'
                nd = float(getattr(mat, "nd", np.nan))
                v = float(getattr(mat, "vd", np.nan))
                nd_list.append(nd)
                v_list.append(v)

        # Normalise with EFL
        EFL = ComputeEFL(Seq, c_list, t_list, nd_list)
        c_array = np.array(c_list)
        t_array = np.array(t_list)
        c_list = (c_array*EFL).tolist()
        t_list = (t_array/EFL).tolist()
        
        MyDict = dict()
        MyDict['c'] = c_list
        MyDict['t'] = t_list
        MyDict['nd'] = nd_list
        MyDict['v'] = v_list
        MyDict['epd'] = [self.system.epd/EFL]
        MyDict['hfov'] = self.system.fieldY
        MyDict['sequence'] = [Seq]
        MyDict['stop_idx'] = [self.system.stop]
        MyDict['efl'] = [EFL]
        return MyDict
    
    @staticmethod
    def SphericalSag_2D(curv, x):
        sag = curv*x**2/(1+np.sqrt(1-curv**2*x**2))
        return sag

    # def rays_1D(self, yo, wavelength=None, axis=1,
    #             clip=False, weight=None, ref=0, nrays=16):
    #     # aperture position and size
    #     z = self.system.track[self.system.stop]
    #     p = np.array([[-1/2, -1/2], [1/2, 1/2]]) * self.system.epd
    #     px = np.linspace(-self.system.epd/2, self.system.epd/2, nrays)
    #     py = np.linspace(-self.system.epd/2, self.system.epd/2, nrays)
    #     p_lin = np.hstack([px.reshape([nrays,1]), py.reshape([nrays,1])])
    #     # field angle
    #     yp = np.zeros((nrays+1, 2))
    #     yp[1:, axis] = p_lin[:, axis]/np.fabs(p_lin).max()
    #     # ray aiming
    #     y, u = self.system.aim(yo, yp, z, p, filter=filter)
    #     self.rays_given(y, u, wavelength, weight, ref)
    #     self.propagate(clip=clip)

    # # trace for ray diagram
    # def rays_raydiagram(self, yo, wavelength=None, axis=1,
    #             clip=False, weight=None, ref=0):
    #     # aperture position and size
    #     z = self.system.track[self.system.stop]
    #     p = np.array([[-1/2, -1/2], [1/2, 1/2]]) * self.system.epd
    #     # field angle
    #     yp = np.zeros((3, 2))
    #     yp[1:, axis] = p[:, axis]/np.fabs(p).max()
    #     # ray aiming
    #     y, u = self.system.aim(yo, yp, z, p, filter=filter)
    #     self.rays_given(y, u, wavelength, weight, ref)
    #     self.propagate(clip=clip)

    # def rays(self, yo, yp, wavelength, filter=None,
    #          clip=False, weight=None, ref=0):
    #     if filter is None:
    #         filter = not clip
    #     # get aperture radius and position
    #     z = self.system.track[self.system.stop]
    #     p = np.array([[-1/2, -1/2], [1/2, 1/2]]) * self.system.epd
    #     # ray aiming
    #     y, u = self.system.aim(yo, yp, z, p, filter=filter)
    #     self.rays_given(y, u, wavelength, weight, ref)
    #     self.propagate(clip=clip)

    # # trace for spot diagram
    # def rays_point(self, yo, wavelength=None, nrays=11,
    #                distribution="meridional", filter=None, stop=None,
    #                clip=False):
    #     ref, yp, weight = pupil_distribution(distribution, nrays)
    #     self.rays(yo, yp, wavelength, filter=filter,
    #               clip=clip, weight=weight, ref=ref)

    # # trace for aberration diagram
    # def rays_line(self, yo, wavelength=None, nrays=21, eps=1e-2):
    #     yi = np.linspace(0, 1, nrays)[:, None]*np.atleast_2d(yo)
    #     y = np.empty((3, nrays, 3))
    #     u = np.empty_like(y)
    #     e = np.zeros((3, 2))  # chief, meridional, sagittal
    #     e[(1, 2), (1, 0)] = eps

    #      # aperture position and size
    #     z = self.system.track[self.system.stop]
    #     p = np.array([[-1/2, -1/2], [1/2, 1/2]]) * self.system.epd
    #     # iterate through...
    #     for i in range(yi.shape[0]):
    #         z = self.system.aim_chief(yi[i], z, np.fabs(p).max(),
    #                                   l=wavelength)
    #         y[:, i], u[:, i] = self.system.aim(yi[i], e, z, p)
    #     self.rays_given(y.reshape(-1, 3), u.reshape(-1, 3), wavelength)
    #     self.propagate()

    def propagate(self, start=1, stop=None, clip=False):
        super(GeometricTrace, self).propagate()
        init = start - 1
        y, u, n, l = self.y[init], self.u[init], self.n[init], self.l
        y, u = self.system[init].from_normal(y, u)
        for j, yunit in enumerate(self.system.propagate(
                y, u, n, l, start, stop, clip)):
            j += start
            
            self.y[j], self.u[j], self.n[j], self.i[j], self.t[j] = yunit
            # add ray failure state
            dev_height = self.system[j].radius-abs(yunit[0][:,1])
            S = np.sign(dev_height)
            fail = np.zeros(S.shape, dtype=bool)
            fail[np.where(S>=0)] = False
            fail[np.where(S==-1)] = True
            fail[np.isnan(S)] = True
            self.fail[j] = fail

    def plot(self, ax, axis=1, **kwargs):
        y = np.array([el.from_normal(yi) + oi for el, yi, oi
                      in zip(self.system, self.y, self.origins)])
        ax.plot(y[:, :, 2], y[:, :, axis], **kwargs)