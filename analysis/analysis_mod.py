import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
from rayopt.utils import tanarcsin

from rayopt.analysis import Analysis, CenteredFormatter
from analysis.geometric_trace_mod import GeometricTrace_Mod
from analysis.w2rgb import wavelength_to_rgb

class Analysis_Mod(Analysis):
    figwidth = 5.
    figheight = 5.
    titlesize = 8
    ticksize = 6
    run = True
    update = False
    print = False
    trace_gaussian = False
    print_gaussian = False
    print_system = False
    print_paraxial = False
    resize_full = False
    refocus_full = False
    print_full = False
    plot_paraxial = False
    plot_gaussian = False
    plot_full = False
    plot_rays = 3
    plot_transverse = False
    plot_spots = False
    defocus = 5
    plot_opds = False
    plot_longitudinal = False

    def __init__(self, system, show_aberration=False, **kwargs):
        self.system = system
        self.text = []
        self.figures = []
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError("no such option %s" % k)
            setattr(self, k, v)
        if self.run:
            self.run()
        if self.print:
            for t in self.text:
                print(t)
        if show_aberration:
            self.plot_transverse = True
            self.plot_longitudinal = True

    def run(self):
        # if self.update:
        #     self.system.update()
        # if self.resize_full:
        #     t = GeometricTrace(self.system)
        #     t.rays_paraxial()
        #     t.resize()
        #     self.system.resize_convex()
        # if self.refocus_full:
        #     t = GeometricTrace(self.system)
        #     t.rays_point((0, 0.), nrays=13, distribution="radau",
        #                  clip=False, filter=False)
        #     t.refocus()
        # if self.print_system:
        #     self.text.append(str(self.system))
        if self.print_paraxial:
            self.text.append(str(self.system.paraxial))
        # g = GaussianTrace(self.system)
        # if self.print_gaussian:
        #     self.text.append(str(g))
        # t = GeometricTrace(self.system)
        # t.rays_paraxial()
        # if self.print_full:
        #     self.text.append(str(t))
        # figure for ray diagram
        fig, ax = plt.subplots(figsize=(self.figwidth, self.figheight),
                               num='Ray Diagram')
        self.figures.append(fig)
        # draw optical elements & optical axis
        self.system.plot(ax, npoints=128, adjust=False)
        # if self.plot_paraxial:
        #     self.system.paraxial.plot(ax)
        # if self.plot_gaussian:
        #     g.plot(ax)
        # if self.plot_full:
        #     t.plot(ax)
        # draw rays, iterate through field angles
        for h in self.system.fields:
            for w in self.system.wavelengths:
                t = GeometricTrace_Mod(self.system)
                t.rays_clipping((0, h), w)
                RGB = wavelength_to_rgb(w*1e+9)
                color = (RGB[0]/255, RGB[1]/255, RGB[2]/255)
                t.plot(ax, 1, color=color)

        if self.plot_transverse:
            # Transverse aberration diagram
            figheight = self.figwidth*len(self.system.fields)/5
            fig = plt.figure(figsize=(self.figwidth, figheight),
                              num='Ray Fan Plot')
            self.figures.append(fig)
            self.transverse(fig, self.system.fields)

        if self.plot_longitudinal:
            # Longitudinal aberration diagram
            num_diag = 3
            fig, ax = plt.subplots(
                1, num_diag, figsize=(self.figwidth, self.figwidth/num_diag),
                num='Longitudinal Aberration Plot')
            self.figures.append(fig)
            self.longitudinal(ax, max(self.system.fields), nrays=128)

        if self.plot_spots:
            # Spot Diagram
            figheight = self.figwidth*len(self.system.fields)/self.defocus
            fig, ax = plt.subplots(len(self.system.fields), self.defocus,
                                   figsize=(self.figwidth, figheight),
                                   sharex=True, sharey=True, squeeze=False)
            fig.suptitle('Spot Diagram', fontsize=16)
            self.figures.append(fig)
            self.spots(ax[::-1], self.system.fields)

        if self.plot_opds:
            figheight = self.figwidth*len(self.system.fields)/4
            fig, ax = plt.subplots(len(self.system.fields), 4,
                                   figsize=(self.figwidth, figheight),
                                   squeeze=False)
            # , sharex=True, sharey=True)
            self.figures.append(fig)
            self.opds(ax[::-1], self.system.fields)

        return self.text, self.figures

    # Draw spot diagram
    def spots(self, ax, heights=[1., .707, 0.],
              wavelengths=None, nrays=128, colors="grbcmyk"):
        paraxial = self.system.paraxial
        if wavelengths is None:
            wavelengths = self.system.wavelengths
        nd = ax.shape[1]
        for axi in ax.flat:
            self.pre_setup_xyplot(axi)
        z = paraxial.rayleigh_range[1]
        z = (np.arange(nd) - nd//2) * z
        for hi, axi in zip(heights, ax[:, 0]):
            axi.text(-.1, .5, "OY=%s" % hi, rotation="vertical",
                     transform=axi.transAxes, verticalalignment="center")
        for zi, axi in zip(z, ax[-1, :]):
            axi.text(.5, -.1, "DZ=%.1g" % zi,
                     transform=axi.transAxes, horizontalalignment="center")
        # iterate through field angles
        for hi, axi in zip(heights, ax):
            # iterate through wavelengths
            for wi, ci in zip(wavelengths, colors):
                r = paraxial.airy_radius[1]/paraxial.wavelength*wi
                t = GeometricTrace_Mod(self.system)
                t.rays_point((0, hi), wi, nrays=nrays,
                             distribution="hexapolar", clip=True)
                # plot transverse image plane hit pattern (ray spot)
                y = t.y[-1, :, :2] - t.y[-1, t.ref, :2]
                u = tanarcsin(t.i[-1])
                for axij, zi in zip(axi, z):
                    axij.add_patch(mpl.patches.Circle(
                        (0, 0), r, edgecolor=ci, facecolor="none"))
                    yi = y + zi*u
                    axij.plot(yi[:, 0], yi[:, 1], ".%s" % ci,
                              markersize=1, markeredgewidth=1, label="%s" % wi)
        for axi in ax:
            for axii in axi:
                self.post_setup_axes(axii)
    

    # draw aberration diagram (distortion/astigmatism/spherical/colors)
    def longitudinal(self, ax, height=1.,
                     wavelengths=None, nrays=21, colors="grbcmyk"):
        # lateral color: image relative to image at wl[0]
        # focus shift paraxial focus vs wl
        # longitudinal spherical: marginal focus vs height (vs wl)
        if wavelengths is None:
            wavelengths = self.system.wavelengths
        # axd, axc, axf, axs, axa = ax
        axd, axf, axa = ax
        for axi, xl, yl, tl in [
                (axd, "EY", "REY", "DIST."),
                # (axc, "EY", "DEY", "TCOLOR"),
                (axf, "EY", "DEZ", "ASTIG."),
                # (axs, "PY", "DEZ", "SPHERE"),
                (axa, "L", "DEZ", "LCOLOR"),
                ]:
            self.setup_axes(axi, xl, yl, tl, yzero=False, xzero=False)
        # iterate through wavelengths
        for i, (wi, ci) in enumerate(zip(wavelengths, colors)):
            t = GeometricTrace_Mod(self.system)
            t.rays_line((0, height), wi, nrays=nrays)
            # field angles
            fa = np.arctan(t.u[0,:nrays,1] / t.u[0,:nrays,2])
            a, b, c = np.split(t.y[-1].T, (nrays, 2*nrays), axis=1)
            p, q, r = np.split(tanarcsin(t.i[-1]).T, (nrays, 2*nrays), axis=1)
            
            # if i == 0:
            # Distortion
            # compute object point
            L = 9e4
            sourceY = t.u[0,:nrays,1] / t.u[0,:nrays,2] * L
            magnif = a[1] / sourceY
            # distance from optical axis
            tracedY = np.abs(a[1])
            targetY = np.abs(sourceY * np.mean(magnif[1:]))
            distor = (tracedY - targetY)/targetY * 100
            axd.plot(distor, fa, ci+"-", label="%s" % wi)
            # axd.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.6f'))
            # a0 = a
            # else:
            #     # Axial color
            #     axc.plot(a[1], a[1] - a0[1], ci+"-", label="%s" % wi)
                
            # Astigmatism (tangential & sagittal)
            # difference of focal point along field angle
            xt = -(b[1]-a[1])/(q[1]-p[1])
            axf.plot(xt, fa, ci+"-", label="EZt %s" % wi)
            xs = -(c[0]-a[0])/(r[0]-p[0])
            axf.plot(xs, fa, ci+"--", label="EZs %s" % wi)
            # # Spherical
            # t = GeometricTrace_Mod(self.system)
            # t.rays_point((0, 0.), wi, nrays=nrays,
            #              distribution="half-meridional", clip=True)
            # p = self.system.track[self.system.stop]
            # py = t.y[0, :, 1] + p*tanarcsin(t.u[0])[:, 1]
            # u = tanarcsin(t.i[-1])[:, 1]
            # u[t.ref] = np.nan
            # z = -t.y[-1, :, 1]/u
            # axs.plot(py, z, ci+"-", label="%s" % wi)

        # Lateral color
        wl, wu = min(wavelengths), max(wavelengths)
        ww = np.linspace(wl - (wu - wl)/4, wu + (wu - wl)/4, nrays)
        zc = []
        # get aperture radius and position
        pd = self.system.track[self.system.stop]
        ph = np.array([[-1/2, -1/2], [1/2, 1/2]]) * self.system.epd
        t = GeometricTrace_Mod(self.system)
        for wwi in np.r_[wavelengths[0], ww]:
            y, u = self.system.aim((0, 0), (0, 1e-3), pd, ph)
            t.rays_given(y, u, wwi)
            t.propagate(clip=False)
            zc.append(-t.y[-1, 0, 1]/tanarcsin(t.i[-1, 0])[1])
        zc = np.array(zc[1:]) - zc[0]
        axa.plot(ww, fa, "-")
        for axi in ax:
            self.post_setup_axes(axi)


    def transverse(self, fig, heights=[0., .707, 1.],
                   wavelengths=None, nrays_line=128,
                   colors="grbcmyk"):
        if wavelengths is None:
            wavelengths = self.system.wavelengths
        ax = self.pre_setup_fanplot(fig, len(heights))
        p = self.system.track[self.system.stop]

        # iterate through field angles
        for hi, axi in zip(heights, ax):
            axm, axsm, axss = axi
            axm.text(-.1, .5, "OY=%s" % hi, rotation="vertical",
                     transform=axm.transAxes,
                     verticalalignment="center",
                     fontsize=self.ticksize)
            for axi, xl, yl, tl in [
                (axm, "", "", "Tangential"),
                (axsm, "", "", "Tangential(?)"),
                (axss, "", "", "Sagittal")
                ]:
                self.setup_axes(axi, xl, yl, tl, yzero=False, xzero=False)
            # iterate through wavelengths
            for wi, ci in zip(wavelengths, colors):
                t = GeometricTrace_Mod(self.system)
                t.rays_point((0, hi), wi, nrays=nrays_line,
                             distribution="tee", clip=True)
                # plot transverse image plane versus entrance pupil
                # coordinates
                y = t.y[-1, :, :2] - t.y[-1, t.ref, :2]
                py = t.y[0, :, :2] + p*tanarcsin(t.u[0])
                py -= py[t.ref]
                axm.plot(py[:t.ref, 1], y[:t.ref, 1], "-%s" % ci,
                         label="%s" % wi)
                axsm.plot(py[t.ref:, 0], y[t.ref:, 1], "-%s" % ci,
                          label="%s" % wi)
                axss.plot(py[t.ref:, 0], y[t.ref:, 0], "-%s" % ci,
                          label="%s" % wi)
        for axi in ax:
            for axii in axi:
                self.post_setup_axes(axii)

    @classmethod
    def pre_setup_fanplot(cls, fig, n):
        gs = gridspec.GridSpec(n, 4)
        axpx, axe, axpy = None, None, None
        ax = []
        for i in range(n):
            axm = fig.add_subplot(gs.new_subplotspec((i, 0), 1, 2),
                                  sharex=axpy, sharey=axe)
            axpy = axpy or axm
            axe = axe or axm
            axsm = fig.add_subplot(gs.new_subplotspec((i, 2), 1, 1),
                                   sharex=axpx, sharey=axe)
            axpx = axpx or axsm
            axss = fig.add_subplot(gs.new_subplotspec((i, 3), 1, 1),
                                   sharex=axpx, sharey=axe)
            ax.append((axm, axsm, axss))
            for axi, xl, yl in [
                    (axm, "PY", "EY"),
                    (axsm, "PX", "EY"),
                    (axss, "PX", "EX"),
                    ]:
                cls.setup_axes(axi, xl, yl)
        return ax[::-1]

    @staticmethod
    def setup_axes(ax, xlabel=None, ylabel=None, title=None,
                   xzero=True, yzero=True):

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if yzero:
            ax.spines["left"].set_position("zero")
            ax.yaxis.set_major_formatter(CenteredFormatter())
        if xzero:
            ax.spines["bottom"].set_position("zero")
            ax.xaxis.set_major_formatter(CenteredFormatter())
        ax.tick_params(bottom=True, top=False, left=True, right=False,
                       labeltop=False, labelright=False,
                       labelleft=True, labelbottom=True,
                       direction="out", axis="both",
                       labelsize=Analysis_Mod.ticksize)
        # ax.xaxis.set_smart_bounds(True)
        # ax.yaxis.set_smart_bounds(True)
        ax.locator_params(tight=True, nbins=5)
        kw = dict(rotation="horizontal")
        if xlabel:
            ax.set_xlabel(xlabel, horizontalalignment="right",
                          verticalalignment="bottom", **kw,
                          fontsize=Analysis_Mod.ticksize)
        if ylabel:
            ax.set_ylabel(ylabel, horizontalalignment="left",
                          verticalalignment="top", **kw,
                          fontsize=Analysis_Mod.ticksize)
        if title:
            ax.set_title(title, fontsize=Analysis_Mod.titlesize)

    