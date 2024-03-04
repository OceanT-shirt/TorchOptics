import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
import yaml
import torchlens.ray_tracing_lite as rt
import torchlens.lens_modeling as lm

from analysis.w2rgb import wavelength_to_rgb


class OpticsSimulator(nn.Module):
    """
        Class to simulate optical aberrations on a given image
        The psfs, distortion shifts, and relative illumination factors
            need to be computed in children classes
            (e.g., with ray tracing or proxy model)
    """

    def __init__(self,
                 initial_lens_path,
                 stop_index: np.ndarray = np.array([1]),
                 sequence: np.ndarray = np.array(["AGA"]),
                 hfov: torch.Tensor = torch.tensor([0.0, 17.5, 25.0]),
                 epd: torch.Tensor = torch.tensor([0.7]),
                 curvature: torch.Tensor = torch.tensor([0.0, -0.242432341, -0.424975232], requires_grad=True),
                 thickness: torch.Tensor = torch.tensor([1.21071062, 0.25, 9.86362667], requires_grad=True),
                 n_refractive: torch.Tensor = torch.tensor([1.5224147149313454], requires_grad=True),
                 abbe_number: torch.Tensor = torch.tensor([59.450346241693694], requires_grad=True),
                 add_bfl=True,
                 scale_factor=1,
                 detach=False,
                 trainable_vars=None,
                 disable_glass_optimization=False,
                 n_sampled_fields=21,
                 sensor_diagonal=16.,
                 psf_shape=(65, 65),
                 psf_abs_pixel_size=4.0e-3,
                 psf_grid_shape=(9, 9),
                 simulated_res_factor=1,
                 distortion_by_warping=True,
                 apply_distortion=True,
                 apply_relative_illumination=True,
                 lazy_init=False,
                 device="cuda"
                 ):
        super(OpticsSimulator, self).__init__()

        # Running Device (CPU / CUDA)
        self.default_device = device

        if trainable_vars is None:
            self.trainable_vars = {'c': True, 't': True, 'g': True}
        elif isinstance(trainable_vars, bool):
            self.trainable_vars = {k: trainable_vars for k in ('c', 't', 'g')}
        else:
            self.trainable_vars = trainable_vars
        if disable_glass_optimization:
            self.trainable_vars['g'] = False
        self.detach = detach

        # Lens variables params
        if len(initial_lens_path) > 0:
            if isinstance(initial_lens_path, dict):
                self.initial_lens = initial_lens_path
            else:
                with open(initial_lens_path, 'r') as f:
                    # Load lens configuration and initial lens parameters
                    self.initial_lens = yaml.safe_load(f)
        else:
            self._stop_index = stop_index
            self._sequence = sequence
            self._hfov = hfov
            self._epd = epd
            self._curvature = curvature
            self._thickness = thickness
            self._n_refractive = n_refractive
            self._abbe_number = abbe_number

        self.add_bfl = add_bfl
        self.scale_factor = scale_factor
        self.sensor_diagonal = sensor_diagonal
        self.n_fields = n_sampled_fields

        # Optics model params
        self.simulated_res_factor = simulated_res_factor
        self.distortion_by_warping = distortion_by_warping
        self.apply_distortion = apply_distortion
        self.apply_relative_illumination = apply_relative_illumination
        # Metrics
        self.logged_metrics = {}

        self.lazy_init = lazy_init

        self.loss_dict = None

        self.structure = None
        self.hfov = None
        self.epd = None
        self.efl = None
        self.specs = None
        self.lensR = None
        self.c = None
        self.t = None
        self.g = None

    def initialize(self):
        # Lens structure
        self.structure = lm.Structure(
            stop_idx=self._stop_index,
            sequence=self._sequence,
            default_device=self.default_device
        )

        # Lens specifications
        self.hfov = torch.nn.Parameter(torch.deg2rad(self._hfov[-1].reshape([1, ]).clone().detach()),
                                       requires_grad=False)
        self.epd = torch.nn.Parameter(self._epd.clone().detach(), requires_grad=False)

        # Compute effective focal length required
        self.efl = self.sensor_diagonal / 2 / torch.tan(self.hfov)
        self.specs = lm.Specs(self.structure, self.epd, self.hfov)

        # Lens variables - with normalized forms
        self.lensR = lm.Lens(self.structure, *[key for key in [self._curvature, self._thickness, self._n_refractive,
                                                               self._abbe_number]])

class RaytracedOptics(OpticsSimulator):
    """
        Class to simulate optical aberrations through exact ray tracing of a compound lens
        For convenience, the class also supports the optimization of the lens
            and computation of losses that act exclusively on the lens
    """

    def __init__(self,
                 initial_lens_path,
                 quantized_continuous_glass_variables=True,
                 wavelengths=(459.0, 520.0, 640.0),
                 penalty_rate = 0.2,
                 n_pupil_rings=32,
                 n_ray_aiming_iter=1,
                 pupil_sampling='skew_uniform_half_jittered',
                 spot_size_weight=1,
                 ray_path_weight=100,
                 ray_path_lower_thresholds=(0.01, 1.0, 12.0),
                 ray_path_upper_thresholds=(None, 3.0, None),
                 ray_angle_weight=100,
                 ray_angle_threshold=60,
                 glass_weight=.01,
                 glass_catalog_path='./glass/selected_glass.csv',
                 loss_multiplier=1,
                 **kwargs
                 ):
        super(RaytracedOptics, self).__init__(initial_lens_path, **kwargs)

        # Lens variable params
        self.quantized_continuous_glass_variables = quantized_continuous_glass_variables

        # Ray tracing params
        self.additional_rt_params = {}
        self.n_pupil_rings = n_pupil_rings
        self.n_ray_aiming_iter = n_ray_aiming_iter
        self.pupil_sampling = pupil_sampling
        self.wavelengths = wavelengths
        self.penalty_rate = penalty_rate

        # Loss params
        self.ray_path_lower_thresholds = ray_path_lower_thresholds
        self.ray_path_upper_thresholds = ray_path_upper_thresholds
        self.ray_angle_threshold = ray_angle_threshold
        self.loss_weights = {
            'glass': glass_weight * loss_multiplier,
            'spot_size': spot_size_weight * loss_multiplier,
            'ray_path': ray_path_weight * loss_multiplier,
            'ray_angle': ray_angle_weight * loss_multiplier,
            'loss_unsup': 1
        }
        # Manage reference glasses
        ref_glasses = torch.tensor(np.loadtxt(glass_catalog_path, delimiter=',', dtype=np.float32))
        # self.catalog_g = tf.reshape(lm.g_from_n_v(*tf.unstack(ref_glasses, axis=1)), (-1, 2))
        self.catalog_g = torch.reshape(lm.g_from_n_v(*torch.unbind(ref_glasses, dim=1)), (-1, 2))

        self.initialize()

    def compute_loss_out(self, rt_outputs):
        """
            From the outputs of the ray-tracing and the lens parameters,
                compute the loss that operates on the lens
               Written by Sasaki-san
        """

        x, y, *_, ray_ok, ray_backward, stacks = rt_outputs
        rms = rt.compute_rms2d(x, y, ray_ok)

        # Compute unsupervised loss function outout
        # compute penalty Q
        numSequence = len(self._sequence[0])
        Q = (torch.stack(stacks['theta_norm'], dim=0).sum(dim=0) +
             torch.stack(stacks['theta_prime_norm'], dim=0).sum(dim=0) +
             torch.stack(stacks['z_RELU'], dim=0).sum(dim=0)) / numSequence
        # replace nan with zero
        Q = torch.where(torch.isnan(Q), torch.zeros_like(Q), Q)
        sumQ = torch.sum(Q)
        Lu = rms + self.penalty_rate*sumQ
        self.loss_dict = {'loss_unsup': Lu,  "rms": rms, "penalty": sumQ}

    # def get_losses(self):
    #     weighted_losses = {k: self.loss_dict[k] * v for k, v in self.loss_weights.items() if v is not None}
    #     return weighted_losses

    def do_ray_tracing(self, lens=None, should_log=True):
        """
            Do the raw ray tracing, whose intermediate results are used to compute
                the spot size, spot diagrams (for PSFs), and penalty terms
        """
        specs = self.specs
        lens = lens or self.lens

        # Log some metrics on the lens
        # if should_log:
        #     self.logged_metrics.update({
        #         'lens/defocus': lens.flat_t[-1] - lens.bfl[0],
        #         'lens/back_focal_length': lens.bfl[0],
        #         'lens/percentage_distortion': 100 * rt.compute_distortion(specs, lens, [1.])[0, 0],
        #         'lens/relative_illumination': rt.compute_relative_illumination(
        #         specs, lens, [0., 1.], None, n_ray_aiming_iter=1)[0, -1, 0]
        #     })

        if self.n_fields==1:
            fields = [1.0]
        else:
            fields = list(np.linspace(0, 1, self.n_fields))

        # wavelengths_flat = [item for k in ('R', 'G', 'B') for item in self.wavelengths[k]]

        # rt_params = dict(
        #     n_rays=(self.n_pupil_rings, 1), rel_fields=fields, vig_fn=None,
        #     n_ray_aiming_iter=self.n_ray_aiming_iter, wavelengths=wavelengths_flat, mode=self.pupil_sampling)
        rt_params = dict(
            n_rays=(self.n_pupil_rings, self.n_pupil_rings), rel_fields=fields, vig_fn=None,
            n_ray_aiming_iter=self.n_ray_aiming_iter, wavelengths=self.wavelengths, mode=self.pupil_sampling,
            default_device=self.default_device)
        rt_params.update(**self.additional_rt_params)
        ray_tracer = rt.RayTracer(**rt_params)
        rt_outputs = ray_tracer.trace_rays(specs, lens, aggregate=True)
        x, y, *_, ray_ok, ray_backward, stacks = rt_outputs

        self.compute_loss_out(rt_outputs)

        # Log some ray tracing metrics
        # if should_log:
        #     self.logged_metrics.update({
        #         'ray_tracing/ray_failures': torch.sum(~ray_ok.to(torch.float32)),
        #         'ray_tracing/backward_rays': torch.sum(ray_backward.to(torch.float32)),
        #         'ray_tracing/max_ray_aiming_error': torch.max(torch.abs(
        #                     rt.compute_ray_aiming_error(specs, lens, fields, None, 1, 'real'))),
        #     })

        return x, y, ray_ok
    
    def do_ray_aiming(self, xp_rel, yp_rel, lens=None):
        # get lens and configurations
        specs = self.specs
        lens = lens or self.lens

        # field rate
        if self.n_fields==1:
            fields = [1.0]
        else:
            fields = list(np.linspace(0, 1, self.n_fields))

        if self._stop_index[0] >= 2:
            rt_params = dict(
                n_rays=(self.n_pupil_rings, self.n_pupil_rings), rel_fields=fields, vig_fn=None,
                n_ray_aiming_iter=self.n_ray_aiming_iter, wavelengths=self.wavelengths, mode=self.pupil_sampling,
                default_device=self.default_device)
            rt_params.update(**self.additional_rt_params)
            ray_tracer = rt.RayTracer(**rt_params)

            xp, delta_xp, yp_scale, yp_offset = ray_tracer.ray_aiming(specs, lens.detach(), True, mode='yscale')
            xp_rel = xp_rel * (xp + delta_xp)/xp
            yp_rel = yp_rel * yp_scale[0][0][0] + yp_offset[0][0][0]
        
        # Normalise in scale of epd
        xp = rt.scale_to_epd(xp_rel, specs.epd)
        yp = rt.scale_to_epd(yp_rel, specs.epd)
        return xp, yp

    def ShowTraceResult(self, x, y, ray_ok, loss_unsup):
        h = plt.figure()
        h.suptitle('Unsupervised Loss Function Output:\n' + str(loss_unsup), fontsize=12)
        ax = h.add_subplot()

        xd = x.detach().numpy()
        yd = y.detach().numpy()

        for f in range(0, xd.shape[1]):
            for w in range(0, xd.shape[3]):
                wave = self.wavelengths[w]
                RGB = wavelength_to_rgb(wave)
                PltColor = (RGB[0] / 255, RGB[1] / 255, RGB[2] / 255)

                for p in range(0, xd.shape[2]):
                    if ray_ok[0, f, p, w]:
                        ax.plot(xd[0, f, p, w], yd[0, f, p, w], '.', color=PltColor, markersize=4)

        ax.axis('equal')
        plt.show()

def compute_ray_path_penalty(lens, z_stack, min_thickness, max_thickness):
    """
        z_stack: z-coordinates of the rays across all surfaces [n_surface, n_lens, n_field, n_pupil, n_wavelength]
        min_thickness/max_thickness: tuple (float/None, float/None, float/None)
    """
    min_thickness = [value if value is not None else -np.inf for value in min_thickness]
    max_thickness = [value if value is not None else np.inf for value in max_thickness]
    min_t_air, min_t_glass, min_t_image = min_thickness
    max_t_air, max_t_glass, max_t_image = max_thickness
    ref_vertex_z = torch.cumsum(torch.cat((torch.reshape(lens.t, (-1,)), torch.zeros(1).to(lens.t.device))), dim=0)
    abs_z = z_stack + torch.reshape(ref_vertex_z, (-1, 1, 1, 1, 1))
    delta_z = abs_z[1:] - abs_z[:-1]
    # Combine the thresholds for air and glass
    min_t_map = torch.where(lens.structure.mask_G_torch, min_t_glass, min_t_air)
    max_t_map = torch.where(lens.structure.mask_G_torch, max_t_glass, max_t_air)
    # Do the same for the surface before the image plane
    min_t_map[:, lens.structure.mask_torch.sum(dim=1) - 1] = min_t_image
    max_t_map[:, lens.structure.mask_torch.sum(dim=1) - 1] = max_t_image
    thickness_penalty = torch.maximum(min_t_map.reshape(-1, 1, 1, 1, 1) - delta_z,
                                      torch.zeros_like(min_t_map.reshape(-1, 1, 1, 1, 1))) + \
                        torch.maximum(delta_z - max_t_map.reshape(-1, 1, 1, 1, 1),
                                      torch.zeros_like(max_t_map.reshape(-1, 1, 1, 1, 1)))
    thickness_penalty = torch.sum(torch.mean(thickness_penalty, dim=(1, 2, 3, 4)))
    return thickness_penalty

def compute_ray_angle_penalty(cos_squared, angle_threshold):
    threshold = torch.cos(torch.deg2rad(torch.tensor(angle_threshold))) ** 2
    return torch.sum(torch.mean(torch.maximum(threshold - cos_squared, torch.zeros_like(threshold)), dim=(1, 2, 3, 4)))

def compute_glass_penalty(structure, g, catalog_g):
    if catalog_g is not None:
        dist = torch.norm(g[:, None, :] - catalog_g[None, :, :], dim=-1)
        min_dist = torch.min(dist, dim=1).values
        agg = rt.mask_replace(structure.mask_G, torch.zeros_like(structure.mask_G_torch, dtype=g.dtype), min_dist)
        glass_penalty = torch.sum(agg ** 2)
        return glass_penalty
    else:
        return torch.tensor(0., dtype=torch.float32)