"""
    Classes that represent batched specifications, lenses and lens structures

    The lens class is used to batch lenses regardless of their structure and number of variables
    This allows the ray-tracing operations to be batched for better use of computational resources

    Underlying tensors have 2D shape (batch x max length)
        curvatures are padded with 0's
        thicknesses are padded with 0's
        refractive indices are padded with 1's

    1D compact forms can be recovered from the *_flat() methods
    2D forms can be updated from the 1D forms
"""
from dataclasses import dataclass
import numpy as np
import torchlens.ray_tracing_lite as rt
import torch


def mask_replace(mask: np.ndarray, src: torch.Tensor, dst: torch.Tensor):
    assert src.shape == mask.shape
    assert src.dtype == dst.dtype
    assert src.device == dst.device
    assert len(dst.shape) == 1
    return src.masked_scatter_(torch.from_numpy(mask).to(dst.device), dst)


def g_from_n_v(n: torch.Tensor, v: torch.Tensor):
    assert len(n.shape) == len(v.shape) == 1
    assert n.device == v.device
    assert n.dtype == v.dtype
    w = n.new_tensor([[-7.497527849096219, -7.49752916467739], [0.07842101471405442, -0.07842100095362642]])
    mean = n.new_tensor([[1.6426209211349487, 48.8505973815918]])

    g = torch.matmul((torch.stack((n, v), dim=-1) - mean), w)

    return g


def n_v_from_g(g: torch.Tensor):
    assert len(g.shape) == 2 and g.shape[1] == 2
    w = g.new_tensor([[-0.06668863644654068, 6.3758429552417315], [-0.0666886481483064, -6.375841836481304]])
    mean = g.new_tensor([[1.6426209211349487, 48.8505973815918]])

    return torch.unbind(torch.matmul(g, w) + mean, dim=1)


def find_valid_curvatures(sequence):
    # Conditions for exclusion: current and previous elements are air, or last curvature of the system
    previous_element = np.concatenate((np.zeros_like(sequence.mask_G[:, 0:1]), sequence.mask_G[:, :-1]), axis=1)
    valid_curvature_mask = sequence.mask_G | previous_element & sequence.mask_except_last & sequence.mask
    return valid_curvature_mask


# def get_normalized_lens_variables(lens, trainable_vars, add_bfl=False, scale_factor=1):
#     """
#         Initialize TF variables from a lens object
#         The choice of "scale_factor" only has an effect during optimization,
#             by changing the relative scale of the variables w.r.t. the gradients
#         With the Adam optimizer, reducing the scale_factor has an effect similar to augmenting the learning rate
#     """
#
#     # First scale the variables to get EFL == 1
#     current_efl = lens.efl
#     if np.isfinite(current_efl.numpy().item()):
#         lens = lens.scale(1 / current_efl)
#     else:
#         # If using a random starting point with bad behaviour, compute the last curvature so that EFL=1
#         lens.flat_c = rt.compute_last_curvature(lens.structure, lens.flat_c_but_last, lens.flat_t, lens.flat_nd)
#
#     # We first define the glass materials by the refractive indices 'nd' and Abbe numbers 'v'
#     # We go to the normalized form 'g' which is the one that we will optimize
#     # Then we go back to the first form 'nd' and 'v' for ray tracing
#     g = torch.nn.Parameter(g_from_n_v(lens.flat_nd, lens.flat_v) * scale_factor)
#
#     t_non_flat = lens.t
#     if add_bfl:
#         # Find last thickness, which corresponds to the defocus
#         last_t_position = lens.structure.mask.sum(axis=1) - 1
#         last_t_indices = torch.stack((torch.arange(len(lens)), torch.from_numpy(last_t_position)), dim=1)
#         last_t = torch.gather(lens.t, 1, last_t_indices)
#
#         # Remove the BFL
#         updated_last_t = last_t - lens.bfl
#
#         # Update
#         t_non_flat.scatter_(1, last_t_indices, updated_last_t)
#     t = torch.nn.Parameter(t_non_flat[lens.structure.mask_torch] * scale_factor)
#
#     # Curvatures are optimized as is
#     # We exclude the last curvature which is computed on the fly
#     # We also exclude the curvatures of the surfaces surrounded by air (usually the aperture stop)
#     valid_curvatures = find_valid_curvatures(lens.structure)
#
#     c = torch.nn.Parameter(lens.c[torch.from_numpy(valid_curvatures)] * scale_factor)
#
#     return c, t, g


def map_glass_to_closest(g, catalog_g):
    dist = torch.norm(g[:, None, :] - catalog_g[None, :, :], dim=-1)
    min_dist_idx = torch.argmin(dist, dim=1)
    return torch.gather(catalog_g, 0, min_dist_idx), catalog_g


# def get_lens_from_normalized(structure, c, t, g, catalog_g, add_bfl=False, scale_factor=1, qc_variables=True):
#     """
#     Calculate the lens variables from the normalized variables
#     *Including calculating the last curvature*
#     """
#     # Undo the scaling operation
#     c = c / scale_factor
#     t = t / scale_factor
#     g = g / scale_factor
#
#     # If quantized continuous glass variables, map the glass variables to the closest catalog glass
#     if qc_variables:
#         g, _ = map_glass_to_closest(g, catalog_g)
#         """
#         PyTorchでは、torch.autograd.grad_modeを使用してTensorのグラデーションパスを変更することはできません。
#         そのため、tf.grad_pass_throughと同様の動作を再現することは困難です。
#         一つのアプローチとしては、qc_variablesがTrueの場合にのみ、map_glass_to_closest関数を呼び出して変数gを更新する方法があります。
#         """
#
#     # Retrieve the lens
#     nd, v = n_v_from_g(g)
#     # Fill the curvature array
#     c2d = torch.zeros_like(structure.mask, dtype=c.dtype)
#     c2d = mask_replace(find_valid_curvatures(structure), c2d, c)
#     flat_c = c2d[structure.mask_except_last]
#     # Compute the last curvature with an algebraic solve to enforce EFL = 1
#     c = rt.compute_last_curvature(structure, flat_c, t, nd)
#     lens = Lens(structure, c, t, nd, v)
#
#     if add_bfl:
#         # Find last thickness, which corresponds to the defocus
#         last_t_position = lens.structure.mask.sum(axis=1) - 1
#         last_t_indices = torch.stack((torch.arange(len(lens)), last_t_position), dim=1)
#         last_t_indices_flat = last_t_position + np.arange(structure.mask.shape[0]) * structure.mask.shape[1]
#         last_t = t[last_t_indices_flat.tolist()]
#
#         # Compute the new value by adding the BFL to the defocus
#         updated_t = lens.bfl + last_t
#
#         # Update
#         lens.t.scatter_(1, last_t_indices, updated_t.unsqueeze(1))
#     return lens


class Structure:

    def __init__(self, stop_idx, mask: np.ndarray = None, mask_G: np.ndarray = None, sequence=None,
                 default_device='cuda'):
        self.stop_idx = stop_idx
        assert len(self.stop_idx.shape) == 1

        if sequence is not None:
            assert mask is None
            assert mask_G is None
            assert isinstance(sequence, np.ndarray)

            n = sequence.shape[0]
            sequence = sequence.view('U1').reshape(n, -1)

            self.mask = np.array(sequence != '')
            self.mask_G = np.array(sequence == 'G')

        else:
            assert mask is not None
            assert mask_G is not None
            self.mask = mask
            self.mask_G = mask_G

        self.default_device = default_device
        self.mask_torch = torch.from_numpy(self.mask).to(self.default_device)
        self.mask_G_torch = torch.from_numpy(self.mask_G).to(self.default_device)

        assert len(self.mask.shape) == 2
        assert len(self.mask_G.shape) == 2

    def __len__(self):
        return self.mask.shape[0]

    def up_to_stop(self):
        """
            Returns the lens structures up to the aperture stop of the systems (used to recover the entrance pupil)
        """
        max_len = self.stop_idx.max()
        mask = np.arange(max_len)[None, :] < self.stop_idx[:, None]
        return Structure(self.stop_idx, self.mask[:, :max_len] & mask, self.mask_G[:, :max_len] & mask,
                         default_device=self.default_device)

    def clone(self):
        return Structure(self.stop_idx.copy(), self.mask.copy(), self.mask_G.copy(), default_device=self.default_device)

    def __getitem__(self, index):
        index = slice(index, index + 1) if isinstance(index, int) else index
        max_len = self.mask[index].sum(axis=1).max()
        return Structure(self.stop_idx[index], self.mask[index, :max_len], self.mask_G[index, :max_len],
                         default_device=self.default_device)

    @property
    def last_g_idx(self):
        # Find the index of the last glass element
        idx = np.broadcast_to(np.arange(self.mask.shape[1], dtype=self.stop_idx.dtype), self.mask.shape)
        return np.where(self.mask_G, idx, 0).argmax(axis=1)

    @property
    def mask_except_last(self):
        mask = self.mask.copy()
        mask[np.arange(len(self)), self.last_g_idx + 1] = 0
        return mask


@dataclass
class Specs:
    structure: Structure
    epd: torch.Tensor
    hfov: torch.Tensor
    vig_up: torch.Tensor = None
    vig_down: torch.Tensor = None
    vig_x: torch.Tensor = None

    def __post_init__(self):
        assert len(self.epd.shape) == 1, 'EPD should be 1-dimensional'
        assert len(self.hfov.shape) == 1, 'HFOV should be 1-dimensional'

        if any((self.vig_up is None, self.vig_down is None)):
            self.vig_up = torch.zeros_like(self.epd)
            self.vig_down = torch.zeros_like(self.epd)
            self.vig_x = torch.zeros_like(self.epd)

    def __len__(self):
        return len(self.structure)

    def scale(self, factor):
        return Specs(self.structure, self.epd * factor, self.hfov, self.vig_up, self.vig_down, self.vig_x)

    def up_to_stop(self):
        return Specs(self.structure.up_to_stop(), self.epd, self.hfov, self.vig_up, self.vig_down, self.vig_x)

    def __getitem__(self, index):
        index = slice(index, index + 1) if isinstance(index, int) else index
        return Specs(
            self.structure[index],
            self.epd[index],
            self.hfov[index],
            self.vig_up[index],
            self.vig_down[index],
            self.vig_x[index]
        )


@dataclass
class Lens:
    structure: Structure
    c: torch.Tensor
    t: torch.Tensor
    nd: torch.Tensor
    v: torch.Tensor

    def __post_init__(self):

        if len(self.c.shape) == 1:
            flat_c = self.c
            self.c = torch.zeros_like(self.structure.mask_torch, dtype=self.c.dtype)
            self.flat_c = flat_c

        if len(self.t.shape) == 1:
            flat_t = self.t
            self.t = torch.zeros_like(self.structure.mask_torch, dtype=self.t.dtype)
            self.flat_t = flat_t

        if len(self.nd.shape) == 1:
            flat_nd = self.nd
            self.nd = torch.ones_like(self.structure.mask_torch, dtype=self.nd.dtype)
            self.flat_nd = flat_nd

        if len(self.v.shape) == 1:
            flat_v = self.v
            self.v = torch.full(self.structure.mask.shape, np.nan).to(self.structure.default_device)
            self.flat_v = flat_v

    def __len__(self):
        return len(self.structure)

    def scale(self, factor):
        return Lens(self.structure, self.c / factor, self.t * factor, self.nd, self.v)

    def up_to_stop(self):
        structure = self.structure.up_to_stop()
        new_len = structure.mask.shape[1]
        return Lens(
            structure,
            self.c[:, :new_len][structure.mask_torch],
            self.t[:, :new_len][structure.mask_torch],
            self.nd[:, :new_len][structure.mask_G_torch],
            self.v[:, :new_len][structure.mask_G_torch],
        )

    def __getitem__(self, index):
        index = slice(index, index + 1) if isinstance(index, int) else index
        structure = self.structure[index]
        max_length = structure.mask.shape[1]
        return Lens(
            structure,
            self.c[index, :max_length],
            self.t[index, :max_length],
            self.nd[index, :max_length],
            self.v[index, :max_length]
        )

    def detach(self):
        return Lens(self.structure, self.c.detach(), self.t.detach(), self.nd.detach(), self.v.detach())

    @property
    def flat_c(self):
        return self.c[self.structure.mask_torch]

    @flat_c.setter
    def flat_c(self, c):
        self.c = mask_replace(self.structure.mask, self.c, c)

    @property
    def flat_c_but_last(self):
        c_mask = self.structure.mask.copy()
        c_mask[np.arange(len(self)), self.structure.mask.sum(axis=1) - 1] = False
        return self.c[c_mask]

    @property
    def flat_t(self):
        return self.t[self.structure.mask_torch]

    @flat_t.setter
    def flat_t(self, t):
        self.t = mask_replace(self.structure.mask, self.t, t)

    @property
    def flat_nd(self):
        return self.nd[self.structure.mask_G_torch]

    @flat_nd.setter
    def flat_nd(self, nd):
        self.nd = mask_replace(self.structure.mask_G, self.nd, nd)

    @property
    def flat_v(self):
        return self.v[self.structure.mask_G_torch]

    @flat_v.setter
    def flat_v(self, v):
        self.v = mask_replace(self.structure.mask_G, self.v, v)

    def get_refractive_indices(self, wavelengths):
        """
            Interpolate the refractive indices at the desired wavelengths [in nm]
            We use a two-parameter model n(lambda) = A + B / lambda**2
            A and B are recovered from the refractive index at the "d" wavelength and the Abbe number
            See "End-to-End Complex Lens Design with Differentiable Ray Tracing" (Sun et al, 2021)
        """
        wc = 656.3
        wd = 587.6
        wf = 486.1
        b = (self.nd - 1) / (self.v * (wf ** -2 - wc ** -2))
        a = self.nd - b / wd ** 2
        assert a.device == b.device
        n = a[..., None] + b[..., None] / torch.tensor([[wavelengths]]).to(a.device) ** 2
        # set refactive index for air
        n = torch.where(self.structure.mask_G_torch[..., None], n, torch.ones_like(n))
        # set original refractive index where abbe number is zero 
        ZeroDisp = torch.transpose(self.v!=0, 0, 1)
        n = torch.where(ZeroDisp, n, torch.transpose(self.nd, 0, 1))
        return n

    @property
    def efl(self):
        return rt.get_first_order(self)[0]

    @property
    def bfl(self):
        return rt.get_first_order(self)[1]

    @property
    def entrance_pupil_position(self):
        return rt.compute_pupil_position(self)
