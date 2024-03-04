import torch
import numpy as np
from typing import Union
from torchlens.optics_simulator_lite import RaytracedOptics
from torchlens.lens_modeling import n_v_from_g, Structure
from torchlens.ray_tracing_lite import compute_last_curvature
from preprocessing.lens_sequence import sequence_decoder
from typing import NamedTuple


"""
命名規則
optical_loss_single_ns/s_v0
"""
class RayTracerInput(NamedTuple):
    stop_idx: int
    sequence: str
    hfov: torch.tensor  # shape: (1,)
    epd: torch.tensor  # shape: (1,)
    c_wo_last: torch.tensor
    t: torch.tensor
    nd: torch.tensor
    v: torch.tensor

def raytracer(input_object: RayTracerInput, device: Union[torch.device, str] = "cuda", glass_catalog_path: str = "glass/selected_ohara_glass.csv", return_penalties: bool = False, penalty_rate: float = 0.2):
    seq_np = np.array([input_object.sequence])
    stop_idx_np = np.array([input_object.stop_idx])
    structure = Structure(stop_idx=stop_idx_np, sequence=seq_np, default_device=device)
    c = compute_last_curvature(structure, input_object.c_wo_last, input_object.t, input_object.nd)
    sim = RaytracedOptics(
        initial_lens_path="",
        stop_index=stop_idx_np,
        sequence=seq_np,
        hfov=torch.stack([input_object.hfov.new_tensor(0), input_object.hfov * 0.7, input_object.hfov]),
        epd=input_object.epd.unsqueeze(0),
        curvature=c,
        thickness=input_object.t,
        n_refractive=input_object.nd,
        abbe_number=input_object.v,
        n_sampled_fields=8,  # Down from 21 (default) to fit into memory
        n_pupil_rings=8,  # Down from 32 (default) to fit into memory
        wavelengths=torch.tensor([459, 520, 640]),
        pupil_sampling='circular',
        simulated_res_factor=1,
        apply_distortion=True,
        apply_relative_illumination=True,
        lazy_init=True,
        glass_catalog_path=glass_catalog_path,
        device=device,
        penalty_rate=penalty_rate,
    )
    sim.do_ray_tracing(sim.lensR)
    if return_penalties:
        return sim.loss_dict["loss_unsup"], sim.loss_dict["rms"], sim.loss_dict["penalty"]
    else:
        return sim.loss_dict["loss_unsup"], sim.loss_dict["rms"]



def optical_loss_single_v1(epd: torch.Tensor, hfov: torch.Tensor, c_wo_last_fixed: torch.Tensor, t_fixed: torch.Tensor,
                            g: torch.Tensor, as_d: torch.Tensor, device: Union[torch.device, str] = "cuda",
                            glass_catalog_path: str = "glass/selected_ohara_glass.csv", loss_on_exception: float = 1.0, rms_on_exception: float = 1.0, debug_mode: bool = False):
    """
    input_object: all tensors (fixed length)
    """
    g_parallel = g.reshape(-1, 2)
    nd_fixed, v_fixed = n_v_from_g(g_parallel)
    try:
        c_w_invalid_last, t, nd, v, seq, stop_idx, _ = sequence_decoder(c_wo_last_fixed, t_fixed, nd_fixed, v_fixed, as_d, debug_mode=debug_mode)
    except ValueError as e:
        # TODO: need change
        if debug_mode:
            print("Exception occurred in sequence_decoder")
            print(e)
        return t_fixed.new_tensor(loss_on_exception, requires_grad=True), t_fixed.new_tensor(rms_on_exception, requires_grad=True)
    seq_np = np.array([seq])
    stop_idx_np = np.array([stop_idx])
    structure = Structure(stop_idx=stop_idx_np, sequence=seq_np, default_device=device)
    c = compute_last_curvature(structure, c_w_invalid_last, t, nd)
    epd_all = epd.unsqueeze(0)
    sim = RaytracedOptics(
        initial_lens_path="",
        stop_index=stop_idx_np,
        sequence=seq_np,
        hfov=torch.stack([hfov.new_tensor(0), hfov * 0.7, hfov]),
        epd=epd_all,
        curvature=c,
        thickness=t,
        n_refractive=nd,
        abbe_number=v,
        n_sampled_fields=8,  # Down from 21 (default) to fit into memory
        n_pupil_rings=8,  # Down from 32 (default) to fit into memory
        wavelengths=torch.tensor([459, 520, 640], device=device),
        pupil_sampling='circular',
        simulated_res_factor=1,
        apply_distortion=True,
        apply_relative_illumination=True,
        lazy_init=True,
        glass_catalog_path=glass_catalog_path,
        device=device
    )
    x, y, ray_ok = sim.do_ray_tracing(sim.lensR)
    return sim.loss_dict["loss_unsup"], sim.loss_dict["rms"]
