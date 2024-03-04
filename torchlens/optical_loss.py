import sys
sys.path.append('C:/Users/KU/Desktop/AI/src')

import numpy as np
import torch
from optics_simulator_lite import RaytracedOptics
from lens_modeling import n_v_from_g, Structure
from ray_tracing_lite import compute_last_curvature
from preprocessing.process_dataframe import sequence_encoder, sequence_decoder

class Optical_Loss():
    def __init__(self, lens_type):
        self.lens_type = lens_type
        self.code_lenstype = sequence_encoder(self.lens_type)
        self.numsurf = len(str(self.code_lenstype))
        self.numglass = sum(map(int, str(self.code_lenstype)))
        self.numin = 2 + 2*self.numsurf
        self.numout = 2*self.numglass + 2*self.numsurf-1

    def optical_loss_unsupervised_single(self, input: torch.Tensor, output: torch.Tensor, penalty_rate, device="cuda"):
        """
        GA
        Example input tensor
        "epd", "hfov", "t1_min", "t1_range", "t2_min", "t2_range", "sequence_encoded", "stop_idx", "as_c", "as_t"
        Example output tensor
        "g11", "g12", "c1", "t1", "t2"
        GGA
        Example input tensor
        "epd", "hfov", "t1_min", "t1_range", "t2_min", "t2_range", "t3_min", "t3_range", "sequence_encoded", "stop_idx", "as_c", "as_t"
        Example output tensor
        "g11", "g12", "g21", "g22", "c1", "c2", "t1", "t2", "t3"
        GAGA
        Example input tensor
        "epd", "hfov", "t1_min", "t1_range", "t2_min", "t2_range", "t3_min", "t3_range", "t4_min", "t4_range", "sequence_encoded", "stop_idx", "as_c", "as_t"
        Example output tensor
        "g11", "g12", "g21", "g22", "c1", "c2", "c3", "t1", "t2", "t3", "t4"
        """
        epd = input[[0]]
        hfov = input[1]
        t = output[self.numglass*2+self.numsurf-1:self.numout]

        # Convert g into n and v
        n = torch.zeros(self.numglass, device=device)
        v = torch.zeros(self.numglass, device=device)
        # iterate through glasses
        for i in range(self.numglass):
            gi = output[2*i:2*i+2].unsqueeze(dim=0)
            # Make sure nv_converter is the same one as the first convert
            ni, vi = n_v_from_g(gi)
            n[i] = ni
            v[i] = vi

        # parameters for ray tracing
        sequence_encoded = input[-4]
        stop_idx = input[-3].detach().cpu().unsqueeze(dim=0).numpy()
        stop_idx = stop_idx.astype("int64")
        as_c = input[-2].unsqueeze(dim=0)
        as_t = input[-1].unsqueeze(dim=0)

        # Decode the lens sequence
        sequence = sequence_decoder(int(sequence_encoded))  # str
        # Calc c
        structure = Structure(stop_idx=stop_idx, sequence=np.array([sequence]), default_device=device)
        c_wo_last = output[self.numglass*2:self.numglass*2+self.numsurf-1]
        c = compute_last_curvature(structure, c_wo_last, t, n)

        # Convert the t
        t = Optical_Loss.t_converter(int(stop_idx), sequence, t, as_t)
        c = Optical_Loss.t_converter(int(stop_idx), sequence, c, as_c)

        sim = RaytracedOptics(
            initial_lens_path="",
            stop_index=stop_idx,
            sequence=np.array([sequence]),
            hfov=torch.tensor([0, hfov*0.5, hfov]).cuda(),
            epd=epd,
            curvature=c,
            thickness=t,
            n_refractive=n,
            abbe_number=v,
            n_sampled_fields=8,  # Down from 21 (default) to fit into memory
            n_pupil_rings=8,  # Down from 32 (default) to fit into memory
            wavelengths = torch.tensor([459, 520, 640]).cuda(),
            penalty_rate = penalty_rate,
            pupil_sampling='circular',
            simulated_res_factor=1,
            apply_distortion=True,
            apply_relative_illumination=True,
            lazy_init=True,
            glass_catalog_path='/content/glass/selected_glass.csv',
            device=device
        )

        sim.do_ray_tracing(sim.lensR)

        return sim.loss_dict["loss_unsup"], sim.loss_dict["rms"], sim.loss_dict["penalty"]


    def optical_loss_unsupervised(self, input: torch.Tensor, output: torch.Tensor, penalty_rate=0.2, device="cuda"):
        loss_sum = torch.tensor(0., device=device)
        rms_sum = torch.tensor(0., device=device)
        penalty_sum = torch.tensor(0., device=device)
        batch_size = input.size(0)  # ミニバッチサイズ

        for i in range(batch_size):
            # 各サンプルごとの入力と出力を取得
            input_sample = input[i]
            output_sample = output[i]

            # 損失関数の計算
            loss, rms, penalty = self.optical_loss_unsupervised_single(input_sample, output_sample, penalty_rate, device)

            loss_sum += loss
            rms_sum += rms
            penalty_sum += penalty

        # バッチ内のサンプルに対する平均損失を計算
        loss_mean = loss_sum / batch_size
        rms_mean = rms_sum / batch_size
        penalty_mean = penalty_sum / batch_size

        return loss_mean, rms_mean, penalty_mean
    

    @ staticmethod
    def t_converter(stop_idx, sequence, t, as_t=None):
        # *1 <= stop_idx <= len(sequence)
        base_t_list = t
        if sequence[stop_idx-1] == "A" and (as_t != None and as_t != -1):
            # as_t == -1 when the base don't have aperture stop variable
            return torch.cat((base_t_list[:stop_idx-1], as_t, base_t_list[stop_idx-1:]))
        else:
            return base_t_list
    

    def optical_loss_supervised(self, input: torch.Tensor, output: torch.Tensor, device="cuda"):
        """
        GA
        Example input tensor
        "epd", "hfov", "t1_min", "t1_range", "t2_min", "t2_range", "sequence_encoded", "stop_idx", "as_c", "as_t"
        Example output tensor
        "g11", "g12", "c1", "t1", "t2"
        GGA
        Example input tensor
        "epd", "hfov", "t1_min", "t1_range", "t2_min", "t2_range", "t3_min", "t3_range", "sequence_encoded", "stop_idx", "as_c", "as_t"
        Example output tensor
        "g11", "g12", "g21", "g22", "c1", "c2", "t1", "t2", "t3"
        GAGA
        Example input tensor
        "epd", "hfov", "t1_min", "t1_range", "t2_min", "t2_range", "t3_min", "t3_range", "sequence_encoded", "stop_idx", "as_c", "as_t"
        Example output tensor
        "g11", "g12", "g21", "g22", "c1", "c2", "t1", "t2", "t3"
        """
        S = self.numsurf
        G = self.numglass
        g1 = list(range(0, 2*G, 2))
        g2 = list(range(1, 2*G+1, 2))
        c_st = G*2
        t_st = G*2 + S-1

        DevG1 = (output[:, g1] - input[:, g1])
        DevG2 = (output[:, g2] - input[:, g2])
        DevC = (output[:, c_st:c_st+S-1] - input[:, c_st:c_st+S-1])
        DevT = (output[:, t_st:t_st+S] - input[:, t_st:t_st+S])

        SumSq_G1 = torch.sum(DevG1**2, 1)
        SumSq_G2 = torch.sum(DevG2**2, 1)
        SumSq_C = torch.sum(DevC**2, 1)
        SumSq_T = torch.sum(DevT**2, 1)

        # compute MSE for each design
        MSEs = (SumSq_G1+SumSq_G2+SumSq_C+SumSq_T)/(2*G+2*S-1)
        # compute mean MSE
        loss_sp = torch.mean(MSEs)

        return loss_sp

########################################################################
# test
if __name__ == '__main__':
    import pandas as pd
    from preprocessing.process_dataframe import DataframeProcessor
    from training.lensdesign_model import LensDesignCreator
    from lens_modeling import g_from_n_v

    # # preprocess dataframe
    # df = pd.read_csv("../design/GA_NS_1/ga_ns_1_raw.csv")
    # DP = DataframeProcessor('GA')
    # df, tensor_X, tensor_y = DP.process_dataframe(df, 1024)
    # X = tensor_X[:,:6]

    # # # model
    # model = LensDesignCreator('GA')
    # y = model(X)

    # # compute loss
    # OL = Optical_Loss('GA')
    # loss_us, rms, penalty = OL.optical_loss_unsupervised(tensor_X, y, device="cpu")
    # # loss_sp = OL.optical_loss_supervised(tensor_y, y, device="cuda")
    # # print(loss_sp)


    # G = g_from_n_v(torch.tensor([0.9812]), torch.tensor([36.0798]))
    # G = G.numpy()
    # tensor_X = torch.tensor([0.1, 5.0, 1, 1, 1, 1, 10, 1, -1, -1])
    # y = torch.tensor([G[0][1], G[0][1], -10.2554, 4.9996, 2.1608])

    # OL = Optical_Loss('GA')
    # loss_us, rms, penalty = OL.optical_loss_unsupervised_single(tensor_X, y, 0.2, device="cpu")
    # print(rms)
    # print(penalty)