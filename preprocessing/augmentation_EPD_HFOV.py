import sys
sys.path.append('C:/Users/KU/Desktop/AI/src')

import os
import numpy as np
import pandas as pd
import scipy

from analysis import design2system
from analysis.geometric_trace_mod import GeometricTrace_Mod

class Augmentation_EPD_HFOV():
    def __init__(self, filepath):
        self.shrinkrate = 0.94

        # len or zemax?
        split_tup = os.path.splitext(filepath)
        ext = split_tup[1]
        if ext in ['.len', '.LEN']:
            # import system from oslo
            with open(filepath, 'r') as file:
                self.sys = design2system.len_to_system(file)
            
        elif ext in ['.zmx', '.ZMX']:
            # import system from zemax
            with open(filepath, 'r') as file:
                data = file.read()
                self.sys = design2system.zmx_to_system(data)

    # Return index of effective apperture stop
    def find_effective_aperture(self):
        # check stop index = 1 and HFOV = 0
        rate = 0.1
        self.sys.stop = 1
        self.sys.object.angle = np.deg2rad(0)
        self.sys.fields = [1]
        self.sys.epd = 2*self.sys[1].radius*rate
        self.sys.update()
        wave = 520e-9

        # trace 1D ray bundle
        t = GeometricTrace_Mod(self.sys)
        t.rays_clipping((0, 1), wave)
        # denormalise intersections [x, y, z]
        y = np.array([el.from_normal(yi) + oi for el, yi, oi
                        in zip(t.system, t.y, t.origins)])
        
        # calculate max y height for each surface and
        # compute ratio of aperture radius - y height
        numsurf = len(self.sys) - 2
        AYRatio = np.zeros(numsurf)
        for i in range(numsurf):
            AYRatio[i] = self.sys[i+1].radius/max(abs(y[i+1,:,1]))

        self.eff_stop_index = AYRatio.argmin() + 1

    # Compute maximum aperture diameter of zero HFOV and zero ray loss
    def effective_maximum_aperture(self):
        # Compute max aperture diameter whose ray loss is zero.

        # trace conditions
        self.sys.stop = self.eff_stop_index
        D_init = 2*self.sys[self.sys.stop].radius*self.shrinkrate
        self.sys.epd = D_init
        self.sys.update()

        # try aperture of lens diameter
        # self.AptYs = np.array([0., 1., -1.]) * D_init/2
        num_fail = self.ray_failure()
        if num_fail == 0:
            self.eff_max_EPD = D_init 
        else:
            opt = scipy.optimize.fminbound(self.optimize_EPD, 1e-4, D_init,
                                            full_output=True, disp=0)
            self.eff_max_EPD = opt[0]

    def EPD_HFOV_coupling(self, res):
        # EPD rate
        rate = np.linspace(1., 0., res+1)

        couple_array = np.zeros([res, 2])
        couple_array[0,:] = [self.eff_max_EPD, 0.]

        # iterate through couples
        for i in range(res-1):
            epd = self.eff_max_EPD * rate[i+1]
            self.sys.epd = epd
            # self.AptYs = np.array([0., 1., -1.]) * epd/2*self.shrinkrate
            # compute maximum HFOV in given epd whose ray loss is zero
            opt = scipy.optimize.fminbound(self.optimize_HFOV, 0.1, 75,
                                            full_output=True, disp=0)
            couple_array[i+1,:] = [epd, opt[0]]

        return couple_array
    
    def optimize_EPD(self, epd):
        # update EPD value
        # self.AptYs = np.array([0., 1., -1.]) * epd/2*self.shrinkrate
        self.sys.epd = epd
        self.sys.update()
        
        # compute ray failure
        num_fail = self.ray_failure()

        D = 2*self.sys[self.sys.stop].radius
        if num_fail == 0:
            objective = 1 - epd/D   # deviation/D
        else:
            objective = 1 + epd/D   # (D + 1/EPD)/D
        return objective
    
    def optimize_HFOV(self, hfov):
        # update HFOV value
        self.sys.object.angle = np.deg2rad(hfov) 
        self.sys.update()

        # compute ray failure
        num_fail = self.ray_failure()

        if num_fail == 0:
            objective = 1 - hfov/90
        else:
            objective = 1 + hfov/90 
        return objective

    def ray_failure(self):
        # compute ray failure
        # t = GeometricTrace_Mod(self.sys)
        # t.rays_clipping((0, 1), min(self.sys.wavelengths))
        # fails_wmin = np.sum(np.any(t.fail, axis=0))
        # t.rays_clipping((0, 1), max(self.sys.wavelengths))
        # fails_wmax = np.sum(np.any(t.fail, axis=0))
        # num_fail = max([fails_wmin, fails_wmax])

        t = GeometricTrace_Mod(self.sys)
        t.rays_clipping((0, 1), np.mean(self.sys.wavelengths))
        num_fail = np.sum(np.any(t.fail, axis=0))
    
        # # check ray aiming error
        # dev = np.mean(np.abs(t.y[self.sys.stop][:,1]-self.AptYs))
        # if num_fail==0 and dev/self.sys.epd >= 1e-2:
        #     num_fail = 1

        return num_fail

# Compute maximum lens diameter for single lens
def maximum_diameter(sequence, Curv, Thick):
    # sequence: string
    # Curv: 1D ndarray
    # Thick: 1D ndarray

    # convert sequence to lens surface group
    mode = False
    lens_list = []
    for i in range(len(sequence)):
        if sequence[i] == 'G':
            if not mode:
                # first surface of lens
                mode = True
                # create lens
                lens = [i]
            else:
                # after second surface of lens
                # append surface
                lens.append(i)
        elif sequence[i] == 'A':
            if mode:
                # last surface of lens
                mode = False
                # close lens
                lens.append(i)
                lens_list.append(lens)

    # Compute diameter for each lens
    Diam = []
    for i in range(len(lens_list)):
        lens = lens_list[i]
        numsurf = len(lens)

        if numsurf == 2:
            list_comb = [[0,1]]
        elif numsurf == 3:
            list_comb = [[0,1], [0,2], [1,2]]
        elif numsurf == 4:
            list_comb = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
        
        NumComb = len(list_comb)
        D_array = np.zeros(NumComb)
        for j in range(NumComb):
            c1 = Curv[lens[list_comb[j][0]]]
            c2 = Curv[lens[list_comb[j][1]]]
            t = 0
            for k in range(list_comb[j][0], list_comb[j][1]):
                t += Thick[lens[k]]
            R1 = abs(1/c1)
            R2 = abs(1/c2)
            
            if c1>0 and c2<0:
                # intersect
                D = diameter_from_intersection(R1, R2, R1, t-R2)
            elif c1<0 and c2>0:
                # hemi-sphere
                D = 2*min([R1,R2])
            elif (c1>0 and c2>0) and abs(c1)>abs(c2):
                # intersect
                D = diameter_from_intersection(R1, R2, R1, t+R2)
            elif (c1<0 and c2<0) and abs(c1)<abs(c2):
                # intersect
                D = diameter_from_intersection(R1, R2, -R1, t-R2)
            elif (c1>0 and c2>0) and abs(c1)<=abs(c2):
                # hemi-sphere
                D = 2*R2
            elif (c1<0 and c2<0) and abs(c1)>=abs(c2):
                # hemi-sphere
                D = 2*R1
            elif c1!=0 and c2==0:
                # sphere-plano
                D = 2*R1
            elif c1==0 and c2!=0:
                # plano-sphere
                D = 2*R2
            elif c1==0 and c2==0:
                # plano-plano
                D = 9e4
            D_array[j] = D
        Diam = Diam + [D_array.max()]*numsurf

    Diam = np.array(Diam)
    idx_inf = np.where(Diam == 9e4)[0]
    if idx_inf.size >= 1:
        D_descend = -np.sort(-np.unique(Diam))
        # Replace inf with largest diameter
        Diam[idx_inf] = D_descend[1]
    return Diam

# distance between two intersections of x shifted two circles
def diameter_from_intersection(R1, R2, C1X, C2X):
    x = (R1**2-R2**2-C1X**2+C2X**2)/(-2*(C1X-C2X))
    in_sqrt = R1**2-x**2+2*C1X*x-C1X**2
    if in_sqrt >= 0:
        y1 = np.sqrt(in_sqrt)
        y2 = -y1
        D = y1 - y2
    else:
        D = min([R1, R2])*2
    return D


def main(df, Seq, res_epd, res_hfov):
    from analysis.tensor2zmx import tensor2zemax
    from analysis.lens_visualization import Reference2DesignTuple
    
    num_design = df.shape[0]
    df_augment = pd.DataFrame(np.repeat(df.values, res_epd*res_hfov, axis=0),
                              columns=df.columns)
    # copy data types
    df_augment = df_augment.astype(df.dtypes)
    st = 0
    # iterate through designs
    for i in range(num_design):
        # duplicate dataframe as number of augments
        df_design = pd.DataFrame(np.repeat(df.values[i].reshape([1,df.shape[1]]), res_epd*res_hfov, axis=0), 
                                 columns=df.columns)
        df_design = df_design.astype(df.dtypes)

        # tensor to zemax
        design = Reference2DesignTuple(df.iloc[i], Seq)
        zmxpath = 'analysis/augment.zmx'
        tensor2zemax(design, zmxpath)

        # compute the set of EPD-HFOV
        AG = Augmentation_EPD_HFOV(zmxpath)
        AG.find_effective_aperture()
        if AG.eff_stop_index == 1:
            AG.effective_maximum_aperture()
            couple = AG.EPD_HFOV_coupling(res_epd)

            # augment for hfov 0 to max angle
            EPD_HFOV = np.zeros([res_epd*res_hfov,2])
            st2 = 0
            for j in range(res_epd):
                EPD = np.ones(res_hfov)*couple[j,0]
                HFOV = np.linspace(0, couple[j,1], res_hfov)
                EPD_HFOV[st2:st2+res_hfov,0] = EPD
                EPD_HFOV[st2:st2+res_hfov,1] = HFOV
                st2 = st2 + res_hfov

            df_design['epd'] = EPD_HFOV[:,0]
            df_design['hfov'] = EPD_HFOV[:,1]
            for j in range(res_epd*res_hfov):
                df_design['hfov_all'].iloc[j] = '[0.0, '+str(df_design['hfov'].iloc[j])+']'
            df_augment.iloc[st:st+res_epd*res_hfov,:] = df_design

            st = st + res_epd*res_hfov
    df_augment = df_augment.iloc[:st]

    # delete duplicates
    df_augment = df_augment.drop_duplicates()
    return df_augment


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    res_epd = 4
    res_hfov = 4
    
    # Import zebase and lensview database
    df_zebase = pd.read_csv("C:/Users/KU/Desktop/AI/design/Zebase/Zebase.csv")
    df_lensview = pd.read_csv("C:/Users/KU/Desktop/AI/design/Lensview/LensView_all.csv")
    # Merge them depending on sequence GA, GGA, GAGA
    df = pd.concat([df_zebase, df_lensview], ignore_index=True, axis=0)

    # Exclude zero EFL
    df = df[df["efl"] != 0.]
    # Exclude config more than 2
    df = df[~df['file_name'].str.contains('conf') | df['file_name'].str.contains('conf1')]

    # database of each sequence 
    Seq = 'GA'
    df_seq = df[df['sequence']==Seq]
    df_seq = df_seq[df_seq['hfov']<5.]
    # reset row indexing starting from zero
    df_seq = df_seq.reset_index(drop=True)
    
    df_augment = main(df_seq, Seq, res_epd, res_hfov)

    from preprocessing.process_dataframe import DataframeProcessor
    DP = DataframeProcessor(Seq)
    df_process, tensor_X, tensor_y = DP.process_dataframe(df_augment)

    # # plot scattered plot
    # fig, ax = plt.subplots()
    # ax.scatter(df_augment['epd'].to_numpy(), df_augment['hfov'].to_numpy(), s=80, facecolors='none', edgecolors='b')
    # ax.scatter(df_seq['epd'].to_numpy(), df_seq['hfov'].to_numpy(), s=80, facecolors='none', edgecolors='r')

    # plt.suptitle(Seq +' - '+ str(df_augment.shape[0])+' designs')
    # ax.set_xlabel('EPD [mm]')
    # ax.set_ylabel('HFOV [deg]')
    # plt.show()