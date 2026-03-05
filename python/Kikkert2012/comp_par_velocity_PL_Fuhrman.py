# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:28:28 2024

@author: kaczmare3m
"""

import os
import sys
import numpy as np
import fluidfoam as ff
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import netcdf

# sys.path.insert(0, 'Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Python')

import warnings  # hide the warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Input data ############################################################

dir_path = os.path.dirname(os.path.realpath(__file__))

path_exp = "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Depth-Averaged Results\\"

name_exp = ["IMP_015_PIV"]  # , 'PER_100_PIV'] #, 'IMP_060_PIV', 'IMP_100_PIV']

paths_XB = []
paths_OF = [
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.0\\3D_rug13_h0=0.59_geomCorr\\",
]

nom_plots = ["XBeach V1 run02", "cas de référence 2D (interIsoFoam V0.2)"]

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
probe_pos = np.array([-1.802, 0.072, 0.772, 1.567, 2.377, 3.177])
timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "vitesse_par_rug13_CP_Fuhrman"

slope = 1 / 10
theta = np.arctan(slope)

seuil_alpha = 0.5


# colormap
clmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
clmap = [clmap[2], "#000000"]
taille_plot = [3, 1.5]
style_trait = ["-", "--"]


# Code ########################################################################


for probeNum, probe_name in enumerate(probe_names):

    print("Création figure " + probe_name)
    plt.figure()

    for ind_path2, curr_path in enumerate(paths_OF):

        # Loading simulation data
        varName, probe_loc, timeProbe, varProbe = ff.readsampling(
            path=curr_path + postProcessFolder + "\\",
            probes_name=probe_name,
            time_name=timeStep,
        )

        probeYLoc = (probe_loc[:, 1] - probe_loc[0, 1]) * np.cos(theta) - (
            probe_loc[:, 0] - probe_loc[0, 0]
        ) * np.sin(theta)

        # Loading experimental data
        mat = scipy.io.loadmat(path_exp + "U_" + name_exp[0] + probe_name[-1] + ".mat")
        texp = np.concatenate(mat.get("t_" + name_exp[0] + probe_name[-1]))
        Uexp = np.concatenate(mat.get("U_" + name_exp[0] + probe_name[-1]))

        # calculating mean velocity on simulation data
        UnumPlot = []
        for ind_t, temps in enumerate(timeProbe):
            Unum = varProbe[ind_t, varName.index("U")]  # get U
            alphaNum = varProbe[ind_t, varName.index("alpha.water")]

            UPlot = np.array(
                [i for ind_r, i in enumerate(Unum) if alphaNum[ind_r] >= seuil_alpha]
            )  # only data corresponding to a ‘water’ cell is retrieved
            alphaPlot = np.array(
                [
                    i
                    for ind_r, i in enumerate(alphaNum)
                    if alphaNum[ind_r] >= seuil_alpha
                ]
            )

            if len(UPlot) > 1:
                UPlotParr = UPlot[:, 0] * np.cos(theta) + UPlot[:, 1] * np.sin(
                    theta
                )  # calculating bed parallel component
                UnumPlot.append(
                    sum((UPlotParr[1:] * np.transpose(alphaPlot[1:])[0]))
                    / sum(alphaPlot[1:])[0]
                )
            else:
                UnumPlot.append(np.nan)

        # Plotting mean bed parallel velocity evolution over time

        plt.plot(
            timeProbe,
            UnumPlot,
            color=clmap[ind_path2],
            label=nom_plots[ind_path2],
            linewidth=taille_plot[ind_path2],
            linestyle=style_trait[ind_path2],
        )

    plt.scatter(texp, Uexp, color="red", marker="+", label="Données expérimentales")

    # Setting axis labels
    plt.xlabel("$t$ (s)", fontsize=20, usetex=True)
    plt.ylabel("$\langle U_{x} \\rangle$ (m/s)", fontsize=20, usetex=True)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.title("Vitesse parallèle moyenne : "+probe_name)

    # add grid and legend
    plt.grid()
    plt.rc("axes", axisbelow=True)
    # plt.ylim((0, 0.2)
    # plt.legend(bbox_to_anchor=(0, -0.18), loc="upper left")

    # show
    # plt.show()

    # Save figure
    if sauvegarde:
        plt.savefig(
            dir_path + "\\" + titre_sauv + "_sonde" + str(probeNum + 1) + ".pdf",
            bbox_inches="tight",
        )
