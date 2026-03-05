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
import matplotlib.gridspec as gridspec

import warnings  # hide the warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


### Input data ################################################################

dir_path = os.path.dirname(os.path.realpath(__file__))

path_exp = "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Depth-Averaged Results\\"

name_exp = ["IMP_015_PIV"]  # , 'PER_100_PIV'] #, 'IMP_060_PIV', 'IMP_100_PIV']

paths_XB = []
paths_OF = [
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.0\\3D_rug13_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V0.2.5\\2D_rug13_CP_lin_h=0.35xd50\\",
]

nom_plots = [
    "Fuhrman 3D (interIso V2.0)",
    "CP 3D (interIso V2.1)",
    "CP 2D (interIso V0.2.5)",
]

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
probe_pos = np.array([-1.802, 0.072, 0.772, 1.567, 2.377, 3.177])
timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "error_vitesse_par"

slope = 1 / 10
theta = np.arctan(slope)

seuil_alpha = 0.5


# colormap
clmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]

gs = gridspec.GridSpec(3, 1)


# Code ########################################################################

RMSE = np.zeros((len(probe_names), len(paths_OF)))
BSS = np.zeros((len(probe_names), len(paths_OF)))  # Brier skill score
WSS = np.zeros((len(probe_names), len(paths_OF)))  # Willmott skill score

probes_x = []


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

        # Calculate mean velocity
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
                )  # calculate the bed parallel component
                UnumPlot.append(
                    sum((UPlotParr[1:] * np.transpose(alphaPlot[1:])[0]))
                    / sum(alphaPlot[1:])[0]
                )
            else:
                UnumPlot.append(np.nan)

        # interpolation of the numerical data
        interp_U = np.interp(texp, timeProbe, UnumPlot)

        # calculate the skill scores
        RMSE[probeNum, ind_path2] = np.sqrt(
            np.nansum((interp_U - Uexp) ** 2) / len(interp_U)
        )
        BSS[probeNum, ind_path2] = 1 - np.nansum((Uexp - interp_U) ** 2) / np.nansum(
            (Uexp - np.nanmean(Uexp)) ** 2
        )
        WSS[probeNum, ind_path2] = 1 - np.nansum((Uexp - interp_U) ** 2) / np.nansum(
            (abs(Uexp - np.nanmean(Uexp)) + abs(interp_U - np.nanmean(Uexp))) ** 2
        )

    # add the probe to the results
    probes_x.append(probe_pos[int(probe_name[-1]) - 1])


probe_x = np.array(probes_x)
for i in range(np.shape(RMSE)[1]):
    plt.scatter(
        probe_x,
        RMSE[:, i],
        color=clmap[i],
        marker="+",
        label=nom_plots[i] + " : mean error = " + str(np.mean(RMSE[:, i])),
    )
    plt.scatter(
        0, np.mean(RMSE[:, i]), color=clmap[i], marker="o", label="_Hidden label"
    )


fig = plt.figure(num=1, figsize=(8, 16), dpi=300)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[2, 0])

for i in range(np.shape(RMSE)[1]):

    ax1.scatter(
        probe_x,
        RMSE[:, i],
        color=clmap[i],
        marker="+",
        label=nom_plots[i] + " : mean error = " + str(np.mean(RMSE[:, i])),
    )
    ax1.scatter(
        0,
        np.mean(RMSE[:, i]),
        color=clmap[i],
        marker="o",
        edgecolor="black",
        label="_Hidden label",
    )

    ax2.scatter(
        probe_x,
        BSS[:, i],
        color=clmap[i],
        marker="+",
        label=nom_plots[i] + " : mean BSS = " + str(np.mean(BSS[:, i])),
    )
    ax2.scatter(
        0,
        np.mean(BSS[:, i]),
        color=clmap[i],
        marker="o",
        edgecolor="black",
        label="_Hidden label",
    )

    ax3.scatter(
        probe_x,
        WSS[:, i],
        color=clmap[i],
        marker="+",
        label=nom_plots[i] + " : mean WSS = " + str(np.mean(WSS[:, i])),
    )
    ax3.scatter(
        0,
        np.mean(WSS[:, i]),
        color=clmap[i],
        marker="o",
        edgecolor="black",
        label="_Hidden label",
    )


# Setting axis labels
ax1.set_xlabel("x (m)")
ax1.set_ylabel("Root Mean Square Error (h (m))")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("Brier Skill Score")
ax3.set_xlabel("x (m)")
ax3.set_ylabel("Willmott Skill Score")


# add grid and legend
ax1.grid()
ax2.grid()
ax3.grid()
# plt.ylim((0, 0.2)
ax1.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
ax2.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
ax3.legend(bbox_to_anchor=(1.1, 1), loc="upper left")


# Save figure
if sauvegarde:
    plt.savefig(dir_path + "\\" + titre_sauv + ".pdf", bbox_inches="tight")
