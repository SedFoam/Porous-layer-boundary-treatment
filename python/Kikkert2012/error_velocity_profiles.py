# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:28:28 2024

@author: kaczmare3m
"""

import os
import numpy as np
import fluidfoam as ff
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec


# Input data
dir_path = os.path.dirname(os.path.realpath(__file__))
path_exp = "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Ensemble-Averaged Results\\"
name_exp = "IMP_015_PIV"

paths = [
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.0\\3D_rug13_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V0.2.5\\2D_rug13_CP_lin_h=0.35xd50\\",
]

nom_legend = [
    "Fuhrman 3D (interIso V2.0)",
    "CP 3D (interIso V2.1)",
    "CP 2D (interIso V0.2.5)",
    "Données expérimentales",
]
colormap = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

color_paths = ["blue", "red", "orange", "green"]
symb_legend = [
    Line2D([0], [0], label=nom_legend[0], color="b"),
    Line2D([0], [0], label=nom_legend[1], color="r"),
    Line2D([0], [0], label=nom_legend[2], color="orange"),
    Line2D(
        [0],
        [0],
        label=nom_legend[3],
        marker="o",
        markerfacecolor="none",
        markeredgecolor="black",
        linestyle="",
    ),
]

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
probe_pos = np.array([-1.802, 0.072, 0.772, 1.567, 2.377, 3.177])
timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "error_profils_vitesse"

slope = 1 / 10
theta = np.arctan(slope)

seuil_alpha = 0.5

allTimeSteps = True  # True to calculate the RMSE on all time steps, False to calculate it only on timePlot times
timePlot = np.linspace(3, 6.9, 7)  # time list

gs = gridspec.GridSpec(3, 1)


### Code ######################################################################

RMSE = np.zeros((len(probe_names), len(paths)))
BSS = np.zeros((len(probe_names), len(paths)))  # Brier skill score
WSS = np.zeros((len(probe_names), len(paths)))  # Willmott skill score

probes_x = []

for ind_name, probe_name in enumerate(probe_names):

    print("Création figure " + probe_name)

    # Loading experimental data
    mat = scipy.io.loadmat(
        path_exp + "U_profiles_" + name_exp + probe_name[-1] + ".mat"
    )
    texp = mat.get("t_" + name_exp + probe_name[-1])[0]
    Uexp = mat.get("U_profiles_" + name_exp + probe_name[-1])
    Yexp = mat.get("z_" + name_exp + probe_name[-1])[0]

    if allTimeSteps:  # if we want to calculate the RMSE over all time steps
        timePlot = texp

    # get the experimental time steps
    UexpPlot = []
    tExpPlot = []
    for temps in timePlot:
        ind = np.abs(texp - temps).argmin()  # closest time index to the target time
        UexpPlot.append(Uexp[:, ind])  # get U
        tExpPlot.append(texp[ind])  # get the exact time

    for ind_path, path in enumerate(paths):
        # Loading simulation data
        varName, probe_loc, timeProbe, varProbe = ff.readsampling(
            path=path + postProcessFolder + "\\",
            probes_name=probe_name,
            time_name=timeStep,
        )

        probeYLoc = (probe_loc[:, 1] - probe_loc[0, 1]) * np.cos(theta) - (
            probe_loc[:, 0] - probe_loc[0, 0]
        ) * np.sin(theta)

        # get the numerical time steps
        UnumPlot = []
        tNumPlot = []
        probeYLocPlot = []
        for temps in timePlot:
            ind = np.abs(
                timeProbe - temps
            ).argmin()  # closest time index to the target time
            Unum = varProbe[ind, varName.index("U")]  # get U
            alphaNum = varProbe[ind, varName.index("alpha.water")]

            UPlot = np.array(
                [i for ind_r, i in enumerate(Unum) if alphaNum[ind_r] >= seuil_alpha]
            )  # only data corresponding to a ‘water’ cell is retrieved
            probeYLocPlot.append(
                [
                    i
                    for ind_r, i in enumerate(probeYLoc)
                    if alphaNum[ind_r] >= seuil_alpha
                ]
            )

            if (
                np.size(np.shape(UPlot)) > 1
            ):  # test if the vector is empty (not enough water)
                UPlotParr = UPlot[:, 0] * np.cos(theta) + UPlot[:, 1] * np.sin(
                    theta
                )  # calculate bed parallel component
                UnumPlot.append(UPlotParr)
            else:
                UnumPlot.append([])
            tNumPlot.append(timeProbe[ind])

        # calculate RMSE and skill scores
        RMSE_part = 0
        BSS_part = 0
        WSS_part = 0

        n_points = 0
        for ind, UNum in enumerate(UnumPlot):
            UExp = UexpPlot[ind]
            UExp_noNan = UExp[
                ~np.isnan(UExp)
            ]  # remove nan values for the interpolation

            try:
                interpNumU = np.interp(
                    Yexp[0 : len(UExp_noNan)], probeYLocPlot[ind], UNum
                )
            except:  # if UExp_noNan is empty
                interpNumU = np.array([])

            RMSE_part += sum((UExp_noNan - interpNumU) ** 2)
            BSS_part += sum((UExp_noNan - np.mean(UExp_noNan)) ** 2)
            WSS_part += sum(
                (
                    abs(UExp_noNan - np.mean(UExp_noNan))
                    + abs(interpNumU - np.mean(UExp_noNan))
                )
                ** 2
            )

            n_points += len(UExp_noNan)

        RMSE[ind_name, ind_path] = np.sqrt(RMSE_part / n_points)
        BSS[ind_name, ind_path] = 1 - RMSE_part / BSS_part
        WSS[ind_name, ind_path] = 1 - RMSE_part / WSS_part

    # add the probe to the results
    probes_x.append(probe_pos[int(probe_name[-1]) - 1])


probe_x = np.array(probes_x)


fig = plt.figure(num=1, figsize=(8, 16), dpi=300)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[2, 0])

for i in range(np.shape(RMSE)[1]):

    ax1.scatter(
        probe_x,
        RMSE[:, i],
        color=colormap[i],
        marker="+",
        label=nom_legend[i] + " : mean error = " + str(np.mean(RMSE[:, i])),
    )
    ax1.scatter(
        0,
        np.mean(RMSE[:, i]),
        color=colormap[i],
        marker="o",
        edgecolor="black",
        label="_Hidden label",
    )

    ax2.scatter(
        probe_x,
        BSS[:, i],
        color=colormap[i],
        marker="+",
        label=nom_legend[i] + " : mean BSS = " + str(np.mean(BSS[:, i])),
    )
    ax2.scatter(
        0,
        np.mean(BSS[:, i]),
        color=colormap[i],
        marker="o",
        edgecolor="black",
        label="_Hidden label",
    )

    ax3.scatter(
        probe_x,
        WSS[:, i],
        color=colormap[i],
        marker="+",
        label=nom_legend[i] + " : mean WSS = " + str(np.mean(WSS[:, i])),
    )
    ax3.scatter(
        0,
        np.mean(WSS[:, i]),
        color=colormap[i],
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
