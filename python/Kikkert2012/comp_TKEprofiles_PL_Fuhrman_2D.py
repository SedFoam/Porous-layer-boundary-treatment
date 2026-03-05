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


# Input data
dir_path = os.path.dirname(os.path.realpath(__file__))
path_exp = "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Turbulence Quantities\\"
name_exp = "IMP_015_PIV"

paths = [
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.0\\3D_rug13_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V0.2.5\\2D_rug13_CP_lin_h=0.35xd50\\",
]

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "profilsTKE_rug13_CP_Fuhrman+2D"

seuil_alpha = 0.5
slope = 1 / 10
theta = np.arctan(slope)

timePlot = np.array([7])  # time to plot
XLim = (0, 0.06)

clmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
clmap = ["#000000", clmap[2], clmap[1]]


### Code ######################################################################

for i_name, probe_name in enumerate(probe_names):

    print("Création figure " + probe_name)

    # Loading experimental data
    mat = scipy.io.loadmat(path_exp + "TKE_" + name_exp + probe_name[-1] + ".mat")
    texp = mat.get("t_" + name_exp + probe_name[-1])[0]
    TKE = mat.get("TKE_" + name_exp + probe_name[-1])
    Yexp = mat.get("z_" + name_exp + probe_name[-1])[0]

    # getting experimental time steps to plot
    TKEexpPlot = []
    tExpPlot = []
    for temps in timePlot:
        ind = np.abs(texp - temps).argmin()  # closest time index to the target time
        TKEexpPlot.append(TKE[:, ind])
        tExpPlot.append(texp[ind])  # getting the exact time

    # Plot TKE
    plt.figure()
    plt.scatter(
        TKEexpPlot[0], Yexp, color="red", marker="+", label="Données expérimentales"
    )

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

        # getting numerical time steps to plot
        knumPlot = []
        tNumPlot = []
        probeYLocPlot = []
        for temps in timePlot:
            ind = np.abs(
                timeProbe - temps
            ).argmin()  # closest time index to the target time
            knum = varProbe[ind, 1]  # getting k

            alphaNum = varProbe[ind, 0]

            kPlot = np.array(
                [i for ind_r, i in enumerate(knum) if alphaNum[ind_r] >= seuil_alpha]
            )  # retrieve the indices of the cells containing water
            probeYLocPlot.append(
                [
                    i
                    for ind_r, i in enumerate(probeYLoc)
                    if alphaNum[ind_r] >= seuil_alpha
                ]
            )

            if (
                np.size(np.shape(kPlot)) > 1
            ):  # test if the vector is empty (not enough water)
                knumPlot.append(kPlot)
            else:
                knumPlot.append([])
            tNumPlot.append(timeProbe[ind])

        plt.plot(knumPlot[0], probeYLocPlot[0], color=clmap[ind_path])

    # plt.yscale('log')

    # Setting axis labels
    plt.xlabel("$k$ (m²/s²)", fontsize=20, usetex=True)
    plt.ylabel("$y$ (m)", fontsize=20, usetex=True)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlim(XLim)
    # plt.title(probe_name)

    # add grid and legend
    plt.grid()
    # plt.ylim((0, 0.2)
    # plt.legend(bbox_to_anchor=(0, -0.18), loc="upper left")

    # show
    # plt.show()

    # Save figure
    if sauvegarde:
        plt.savefig(
            dir_path
            + "\\"
            + titre_sauv
            + "_t="
            + str(timePlot[0])
            + "_sonde"
            + str(i_name + 1)
            + ".pdf",
            bbox_inches="tight",
        )
