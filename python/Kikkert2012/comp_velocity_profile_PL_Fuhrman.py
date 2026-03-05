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


# Input data
dir_path = os.path.dirname(os.path.realpath(__file__))
path_exp = "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Ensemble-Averaged Results\\"
name_exp = "IMP_015_PIV"

paths = [
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.0\\3D_rug13_h0=0.59_geomCorr\\",
]

nom_legend = [
    "Référence 2D - density variable (interFoam V2.7.1)",
    "sed original (sedInterFoam V0.2)",
    "sed avec terme puit de k (sedInterFoam V1.0 noOmegaSourceTerm)",
    "Données expérimentales",
]

color_paths = ["blue", "red", "orange", "green", "cyan", "m", "y"]

clmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
clmap = [clmap[2], "#000000"]
taille_plot = [3, 1.5]
style_trait = ["-", "--"]

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "profils_vitesse_CP_Fuhrman_rug13"

slope = 1 / 10
theta = np.arctan(slope)

seuil_alpha = 0.5

timePlot = np.linspace(3, 6.9, 7)  # time steps to plot


### Code ######################################################################

for i_name, probe_name in enumerate(probe_names):

    print("Création figure " + probe_name)

    # Loading experimental data
    mat = scipy.io.loadmat(
        path_exp + "U_profiles_" + name_exp + probe_name[-1] + ".mat"
    )
    texp = mat.get("t_" + name_exp + probe_name[-1])[0]
    Uexp = mat.get("U_profiles_" + name_exp + probe_name[-1])
    Yexp = mat.get("z_" + name_exp + probe_name[-1])[0]

    # getting experimental time steps to plot
    UexpPlot = []
    tExpPlot = []
    for temps in timePlot:
        ind = np.abs(texp - temps).argmin()  # closest time index to the target time
        UexpPlot.append(Uexp[:, ind])  # get experimental U
        tExpPlot.append(texp[ind])  # getting the exact time

    fig = plt.figure()
    ax = fig.add_subplot()

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
        UnumPlot = []
        tNumPlot = []
        probeYLocPlot = []
        laminar = 0
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
                )  # calculate bed-parallel velocity
                UnumPlot.append(UPlotParr)
            else:
                UnumPlot.append([])
            tNumPlot.append(timeProbe[ind])

        # Plot simulation data
        for ind_t, t in enumerate(timePlot):
            ax.plot(
                UnumPlot[ind_t],
                probeYLocPlot[ind_t],
                color=clmap[ind_path],
                label="_Hidden label",
                linewidth=taille_plot[ind_path],
                linestyle=style_trait[ind_path],
            )

    # Plot experimental data
    for ind_t, t in enumerate(timePlot):
        ax.scatter(
            UexpPlot[ind_t],
            Yexp,
            80,
            marker="+",
            facecolors="red",
            label="_Hidden label",
        )

        ind_Uexp = (
            (~np.isnan(UexpPlot[ind_t])).cumsum(0).argmax(0)
        )  # retrieves the index of the last non-nan value
        if (
            ind_Uexp != 0
        ):  # The time step is only plotted if there is a non-nan value in the serie
            try:
                ax.text(
                    UexpPlot[ind_t][ind_Uexp],
                    np.max((np.max(probeYLocPlot[ind_t]), np.max(Yexp[ind_Uexp])))
                    + 0.02
                    - 0.01 / 6 * i_name,
                    str(round(tExpPlot[ind_t], 2)),
                    verticalalignment="center",
                    horizontalalignment="center",
                    bbox=dict(facecolor="white", edgecolor="white"),
                )
            except:
                1

    # Setting axis labels
    ax.set_xlabel("$U_x$ (m/s)", fontsize=20, usetex=True)
    ax.set_ylabel("$y$ (m)", fontsize=20, usetex=True)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # ax.set_title("Profils de vitesse : "+probe_name)

    # add grid and legend
    maxX1, maxX2 = ax.get_xlim()
    ax.set_xlim([maxX1, maxX2 + 0.023])
    maxY1, maxY2 = ax.get_ylim()
    ax.set_ylim([maxY1, maxY2 + 0.023])
    # Plot x=0
    ax.plot(
        [0, 0], [-1, 1], "--", color="dimgrey", scaley=False, zorder=-10, linewidth=2
    )
    plt.grid()
    # plt.ylim((0, 0.2)
    # plt.legend(symb_legend, nom_legend, bbox_to_anchor=(0, -0.18), loc="upper left")

    # show
    # plt.show()

    # Save figure
    if sauvegarde:
        plt.savefig(
            dir_path + "\\" + titre_sauv + "_sonde" + str(i_name + 1) + ".pdf",
            bbox_inches="tight",
        )
