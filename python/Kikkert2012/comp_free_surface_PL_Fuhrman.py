# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:28:28 2024

@author: kaczmare3m
"""

import os
import inspect
import numpy as np
import fluidfoam as ff
import matplotlib
import matplotlib.pyplot as plt
import scipy.io


# Input data
dir_path = os.path.dirname(os.path.realpath(__file__))
path_exp = "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Depth-Averaged Results\\"
name_exp = "IMP_015_PIV"

paths = [
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.0\\3D_rug13_h0=0.59_geomCorr\\",
]

nom_plots = [
    "Référence 2D - density variable (interFoam V2.7.1)",
    "sed original (sedInterFoam V0.2)",
    "sed, k complet, sans omega (sedInterFoam V1.0 kComplet noOmega)",
]

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "surface_libre_CP_Fuhrman_rug13"

slope = 1 / 10
theta = np.arctan(slope)
errorBar = 0.01  # mesh size around the free surface

clmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
clmap = [clmap[2], "#000000"]
taille_plot = [3, 1.5]
style_trait = ["-", "--"]

# error bars
tBarre = [[8.1, 3, 3.7, 4.2, 4.4, 4.25], [7.70, 2.6, 3.3, 3.7, 4.05, 4.75]]

############### Code ##########################################################

for i_name, probe_name in enumerate(probe_names):

    print("Création figure " + probe_name)

    # Loading experimental data
    mat = scipy.io.loadmat(path_exp + "U_" + name_exp + probe_name[-1] + ".mat")
    t = mat.get("t_" + name_exp + probe_name[-1])
    h = mat.get("h_" + name_exp + probe_name[-1])
    if probe_name[-1] == "6":  # aligning points of probe 6
        h = h - np.mean(h[0, 0:14])

    # Plotting free surface height evolution over time
    plt.figure()

    for ind_nom, path in enumerate(paths):
        # Loading simulation data
        varName, probe_loc, timeProbe, varProbe = ff.readsampling(
            path=path + postProcessFolder + "\\",
            probes_name=probe_name,
            time_name=timeStep,
        )

        surf = np.zeros(len(timeProbe))
        for i in range(len(timeProbe)):
            alpha = np.concatenate(varProbe[i, 0])
            try:
                ind_surf = max(np.where(np.diff(np.sign(alpha - 0.5)))[0])
            except:
                ind_surf = 0
            # Rotating the reference frame by ~5.71° to align it with the experimental reference frame
            surf[i] = (probe_loc[ind_surf, 1] - probe_loc[0, 1]) * np.cos(theta) - (
                probe_loc[ind_surf, 0] - probe_loc[0, 0]
            ) * np.sin(theta)

        plt.plot(
            timeProbe,
            surf,
            color=clmap[ind_nom],
            label=nom_plots[ind_nom],
            linewidth=taille_plot[ind_nom],
            linestyle=style_trait[ind_nom],
        )
        # plt.fill_between(timeProbe, surf-errorBar, surf+errorBar, alpha=0.25, label='_Hidden label')

        yBarre = surf[int((tBarre[ind_nom][i_name]) * 20 - 1)]
        plt.errorbar(
            tBarre[ind_nom][i_name],
            yBarre,
            xerr=[0],
            yerr=[0.005],
            capsize=4,
            capthick=3,
            ecolor=clmap[ind_nom],
            marker="none",
            linestyle="none",
            zorder=10,
        )  # plot error bars

    plt.scatter(
        t,
        h,
        color="red",
        marker="+",
        label="Données expérimentales",
        linewidths=2,
        s=80,
    )
    # plt.fill_between(t[0], h[0]-0.01, h[0]+0.01, alpha=0.2, color=clmap[2], label='_Hidden label')

    # Setting axis labels
    plt.xlabel("$t$ (s)", fontsize=20, usetex=True)
    plt.ylabel("$y$ (m)", fontsize=20, usetex=True)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
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
            dir_path + "\\" + titre_sauv + "_sonde" + str(i_name + 1) + ".pdf",
            bbox_inches="tight",
        )
