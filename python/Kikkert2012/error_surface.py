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
import matplotlib.gridspec as gridspec


# Input data
dir_path = os.path.dirname(os.path.realpath(__file__))
path_exp = "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Depth-Averaged Results\\"
name_exp = "IMP_015_PIV"

paths = [
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.0\\3D_rug13_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V0.2.5\\2D_rug13_CP_lin_h=0.35xd50\\",
]

nom_plots = [
    "Fuhrman 3D (interIso V2.0)",
    "CP 3D (interIso V2.1)",
    "CP 2D (interIso V0.2.5)",
]
colormap = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
probe_pos = np.array([-1.802, 0.072, 0.772, 1.567, 2.377, 3.177])
timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "error_surface"

slope = 1 / 10
theta = np.arctan(slope)

gs = gridspec.GridSpec(3, 1)

### Code ######################################################################

RMSE = np.zeros((len(probe_names), len(paths)))
BSS = np.zeros((len(probe_names), len(paths)))  # Brier skill score
WSS = np.zeros((len(probe_names), len(paths)))  # Willmott skill score

probes_x = []

for ind_name, probe_name in enumerate(probe_names):

    print("Création figure " + probe_name)

    # Loading experimental data
    try:
        mat = scipy.io.loadmat(path_exp + "U_" + name_exp + probe_name[-1] + ".mat")
        t = mat.get("t_" + name_exp + probe_name[-1])[0]
        h = mat.get("h_" + name_exp + probe_name[-1])[0]
    except:
        break

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

        # Calculation of error and skill scores
        interpNumH = np.interp(t, timeProbe, surf)
        RMSE[ind_name, ind_nom] = np.sqrt(sum((h - interpNumH) ** 2) / len(interpNumH))
        BSS[ind_name, ind_nom] = 1 - sum((h - interpNumH) ** 2) / sum(
            (h - np.mean(h)) ** 2
        )
        WSS[ind_name, ind_nom] = 1 - sum((h - interpNumH) ** 2) / sum(
            (abs(h - np.mean(h)) + abs(interpNumH - np.mean(h))) ** 2
        )

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
        label=nom_plots[i] + " : mean error = " + str(np.mean(RMSE[:, i])),
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
        label=nom_plots[i] + " : mean BSS = " + str(np.mean(BSS[:, i])),
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
        label=nom_plots[i] + " : mean WSS = " + str(np.mean(WSS[:, i])),
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

# show
# plt.show()


# Save figure
if sauvegarde:
    plt.savefig(dir_path + "\\" + titre_sauv + ".pdf", bbox_inches="tight")
