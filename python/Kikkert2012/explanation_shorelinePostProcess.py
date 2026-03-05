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
from intersect import intersect


# Input data
dir_path = os.path.dirname(os.path.realpath(__file__))
path_exp = (
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Shoreline Position\\"
)
name_exp = "ShorePos_IMP_015"

path_exp2 = "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Ensemble-Averaged Results\\"
name_exp2 = "IMP"
name_exp_size2 = "015"


nom_plots = [
    "interIsoFoam Fuhrman (ref 2D) (interIsoFoam V0.2)",
    "interIsoFoam couchePoreuse 0.2xd50 (interIsoFoam V0.2.2 0.2xd50)",
    "Données expérimentales",
]

sauvegarde = True
titre_sauv = "explications_shorelineMoyenneGlissante_PlusGrande"


bin_width = 380

clmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
clmap = ["#000000", clmap[0]]

symb_legend = [
    Line2D([0], [0], label=nom_plots[0], color=clmap[0]),
    Line2D([0], [0], label=nom_plots[1], color=clmap[1]),
    # Line2D([0], [0], label=nom_plots[2], color='orange'),
    # Line2D([0], [0], label=nom_plots[3], color='green'),
    # Line2D([0], [0], label=nom_plots[4], color='c'),
    # Line2D([0], [0], label=nom_plots[5], color='m'),
    # Line2D([0], [0], label=nom_plots[6], color='y'),
    Line2D([0], [0], label=nom_plots[2], marker="+", color="red", linestyle=""),
]

# beach/probe distance
L_sonde = 0.005
sonde_plage = np.array([[-1, L_sonde], [8, L_sonde]])


temps_fig = np.array([60, 133])
XLim = [(4, 4.4), (-0.5, 1.5)]
YLim = [(-0.0025, 0.02), (0, 0.015)]


### Code ######################################################################


# Calculation of the shoreline from smoothed swash lenses
mat = scipy.io.loadmat(path_exp2 + "h_lens_" + name_exp2 + ".mat")
tExp = mat.get("t_lens_" + name_exp2)[0]
hExp = mat.get("h_lens_" + name_exp2 + "_" + name_exp_size2)
xExp = mat.get("x_lens_" + name_exp2 + "_" + name_exp_size2)[0]


list_intersec_res = []
for indFig, tFig in enumerate(temps_fig):
    i_exp = tFig
    val_exp = hExp[i_exp]

    for j in range(len(val_exp) - 1):
        intersecBrut = intersect(
            sonde_plage[0],
            sonde_plage[1],
            np.array([xExp[j], val_exp[j]]),
            np.array([xExp[j + 1], val_exp[j + 1]]),
        )
        if intersecBrut is not None:
            break

    plt.figure(figsize=(7, 6), dpi=300)

    # bin experimental data
    xExpBin = []
    val_expBin = []
    compt = 0
    tempX = 0
    tempH = 0
    for ind, val in enumerate(xExp):
        if compt < bin_width:
            compt += 1
            tempX += xExp[ind]
            tempH += val_exp[ind]
        else:
            compt = 0
            xExpBin.append(tempX / bin_width)
            val_expBin.append(tempH / bin_width)
            tempX = 0
            tempH = 0

    inter = 0
    for i in range(len(val_expBin) - 1):
        intersec = intersect(
            sonde_plage[0],
            sonde_plage[1],
            np.array([xExpBin[i], val_expBin[i]]),
            np.array([xExpBin[i + 1], val_expBin[i + 1]]),
        )
        if intersec is not None:
            inter = 1

            ind_brut_min = None
            dist_min = 1000
            for ind_brut, val_brut in enumerate(val_exp):
                dist = (val_brut - intersec[1]) ** 2 + (
                    xExp[ind_brut] - intersec[0]
                ) ** 2
                if dist < dist_min:
                    dist_min = dist
                    ind_brut_min = ind_brut

                # Retrieve experimental points at ±0.5 m from the intersection found with the binned data
                xRed = xExp[xExp > xExp[ind_brut_min] - 0.5]
                valRed = val_exp[xExp > xExp[ind_brut_min] - 0.5]
                valRed = valRed[xRed < xExp[ind_brut_min] + 0.5]
                xRed = xRed[xRed < xExp[ind_brut_min] + 0.5]

                # fit raw data
                fit = np.polyfit(xRed, valRed, 1)
                xFit = np.array([xRed[0], xRed[-1]])
                valFit = fit[0] * xFit + fit[1]

                # calculation of the intersection on the fitted raw data
                intersecBrutFit = intersect(
                    sonde_plage[0],
                    sonde_plage[1],
                    np.array([xFit[0], valFit[0]]),
                    np.array([xFit[1], valFit[1]]),
                )

            break

    plt.plot(xExp, val_exp, color="grey", zorder=1, linewidth=2)
    plt.plot([-1, 6], [L_sonde, L_sonde], "--g", zorder=3, linewidth=2)
    plt.scatter(
        intersecBrut[0],
        intersecBrut[1],
        zorder=10,
        color="green",
        edgecolor="white",
        s=80,
        marker="s",
    )

    if indFig == 1:
        plt.plot(
            xExpBin,
            val_expBin,
            color="orange",
            marker="o",
            markersize=3,
            zorder=8,
            linewidth=2,
        )
        plt.plot(xFit, valFit, color="red", zorder=9, linewidth=2)
        plt.scatter(
            intersec[0],
            intersec[1],
            zorder=10,
            color="darkorange",
            marker="d",
            edgecolor="white",
            s=80,
        )
        plt.scatter(
            intersecBrutFit[0],
            intersecBrutFit[1],
            zorder=10,
            color="red",
            edgecolor="white",
            s=80,
        )
        plt.plot(xRed, valRed, color="black", zorder=2, linewidth=2)

    plt.xlim(XLim[indFig])
    plt.ylim(YLim[indFig])

    # Setting axis labels
    plt.xlabel("$x$ (m)", fontsize=18, usetex=True)
    plt.ylabel("$y$ (m)", fontsize=18, usetex=True)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.title(probe_name)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))

    # add grid and legend
    plt.grid()
    plt.rc("axes", axisbelow=True)
    # plt.ylim((0, 0.2)
    # plt.legend(symb_legend, nom_plots, bbox_to_anchor=(0, -0.18), loc="upper left")

    # show
    # plt.show()

    # Save figure
    if sauvegarde:
        plt.savefig(
            dir_path + "\\" + titre_sauv + "_" + str(indFig) + ".pdf",
            bbox_inches="tight",
        )
