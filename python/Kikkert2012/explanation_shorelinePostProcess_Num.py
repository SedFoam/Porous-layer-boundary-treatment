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

path = "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\"

nom_plots = [
    "interIsoFoam Fuhrman (ref 2D) (interIsoFoam V0.2)",
    "interIsoFoam couchePoreuse 0.2xd50 (interIsoFoam V0.2.2 0.2xd50)",
    "Données expérimentales",
]

sauvegarde = True
titre_sauv = "explications_shorelineMoyenneGlissante_Num"


bin_width = 380
bin_width_num = 28

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


XLim = [(4, 4.4), (-0.5, 1.5)]
YLim = [(-0.0025, 0.02), (0, 0.015)]

temps_str = ["4.5", "9.95"]

seuil_alpha = 0.5
pm_seuil = 0.005

xOrigine = 5.806
yOrigine = 0.062

### Code ######################################################################


list_intersec_num = []

# beach probe construction
sonde_plage = np.array([[-1, L_sonde], [8, L_sonde]])

# Loading simulation data
x, y, z = ff.readmesh(path)

# projection in 2D if needed
z = np.round(z, 8)
if len(np.unique(z)) > 1:
    z0 = min(abs(np.unique(z)))
    x = x[z == z0]
    y = y[z == z0]

# moving the origin to that of the simulation data
x0 = x - xOrigine
y0 = y - yOrigine
hlens = [0] * len(y0)

# from h to h_lens
for ind_y, y_curr in enumerate(y0):
    hlens[ind_y] = y_curr - x0[ind_y] / 10


for ind_t, time in enumerate(temps_str):

    alpha = ff.readscalar(path, time, "alpha.water")

    if len(np.unique(z)) > 1:
        alpha = alpha[z == z0]

    plt.figure(2)
    cont = plt.tricontourf(
        x0,
        hlens,
        alpha,
        [seuil_alpha - pm_seuil, seuil_alpha + pm_seuil],
        colors=clmap[ind_t],
    )

    lim_contours = np.where(cont.collections[0].get_paths()[0].codes == 1)[0]
    vertices_contours = cont.collections[0].get_paths()[0].vertices

    taille_contours = lim_contours[1:] - lim_contours[0:-1]

    if len(taille_contours) > 0:
        pos_biggest_contours = np.argmax(
            taille_contours
        )  # retrieve the position of the biggest contour
        contour = vertices_contours[
            lim_contours[pos_biggest_contours] : lim_contours[pos_biggest_contours + 1]
        ]
    else:
        contour = vertices_contours

    # we use half of the contour to avoid interpolation issues
    ind_half_contour = int((len(contour[:, 0]) - 1) / 2)
    xCont = contour[ind_half_contour:, 0]
    yCont = contour[ind_half_contour:, 1]

    if xCont[0] > xCont[-1]:  # flip vectors if needed
        xCont = np.flip(xCont)
        yCont = np.flip(yCont)

    # # remove the contour below the swash lens
    # ind_max = np.argmax(xCont)

    # xCont = xCont[0:ind_max]
    # yCont = yCont[0:ind_max]

    for j in range(len(yCont) - 1):
        intersecBrut = intersect(
            sonde_plage[0],
            sonde_plage[1],
            np.array([xCont[j], yCont[j]]),
            np.array([xCont[j + 1], yCont[j + 1]]),
        )
        if intersecBrut is not None:
            break

    # bin experimental data
    xNumBin = []
    val_numBin = []
    compt = 0
    tempX = 0
    tempH = 0
    for ind, val in enumerate(xCont):
        if compt < bin_width_num:
            compt += 1
            tempX += xCont[ind]
            tempH += yCont[ind]
        else:
            compt = 0
            xNumBin.append(tempX / bin_width_num)
            val_numBin.append(tempH / bin_width_num)
            tempX = 0
            tempH = 0

    if float(time) > 6.5:

        inter = 0
        for i in range(len(val_numBin) - 1):
            intersec = intersect(
                sonde_plage[0],
                sonde_plage[1],
                np.array([xNumBin[i], val_numBin[i]]),
                np.array([xNumBin[i + 1], val_numBin[i + 1]]),
            )
            if intersec is not None:
                inter = 1

                ind_brut_min = None
                dist_min = 1000
                for ind_brut, val_brut in enumerate(yCont):
                    dist = (val_brut - intersec[1]) ** 2 + (
                        xCont[ind_brut] - intersec[0]
                    ) ** 2
                    if dist < dist_min:
                        dist_min = dist
                        ind_brut_min = ind_brut

                # Retrieve experimental points at ±0.5 m from the intersection found with the binned data
                xRed = xCont[xCont > xCont[ind_brut_min] - 0.5]
                valRed = yCont[xCont > xCont[ind_brut_min] - 0.5]
                valRed = valRed[xRed < xCont[ind_brut_min] + 0.5]
                xRed = xRed[xRed < xCont[ind_brut_min] + 0.5]

                # fit raw data
                fit = np.polyfit(xRed, valRed, 1)
                xFit = np.array([xRed[0], xRed[-1]])
                valFit = fit[0] * xFit + fit[1]

                # calculation of the intersection on the fitted raw data
                intersecFit = intersect(
                    sonde_plage[0],
                    sonde_plage[1],
                    np.array([xFit[0], valFit[0]]),
                    np.array([xFit[1], valFit[1]]),
                )

                break

    plt.figure(ind_t, figsize=(7, 6), dpi=300)

    plt.plot(xCont, yCont, color="grey", zorder=1, linewidth=2)
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

    if ind_t == 1:
        plt.plot(
            xNumBin,
            val_numBin,
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
            intersecFit[0],
            intersecFit[1],
            zorder=10,
            color="red",
            edgecolor="white",
            s=80,
        )
        plt.plot(xRed, valRed, color="black", zorder=2, linewidth=2)

    plt.xlim(XLim[ind_t])
    plt.ylim(YLim[ind_t])

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
            dir_path + "\\" + titre_sauv + "_" + str(ind_t) + ".pdf",
            bbox_inches="tight",
        )
