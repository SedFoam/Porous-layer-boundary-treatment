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
path_exp = "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Ensemble-Averaged Results\\"
name_exp = "IMP"
name_exp_size = "015"

paths = [
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.0\\3D_rug13_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V0.2.5\\2D_rug13_CP_lin_h=0.35xd50\\",
]

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]

nom_plots = [
    "Fuhrman 3D (interIso V2.0)",
    "CP 3D (interIso V2.1)",
    "CP 2D (interIso V0.2.5)",
]
colormap = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "error_swash_lens"

seuil_alpha = 0.5
slope = 1 / 10
theta = np.arctan(slope)

xOrigine = 5.806
yOrigine = 0.062


clmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
clmap = ["#000000", clmap[0], clmap[1], clmap[2]]

temps = np.linspace(0.05, 10, 200)  # time list
temps_str = []
for t in temps:
    t = round(t, 2)
    if str(t)[-2::] == ".0":
        temps_str.append(str(t)[0:-2])
    else:
        temps_str.append(str(t))

gs = gridspec.GridSpec(3, 1)


### Code ######################################################################

# Loading experimental data
mat = scipy.io.loadmat(path_exp + "h_lens_" + name_exp + ".mat")
tExp = mat.get("t_lens_" + name_exp)[0]
hExp = mat.get("h_lens_" + name_exp + "_" + name_exp_size)
xExp = mat.get("x_lens_" + name_exp + "_" + name_exp_size)[0]

RMSE = np.zeros((len(tExp), len(paths)))
BSS = np.zeros((len(tExp), len(paths)))  # Brier skill score
WSS = np.zeros((len(tExp), len(paths)))  # Willmott skill score

for ind_temps, tempsExp in enumerate(tExp):  # boucle sur le temps des expé

    print("t = " + str(tempsExp))
    hTempsExp = hExp[ind_temps]

    for ind_path, path in enumerate(paths):

        ind = np.abs(temps - tempsExp).argmin()  # closest time index to the target time

        # Loading simulation data
        x, y, z = ff.readmesh(path)
        alpha = ff.readscalar(path, temps_str[ind], "alpha.water")

        # projection to 2D if needed
        z = np.round(z, 8)
        if len(np.unique(z)) > 1:
            z0 = min(abs(np.unique(z)))
            x = x[z == z0]
            y = y[z == z0]
            alpha = alpha[z == z0]

        # moving the origin to that of the simulation data
        x0 = x - xOrigine
        y0 = y - yOrigine
        hlens = [0] * len(y0)

        # from h to h_lens
        for ind_y, y_curr in enumerate(y0):
            hlens[ind_y] = y_curr - x0[ind_y] / 10

        plt.figure(2)
        cont = plt.tricontourf(x0, hlens, alpha, [0.45, 0.55], colors=clmap[ind_path])

        lim_contours = np.where(cont.collections[0].get_paths()[0].codes == 1)[0]
        vertices_contours = cont.collections[0].get_paths()[0].vertices

        if len(lim_contours) == 1:  # if there is only one contour at this time step
            contour = vertices_contours
        else:
            taille_contours = lim_contours[1:] - lim_contours[0:-1]
            pos_biggest_contours = np.argmax(
                taille_contours
            )  # retrieve the position of the biggest contour
            contour = vertices_contours[
                lim_contours[pos_biggest_contours] : lim_contours[
                    pos_biggest_contours + 1
                ]
            ]

        # we use only half the contour to avoid interpolation issues
        ind_half_contour = int((len(contour[:, 0]) - 1) / 2)
        xCont = contour[ind_half_contour:, 0]
        yCont = contour[ind_half_contour:, 1]

        # removes the contour below the swash lens
        ind_max = np.argmax(xCont)

        if xCont[0] > xCont[-1]:
            xCont = xCont[ind_max:]
            yCont = yCont[ind_max:]
        else:
            xCont = xCont[0:ind_max]
            yCont = yCont[0:ind_max]

        # search for the overlap zone of the result vectors (num and exp)
        xContRedMin = xCont[xCont > xExp[0]]
        yContRedMin = yCont[xCont > xExp[0]]
        xContRedMinMax = xContRedMin[xContRedMin < xExp[-1]]
        yContRedMinMax = yContRedMin[xContRedMin < xExp[-1]]

        if xContRedMinMax[0] > xContRedMinMax[-1]:  # flip if needed
            xContRedMinMax = np.flip(xContRedMinMax)
            yContRedMinMax = np.flip(yContRedMinMax)

        xExpRed = xExp[xExp > xContRedMinMax[0]]
        hTempsExpRed = hTempsExp[xExp > xContRedMinMax[0]]
        xExpRedRed = xExpRed[xExpRed < xContRedMinMax[-1]]
        hTempsExpRedRed = hTempsExpRed[xExpRed < xContRedMinMax[-1]]

        lensInterp = np.interp(xExpRedRed, xContRedMinMax, yContRedMinMax)

        # calculate error and skill scores
        RMSE[ind_temps, ind_path] = np.sqrt(
            sum((hTempsExpRedRed - lensInterp) ** 2) / len(lensInterp)
        )
        BSS[ind_temps, ind_path] = 1 - sum((hTempsExpRedRed - lensInterp) ** 2) / sum(
            (hTempsExpRedRed - np.mean(hTempsExpRedRed)) ** 2
        )
        WSS[ind_temps, ind_path] = 1 - sum((hTempsExpRedRed - lensInterp) ** 2) / sum(
            (
                abs(hTempsExpRedRed - np.mean(hTempsExpRedRed))
                + abs(lensInterp - np.mean(hTempsExpRedRed))
            )
            ** 2
        )


fig = plt.figure(num=1, figsize=(8, 16), dpi=300)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[2, 0])

for i in range(np.shape(RMSE)[1]):

    ax1.plot(
        tExp,
        RMSE[:, i],
        color=colormap[i],
        marker="+",
        label=nom_plots[i] + " : mean error = " + str(np.mean(RMSE[:, i])),
    )
    ax1.scatter(
        5,
        np.mean(RMSE[:, i]),
        color=colormap[i],
        marker="o",
        edgecolor="black",
        label="_Hidden label",
    )

    ax2.plot(
        tExp,
        BSS[:, i],
        color=colormap[i],
        marker="+",
        label=nom_plots[i] + " : mean BSS = " + str(np.mean(BSS[:, i])),
    )
    ax2.scatter(
        5,
        np.mean(BSS[:, i]),
        color=colormap[i],
        marker="o",
        edgecolor="black",
        label="_Hidden label",
    )

    ax3.plot(
        tExp,
        WSS[:, i],
        color=colormap[i],
        marker="+",
        label=nom_plots[i] + " : mean WSS = " + str(np.mean(WSS[:, i])),
    )
    ax3.scatter(
        5,
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
