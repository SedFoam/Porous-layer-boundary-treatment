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

plt.rcParams["figure.dpi"] = 300

# Input data
dir_path = os.path.dirname(os.path.realpath(__file__))
path_exp = "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Ensemble-Averaged Results\\"
name_exp = "IMP"
name_exp_size = "015"

paths = [
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V0.2.5\\2D_rug13_CP_lin_h=0.35xd50\\",
]

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "Swash_lens_rug13_2D_3D"

seuil_alpha = 0.5
slope = 1 / 10
theta = np.arctan(slope)
errorBar = 0.01  # mesh size around the free surface

temps = 9.5  # plot time
xOrigine = 5.806
yOrigine = 0.062

bin_width = 200

clmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
clmap = [clmap[2], "#000000"]
taille_plot = [3, 1.5]
style_trait = ["-", "-."]


### Code ######################################################################

# Loading experimental data
mat = scipy.io.loadmat(path_exp + "h_lens_" + name_exp + ".mat")
tExp = mat.get("t_lens_" + name_exp)
hExp = mat.get("h_lens_" + name_exp + "_" + name_exp_size)
xExp = mat.get("x_lens_" + name_exp + "_" + name_exp_size)

# getting the time steps to plot
ind = np.abs(tExp - temps).argmin()  # closest time index to the target time
hExpPlot = hExp[ind]
tExpPlot = tExp[0, ind]  # getting exact time

# bin experimental data
xExpBin = []
hExpPlotBin = []
compt = 0
tempX = 0
tempH = 0
for ind, val in enumerate(xExp[0]):
    if compt < bin_width:
        compt += 1
        tempX += xExp[0, ind]
        tempH += hExpPlot[ind]
    else:
        compt = 0
        xExpBin.append(tempX / bin_width)
        hExpPlotBin.append(tempH / bin_width)
        tempX = 0
        tempH = 0


# Plot swash lens
plt.figure(1)
plt.scatter(
    xExpBin,
    hExpPlotBin,
    color="red",
    marker="+",
    label="Données expérimentales",
    linewidths=2,
    s=80,
)

for ind_path, path in enumerate(paths):
    # Loading simulation data
    x, y, z = ff.readmesh(path)
    alpha = ff.readscalar(path, str(temps), "alpha.water")

    # projection from 3D to 2D if needed
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

    taille_contours = lim_contours[1:] - lim_contours[0:-1]
    pos_biggest_contours = np.argmax(
        taille_contours
    )  # retrieve the position of the biggest contour

    contour = vertices_contours[
        lim_contours[pos_biggest_contours] : lim_contours[pos_biggest_contours + 1]
    ]

    ind_half_contour = int((len(contour[:, 0]) - 1) / 2)
    xCont = contour[ind_half_contour:, 0]
    yCont = contour[ind_half_contour:, 1]

    plt.figure(1)
    plt.plot(
        xCont,
        yCont,
        color=clmap[ind_path],
        linewidth=taille_plot[ind_path],
        linestyle=style_trait[ind_path],
    )

plt.xlim((-1, 6.5))
plt.ylim((-0.01, 0.25))

# Setting axis labels
plt.xlabel("$x$ (m)", fontsize=20, usetex=True)
plt.ylabel("$h$ (m)", fontsize=20, usetex=True)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
# plt.title(probe_name)

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
        dir_path + "\\" + titre_sauv + "_t=" + str(temps) + ".pdf", bbox_inches="tight"
    )
