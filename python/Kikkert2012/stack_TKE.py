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
import scipy.interpolate as si
import netCDF4 as nc
import math as mt


def roundPartial(value, resolution):
    return mt.floor(value / resolution) * resolution


# Input data
dir_path = os.path.dirname(os.path.realpath(__file__))

paths = [
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V0.2.5\\2D_rug13_CP_lin_h=0.35xd50\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.0\\3D_rug13_h0=0.59_geomCorr\\",
]

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "stack_TKE"
nom_sauv = ["CP_3D", "CP_2D", "Fuhrman_3D"]

seuil_alpha = 0.5
slope = 1 / 10
theta = np.arctan(slope)

clmap = "turbo"

# time list
temps = np.linspace(0.05, 10, 200)
temps_str = []
for t in temps:
    t = round(t, 2)
    if str(t)[-2::] == ".0":
        temps_str.append(str(t)[0:-2])
    else:
        temps_str.append(str(t))


### Code ######################################################################


plt.figure()

for ind_path, path in enumerate(paths):

    # Loading shoreline data
    NCfile_SL = nc.Dataset(path + "netcdf\\shoreline.nc")
    SL = np.array(NCfile_SL["shoreline position"])

    # Loading simulation data
    print("Chargement des données de simu :")

    NCfile = nc.Dataset(path + "netcdf\\interp_k.nc")
    print("  - chargement x...")
    x = np.array(NCfile["x"])
    print("  - chargement k...")
    k = np.array(NCfile["k"])
    # print('  - chargement gradU...')
    # gradU = np.array(NCfile['gradU'])

    k_cut = k

    for ind_t, k_sondes in enumerate(k):
        k_cut[ind_t, x > SL[ind_t]] = np.nan  # cut off values that exceed the shoreline

    k_cut = np.transpose(k_cut)

    # Plot results
    fig, ax = plt.subplots(dpi=300)
    cax = fig.add_axes([0.95, 0.125, 0.05, 0.75])

    if ind_path == 0:
        k_save = k_cut

    im = ax.contourf(
        temps,
        x / max(SL),
        k_cut,
        200,
        cmap=clmap,
        vmin=np.nanmin(k_save),
        vmax=np.nanmax(k_save),
    )

    if ind_path == 0:
        im_0 = im

    cbar = fig.colorbar(im_0, cax=cax, orientation="vertical")
    cbar.ax.set_title("<k> (m²/s²)", fontsize=16)

    if ind_path == 0:
        # colorbar settings
        max_k = np.nanmax(k_cut)
        closest = roundPartial(max_k, 0.02)
        Yticks = np.linspace(0, closest, int(closest / 0.02) + 1)
        Ylabel = [str(val) for val in Yticks]

    cbar.ax.set_yticks(Yticks, Ylabel, fontsize=13)

    # Setting axis labels
    ax.set_xlabel("t (s)", fontsize=16)
    ax.set_ylabel("x/x_max", fontsize=16)
    ax.set_xticks(np.array([2, 4, 6, 8, 10]), ["2", "4", "6", "8", "10"], fontsize=13)
    ax.set_yticks(
        np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.1]),
        ["0", "0.2", "0.4", "0.6", "0.8", "1", ""],
        fontsize=13,
    )
    ax.set_xlim((1.5, 10.5))
    # plt.title(probe_name)

    # add grid and legend
    # plt.ylim((0, 0.2)
    # plt.legend(bbox_to_anchor=(0, -0.18), loc="upper left")

    # show
    # plt.show()

    # Save figure
    if sauvegarde:
        plt.savefig(
            dir_path + "\\" + titre_sauv + "_" + nom_sauv[ind_path] + ".pdf",
            bbox_inches="tight",
        )
