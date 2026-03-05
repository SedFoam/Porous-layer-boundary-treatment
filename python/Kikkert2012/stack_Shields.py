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
import scipy.ndimage as nd
import netCDF4 as nc
import math as mt
from mpl_toolkits.axes_grid1 import AxesGrid

import matplotlib.cm as cm
import matplotlib.colors as mcolors


### Fonctions #################################################################


def roundPartial(value, resolution):
    return mt.floor(value / resolution) * resolution


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si2 in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si2, r, r))
        cdict["green"].append((si2, g, g))
        cdict["blue"].append((si2, b, b))
        cdict["alpha"].append((si2, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap, override_builtin=True)

    return newcmap


###############################################################################


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

sauvegarde = False
titre_sauv = "stack_tau_Shields"
nom_sauv = ["CP_3D", "CP_2D", "Fuhrman_3D"]
lissage = 1  # 1 to smooth the data, 0 otherwise
cut_SL = 1
normalise_long = 0  # 1 to normalise by the maximum runup length, 0 otherwise

seuil_alpha = 0.5
slope = 1 / 10
theta = np.arctan(slope)

clmap = "turbo"  # turbo
# clmap = shiftedColorMap(plt.get_cmap(clmap), 0, 0.2, 1)

# time list
temps = np.linspace(0.05, 10, 200)
temps_str = []
for t in temps:
    t = round(t, 2)
    if str(t)[-2::] == ".0":
        temps_str.append(str(t)[0:-2])
    else:
        temps_str.append(str(t))


str_tau = ""  # figure to plot : '' (max absolute value in the column - positive and negative) '_max' (max value) '_min' (min value)

clm_val = [-1, 4]  # colormap trim

rho_s = 2650  # kg/m^3
rho_w = 1000  # kg/m^3
g = 9.81  # m/s^-2
D = 0.0013  # m

### Code ######################################################################


plt.figure()

for ind_path, path in enumerate(paths):

    # Loading shoreline data
    NCfile_SL = nc.Dataset(path + "netcdf\\shoreline.nc")
    SL = np.array(NCfile_SL["shoreline position"])

    # Loading simulation data
    print("Chargement des données de simu :")

    NCfile = nc.Dataset(path + "netcdf\\interp_tau.nc")

    print("  - chargement x...")
    x = np.array(NCfile["x"])
    print("  - chargement tau" + str_tau + "...")
    tau = np.array(NCfile["tau" + str_tau])

    # print('  - chargement gradU...')
    # gradU = np.array(NCfile['gradU'])

    tau_cut = tau

    tau_cut = np.transpose(tau_cut)
    # interpolation to fill gaps in tau_cut
    tau_cut_int = tau_cut
    for i in range(np.shape(tau_cut)[0]):
        for j in range(np.shape(tau_cut)[1]):
            if tau_cut[i, j] == 0:
                if j > 0 & j < np.shape(tau_cut)[1] - 1:
                    tau_cut_int[i, j] = (tau_cut[i, j - 1] + tau_cut[i, j + 1]) / 2
                elif j == 0:
                    tau_cut_int[i, j] = 2 * tau_cut[i, j + 1] - tau_cut[i, j + 2]
                elif j == np.shape(tau_cut)[1] - 1:
                    tau_cut_int[i, j] = 2 * tau_cut[i, j - 1] - tau_cut[i, j - 2]

    tau_cut = tau_cut_int

    # smooth data if needed
    if lissage:
        tau_cut_filter = nd.filters.gaussian_filter(
            tau_cut, [1.5, 1.5], mode="nearest", truncate=4
        )

        # We put the smoothed values in tau_cut (and keep the initial values at the edges that are cut off by the smoothing)
        for i in range(np.shape(tau_cut_filter)[0]):
            for j in range(np.shape(tau_cut_filter)[1]):
                if not np.isnan(tau_cut_filter[i, j]):
                    tau_cut[i, j] = tau_cut_filter[i, j]

    # values beyond the SL are cut if necessary
    if cut_SL:
        for ind_t, tau_sondes in enumerate(tau):
            tau_cut[x > SL[ind_t], ind_t] = np.nan  # cut values beyond the shoreline

    print("  - plot...")

    # Plot results
    fig, ax = plt.subplots(dpi=300)

    if ind_path == 0:
        tau_save = tau_cut

    tau_cut = tau_cut / ((rho_s - rho_w) * g * D)  # calculate Shields number

    # plot
    if normalise_long == 1:
        im = ax.contourf(
            temps,
            x / max(SL),
            tau_cut,
            1000,
            cmap=clmap,
            vmin=clm_val[0],
            vmax=clm_val[1],
        )
        im2 = ax.contour(
            temps,
            x / max(SL),
            tau_cut,
            1000,
            cmap=clmap,
            vmin=clm_val[0],
            vmax=clm_val[1],
            linewidths=0.1,
        )
    else:
        im = ax.contourf(
            temps, x, tau_cut, 1000, cmap=clmap, vmin=clm_val[0], vmax=clm_val[1]
        )
        im2 = ax.contour(
            temps,
            x,
            tau_cut,
            1000,
            cmap=clmap,
            vmin=clm_val[0],
            vmax=clm_val[1],
            linewidths=0.1,
        )
        im3 = ax.contour(
            temps, x, tau_cut, levels=[-0.1, 0.1], colors="k", linewidths=1
        )
        # ax.clabel(im3, fontsize=9)

    # ax.text(2, 1.1, 'tau '+str_tau, horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='white'))

    if ind_path == 0:
        im_0 = im

    norm = mcolors.Normalize(vmin=clm_val[0], vmax=clm_val[1])
    mappable = cm.ScalarMappable(norm=norm, cmap=clmap)
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.ax.set_title("$\\theta$", fontsize=18, usetex=True, position=(0.5, 0))

    # if ind_path == 0:
    #     # colorbar settings
    #     max_tau = np.nanmax(tau_cut)
    #     closest = roundPartial(max_tau, 0.02)
    #     Yticks = np.linspace(0, closest, int(closest/0.02)+1)
    #     Ylabel = [str(val) for val in Yticks]

    # cbar.ax.set_yticks(Yticks, Ylabel, fontsize=13)

    # Setting axis labels
    ax.set_xlabel("$t$ (s)", fontsize=18, usetex=True)

    if normalise_long == 1:
        ax.set_ylabel("$x/x_{max}$", fontsize=18, usetex=True)
        ax.set_yticks(
            np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.1]),
            ["0", "0.2", "0.4", "0.6", "0.8", "1", ""],
            fontsize=13,
        )
    else:
        ax.set_ylabel("$x$ (m)", fontsize=18, usetex=True)

    ax.set_xticks(np.array([2, 4, 6, 8, 10]), ["2", "4", "6", "8", "10"], fontsize=13)
    ax.set_xlim((1.5, 10.5))
    # plt.title(probe_name)

    # add grid and legend
    # plt.ylim((0, 0.2)
    # plt.legend(bbox_to_anchor=(0, -0.18), loc="upper left")

    # show
    # plt.show()

    # Save figure
    if sauvegarde:
        print("  - sauvegarde...")
        plt.savefig(
            dir_path + "\\" + titre_sauv + str_tau + "_" + nom_sauv[ind_path] + ".pdf",
            bbox_inches="tight",
        )
