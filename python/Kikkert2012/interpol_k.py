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


# Input data
dir_path = os.path.dirname(os.path.realpath(__file__))

paths = [  #'Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V0.2.5\\2D_rug13_CP_lin_h=0.35xd50\\',
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.0\\3D_rug13_h0=0.59_geomCorr\\"
]  # ,
#'Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\']

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
timeStep = "mergeTime"

sauvegarde = False
titre_sauv = "interp_k.nc"

seuil_alpha = 0.5
slope = 1 / 10
theta = np.arctan(slope)

clmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
clmap = ["#000000", clmap[0]]

# time list
temps = np.linspace(0.05, 10, 200)
temps_str = []
for t in temps:
    t = round(t, 2)
    if str(t)[-2::] == ".0":
        temps_str.append(str(t)[0:-2])
    else:
        temps_str.append(str(t))


# Parameters
N = 601  # 501
pente = 0.1

x0 = 0  # initial water position
y0 = 0

xf = 6  # 5 # position of the end of the beach
yf = 0.7

h = 1  # maximum size of the probes

theta = np.arctan(pente)


# probes positions
xS = np.linspace(x0, xf, N)
yS = np.linspace(0, 0.2, 100)

list_sondes = []

for ind, xloc in enumerate(xS):
    list_sondes.append(np.transpose(np.array([[xloc] * 100, yS])))


### Code ######################################################################


for ind_path, path in enumerate(paths):

    # Loading simulation data
    print("Chargement des données de simu :")

    NCfile = nc.Dataset(path + "netcdf\\k_nut_gradU.nc")

    print("  - chargement x et y...")
    x = np.array(NCfile["x"])
    y = np.array(NCfile["y"])
    print("  - chargement alpha...")
    alpha = np.array(NCfile["alpha"])
    print("  - chargement k...")
    k = np.array(NCfile["k"])
    # print('  - chargement gradU...')
    # gradU = np.array(NCfile['gradU'])

    k_res = np.zeros((len(temps), len(list_sondes)))

    for ind_t, t in enumerate(temps):

        for ind_sonde, sonde in enumerate(list_sondes):

            print(
                "interpolation : t = "
                + str(t)
                + " s, sonde = "
                + str(ind_sonde + 1)
                + "/"
                + str(N)
            )

            k_interp = si.griddata((x, y), k[ind_t], sonde)
            alpha_interp = si.griddata((x, y), alpha[ind_t], sonde)

            k_eau = k_interp[alpha_interp >= 0.5]

            if len(k_eau) == 0:
                k_res[ind_t, ind_sonde] = np.nan
            else:
                k_res[ind_t, ind_sonde] = np.mean(k_eau)

    # save results
    if not os.path.exists(path + "netcdf\\"):
        os.makedirs(path + "netcdf\\")

    file = nc.Dataset(path + "netcdf\\" + titre_sauv, "w", format="NETCDF4")

    space = file.createDimension("m", None)
    timeDim = file.createDimension("s", None)
    kDim = file.createDimension("m2.s-2", None)

    xNC = file.createVariable("x", "f4", "m")
    kNC = file.createVariable("k", "f4", ("s", "m"))

    xNC[:] = xS
    kNC[:] = k_res

    file.close()
