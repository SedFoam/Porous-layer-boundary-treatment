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

paths = [  #'Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V0.2.5\\2D_rug13_CP_lin_h=0.35xd50\\']#,
    #'Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.0\\3D_rug13_h0=0.59_geomCorr\\']#,
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\"
]

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "interp_tau"

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

# physical parameters
rho_eau = 1000
rho_air = 1
nu_eau = 1e-6
nu_air = 1.48e-5


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
    print("  - chargement nut...")
    nut = np.array(NCfile["nut"])
    print("  - chargement gradU...")
    gradU = np.array(NCfile["gradU"])

    tau_res = np.zeros((len(temps), len(list_sondes)))
    tau_res_max = np.zeros((len(temps), len(list_sondes)))
    tau_res_min = np.zeros((len(temps), len(list_sondes)))

    for ind_t, t in enumerate(temps):

        print("interpolation de la simu : " + path)

        for ind_sonde, sonde in enumerate(list_sondes):

            print(
                "interpolation : t = "
                + str(t)
                + " s, sonde = "
                + str(ind_sonde + 1)
                + "/"
                + str(N)
            )

            nut_interp = si.griddata((x, y), nut[ind_t], sonde)
            alpha_interp = si.griddata((x, y), alpha[ind_t], sonde)
            gradUSym_interp = si.griddata(
                (x, y), 0.5 * (gradU[ind_t, 1] + gradU[ind_t, 3]), sonde
            )

            alpha_eau = alpha_interp[alpha_interp >= 0.5]
            nut_eau = nut_interp[alpha_interp >= 0.5]
            gradUSym_eau = gradUSym_interp[alpha_interp >= 0.5]

            if len(nut_eau) == 0 or len(gradUSym_eau) == 0:
                tau_res[ind_t, ind_sonde] = np.nan
                tau_res_max[ind_t, ind_sonde] = np.nan
                tau_res_min[ind_t, ind_sonde] = np.nan
            else:
                # Retrieves the maximum shear stress (positive or negative) in the water column
                tau_brut = (
                    2
                    * (nut_eau + nu_eau * alpha_eau + nu_air * (1 - alpha_eau))
                    * (rho_eau * alpha_eau + rho_air * (1 - alpha_eau))
                    * gradUSym_eau
                )

                tau_abs = abs(tau_brut)
                ind_tau_abs_max = list(tau_abs).index(np.nanmax(tau_abs))
                tau_res[ind_t, ind_sonde] = tau_brut[ind_tau_abs_max]

                # Retrieves the maximum shear stress in the water column
                tau_res_max[ind_t, ind_sonde] = np.nanmax(tau_brut)

                # Retrieves the minimum shear stress in the water column
                tau_res_min[ind_t, ind_sonde] = np.nanmin(tau_brut)

        # save results at each time step
        if not os.path.exists(path + "netcdf\\"):
            os.makedirs(path + "netcdf\\")

        if ind_t % 2 == 0:
            file = nc.Dataset(
                path + "netcdf\\" + titre_sauv + ".nc", "w", format="NETCDF4"
            )

            space = file.createDimension("m", None)
            timeDim = file.createDimension("s", None)
            tauDim = file.createDimension("N.m-2", None)

            xNC = file.createVariable("x", "f4", "m")
            tauNC = file.createVariable("tau", "f4", ("s", "N.m-2"))
            tauMaxNC = file.createVariable("tau_max", "f4", ("s", "N.m-2"))
            tauMinNC = file.createVariable("tau_min", "f4", ("s", "N.m-2"))

            xNC[:] = xS
            tauNC[:] = tau_res
            tauMaxNC[:] = tau_res_max
            tauMinNC[:] = tau_res_min

            file.close()

            if ind_t > 0:
                os.remove(
                    path + "netcdf\\" + titre_sauv + "_bis.nc"
                )  # Deletes the existing backup file

        else:
            file = nc.Dataset(
                path + "netcdf\\" + titre_sauv + "_bis.nc", "w", format="NETCDF4"
            )

            space = file.createDimension("m", None)
            timeDim = file.createDimension("s", None)
            tauDim = file.createDimension("N.m-2", None)

            xNC = file.createVariable("x", "f4", "m")
            tauNC = file.createVariable("tau", "f4", ("s", "N.m-2"))
            tauMaxNC = file.createVariable("tau_max", "f4", ("s", "N.m-2"))
            tauMinNC = file.createVariable("tau_min", "f4", ("s", "N.m-2"))

            xNC[:] = xS
            tauNC[:] = tau_res
            tauMaxNC[:] = tau_res_max
            tauMinNC[:] = tau_res_min

            file.close()

            os.remove(
                path + "netcdf\\" + titre_sauv + ".nc"
            )  # Deletes the existing backup file
