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
import netCDF4 as nc


# Input data
dir_path = os.path.dirname(os.path.realpath(__file__))

paths = [
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V0.2.5\\2D_rug13_CP_lin_h=0.35xd50\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.0\\3D_rug13_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\",
]

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
timeStep = "mergeTime"


seuil_alpha = 0.5
slope = 1 / 10
theta = np.arctan(slope)

clmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
clmap = ["#000000", clmap[0]]

# initial shoreline position
xOrigine = 5.806
yOrigine = 0.062

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


for ind_path, path in enumerate(paths):

    alphaTot = []
    kTot = []
    nutTot = []
    gradUTot = []

    x, y, z = ff.readmesh(path)

    # projection to 2D if needed
    z = np.round(z, 8)
    if len(np.unique(z)) > 1:
        z0 = min(abs(np.unique(z)))
        x = x[z == z0]
        y = y[z == z0]

    # moving the origin to that of the simulation data
    x0 = x - xOrigine
    y0 = y - yOrigine

    x0Red = x0[x0 >= -0.5]
    y0Red = y0[x0 >= -0.5]

    xplage = [0] * len(x0Red)
    yplage = [0] * len(y0Red)

    # changing coordinates from the simulation reference frame to the experimental reference frame
    for ind_y, y_curr in enumerate(y0Red):
        xplage[ind_y] = x0Red[ind_y] * np.cos(theta) + y0Red[ind_y] * np.sin(theta)
        yplage[ind_y] = y0Red[ind_y] * np.cos(theta) - x0Red[ind_y] * np.sin(theta)

    for ind_t, t in enumerate(temps):

        # Loading simulation data
        try:
            alpha = ff.readscalar(path, temps_str[ind_t], "alpha.water")
        except:
            alpha = np.zeros(len(z))

        try:
            nut = ff.readscalar(path, temps_str[ind_t], "nut")
        except:
            nut = np.zeros(len(z))

        try:
            k = ff.readscalar(path, temps_str[ind_t], "k")
        except:
            k = np.zeros(len(z))

        try:
            gradU = ff.readtensor(path, temps_str[ind_t], "grad(U)")
        except:
            gradU = np.zeros((9, len(z)))

        # projection to 2D if needed
        if len(np.unique(z)) > 1:
            alpha = alpha[z == z0]
            k = k[z == z0]
            nut = nut[z == z0]
            gradU = gradU[:, z == z0]

        # only values above the initial shoreline are kept
        alphaRed = alpha[x0 >= -0.5]
        kRed = k[x0 >= -0.5]
        nutRed = nut[x0 >= -0.5]
        gradURed = gradU[:, x0 >= -0.5]

        alphaTot.append(alphaRed)
        kTot.append(kRed)
        nutTot.append(nutRed)
        gradUTot.append(gradURed)

    alphaTot = np.array(alphaTot)
    kTot = np.array(kTot)
    nutTot = np.array(nutTot)
    gradUTot = np.array(gradUTot)

    # save results
    if not os.path.exists(path + "netcdf\\"):
        os.makedirs(path + "netcdf\\")

    file = nc.Dataset(path + "netcdf\\" + "k_nut_gradU.nc", "w", format="NETCDF4")

    space = file.createDimension("m", None)
    timeDim = file.createDimension("s", None)
    noDim = file.createDimension("None", None)
    noDim2 = file.createDimension("None2", None)
    kDim = file.createDimension("m2.s-2", None)
    nutDim = file.createDimension("m2.s-1", None)
    gradUDim = file.createDimension("s-1", None)

    xNC = file.createVariable("x", "f4", "m")
    yNC = file.createVariable("y", "f4", "m")
    alphaNC = file.createVariable("alpha", "f4", ("s", "None"))
    kNC = file.createVariable("k", "f4", ("s", "m2.s-2"))
    nutNC = file.createVariable("nut", "f4", ("s", "m2.s-1"))
    gradUNC = file.createVariable("gradU", "f4", ("s", "None2", "s-1"))

    print("saving x and y...")
    xNC[:] = xplage
    yNC[:] = yplage
    print("saving alpha...")
    alphaNC[:] = alphaTot
    print("saving k...")
    kNC[:] = kTot
    print("saving nut...")
    nutNC[:] = nutTot
    print("saving gradU...")
    gradUNC[:] = gradUTot

    file.close()

###############################################################################
