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


# Input data
dir_path = os.path.dirname(os.path.realpath(__file__))
path_exp = "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Turbulence Quantities\\"
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

postProcessFolder = "postProcessing"
probe_names = ["sonde1", "sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
probe_pos = np.array([-1.802, 0.072, 0.772, 1.567, 2.377, 3.177])
timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "error_TKE"

seuil_alpha = 0.5
slope = 1 / 10
theta = np.arctan(slope)

clmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]


### Code ######################################################################

RMSE = np.zeros((len(probe_names), len(paths)))
probes_x = []

for probeNum, probe_name in enumerate(probe_names):

    print("Création figure " + probe_name)

    # Loading experimental data
    mat = scipy.io.loadmat(path_exp + "TKE_" + name_exp + probe_name[-1] + ".mat")
    t = mat.get("t_" + name_exp + probe_name[-1])[0]
    TKE = np.nanmean(mat.get("TKE_" + name_exp + probe_name[-1]), 0)

    plt.figure()

    for ind_path, curr_path in enumerate(paths):

        # Loading simulation data
        varName, probe_loc, timeProbe, varProbe = ff.readsampling(
            path=curr_path + postProcessFolder + "\\",
            probes_name=probe_name,
            time_name=timeStep,
        )

        # calculate mean TKE
        TKE_num = []
        for i in range(len(varProbe)):
            tke = varProbe[i, 1]
            alpha = varProbe[i, 0]
            ind_eau = np.where(np.diff(np.sign(alpha - seuil_alpha), 0) > 0)[
                0
            ]  # retrieve the indices of the cells containing water
            TKE_num.append(np.mean(tke[ind_eau]))

        # interpolation on the experimental data
        interp_TKE = np.interp(t, timeProbe, TKE_num)

        RMSE[probeNum, ind_path] = np.sqrt(
            np.nansum((interp_TKE - TKE) ** 2) / len(interp_TKE)
        )

        if probeNum == 3:
            toto = 12

    # add the probe to the results
    probes_x.append(probe_pos[int(probe_name[-1]) - 1])


probe_x = np.array(probes_x)
for i in range(np.shape(RMSE)[1]):
    plt.scatter(
        probe_x,
        RMSE[:, i],
        color=clmap[i],
        marker="+",
        label=nom_plots[i] + " : mean error = " + str(np.mean(RMSE[:, i])),
    )
    plt.scatter(
        0, np.mean(RMSE[:, i]), color=clmap[i], marker="o", label="_Hidden label"
    )


# Setting axis labels
plt.xlabel("x (m)", fontsize=16)
plt.ylabel("RMSE (m²/s²)", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
# plt.title(probe_name)

# add grid and legend
plt.grid()
# plt.ylim((0, 0.2)
plt.legend(bbox_to_anchor=(0, -0.18), loc="upper left")

# show
# plt.show()


# Save figure
if sauvegarde:
    plt.savefig(dir_path + "\\" + titre_sauv + ".pdf", bbox_inches="tight")
