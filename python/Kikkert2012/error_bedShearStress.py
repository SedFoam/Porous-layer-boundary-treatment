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
path_exp = (
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Bed Shear Stress\\"
)
name_exp = "IMP_015"

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
probe_names = ["sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
probe_X = np.array([0.072, 0.772, 1.567, 2.377, 3.177])
timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "error_friction_parietale"

seuil_alpha = 0.99
slope = 1 / 10
theta = np.arctan(slope)
errorBar = 0.01  # mesh around the free surface

# physical parameters
rho_eau = 1000
rho_air = 1
nu_eau = 1e-6
nu_air = 1.48e-5

methode_friction = ["Tau_M", "Tau_Rmax"]
timeList = np.linspace(0.05, 10, 200)

# colormap
clmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]

gs = gridspec.GridSpec(3, 1)


######### Code ################################################################

RMSE = np.zeros((len(probe_names), len(paths)))
BSS = np.zeros((len(probe_names), len(paths)))  # Brier skill score
WSS = np.zeros((len(probe_names), len(paths)))  # Willmott skill score

probes_x = []

for probeNum, probe_name in enumerate(probe_names):

    print("Création figure " + probe_name)

    # Loading experimental data
    mat = scipy.io.loadmat(
        path_exp + "Tau_" + name_exp + "_PIV" + probe_name[-1] + ".mat"
    )
    t = mat.get("t_" + name_exp + "_PIV" + probe_name[-1])[0]
    Tau = mat.get(methode_friction[0] + "_" + name_exp + "_PIV" + probe_name[-1])[0]
    Tau2 = mat.get(methode_friction[1] + "_" + name_exp + "_PIV" + probe_name[-1])[0]

    for ind_path, path in enumerate(paths):
        # Loading simulation data
        varName, probe_loc, timeProbe, varProbe = ff.readsampling(
            path=path + postProcessFolder + "\\",
            probes_name=probe_name,
            time_name=timeStep,
        )

        probeYLoc = (probe_loc[:, 1] - probe_loc[0, 1]) * np.cos(theta) - (
            probe_loc[:, 0] - probe_loc[0, 0]
        ) * np.sin(theta)

        # getting numerical time steps to plot
        tNumPlot = []
        probeYLocPlot = []
        tauPlot = []
        tauPlot2 = []
        for temps in timeList:
            ind = np.abs(
                timeProbe - temps
            ).argmin()  # closest time index to the target time
            gradUnum = varProbe[ind, varName.index("grad(U)")]  # get gradU
            nutNum = varProbe[ind, varName.index("nut")]
            alphaNum = varProbe[ind, varName.index("alpha.water")]

            gradUPlot = np.array(
                [
                    i
                    for ind_r, i in enumerate(gradUnum)
                    if alphaNum[ind_r] >= seuil_alpha
                ]
            )  # only data corresponding to a ‘water’ cell is retrieved
            probeYLocPlot.append(
                [
                    i
                    for ind_r, i in enumerate(probeYLoc)
                    if alphaNum[ind_r] >= seuil_alpha
                ]
            )
            alphaPlot = np.array(
                [
                    i
                    for ind_r, i in enumerate(alphaNum)
                    if alphaNum[ind_r] >= seuil_alpha
                ]
            )
            nutPlot = np.array(
                [i for ind_r, i in enumerate(nutNum) if alphaNum[ind_r] >= seuil_alpha]
            )

            if np.size(gradUPlot) > 1:  # test if the vector is empty (not enough water)

                # calculate symm(grad(U))
                symmGradU = np.zeros(np.shape(gradUPlot))
                symmGradU[:, 0] = gradUPlot[:, 0]
                symmGradU[:, 4] = gradUPlot[:, 4]
                symmGradU[:, 8] = gradUPlot[:, 8]
                symmGradU[:, 1] = 0.5 * (gradUPlot[:, 1] + gradUPlot[:, 3])
                symmGradU[:, 3] = symmGradU[:, 1]
                symmGradU[:, 2] = 0.5 * (gradUPlot[:, 2] + gradUPlot[:, 6])
                symmGradU[:, 6] = symmGradU[:, 2]
                symmGradU[:, 5] = 0.5 * (gradUPlot[:, 5] + gradUPlot[:, 7])
                symmGradU[:, 7] = symmGradU[:, 5]

                tau = (
                    2
                    * (nutPlot + nu_eau * alphaPlot + nu_air * (1 - alphaPlot))
                    * (rho_eau * alphaPlot + rho_air * (1 - alphaPlot))
                    * symmGradU
                )
                tau = tau[:, 1]  # get the component dux/dy = duy/dx
                tauPlot.append(max(abs(tau)))
                tauPlot2.append(abs(tau[0]))

            else:
                tauPlot.append(np.NaN)
                tauPlot2.append(np.NaN)
            tNumPlot.append(timeProbe[ind])

        # interpolation of the experimental data
        interp_tau = np.interp(t, timeList, tauPlot)

        delta_tau = interp_tau - Tau
        delta_tau2 = interp_tau - Tau2

        delta_tau_moy1 = interp_tau - np.nanmean(Tau)
        delta_tau_moy2 = Tau - np.nanmean(Tau)
        delta_tau2_moy1 = interp_tau - np.nanmean(Tau2)
        delta_tau2_moy2 = Tau2 - np.nanmean(Tau2)

        Delta_tau = np.array([delta_tau, delta_tau2])
        Delta_tau[np.isnan(Delta_tau)] = 1000
        delta_tau_min = Delta_tau.min(axis=0)
        delta_tau_min[delta_tau_min == 1000] = np.nan

        Delta_tau_moy1 = np.array([delta_tau_moy1, delta_tau2_moy1])
        Delta_tau_moy2 = np.array([delta_tau_moy2, delta_tau2_moy2])
        Delta_tau_moy1[np.isnan(Delta_tau_moy1)] = 1000
        Delta_tau_moy2[np.isnan(Delta_tau_moy2)] = 1000
        delta_tau_moy1_min = Delta_tau_moy1.min(axis=0)
        delta_tau_moy2_min = Delta_tau_moy2.min(axis=0)
        delta_tau_moy1_min[delta_tau_moy1_min == 1000] = np.nan
        delta_tau_moy2_min[delta_tau_moy2_min == 1000] = np.nan

        # calculate the skill scores
        RMSE[probeNum, ind_path] = np.sqrt(
            np.nansum((delta_tau_min) ** 2) / len(interp_tau)
        )
        BSS[probeNum, ind_path] = 1 - np.nansum((delta_tau_min) ** 2) / np.nansum(
            (delta_tau_moy2_min) ** 2
        )
        WSS[probeNum, ind_path] = 1 - np.nansum((delta_tau_min) ** 2) / np.nansum(
            (abs(delta_tau_moy2_min) + abs(delta_tau_moy1_min)) ** 2
        )

    # add the probe to the results
    probes_x.append(probe_X[int(probe_name[-1]) - 2])


probe_x = np.array(probes_x)


fig = plt.figure(num=1, figsize=(8, 16), dpi=300)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[2, 0])

for i in range(np.shape(RMSE)[1]):

    ax1.scatter(
        probe_x,
        RMSE[:, i],
        color=clmap[i],
        marker="+",
        label=nom_plots[i] + " : mean error = " + str(np.mean(RMSE[:, i])),
    )
    ax1.scatter(
        0,
        np.mean(RMSE[:, i]),
        color=clmap[i],
        marker="o",
        edgecolor="black",
        label="_Hidden label",
    )

    ax2.scatter(
        probe_x,
        BSS[:, i],
        color=clmap[i],
        marker="+",
        label=nom_plots[i] + " : mean BSS = " + str(np.mean(BSS[:, i])),
    )
    ax2.scatter(
        0,
        np.mean(BSS[:, i]),
        color=clmap[i],
        marker="o",
        edgecolor="black",
        label="_Hidden label",
    )

    ax3.scatter(
        probe_x,
        WSS[:, i],
        color=clmap[i],
        marker="+",
        label=nom_plots[i] + " : mean WSS = " + str(np.mean(WSS[:, i])),
    )
    ax3.scatter(
        0,
        np.mean(WSS[:, i]),
        color=clmap[i],
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
