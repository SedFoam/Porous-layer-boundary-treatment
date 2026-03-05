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
path_exp = (
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\Données Kikkert\\Bed Shear Stress\\"
)
name_exp = "IMP_015"

paths = [
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V2.1\\3D_rug13_CP_lin_h=0.35xd50_h0=0.59_geomCorr\\",
    "Z:\\project\\meige\\2024\\24SWASH\\Maxime\\OpenFOAM\\Simulation\\Kikkert2012\\interIsoFoam\\V0.2.5\\2D_rug13_CP_lin_h=0.35xd50\\",
]
postProcessFolder = "postProcessing"
probe_names = ["sonde2", "sonde3", "sonde4", "sonde5", "sonde6"]
probe_X = np.array([0.072, 0.772, 1.567, 2.377, 3.177])
timeStep = "mergeTime"

sauvegarde = True
titre_sauv = "Friction_parietale_tau_rug13_2D_3D_log"

seuil_alpha = 0.99
slope = 1 / 10
theta = np.arctan(slope)
errorBar = 0.01  # mesh size around the free surface

timePlot = np.array([5])  # plot time
rho_eau = 1000
rho_air = 1
nu_eau = 1e-6
nu_air = 1.48e-5

methode_friction = ["Tau_M", "Tau_Rmax"]  # , 'Tau_L']
timeList = np.linspace(0.05, 10, 200)

# colormap
clmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
clmap = [clmap[2], "#000000"]
taille_plot = [3, 1.5]
style_trait = ["-", "-."]

YLim = np.array([[0.03, 200], [0.03, 200], [0.03, 200], [0.03, 200], [0.03, 200]])


######### Code ################################################################

for probeNum, probe_name in enumerate(probe_names):

    print("Création figure " + probe_name)

    plt.figure()

    # Loading experimental data
    mat = scipy.io.loadmat(
        path_exp + "Tau_" + name_exp + "_PIV" + probe_name[-1] + ".mat"
    )
    t = mat.get("t_" + name_exp + "_PIV" + probe_name[-1])[0]
    Tau = mat.get(methode_friction[0] + "_" + name_exp + "_PIV" + probe_name[-1])[0]
    Tau2 = mat.get(methode_friction[1] + "_" + name_exp + "_PIV" + probe_name[-1])[0]
    # Tau3 = mat.get(methode_friction[2]+'_'+name_exp+'_PIV'+probe_name[-1])[0]

    # plot experimental data
    # plt.scatter(t, Tau3, color='green', marker='s', label="Données expérimentales")
    plt.scatter(
        t,
        Tau2,
        color="darkorange",
        marker="o",
        facecolors="none",
        label="Données expérimentales",
    )
    plt.scatter(t, Tau, color="red", marker="+", label="Données expérimentales")

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

        # getting the time steps to plot
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
                tau = tau[:, 1]  # get the matrix component dux/dy = duy/dx
                tauPlot.append(max(abs(tau)))
                tauPlot2.append(abs(tau[0]))

            else:
                tauPlot.append(np.NaN)
                tauPlot2.append(np.NaN)
            tNumPlot.append(timeProbe[ind])

        # plt.plot(timeList, tauPlot2, color=clmap[ind_path])
        plt.plot(
            timeList,
            tauPlot,
            color=clmap[ind_path],
            linewidth=taille_plot[ind_path],
            linestyle=style_trait[ind_path],
        )

    plt.yscale("log")

    # Setting axis labels
    plt.xlabel("$t$ (s)", fontsize=20, usetex=True)
    plt.ylabel("$\\tau$ (N/m²)", fontsize=20, usetex=True)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylim((YLim[probeNum, 0], YLim[probeNum, 1]))
    # plt.title(probe_name)

    # add grid and legend
    plt.grid()
    # plt.ylim((0, 0.2)
    # plt.legend(bbox_to_anchor=(0, -0.18), loc="upper left")

    # show
    # plt.show()

    # Save figure
    if sauvegarde:
        plt.savefig(
            dir_path + "\\" + titre_sauv + "_sonde" + str(probeNum + 2) + ".pdf",
            bbox_inches="tight",
        )
