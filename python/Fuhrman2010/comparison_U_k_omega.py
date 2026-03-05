#
# Import section
#
import numpy as np
import fluidfoam
import os, sys, subprocess
import matplotlib.pyplot as plt

#
# Change fontsize
#
plt.rcParams.update({"font.size": 20})
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 10
#

# Main figures axis
xmin, xmax_U, xmax_tke, xmax_omega = 0.001, 15, 5.5, 100
# Insert figures axis
zmX1, zmX2_U, zmX2_tke, zmX2_omega, zmY1, zmY2 = 0.001, 15, 3.5, 100, 0.01, 10


#
# Physical parameters Fuhrman et al. (2010)
#
rhof = 1e3
nu = 9.6e-7

HF = 0.062
Reb = 1.4e4
utauF = 0.021
UbarF = Reb * nu / HF
kn = 9.9e-3
knplus = utauF * kn / nu

print("Fuhrman et al. (2010) - rough wall experiments")
print("Reb=", Reb, "    - U=", UbarF, " m/s")
print("knplus=", knplus, "    - u*=", utauF, " m/s")

#
# ----------Loading literature results-------------------
#

kF, zkF = np.loadtxt("DATA/k_rough.txt", unpack=True, skiprows=1)
vF, zvF = np.loadtxt("DATA/v_rough.txt", unpack=True, skiprows=1)

# compute pressure jump
Lx = 0.2 * HF  # numerical domain length
dP = utauF**2 / HF * Lx
print("dP=", dP)


#
# ---------------Loading OpenFoam results--------------------
#


basepath = "../../Fuhrman2010/"
pathFig = "../../Fuhrman2010/figures/"
fig_name = "Fuhrman2010"

if not os.path.exists(pathFig):
    os.makedirs(pathFig)


caseList = ["Fuhrman2010_boundaryCondition", "PorousLayer_boundaryCondition"]

labelList = [
    "Fuhrman et al. (2010) boundary condition",
    "Porous layer boundary condition",
]

lineList = ["-", "--"]
colorList = ["r", "b"]


Ncase = -1
maxCase = len(caseList)


###############################################################################
### Plot of the velocity ######################################################
###############################################################################

fig, ax1 = plt.subplots(figsize=(8, 12), dpi=300, facecolor="w", edgecolor="w")
ax2 = ax1.inset_axes([0.15, 0.5, 0.5, 0.45], xlim=(zmX1, zmX2_U), ylim=(zmY1, zmY2))

for i_case, case in enumerate(caseList):
    Ncase = Ncase + 1
    sol = basepath + case + "/"

    # look for last calculated time step
    valMax = 0
    strMax = 0
    for i in os.listdir(sol):
        try:
            a = float(i)
            if a > valMax:
                valMax = a
                strMax = i
        except:
            1

    try:
        proc = subprocess.Popen(
            ["foamListTimes", "-latestTime", "-case", sol], stdout=subprocess.PIPE
        )
    except:
        print("foamListTimes : command not found")
        print("Do you have loaded OpenFoam environement?")

    tout = strMax
    x, z, y = fluidfoam.readmesh(sol)
    k = fluidfoam.readscalar(sol, tout, "k")

    U = fluidfoam.readvector(sol, tout, "U")
    Tau = fluidfoam.readtensor(sol, tout, "Tau")
    try:
        omega = fluidfoam.readscalar(sol, tout, "omega")
    except:
        epsilon = fluidfoam.readscalar(sol, tout, "epsilon")
        omega = epsilon / k
    u = U[0, :]
    Tau_xz = Tau[3, :]
    wallShear = np.max(Tau[3, :])

    H = 0.062
    Umax = np.max(U)
    Um = np.trapz(u, z) / H

    print(" Reb=", Um * H / nu, " Um=", Um, " m/s")
    utau = np.sqrt(np.max(np.abs(wallShear)))
    print(r"$k_n^+$ =", utau * kn / nu, " u*=", utau, " m/s")
    print(r"$z/k_n$=", z[0] / kn, " z+=", utau * z[0] / nu)

    #
    # ---------------------------------Figures----------------------------------
    #

    # main plot
    ax1.plot(
        u / utauF,
        z / kn,
        ls=lineList[Ncase],
        color=colorList[Ncase],
        label=labelList[i_case],
    )
    if Ncase == maxCase - 1:
        ax1.plot([0, 100000], [0.00203 / kn, 0.00203 / kn], ":", color="grey", zorder=0)
        pw = ax1.scatter(
            vF,
            zvF,
            150,
            color="k",
            marker="+",
            linewidths=2,
            zorder=10,
            label="Fuhrman et al. (2010) experimental data",
        )
        handles, labels = ax1.get_legend_handles_labels()

    ax1.set_xlabel(r"$u/U_{f}$", fontsize=25, usetex=True)
    ax1.set_ylabel(r"$z/k_N$", fontsize=25, usetex=True)
    if i_case == 0:
        ax1.axis([0, 1.2 * Umax / utauF, 0, 6])
    ax1.grid(1)
    ax1.tick_params(axis="both", labelsize=18)
    ax1.set_xlim(xmin, xmax_U)

    # insert background
    rect2 = plt.Rectangle((zmX1, zmY1), zmX2_U - zmX1, zmY2 - zmY1, fc="lavender")
    ax2.add_patch(rect2)

    # insert plot
    ax2.plot(u / utau, z / kn, ls=lineList[Ncase], color=colorList[Ncase])
    if Ncase == maxCase - 1:
        ax2.plot([0, 100000], [0.00203 / kn, 0.00203 / kn], ":", color="grey", zorder=1)
        pw = ax2.scatter(vF, zvF, 150, color="k", marker="+", linewidths=2, zorder=10)
        zBL = np.linspace(1e-2, 1e1, 1000)
        ax2.semilogy(1 / 0.41 * np.log(zBL * 30), zBL, ":k")

    ax2.grid(1)
    ax2.tick_params(axis="both", labelsize=18)

    plt.legend(bbox_to_anchor=(0, -0.1), loc="upper left")

    # save
    fig.savefig(pathFig + fig_name + "_U.pdf", transparent=True, bbox_inches="tight")


###############################################################################
### Plot of the tke ###########################################################
###############################################################################

fig, ax1 = plt.subplots(figsize=(8, 12), dpi=300, facecolor="w", edgecolor="w")
ax2 = ax1.inset_axes([0.45, 0.5, 0.5, 0.45], xlim=(zmX1, zmX2_tke), ylim=(zmY1, zmY2))

Ncase = -1
for i_case, case in enumerate(caseList):
    Ncase = Ncase + 1
    sol = basepath + case + "/"

    # look for last calculated time step
    valMax = 0
    for i in os.listdir(sol):
        try:
            a = float(i)
            if a > valMax:
                valMax = a
                strMax = i
        except:
            1

    try:
        proc = subprocess.Popen(
            ["foamListTimes", "-latestTime", "-case", sol], stdout=subprocess.PIPE
        )
    except:
        print("foamListTimes : command not found")
        print("Do you have loaded OpenFoam environement?")

    tout = strMax
    x, z, y = fluidfoam.readmesh(sol)
    k = fluidfoam.readscalar(sol, tout, "k")

    U = fluidfoam.readvector(sol, tout, "U")
    Tau = fluidfoam.readtensor(sol, tout, "Tau")
    try:
        omega = fluidfoam.readscalar(sol, tout, "omega")
    except:
        epsilon = fluidfoam.readscalar(sol, tout, "epsilon")
        omega = epsilon / k
    u = U[0, :]
    Tau_xz = Tau[3, :]
    wallShear = np.max(Tau[3, :])

    H = 0.062
    Umax = np.max(U)
    Um = np.trapz(u, z) / H

    print(" Reb=", Um * H / nu, " Um=", Um, " m/s")
    utau = np.sqrt(np.max(np.abs(wallShear)))
    print(r"$k_n^+$ =", utau * kn / nu, " u*=", utau, " m/s")
    print(r"$z/k_n$=", z[0] / kn, " z+=", utau * z[0] / nu)

    #
    # ---------------------------------Figures----------------------------------
    #

    # main plot
    ax1.plot(
        k / utauF**2,
        z / kn,
        ls=lineList[Ncase],
        color=colorList[Ncase],
        label=labelList[i_case],
    )
    if Ncase == maxCase - 1:
        ax1.plot([0, 100000], [0.00203 / kn, 0.00203 / kn], ":", color="grey", zorder=0)
        pw = ax1.scatter(
            kF,
            zkF,
            150,
            color="k",
            marker="+",
            linewidths=2,
            zorder=10,
            label="Fuhrman et al. (2010) experimental data",
        )
        handles, labels = ax1.get_legend_handles_labels()

    ax1.set_xlabel(r"$k/U_{f}^2$", fontsize=25, usetex=True)
    if i_case == 0:
        ax1.axis([0, 3.5, 0, 6])
    ax1.grid(1)
    ax1.tick_params(axis="both", labelsize=18)
    ax1.set_xlim(xmin, xmax_tke)
    ax1.set_yticklabels([])

    # insert background
    rect2 = plt.Rectangle((zmX1, zmY1), zmX2_tke - zmX1, zmY2 - zmY1, fc="lavender")
    ax2.add_patch(rect2)

    # insert plot
    ax2.plot(k / utau**2, z / kn, ls=lineList[Ncase], color=colorList[Ncase])
    if Ncase == maxCase - 1:
        ax2.plot([0, 100000], [0.00203 / kn, 0.00203 / kn], ":", color="grey", zorder=1)
        pw = ax2.scatter(kF, zkF, 150, color="k", marker="+", linewidths=2, zorder=10)

    ax2.set_yscale("log")

    ax2.grid(1)
    ax2.tick_params(axis="both", labelsize=18)

    plt.legend(bbox_to_anchor=(0, -0.1), loc="upper left")

    # save
    fig.savefig(pathFig + fig_name + "_tke.pdf", transparent=True, bbox_inches="tight")


###############################################################################
### Plot of the dissipation rate omega ########################################
###############################################################################

fig, ax1 = plt.subplots(figsize=(8, 12), dpi=300, facecolor="w", edgecolor="w")
ax2 = ax1.inset_axes([0.45, 0.5, 0.5, 0.45], xlim=(zmX1, zmX2_omega), ylim=(zmY1, zmY2))

Ncase = -1
for i_case, case in enumerate(caseList):
    Ncase = Ncase + 1
    sol = basepath + case + "/"

    # look for last calculated time step
    valMax = 0
    for i in os.listdir(sol):
        try:
            a = float(i)
            if a > valMax:
                valMax = a
                strMax = i
        except:
            1

    try:
        proc = subprocess.Popen(
            ["foamListTimes", "-latestTime", "-case", sol], stdout=subprocess.PIPE
        )
    except:
        print("foamListTimes : command not found")
        print("Do you have loaded OpenFoam environement?")

    tout = strMax
    x, z, y = fluidfoam.readmesh(sol)
    k = fluidfoam.readscalar(sol, tout, "k")

    U = fluidfoam.readvector(sol, tout, "U")
    Tau = fluidfoam.readtensor(sol, tout, "Tau")
    try:
        omega = fluidfoam.readscalar(sol, tout, "omega")
    except:
        epsilon = fluidfoam.readscalar(sol, tout, "epsilon")
        omega = epsilon / k
    u = U[0, :]
    Tau_xz = Tau[3, :]
    wallShear = np.max(Tau[3, :])

    H = 0.062
    Umax = np.max(U)
    Um = np.trapz(u, z) / H

    print(" Reb=", Um * H / nu, " Um=", Um, " m/s")
    utau = np.sqrt(np.max(np.abs(wallShear)))
    print(r"$k_n^+$ =", utau * kn / nu, " u*=", utau, " m/s")
    print(r"$z/k_n$=", z[0] / kn, " z+=", utau * z[0] / nu)

    #
    # ---------------------------------Figures----------------------------------
    #

    # main plot
    ax1.semilogx(
        omega * nu / utauF**2,
        z / kn,
        ls=lineList[Ncase],
        color=colorList[Ncase],
        label=labelList[i_case],
    )
    if Ncase == maxCase - 1:
        ax1.plot([0, 100000], [0.00203 / kn, 0.00203 / kn], ":", color="grey", zorder=0)

    ax1.set_xlabel(r"$\omega\ \nu/U_{f}^2$", fontsize=25, usetex=True)
    if i_case == 0:
        ax1.axis([1e-3, 5e1, 0, 6])
    ax1.grid(1)
    ax1.tick_params(axis="both", labelsize=18)
    ax1.set_xlim(xmin, xmax_omega)
    ax1.set_yticklabels([])

    # insert background
    rect2 = plt.Rectangle((zmX1, zmY1), zmX2_omega - zmX1, zmY2 - zmY1, fc="lavender")
    ax2.add_patch(rect2)

    # insert plot
    ax2.semilogx(
        omega * nu / utau**2, z / kn, ls=lineList[Ncase], color=colorList[Ncase]
    )
    if Ncase == maxCase - 1:
        ax2.plot([0, 100000], [0.00203 / kn, 0.00203 / kn], ":", color="grey", zorder=1)

    ax2.set_yscale("log")

    ax2.grid(1)
    ax2.tick_params(axis="both", labelsize=18)

    plt.legend(bbox_to_anchor=(0, -0.1), loc="upper left")

    # save
    fig.savefig(
        pathFig + fig_name + "_omega.pdf", transparent=True, bbox_inches="tight"
    )
