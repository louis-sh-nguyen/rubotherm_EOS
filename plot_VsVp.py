"""
Script to plot Vs and Vp calcualted from each pmv method.
Louis Nguyen
sn621@ic.ac.uk
09 Apr 2024
"""

import solubility_master as S
from numpy import *
from datetime import datetime
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib

# Plotting master configuration
matplotlib.rcParams["figure.figsize"] = [4.0, 3.5]  # in inches
matplotlib.rcParams["mathtext.default"] = "regular"  # same as regular text
matplotlib.rcParams["font.family"] = "DejaVu Sans"  # alternative: "serif"
matplotlib.rcParams["font.size"] = 10.0
matplotlib.rcParams["axes.titlesize"] = "small"  # relative to font.size
matplotlib.rcParams["axes.labelsize"] = "small"  # relative to font.size
matplotlib.rcParams["xtick.labelsize"] = "x-small"  # relative to font.size
matplotlib.rcParams["ytick.labelsize"] = "x-small"  # relative to font.size
matplotlib.rcParams["legend.fontsize"] = "xx-small"  # relative to font.size
matplotlib.rcParams["legend.frameon"] = False
matplotlib.rcParams["grid.linestyle"] = "-."
matplotlib.rcParams["grid.linewidth"] = 0.15  # in point units
matplotlib.rcParams["figure.autolayout"] = True

def plot_VsVp_pmv(base_obj, T: float, display_fig:bool=True, save_fig:bool=False):
    """Function to plot partial molar volume isotherms at single temperature.

    Args:
        spm (class object): class object representing the sol-pol mixture.
        T (float): temperature [K].  
    """
    now = datetime.now()  # current time
    time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM    
    
    data = S.SolPolExpData(base_obj.sol, base_obj.pol)    
    _df=data.get_sorption_data(T)
    P_list= linspace(1, _df["P[bar]"].values[-1]*1e5, 50) #[Pa]
    objects = []
    
    # pmv 1
    Vs_pmv1 = []
    Vp_pmv1 = []
    for j, _p in enumerate(P_list):
        if j == 0:
            obj = S.DetailedSolPol(base_obj, T, _p,)
        else:
            x0_list = S.update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
            obj = S.DetailedSolPol(base_obj, T, _p, x0_sol_range = x0_list,)
        Vspmv1, Vppmv1 = obj.Vs_Vp_pmv1()
        
        objects.append(obj)
        Vs_pmv1.append(Vspmv1)
        Vp_pmv1.append(Vppmv1)
    
    # pmv 2
    Vs_pmv2, Vp_pmv2 = zip(*[S.DetailedSolPol(base_obj, T, _p).Vs_Vp_pmv2() for _p in P_list])
    
    # pmv 3
    Vs_pmv3, Vp_pmv3 = zip(*[S.DetailedSolPol(base_obj, T, _p).Vs_Vp_pmv3() for _p in P_list])
    
    # Converting to np array for calculations
    Vs_pmv1 = array(Vs_pmv1)
    Vp_pmv1 = array(Vp_pmv1)
    Vs_pmv2 = array(Vs_pmv2)
    Vp_pmv2 = array(Vp_pmv2)
    Vs_pmv3 = array(Vs_pmv3)
    Vp_pmv3 = array(Vp_pmv3)    
    
    # Vs and Vp 
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(P_list*1e-5, Vs_pmv1*1e6, color=S.custom_colours[1], linestyle="solid", marker="None")
    ax1.plot(P_list*1e-5, Vs_pmv2*1e6, color=S.custom_colours[2], linestyle="solid", marker="None")
    ax1.plot(P_list*1e-5, Vs_pmv3*1e6, color=S.custom_colours[3], linestyle="solid", marker="None")
    ax1.plot(P_list*1e-5, Vp_pmv1*1e6, color=S.custom_colours[1], linestyle="dashed", marker="None")
    ax1.plot(P_list*1e-5, Vp_pmv2*1e6, color=S.custom_colours[2], linestyle="dashed", marker="None")
    ax1.plot(P_list*1e-5, Vp_pmv3*1e6, color=S.custom_colours[3], linestyle="dashed", marker="None")
    ax1.set_xlabel("P [bar]")
    ax1.set_ylabel(r"$\hat{V}$ [$cm^{3}/g$]")
    ax1.set_ylim(top=1.25, bottom=0.85)
    ax1.tick_params(direction="in")
    # Legends
    legend_colours = [Line2D([0], [0], linestyle="None", marker=".", color=S.custom_colours[i+1],
                             label=f"{T-273}°C pmv{pmv}") for i, pmv in enumerate(["1", "2", "3"])]
    legend_linestyles = [Line2D([0], [0], linestyle=line, marker="None", color="black",
                             label=f"{label}") for line, label in zip(["solid", "dashed"],["sol","pol"])]
    legend = legend_colours + legend_linestyles
    ax1.legend(handles=legend)
    
    if display_fig == True:
        plt.show()
    if save_fig == True:
        save_fig_path = f"{data.path}/PlotVsVp_{time_str}.png"
        plt.savefig(save_fig_path, dpi=1200)
        print(f"Plot successfully exported to {save_fig_path}.")

def plot_VsVp_vs_pressure(base_obj, T):
    Vs_values = []
    Vp_values = []
    p_values = linspace(1, 20e6, 10)    # [Pa]
    for p in linspace(1, 20e6, 10):
        obj = S.DetailedSolPol(base_obj, T, p)
        Vs, Vp = obj.Vs_Vp_pmv2()
        Vs_values.append(Vs)
        Vp_values.append(Vp)
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(p_values*1e-6, Vs_values, linestyle='None', marker='x')
    plt.xlabel('Pressure [MPa]')
    plt.ylabel(r'Vs [$m^{3}$/g]')
    
    plt.subplot(2, 1, 2)
    plt.plot(p_values*1e-6, Vp_values, linestyle='None', marker='x')
    plt.xlabel('Pressure [MPa]')
    plt.ylabel(r'Vp [$m^{3}$/g]')
    plt.show()

def plot_VsVp_vs_Sam_multiT(base_obj, T_list, p):
    S_values = linspace(0., 0.1, 20)    # [g/g]
    Vs_values = {}
    Vp_values = {}
    for T in T_list:
        Vs_values[T] = []
        Vp_values[T] = []
        for solubility in S_values:
            obj = S.DetailedSolPol(base_obj, T, p)
            Vs, Vp = obj.Vs_Vp_pmv1(obj.T, obj.P, solubility)
            Vs_values[T].append(Vs)
            Vp_values[T].append(Vp)
    
    plt.figure()
    plt.subplot(2, 1, 1)
    for T in T_list:
        plt.plot(S_values, Vs_values[T], linestyle='None', marker='x', label=f'{T-273} °C')
    plt.xlabel(r'$S_{am}$ [g/g]')
    plt.ylabel(r'$V_{s}$ [$m^{3}$/g]')
    
    plt.subplot(2, 1, 2)
    for T in T_list:
        plt.plot(S_values, Vp_values[T], linestyle='None', marker='x', label=f'{T-273} °C')
    plt.xlabel(r'$S_{am}$ [g/g]')
    plt.ylabel(r'$V_{p}$ [$m^{3}$/g]')
    
    plt.legend()
    plt.show()

def plot_VsVp_vs_Sam_multiP(base_obj, T, p_list):
    p_values = p_list   # [Pa]
    S_values = linspace(0., 0.1, 10)    # [g/g]
    Vs_values = {}
    Vp_values = {}
    for p in p_values:
        Vs_values[p] = []
        Vp_values[p] = []
        for solubility in S_values:
            obj = S.DetailedSolPol(base_obj, T, p)
            Vs, Vp = obj.Vs_Vp_pmv1(obj.T, obj.P, solubility)
            Vs_values[p].append(Vs)
            Vp_values[p].append(Vp)
    
    plt.figure()
    plt.subplot(2, 1, 1)
    for p in p_values:
        plt.plot(S_values, Vs_values[p], linestyle='None', marker='x', label=f'{p*1e-6} MPa')
    plt.xlabel(r'$S_{am}$ [g/g]')
    plt.ylabel(r'$V_{s}$ [$m^{3}$/g]')
    
    plt.subplot(2, 1, 2)
    for p in p_values:
        plt.plot(S_values, Vp_values[p], linestyle='None', marker='x', label=f'{p*1e-6} MPa')
    plt.xlabel(r'$S_{am}$ [g/g]')
    plt.ylabel(r'$V_{p}$ [$m^{3}$/g]')
    
    plt.legend()
    plt.show()


if __name__ == "__main__":
    mix = S.BaseSolPol("CO2","HDPE")
    for T in array([25, 35, 50]) + 273:
        plot_VsVp_pmv(mix, T, display_fig=False, save_fig=True)

    #* Test Vs and Vp vs. S_am at different pressures
    # plot_VsVp_vs_Sam_multiP(mix, T=35+273, p_list=linspace(1e6, 15e6, 10))