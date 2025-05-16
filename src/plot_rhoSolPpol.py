"""
Script to plot and compare density of CO2 and polymer between SAFT and Rubotherm experiments.
25 March 2024
Louis Nguyen
sn621@ic.ac.uk
"""

import solubility_master as S
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

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

custom_colours = ['green','blue','red','purple','orange','brown','plum','indigo','olive','grey']
custom_markers = ['x', 'o', 's', '^', '*', 'D', '.']

def plot_rho_sol(base_obj, T_list:list[float], display_fig:bool=True, save_fig:bool=False):
    now = datetime.now()  # current time
    time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM            
    
    data = S.SolPolExpData(base_obj.sol, base_obj.pol)
    df = {}
    P_list = {}
    rhoCO2_SAFT = {}
    for i, T in enumerate(T_list):    
        df[T]=data.get_sorption_data(T)
        max_i = df[T]["P[bar]"].max()
        if i == 0:
            pbar_max = max_i
        else:
            pbar_max = max(pbar_max, max_i)
    
    def get_rhoCO2_SAFT(T, P):
        eos = base_obj.eos_sol
        rhoL = eos.density(T, P, "L")   # [mol/m^3]
        rhoV = eos.density(T, P, "V")   # [mol/m^3]
        if isclose(rhoL, rhoV):
            rho = rhoV
            print("SC phase")
            
        elif isclose(P, eos.pressure(rhoV,T), rtol=1e-5):
            rho = rhoV
            print("V phase")
            
        elif isclose(P, eos.pressure(rhoL,T), rtol=1e-5):
            rho = rhoL
            print("L phase")
        
        return rho * base_obj.MW_sol  * 1e-6  # [g/cm3]
    
    for i, T in enumerate(T_list):
        #* Using fixed range
        # P_list[T] = linspace(1, pbar_max*1e5, 100)  # [Pa]    
        
        #* Using matching pressure range with exp data
        # P_list[T] = linspace(1, df[T]["P[bar]"].max()*1e5, 10)  # [Pa]
        
        #* Using experimental pressure values
        P_list[T] = df[T]["P[bar]"]*1e5  # [Pa]
        
        # SAFT prediciton
        rhoCO2_SAFT[T] = [get_rhoCO2_SAFT(T,P) for P in P_list[T]] # [g/cm3]

        print(f"Temperature: {T-273} °C")
        print(F'P_list [Pa] = {P_list[T]}')
        print(f"rhoCO2_SAFT [g/cm3] = {rhoCO2_SAFT[T]}")
        print('')

    # Plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, T in enumerate(T_list):
        
        # Exp measurement
        ax.plot(df[T]["P[bar]"], df[T]["ρ[g/cc]"], \
            color=S.custom_colours[i], linestyle="None", marker=S.custom_markers[-1], label=f"{T-273}°C - exp")
        
        # SAFT predictions
        ax.plot(P_list[T]*1e-5, rhoCO2_SAFT[T], \
            color=S.custom_colours[i], linestyle="solid", marker="None", label=f"{T-273}°C - SAFT")

    ax.set_xlabel("P [bar]")
    ax.set_ylabel(r"$\rho_{sol}$ [g/$cm^{3}$]")
    # ax.set_title("Comparison of CO2 density")
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    if save_fig == True:
        save_fig_path = f"{data.path}/PlotRhoSol_{time_str}.png"
        plt.savefig(save_fig_path, dpi=1200)
        print(f"Plot successfully exported to {save_fig_path}.")
    if display_fig == True:
        plt.show()

def plot_rho_pol(base_obj, T_list:list[float], display_fig:bool=True, save_fig:bool=False):
    now = datetime.now()  # current time
    time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM            
    
    data = S.SolPolExpData(base_obj.sol, base_obj.pol)
    df = {}
    P_list = {}
    rhoPol_SAFT = {}
    for i, T in enumerate(T_list):    
        df[T]=data.get_sorption_data(T)
        max_i = df[T]["P[bar]"].max()
        if i == 0:
            pbar_max = max_i
        else:
            pbar_max = max(pbar_max, max_i)

    for i, T in enumerate(T_list):
        #* Using fixed range
        # P_list[T] = linspace(1, pbar_max*1e5, 100)  # [Pa]
        
        #* Using matching pressure range with exp data
        P_list[T] = linspace(1, df[T]["P[bar]"].max()*1e5, 100)  # [Pa]
        
        # SAFT prediciton
        _rhoPol_SAFT = [S.DetailedSolPol(base_obj, T, P).SinglePhaseDensity(array([0., 1.]),T,P) for P in P_list[T]] # [mol/m3]
        rhoPol_SAFT[T] = array(_rhoPol_SAFT) * base_obj.MW_sol  * 1e-6  # [g/cm3]


    # Plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, T in enumerate(T_list):
        
        # SAFT predictions
        ax.plot(P_list[T]*1e-5, rhoPol_SAFT[T], \
            color=S.custom_colours[i], linestyle="solid", marker="None", label=f"{T-273}°C - SAFT")

    ax.set_xlabel("P [bar]")
    ax.set_ylabel(r"$\rho_{pol}$ [g/$cm^{3}$]")
    # ax.set_title("Comparison of CO2 density")
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    if save_fig == True:
        save_fig_path = f"{data.path}/PlotRhoPol_{time_str}.png"
        plt.savefig(save_fig_path, dpi=1200)
        print(f"Plot successfully exported to {save_fig_path}.")
    if display_fig == True:
        plt.show()

def plot_rhoCO2_comparison(base_obj, T_list:list[float], display_fig:bool=True, save_fig:bool=False):
    data = S.SolPolExpData(base_obj.sol, base_obj.pol)
    df = {}
    for i, T in enumerate(T_list):    
        df[T]=data.get_sorption_data(T)
        # print(df[T].info())   
    
    rho_types = ['exp', 'SW EoS', 'SAFT-γ Mie EoS']
    # Plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, T in enumerate(T_list):
        ax.plot(df[T]["P[bar]"], df[T]["ρ[g/cc]"], color=custom_colours[i], marker=custom_markers[0], markerfacecolor='None', linestyle='None', label=rho_types[0])
        ax.plot(df[T]["P[bar]"], df[T]["ρSW[g/cc]"], color=custom_colours[i], marker=custom_markers[1], markerfacecolor='None', linestyle='None', label=rho_types[1])
        ax.plot(df[T]["P[bar]"], df[T]["ρSAFT[g/cc]"], color=custom_colours[i], marker=custom_markers[2], markerfacecolor='None', linestyle='None', label=rho_types[2])
    
    ax.set_xlabel('P [bar]')
    ax.set_ylabel(r'$\rho_{CO2}$ [g/$cm^{3}$]')
    ax.tick_params(direction="in")
    
    # Create custom legend for marker colours and shapes
    colour_legend = [matplotlib.lines.Line2D([], [], color=custom_colours[i], marker='None', linestyle='solid', linewidth=3, label=f'{T-273}°C') for i, T in enumerate(T_list)]
    marker_legend = [matplotlib.lines.Line2D([], [], color='black', marker=custom_markers[i], markerfacecolor='None', linestyle='None', label=f'{type}') for i, type in enumerate(rho_types)]
    custom_legend = colour_legend + marker_legend
    
    # Add custom legend to the plot
    ax.legend(handles=custom_legend, loc='upper left').set_visible(True)
    
    # Update ticks 
    S.update_subplot_ticks(ax, x_lo=0, y_lo=0)
    
    if save_fig == True:
        save_fig_path = f"{data.path}/rhoCO2_comparison.png"
        plt.savefig(save_fig_path, dpi=1200)
        print(f"Plot successfully exported to {save_fig_path}.")
        
    if display_fig == True:
        plt.show()
        
    
if __name__ == "__main__":
    mix = S.BaseSolPol("CO2","HDPE")
    # plot_rho_sol(mix, [25+273, 35+273, 50+273], display_fig=True, save_fig=None)
    # plot_rho_pol(mix, [25+273, 35+273, 50+273], display_fig=False, save_fig=True)
    # plot_rhoCO2_comparison(mix, [25+273, 35+273, 50+273], display_fig=False, save_fig=True)
    plot_rhoCO2_comparison(mix, [35+273, 50+273], display_fig=True, save_fig=False)