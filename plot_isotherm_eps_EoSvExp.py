"""
Script to plot isotherm from EoS prediction and corrected sorption for multiple epsilon_kl .
11 Apr 2024
Louis Nguyen
sn621@ic.ac.uk
"""

import solubility_master as S
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from matplotlib.lines import Line2D

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

def plot_isotherm_epskl_EOSvExp(base_obj, T_list: list, eps_list:list, export_data:bool=False, display_fig:bool=True, save_fig:bool=False):
    """Functio to plot solubility isotherms for chosen eps_kl values at different 

    Args:
        base_obj (_type_): _description_
        T_list (list): _description_
        eps_list (list): _description_
        export_data (bool, optional): _description_. Defaults to False.
    """
    
    data = S.SolPolExpData(base_obj.sol, base_obj.pol)
    now = datetime.now()  # current time
    time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM    
    
    df = pd.DataFrame(columns=['T [K]', 'eps_kl', 'P [Pa]', 'S_sc_SAFT [g/g]', 'S_sc_exp_corrected [g/g]'])
    for T in T_list:        
        _df1=data.get_sorption_data(T)
        P_list = _df1["P[bar]"].values * 1e5    # [Pa]
        
        for eps in eps_list:
            base_obj.modify_kl(eps)
            
            objects = []
            solubility_SAFT = []
            solubility_exp_corrected= []
            for j, _p in enumerate(P_list):
                # Create objects at each T and P
                if j == 0:
                    obj = S.DetailedSolPol(base_obj, T, _p,)
                else:
                    x0_list = S.update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                    obj = S.DetailedSolPol(base_obj, T, _p, x0_sol_range = x0_list,)
                
                # Calculate solubility prediction from SAFT
                try:
                    _S_SAFT = obj.S_sc_EOS
                except:
                    _S_SAFT = None
                
                # Calculate corrected swelling using SAFT and exp data
                mask = abs(_df1["P[bar]"]*1e5 - _p) <= (_p*0.01)
                try:
                    _SwR = obj.SwellingRatio
                    _S_exp_corrected = (_df1[mask]["MP1*[g]"]-data.m_met_filled+_df1[mask]["ρ[g/cc]"]*(data.Vs*(1+_SwR)+data.Vbasket)) / data.ms
                    _S_exp_corrected = _S_exp_corrected.values[0]
                except:
                    _S_exp_corrected = None
                
                objects.append(obj)
                solubility_SAFT.append(_S_SAFT)
                solubility_exp_corrected.append(_S_exp_corrected)
                
            _df=pd.DataFrame({'T [K]': [T for _p in P_list],
                              'eps_kl': [eps for _p in P_list],
                              'P [Pa]':P_list,
                              'S_sc_SAFT [g/g]':solubility_SAFT,
                              'S_sc_exp_corrected [g/g]':solubility_exp_corrected})
            df = pd.concat([df,_df],ignore_index=True)
    print(df)
    
    
    if export_data == True:
        data_export_path = f"{data.path}/PlotIsothermEpsEOSvExp_{time_str}.xlsx"
        with pd.ExcelWriter(data_export_path) as writer:
            df.to_excel(writer, index=False)
        print("Data successfully exported to: ", data_export_path)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)  
    for i, T in enumerate(T_list):
        for j, eps in enumerate(eps_list):            
            mask = (df['T [K]'] == T) & (df['eps_kl'] == eps)
            ax.plot(df[mask]['P [Pa]']*1e-5, df[mask]['S_sc_SAFT [g/g]'], color=S.custom_colours[i+1], linestyle="solid", marker=S.custom_markers[j], label=f"{T-273}°C eps={eps}")
            ax.plot(df[mask]['P [Pa]']*1e-5, df[mask]['S_sc_exp_corrected [g/g]'], color=S.custom_colours[i+1], linestyle="solid", marker=S.custom_markers[j], label=f"{T-273}°C eps={eps}")
    
    ax.set_xlabel("P [bar]")
    ax.set_ylabel(r"$S_{sc}$ [$g_{sol}$/$g_{pol \: sc}$]")
    ax.tick_params(direction="in")

    # ax1.set_ylim(top=1.)
    # ax2.set_ylim(top=2.)
    
    # Legends setting
    legend_markers = [Line2D([0], [0], linestyle="None", marker=S.custom_markers[i], color="black",
                             label=f"eps = {eps}") for i, eps in enumerate(eps_list)]
    legend_colours = [Line2D([0], [0], marker="None", color=S.custom_colours[i+1],
                             label=f"T = {T-273}°C") for i, T in enumerate(T_list)]
    legend = legend_colours + legend_markers
    ax.legend(handles=legend)
    
    if save_fig == True:
        save_fig_path = f"{data.path}/IsothermEpsEOSvExp_{time_str}.png"
        plt.savefig(save_fig_path, dpi=1200)
        print(f"Plot successfully exported to {save_fig_path}.")
    if display_fig == True:
        plt.show()
