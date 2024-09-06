"""
Script to plot absolute solubility using optimised epsilon_kl obtained from different pmv methods.
25 March 2024
Louis Nguyen
sn621@ic.ac.uk
"""
import solubility_master as S
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
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

mix = S.BaseSolPol("CO2","HDPE")

epskl_pmv1_dict = {"25°C": 259.78, "35°C": 244.23, "50°C": 251.05}
epskl_pmv2_dict = {"25°C": 259.42, "35°C": 247.37, "50°C": 254.28}
epskl_pmv3_dict = {"25°C": 259.08, "35°C": 225.12, "50°C": 231.72}


# T_list = [25+273, 35+273, 50+273]
pmv_list = ["1", "2", "3"]

def plot_eps_pmv(T: float, export_data:bool=False, display_fig:bool=True, save_fig:bool=False):
    data = S.SolPolExpData(mix.sol, mix.pol)
    _df1=data.get_sorption_data(T)

    now = datetime.now()  # current time
    time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM    

    df = pd.DataFrame(columns=['pmv_method', 'T [K]', 'P [Pa]', 'eps_kl', 'S_sc_SAFT [g/g]', 'S_sc_exp_NotCorrected [g/g]','S_sc_exp_corrected [g/g]'])

    for k, pmv in enumerate(pmv_list):    
                
        P_list = _df1["P[bar]"].values * 1e5    # [Pa]
        
        if pmv == "1":
            eps = epskl_pmv1_dict[f"{T-273}°C"]
        elif pmv == "2":
            eps = epskl_pmv2_dict[f"{T-273}°C"]
        elif pmv == "3":
            eps = epskl_pmv3_dict[f"{T-273}°C"]
        
        # Change epskl in the database
        mix.modify_kl(eps)
            
        objects = []
        solubility_SAFT = []
        solubility_exp_corrected= []
        for j, _p in enumerate(P_list):
            
            # Create objects at each T and P
            if j == 0:
                obj = S.DetailedSolPol(mix, T, _p, pmv_method=pmv)
            else:
                x0_list = S.update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                obj = S.DetailedSolPol(mix, T, _p, x0_sol_range = x0_list, pmv_method=pmv)
            
            # Sorption without swelling correction
            solubility_exp_NotCorrected = (_df1["MP1*[g]"]-data.m_met_filled+_df1["ρ[g/cc]"]*(data.Vs+data.Vbasket)) / data.ms            
            solubility_exp_NotCorrected = solubility_exp_NotCorrected.values
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
                    
        _df=pd.DataFrame({'pmv_method': [pmv for _p in P_list],
                        'T [K]': [T for _p in P_list],
                        'eps_kl': [eps for _p in P_list],
                        'P [Pa]':P_list,
                        'S_sc_SAFT [g/g]':solubility_SAFT,
                        'S_sc_exp_NotCorrected [g/g]': solubility_exp_NotCorrected,
                        'S_sc_exp_corrected [g/g]': solubility_exp_corrected})
        df = pd.concat([df,_df],ignore_index=True)
    print(df)

    # Expore data
    if export_data == True:
        data_export_path = f"{data.path}/PlotEpsPmv_{time_str}.xlsx"
        with pd.ExcelWriter(data_export_path) as writer:
            df.to_excel(writer, index=False)
        print("Data successfully exported to: ", data_export_path)


    # Calculate the continuous curve in SAFT model
    P_SAFT = linspace(_df1["P[bar]"].values[0],_df1["P[bar]"].values[-1], 30) * 1e5   # [Pa]
    S_SAFT = {}
    for k, pmv in enumerate(pmv_list):
        if pmv == "1":
            eps = epskl_pmv1_dict[f"{T-273}°C"]
        elif pmv == "2":
            eps = epskl_pmv2_dict[f"{T-273}°C"]
        elif pmv == "3":
            eps = epskl_pmv3_dict[f"{T-273}°C"]
        
        # Change epskl in the database
        mix.modify_kl(eps)
        objects = []
        S_SAFT[pmv] = []
        
        for j, _p in enumerate(P_SAFT):
            if j == 0:
                obj = S.DetailedSolPol(mix, T, _p, pmv_method=pmv)
            else:
                x0_list = S.update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                obj = S.DetailedSolPol(mix, T, _p, x0_sol_range = x0_list, pmv_method=pmv)
            objects.append(obj)
            S_SAFT[pmv].append(obj.S_sc_EOS)
            
    # Plot data
    fig = plt.figure()
    ax = fig.add_subplot(111)  # SAFT only
    
    # Uncorrected exp
    mask = (df['T [K]'] == T) & (df['pmv_method'] == "1")
    # ax.plot(df[mask]['P [Pa]']*1e-5, df[mask]['S_sc_exp_NotCorrected [g/g]'], 
    #         color=S.custom_colours[0], linestyle="None", marker=S.custom_markers[0], label=f"{T-273}°C, exp - uncorrected")
    for k, pmv in enumerate(pmv_list):    
        mask = (df['T [K]'] == T) & (df['pmv_method'] == pmv)
        
        # Corrected exp
        ax.plot(df[mask]['P [Pa]']*1e-5, df[mask]['S_sc_exp_corrected [g/g]'], 
                color=S.custom_colours[k+1], linestyle="None", marker=S.custom_markers[1], label=f"{T-273}°C, pmv{pmv} exp - corrected")
        
        # SAFT predictions
        ax.plot(P_SAFT*1e-5, S_SAFT[pmv], 
                color=S.custom_colours[k+1], linestyle="solid", marker="None", label=f"{T-273}°C, pmv{pmv} SAFT")
        

    ax.set_xlabel("P [bar]")
    ax.set_ylabel(r"$S_{sc}$ [$g_{sol}$/$g_{pol \: sc}$]")
    ax.set_ylim(top=0.025)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    if display_fig == True:
        plt.show()
    if save_fig == True:
        save_fig_path = f"{data.path}/Anals/IsothermEpsEOSvExp_{time_str}.png"
        plt.savefig(save_fig_path, dpi=1200,)
        print(f"Plot successfully exported to {save_fig_path}.")
        

if __name__ == "__main__":
    for T in [25+273, 35+273, 50+273]:
        plot_eps_pmv(T, display_fig=False, export_data=False, save_fig=True)