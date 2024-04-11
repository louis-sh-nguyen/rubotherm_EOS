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

def plot_isotherm_pmv(base_obj, T_list:list[float], export_data:bool = False, display_fig:bool=True, save_fig:bool=False):
    """Function to plot sorption of EOS and experimental data to compare different partial molar volume approaches.

    Args:
        spm (class object): class object representing the sol-pol mixture.
        T (float): temperature [K].
        P (float): pressure [Pa].
        export_data (bool): export data.
    """
    now = datetime.now()  # current time
    time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM
    data = S.SolPolExpData(base_obj.sol, base_obj.pol)
    
    df = pd.DataFrame()
    df_SAFT = pd.DataFrame(columns=['T [K]', 'P [Pa]', 'S_sc_pmv1 [g/g]', 'S_sc_pmv2 [g/g]', 'S_sc_pmv3 [g/g]'])
    P_SAFT={}
    S_SAFT_pmv={}
    
    for i, T in enumerate(T_list):
        _df=data.get_sorption_data(T)
        
        # Sorption without swelling correction
        S_exp_woSW = (_df["MP1*[g]"]-data.m_met_filled+_df["ρ[g/cc]"]*(data.Vs+data.Vbasket)) / data.ms
        _df['S_NoCorrection[g/g]'] = S_exp_woSW
        
        # Iterate through each pmv method
        for k in ["1", "2", "3"]:
            
            # Calculate swelling ratio from SAFT at pmv
            objects = []
            SwR_SAFT_pmv = []
            for j, _p in enumerate(_df["P[bar]"].values *1e5):
                if j == 0:
                    obj = S.DetailedSolPol(base_obj, T, _p, pmv_method=k)                
                else:
                    x0_list = S.update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                    obj = S.DetailedSolPol(base_obj, T, _p, pmv_method=k, x0_sol_range = x0_list,)
                objects.append(obj)
                SwR_SAFT_pmv.append(obj.SwellingRatio)                    
            _df[f'SwR_SAFT_pmv{k}[cc/cc]'] = SwR_SAFT_pmv
        
            # Calculate corrected sorption from exp
            S_exp_SW_pmv = (_df["MP1*[g]"]-data.m_met_filled+_df["ρ[g/cc]"]*(data.Vs*(1+_df[f'SwR_SAFT_pmv{k}[cc/cc]'])+data.Vbasket)) / data.ms
            _df[f'S_Corrected_pmv{k}[g/g]'] = S_exp_SW_pmv
        
            # Calculates sorption from SAFT prediction at exp pressure data
            objects = []
            S_SAFT_pmv_pexp = []
            for j, _p in enumerate(_df["P[bar]"].values *1e5):
                if j == 0:
                    obj = S.DetailedSolPol(base_obj, T, _p, pmv_method=k)                
                else:
                    x0_list = S.update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                    obj = S.DetailedSolPol(base_obj, T, _p, pmv_method=k, x0_sol_range = x0_list,)
                objects.append(obj)
                S_SAFT_pmv_pexp.append(obj.S_sc)
            _df[f'S_SAFT_pmv{k}[g/g]'] = S_SAFT_pmv_pexp            
            
            # Calculates sorption from SAFT predictions only
            P_SAFT[T] = linspace(_df["P[bar]"].values[0],_df["P[bar]"].values[-1], 30) * 1e5   # [Pa]
            objects = []
            S_SAFT_pmv[k] = []
            for j, _p in enumerate(P_SAFT[T]):
                if j == 0:
                    obj = S.DetailedSolPol(base_obj, T, _p, pmv_method=k)                
                else:
                    x0_list = S.update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                    obj = S.DetailedSolPol(base_obj, T, _p, pmv_method=k, x0_sol_range = x0_list,)
                objects.append(obj)
                S_SAFT_pmv[k].append(obj.S_sc)
        
        df = pd.concat([df, _df], ignore_index=True)
        
        _df_SAFT = pd.DataFrame({'T [K]': [T for _p in P_SAFT[T]],
                                    'P [Pa]': [_p for _p in P_SAFT[T]],
                                    'S_sc_pmv1 [g/g]': S_SAFT_pmv['1'],
                                    'S_sc_pmv2 [g/g]': S_SAFT_pmv['2'],
                                    'S_sc_pmv3 [g/g]': S_SAFT_pmv['3'],})
        df_SAFT = pd.concat([df_SAFT,_df_SAFT], ignore_index=True)        
    
    print(df)
    print(df_SAFT)
    
    if export_data == True:
        now = datetime.now()  # current time
        time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM
        export_path = f"{data.path}/PlotIsothermPmv_{time_str}.xlsx"
        with pd.ExcelWriter(export_path) as writer:            
            df.to_excel(writer, sheet_name="exp", index=False)
            df_SAFT.to_excel(writer, sheet_name="SAFT", index=False)
        print("Data successfully exported to: ", export_path)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, T in enumerate(T_list):
        mask1 = abs(df['T[C]']+273 - T) <= (T*0.05)
        # Uncorrected
        # ax.plot(df[mask1]["P[bar]"], df[mask1]['S_NoCorrection[g/g]'],color=S.custom_colours[0], linestyle="None", marker="o",label=f"{T-273}°C exp - not corrected")
        
        # Corrected 
        ax.plot(df[mask1]["P[bar]"], df[mask1]['S_Corrected_pmv1[g/g]'],color=S.custom_colours[1], linestyle="None", marker="x",label=f"{T-273}°C exp - corrected with pmv1")
        ax.plot(df[mask1]["P[bar]"], df[mask1]['S_Corrected_pmv2[g/g]'],color=S.custom_colours[2], linestyle="None", marker="x",label=f"{T-273}°C exp - corrected with pmv2")
        ax.plot(df[mask1]["P[bar]"], df[mask1]['S_Corrected_pmv3[g/g]'],color=S.custom_colours[3], linestyle="None", marker="x",label=f"{T-273}°C exp - corrected with pmv3")
        
        # SAFT prediction
        mask2 = (df_SAFT['T [K]'] == T)
        ax.plot(df_SAFT[mask2]["P [Pa]"]*1e-5, df_SAFT[mask2]["S_sc_pmv1 [g/g]"],color=S.custom_colours[1], linestyle="solid",marker="None",label=f"{T-273}°C SAFT pmv1")
        ax.plot(df_SAFT[mask2]["P [Pa]"]*1e-5, df_SAFT[mask2]["S_sc_pmv2 [g/g]"],color=S.custom_colours[2], linestyle="solid",marker="None",label=f"{T-273}°C SAFT pmv2")
        ax.plot(df_SAFT[mask2]["P [Pa]"]*1e-5, df_SAFT[mask2]["S_sc_pmv3 [g/g]"],color=S.custom_colours[3], linestyle="solid",marker="None",label=f"{T-273}°C SAFT pmv3")
    ax.set_xlabel("P [bar]")
    ax.set_ylabel(r"$S_{sc}$ [$g_{sol}$/$g_{pol \: sc}$]")
    ax.set_ylim(top=0.15)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    
    if save_fig == True:
        save_fig_path = f"{data.path}/Anals/IsothermPmv_{time_str}.png"
        plt.savefig(save_fig_path, dpi=1200)
        print(f"Plot successfully exported to {save_fig_path}.")
    if display_fig == True:
        plt.show()

if __name__ == "__main__":
    mix = S.BaseSolPol("CO2","HDPE")
    for T in [25+273, 35+273, 50+273]:
        plot_isotherm_pmv(mix, [T], export_data=False, display_fig=False, save_fig=True)