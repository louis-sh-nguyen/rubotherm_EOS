import solubility_master as S
from numpy import *
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

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

def plot_isotherm_EOSvExp(base_obj, T_list:list[float], export_data:bool = False, display_fig:bool=True, save_fig:bool=False):
    """Function to plot sorption of EOS and experimental data (not corrected for swelling).

    Args:
        spm (class object): class object representing the sol-pol mixture.
        T (float): temperature [K].
        P (float): pressure [Pa].
        export_data (bool): export data.
    """
    data = S.SolPolExpData(base_obj.sol, base_obj.pol)
    now = datetime.now()  # current time
    time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM
    
    P_SAFT={}
    S_SAFT={}
    
    # Finding largest pressure value in experimental data
    _df = {}
    for i, T in enumerate(T_list):    
        _df[T]=data.get_sorption_data(T)
        max_i = _df[T]["P[bar]"].max()
        if i == 0:
            pbar_max = max_i
        else:
            pbar_max = max(pbar_max, max_i)
    
    # Set pressure ponits for SAFT predictions
    for i, T in enumerate(T_list):
        P_SAFT[T] = linspace(1, pbar_max*1e5, 100)  # [Pa]    
    
    df={}
    for i, T in enumerate(T_list):
        
        _df=data.get_sorption_data(T)
        
        # Sorption without swelling correction
        S_exp_woSW = (_df["MP1*[g]"]-data.m_met_filled+_df["ρ[g/cc]"]*(data.Vs+data.Vbasket)) / data.ms
        _df['S_exp_woSW[g/g]']=S_exp_woSW
        
        # Calculate swelling ratio from SAFT
        objects = []
        SwR_SAFT = []
        for j, _p in enumerate(_df["P[bar]"].values *1e5):
            if j == 0:
                obj = S.DetailedSolPol(base_obj, T, _p,)
            else:
                x0_list = S.update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                obj = S.DetailedSolPol(base_obj, T, _p, x0_sol_range = x0_list,)
            objects.append(obj)
            SwR_SAFT.append(obj.SwellingRatio)

        _df['SwR_SAFT[cc/cc]'] = SwR_SAFT
        # Sorption with swelling correction
        S_exp_SW = (_df["MP1*[g]"]-data.m_met_filled+_df["ρ[g/cc]"]*(data.Vs*(1+_df['SwR_SAFT[cc/cc]'])+data.Vbasket)) / data.ms
        _df['S_exp_SW[g/g]'] = S_exp_SW
        
        # Calculate S_sc for continuous SAFT line        
        objects = []
        S_SAFT[T] = []        
        for j, _p in enumerate(P_SAFT[T]):
            if j == 0:
                obj = S.DetailedSolPol(base_obj, T, _p,)
            else:
                x0_list = S.update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                obj = S.DetailedSolPol(base_obj, T, _p, x0_sol_range = x0_list,)
            objects.append(obj)
            S_SAFT[T].append(obj.S_sc)
            
        # Calculate S_sc at experimental pressure points
        objects = []
        S_SAFT_pexp = []
        for j, _p in enumerate(_df["P[bar]"]*1e5):
            if j == 0:
                obj = S.DetailedSolPol(base_obj, T, _p,)
            else:
                x0_list = S.update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                obj = S.DetailedSolPol(base_obj, T, _p, x0_sol_range = x0_list,)
            objects.append(obj)
            S_SAFT_pexp.append(obj.S_sc)
        _df['S_SAFT[g/g]'] = S_SAFT_pexp
        
        df[T] = _df
        print("")
        print("T = ", T)
        print(df[T])
        print("")
    
    
    if export_data == True:
        export_path = f"{data.path}/PlotIsothermEOSvExp_{time_str}.xlsx"
        with pd.ExcelWriter(export_path) as writer:
            for T in T_list:
                df[T].to_excel(writer, sheet_name=f"{T-273}C", index=False)
        print("Data successfully exported to: ", export_path)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, T in enumerate(T_list):
        # Uncorrected
        # ax.plot(df[T]["P[bar]"],df[T]['S_exp_woSW[g/g]'],color=S.custom_colours[i], linestyle="None", marker="o",label=f"{T-273}°C exp - uncorrected")
        
        # Corrected
        ax.plot(df[T]["P[bar]"],df[T]['S_exp_SW[g/g]'],color=S.custom_colours[i], linestyle="None", marker="x",label=f"{T-273}°C exp - corrected")
        
        # EoS Prediction
        ax.plot(P_SAFT[T]*1e-5,S_SAFT[T],color=S.custom_colours[i], linestyle="solid",marker="None",label=f"{T-273}°C SAFT")
    ax.set_xlabel("P [bar]")
    ax.set_ylabel(r"$S_{sc}$ [$g_{sol}$/$g_{pol \: sc}$]")
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    if display_fig == True:
        plt.show()
    if save_fig == True:
        save_fig_path = f"{data.path}/IsothermEpsEOSvExp_{time_str}.png"
        plt.savefig(save_fig_path, dpi=1200)
        print(f"Plot successfully exported to {save_fig_path}.")

if __name__ == "__main__":
    mix = S.BaseSolPol("CO2","HDPE")
    # plot_isotherm_EOSvExp(mix,[25+273, 35+273, 50+273], export_data=False, display_fig=False, save_fig=True)

    mix.modify_kl(259.78)
    plot_isotherm_EOSvExp(mix, T_list=[25+273], export_data=False, display_fig=False, save_fig=True)
    mix.modify_kl(244.23)
    plot_isotherm_EOSvExp(mix, T_list=[35+273], export_data=False, display_fig=False, save_fig=True)
    mix.modify_kl(251.05)
    plot_isotherm_EOSvExp(mix, T_list=[50+273], export_data=False, display_fig=False, save_fig=True)