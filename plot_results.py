# Mendatory for addcopyfighandler package
import warnings
warnings.filterwarnings("ignore")
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

# Import libraries
import solubility_master as S
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os   
import numpy as np
import addcopyfighandler

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

custom_colours = ['green','blue','red','purple', 'grey', 'brown','plum','indigo','olive' 'orange']
custom_markers = ['x', 'o', 's',  '^', '*', 'D', '.']

def plot_litData_pressure(display_fig = True, save_fig = False):
    try:
        # Read exp file
        path = os.path.dirname(__file__)
        filepath = path + '/model_lit_data.xlsx'
        file = pd.ExcelFile(filepath, engine='openpyxl')
        df_model = pd.read_excel(file, sheet_name='model')
        df_lit = pd.read_excel(file, sheet_name='lit')
        
    except Exception as e:
        print("Error: ")
        print(e)

    else:
        # Get common list of temperatures in df_model and df_lit, starting with T values from models ([25, 35, 50])
        T_values_model_unique = df_model['T [°C]'].unique()
        T_values_lit_unique = df_lit['T [°C]'].unique()
        T_values_diff = np.setdiff1d(T_values_lit_unique, T_values_model_unique)
        T_values_diff = np.sort(T_values_diff)
        T_values_unique = np.concatenate((T_values_model_unique, T_values_diff))
        
        # Plot
        fig, ax = plt.subplots()
        
        for i, T in enumerate(T_values_unique):
            # Plot model data
            df_model_T = df_model[df_model['T [°C]'] == T]
            
            if not df_model_T.empty:
                ax.plot(df_model_T['p [bar]'], df_model_T['S_am [g/g_am]'], label=f'model {T}°C - current work', color=custom_colours[i], marker=custom_markers[0], markerfacecolor='None', linestyle='None')
                            
            # Plot literature data
            df_lit_T = df_lit[df_lit['T [°C]'] == T]
            refID_unique = df_lit_T['ref ID'].unique()
            if not df_lit_T.empty:
                for j, refID in enumerate(refID_unique):
                    df_T_refID = df_lit_T[df_lit_T['ref ID'] == refID]
                    ax.plot(df_T_refID['p [bar]'], df_T_refID['S_am [g/g_am]'], label=f'lit {T}°C - {refID}', color=custom_colours[i], marker=custom_markers[j+1], markerfacecolor='None', linestyle='None')

        ax.set_xlabel('p [bar]')
        ax.set_ylabel(r'$S_{am}$ [$g_{sol}$/$g_{pol\,am}$]')
        ax.set_title(r'$CO_{2}$-HDPE')
        ax.legend(fontsize='10').set_visible(True)
        
        S.update_subplot_ticks(ax, x_lo=0, y_lo=0)
        
        if save_fig == True:
            plt.savefig(path + '/CO2-HDPE_lit_vs_model.png', dpi=1200)
        
        if display_fig == True:
            plt.show()

def plot_litData_fugacity(display_fig = True, save_fig = False):
    try:
        # Read exp file
        path = os.path.dirname(__file__)
        filepath = path + '/model_lit_data.xlsx'
        file = pd.ExcelFile(filepath, engine='openpyxl')
        df_model = pd.read_excel(file, sheet_name='model')
        df_lit = pd.read_excel(file, sheet_name='lit')
        
    except Exception as e:
        print("Error: ")
        print(e)

    else:
        # Get common list of temperatures in df_model and df_lit, starting with T values from models ([25, 35, 50])
        T_values_model_unique = df_model['T [°C]'].unique()
        T_values_lit_unique = df_lit['T [°C]'].unique()
        T_values_diff = np.setdiff1d(T_values_lit_unique, T_values_model_unique)
        T_values_diff = np.sort(T_values_diff)
        T_values_unique = np.concatenate((T_values_model_unique, T_values_diff))
        
        # Plot
        fig, ax = plt.subplots()
        
        for i, T in enumerate(T_values_unique):
            # Plot model data
            df_model_T = df_model[df_model['T [°C]'] == T]
            
            if not df_model_T.empty:
                ax.plot(df_model_T['f [bar]'], df_model_T['S_am [g/g_am]'], label=f'model {T}°C - current work', color=custom_colours[i], marker=custom_markers[0], markerfacecolor='None', linestyle='None')
                            
            # Plot literature data
            df_lit_T = df_lit[df_lit['T [°C]'] == T]
            refID_unique = df_lit_T['ref ID'].unique()
            if not df_lit_T.empty:
                for j, refID in enumerate(refID_unique):
                    df_T_refID = df_lit_T[df_lit_T['ref ID'] == refID]
                    ax.plot(df_T_refID['f [bar]'], df_T_refID['S_am [g/g_am]'], label=f'lit {T}°C - {refID}', color=custom_colours[i], marker=custom_markers[j+1], markerfacecolor='None', linestyle='None')

        ax.set_xlabel('f [bar]')
        ax.set_ylabel(r'$S_{am}$ [$g_{sol}$/$g_{pol\,am}$]')
        ax.set_title(r'$CO_{2}$-HDPE')
        ax.legend(fontsize='xx-small').set_visible(True)
        
        S.update_subplot_ticks(ax, x_lo=0, y_lo=0)
        
        if save_fig == True:
            plt.savefig(path + '/CO2-HDPE_lit_vs_model_fugacity.png', dpi=1200)
        
        if display_fig == True:
            plt.show()


def plot_modelResults(T: float, exp_results=True, sw_results=True, saft_results=True, trimmed_data=False, plot_pcv=False, plot_rhoTPS=False, display_fig = True, save_fig = False):
    try:
        # Read data file
        path = os.path.dirname(__file__)
        filepath = path + '/model_lit_data.xlsx'
        file = pd.ExcelFile(filepath, engine='openpyxl')
        df_exp = pd.read_excel(file, sheet_name='EXP')
        df_sw = pd.read_excel(file, sheet_name='SW')
        df_saft = pd.read_excel(file, sheet_name='SAFT')
        
    except Exception as e:
        print("Error: ")
        print(e)

    else:
        # trimmed data
        if trimmed_data == True:
            df_exp = df_exp[df_exp['trimmed'] == 'Yes']
            df_sw = df_sw[df_sw['trimmed'] == 'Yes']
            df_saft = df_saft[df_saft['trimmed'] == 'Yes']
        
        # Filter by temperature
        df_exp_T = df_exp[df_exp['T [°C]'] == (T-273)]
        df_sw_T = df_sw[df_sw['T [°C]'] == (T-273)]
        df_saft_T = df_saft[df_saft['T [°C]'] == (T-273)]
        
        # Plot
        n_plot = 2
        if plot_pcv == True:
            n_plot += 2
        if plot_rhoTPS == True:
            n_plot += 1
        
        fig, ax = plt.subplots(n_plot, 1, figsize=(4, 2*n_plot))       
        
        # Plot S_sc vs. Pressure
        if exp_results == True:
            ax[0].plot(df_exp_T['p [bar]'], df_exp_T['S_sc [g/g_overall]'], label=f'exp', color='black', marker=custom_markers[0], markerfacecolor='None', linestyle='None')
        if sw_results == True:
            ax[0].plot(df_sw_T['p [bar]'], df_sw_T['S_sc [g/g_overall]'], label=f'SW EoS', color='black', marker=custom_markers[1], markerfacecolor='None', linestyle='None')                
        if saft_results == True:
            ax[0].plot(df_saft_T['p [bar]'], df_saft_T['S_sc [g/g_overall]'], label=f'SAFT-γ Mie EoS', color='black', marker=custom_markers[2], markerfacecolor='None', linestyle='None')                
        
        ax[0].set_xlabel('p [bar]')
        ax[0].set_ylabel(r'$S_{sc}$ [$g_{sol}$/$g_{pol}$]')
        ax[0].set_title(f'T = {T-273} °C')
        # ax[0].legend().set_visible(True)
        
        # Plot SwR vs. Pressure
        if exp_results == True:
            ax[1].plot(df_exp_T['p [bar]'], df_exp_T['SwellingRatio [m3/m3]'], label=f'exp', color='black', marker=custom_markers[0], markerfacecolor='None', linestyle='None')
        if sw_results == True:
            ax[1].plot(df_sw_T['p [bar]'], df_sw_T['SwellingRatio [m3/m3]'], label=f'SW EoS', color='black', marker=custom_markers[1], markerfacecolor='None', linestyle='None')
        if saft_results == True:
            ax[1].plot(df_saft_T['p [bar]'], df_saft_T['SwellingRatio [m3/m3]'], label=f'SAFT-γ Mie EoS', color='black', marker=custom_markers[2], markerfacecolor='None', linestyle='None')
        
        ax[1].set_xlabel('p [bar]')
        ax[1].set_ylabel(r'SwR [$cm^{3}$/$cm^{3}$]')
        # ax[1].legend().set_visible(True)
        
        # Update ticks to cover all data points
        S.update_subplot_ticks(ax[0], x_lo=0, y_lo=0)
        S.update_subplot_ticks(ax[1], x_lo=0, y_lo=0)
        
        i = 1
        
        # Plot PCV vs. Pressure
        if plot_pcv == True:            
            if exp_results == True:
                ax[i+1].plot(df_exp_T['p [bar]'], df_exp_T['Vs [m3/g]']*1e6, label=f'exp', color='black', marker=custom_markers[0], markerfacecolor='None', linestyle='None')
                ax[i+2].plot(df_exp_T['p [bar]'], df_exp_T['Vp [m3/g]']*1e6, label=f'exp', color='black', marker=custom_markers[0], markerfacecolor='None', linestyle='None')
            if sw_results == True:
                ax[i+1].plot(df_sw_T['p [bar]'], df_sw_T['Vs [m3/g]']*1e6, label=f'SW EoS', color='black', marker=custom_markers[1], markerfacecolor='None', linestyle='None')
                ax[i+2].plot(df_sw_T['p [bar]'], df_sw_T['Vp [m3/g]']*1e6, label=f'SW EoS', color='black', marker=custom_markers[1], markerfacecolor='None', linestyle='None')
            if saft_results == True:
                ax[i+1].plot(df_saft_T['p [bar]'], df_saft_T['Vs [m3/m3]']*1e6, label=f'SAFT-γ Mie EoS', color='black', marker=custom_markers[2], markerfacecolor='None', linestyle='None')
                ax[i+2].plot(df_saft_T['p [bar]'], df_saft_T['Vp [m3/m3]']*1e6, label=f'SAFT-γ Mie EoS', color='black', marker=custom_markers[2], markerfacecolor='None', linestyle='None')
            
            ax[i+1].set_xlabel('p [bar]')
            ax[i+1].set_ylabel(r'$\bar{V}_{s}$ $(T,p,S_{sc})$ [$cm^{3}$/g]')
            ax[i+2].set_xlabel('p [bar]')
            ax[i+2].set_ylabel(r'$\bar{V}_{p,am}$ $(T,p,S_{sc})$ [$cm^{3}$/g]')
            
            # Update ticks to cover all data points
            S.update_subplot_ticks(ax[i+1], x_lo=0)
            S.update_subplot_ticks(ax[i+2], x_lo=0)
            
            i += 2
        
        # Plot rhoTPS vs. Pressure
        if plot_rhoTPS == True:
            if exp_results == True:
                ax[i+1].plot(df_exp_T['p [bar]'], df_exp_T['rho_TPS [g/m3]']*1e-6, label=f'exp', color='black', marker=custom_markers[0], markerfacecolor='None', linestyle='None')
            if sw_results == True:
                ax[i+1].plot(df_sw_T['p [bar]'], df_sw_T['rho_TPS [g/m3]']*1e-6, label=f'SW EoS', color='black', marker=custom_markers[1], markerfacecolor='None', linestyle='None')
            if saft_results == True:
                ax[i+1].plot(df_saft_T['p [bar]'], df_saft_T['rho_TPS [g/m3]']*1e-6, label=f'SAFT-γ Mie EoS', color='black', marker=custom_markers[2], markerfacecolor='None', linestyle='None')
            
            ax[i+1].set_xlabel('p [bar]')
            ax[i+1].set_ylabel(r'$\rho_{tot}$ $(T,p,S_{sc})$ [g/$cm^{3}$]')
            
            # Update ticks to cover all data points
            S.update_subplot_ticks(ax[i+1], x_lo=0)
            
            i += 1
        
        if T in [35+273, 50+273]:
            for _ax in ax:
                _ax.set_xlim(right=250)
        
        name = f'CO2-HDPE_modelResults_{T-273}C'
        if exp_results == True:
            name += '_EXP'
        if sw_results == True:
            name += '_SW'
        if saft_results == True:
            name += '_SAFT'
        if plot_pcv == True:
            name += '_PCV'
        if plot_rhoTPS == True:
            name += '_rhoTPS'
        if trimmed_data == True:
            name += '_trimmed'
        
        if save_fig == True:
            plt.savefig(path + f'/{name}.png', dpi=1200)
        if display_fig == True:
            plt.show()

def plot_rawData(T: float, display_fig = True, save_fig = False):
    try:
        # Read data file
        path = os.path.dirname(__file__)
        filepath = path + '/model_lit_data.xlsx'
        file = pd.ExcelFile(filepath, engine='openpyxl')
        df_raw = pd.read_excel(file, sheet_name='raw')
        
    except Exception as e:
        print("Error: ")
        print(e)

    else:
        # Filter by temperature
        df_raw_T = df_raw[df_raw['T [°C]'] == (T-273)]
        
        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot m_raw vs. Pressure
        ax.plot(df_raw_T['p [bar]'], df_raw_T['m_raw [g]'], label=f'raw', color='black', marker='o', markerfacecolor='None', linestyle='None')                
        
        ax.set_xlabel('p [bar]')
        ax.set_ylabel(r'$m_{raw}$ [g]')
        ax.set_title(f'T = {T-273} °C')
        # ax.legend().set_visible(True)
        
        # Update ticks to cover all data points
        S.update_subplot_ticks(ax, x_lo=0)
        
        if T in [35+273, 50+273]:
            for _ax in ax:
                _ax.set_xlim(right=250)        
        
        if save_fig == True:
            plt.savefig(path + f'/CO2-HDPE_rawData_{T-273}C.png', dpi=1200)
        if display_fig == True:
            plt.show()

if __name__ == '__main__':
    # plot_litData_fugacity(display_fig=True, save_fig=False)
    # for T in [25+273, 35+273, 50+273]:
    #     plot_modelResults(T, display_fig=False, save_fig=True)
    plot_modelResults(50+273, trimmed_data=True,
                      plot_pcv=False, plot_rhoTPS=False,
                      sw_results=True, exp_results=False, saft_results=False, 
                      display_fig=True, save_fig=True)
    # plot_rawData(T=50+273, display_fig=True, save_fig=True)