import solubility_master as S
from numpy import *
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import time
import pandas as pd
from scipy.optimize import fsolve

# Change subplot styling
def Test_solve_parameters_rootEvaluation(base_obj, T: float, p: float):
    print('Experimental p:', p)
    
    obj = S.DetailedSolPol(base_obj, T, p,)
    
    #* Using solve_parameters_plots_NEW
    # obj.solve_parameters_plots_NEW()
    
    #* Using solve_parametersplots_NEW2
    obj.solve_parameters_plots_NEW2() 
    
def _TestMultiplePressure_solve_parameters(base_obj, T:float,):
    data = S.SolPolExpData(base_obj.sol, base_obj.pol)
    # Finding largest pressure value in experimental data
    _df = {}
    _df[T]=data.get_sorption_data(T)
    pressures = _df[T]["P[bar]"] * 1e5 # [Pa]
    S_sc_exp_values = []
    SwellingRatio_values = []
    for p in pressures:
        print('p:', p)
        obj = S.DetailedSolPol(base_obj, T, p)
        
        #* Using swelling
        SwR, S_sc = obj.solve_parameters_fsolve_NEW()
        
        #* Without Swelling
        # SwR, S_sc = obj.solve_parameters_fsolve_NEW_noSwelling()
        
        print('S_sc:', S_sc)
        print('SwellingRatio:', SwR)
        S_sc_exp_values.append(S_sc)
        SwellingRatio_values.append(SwR)
        print('')
    print('S_sc_exp_values:', *S_sc_exp_values)
    print('SwellingRatio_values:', *SwellingRatio_values)
    print('')
    
    # Plotting S_sc_exp against pressures
    fig=plt.figure(figsize=(4, 7))  # (width, height)
    ax1=fig.add_subplot(211)
    ax1.plot(pressures, S_sc_exp_values, color='black', marker='x', linestyle='None')
    ax1.set_xlabel('Pressure [Pa]')
    ax1.set_ylabel(r'$S_{sc}^{exp}$ [g/g]')
    ax1.set_title(r'$S_{sc}^{exp}$ vs Pressure')

    # Get the length of major ticks on the x-axis
    ax1_x_major_tick_length = ax1.get_xticks()[1] - ax1.get_xticks()[0]
    ax1_y_major_tick_length = ax1.get_yticks()[1] - ax1.get_yticks()[0]
    
    # Set adjust x and y tick to cover all data
    ax1.set_xlim(left=0, right=max(pressures) + ax1_x_major_tick_length)
    ax1.set_ylim(bottom=0, top=max(S_sc_exp_values) + ax1_y_major_tick_length)    
    
    # Plotting SwellingRatio against pressures
    ax2=fig.add_subplot(212)
    ax2.plot(pressures, SwellingRatio_values, color='black', marker='x', linestyle='None')
    ax2.set_xlabel('Pressure [Pa]')
    ax2.set_ylabel(r'SwellingRatio [$m^{3}$/$m^{3}$]')
    ax2.set_title('SwellingRatio vs Pressure')
    
    # Get the length of major ticks on the x-axis
    ax2_x_major_tick_length = ax2.get_xticks()[1] - ax2.get_xticks()[0]
    ax2_y_major_tick_length = ax2.get_yticks()[1] - ax2.get_yticks()[0]
    
    # Set adjust x and y tick to cover all data
    ax2.set_xlim(left=0, right=max(pressures) + ax2_x_major_tick_length)
    ax2.set_ylim(bottom=0, top=max(SwellingRatio_values) + ax2_y_major_tick_length)

    # plt.suptitle(f'Temperature: {T-273} °C')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{base_obj.sol}_{base_obj.pol}_T{T-273}.png', dpi=1200)

def _Test_SinglePhaseDensity(base_obj, T):
    data = S.SolPolExpData(base_obj.sol, base_obj.pol)
    
    # Finding largest pressure value in experimental data
    _df = {}
    _df=data.get_sorption_data(T)
    pressures = _df["P[bar]"] * 1e5 # [Pa]
    S_sc_exp_values = []
    SwellingRatio_values = []
    
    for p in pressures:
        print('p:', p)
        obj = S.DetailedSolPol(base_obj, T, p)
        
        #* Using swelling
        SwR, S_sc = obj.solve_parameters_fsolve_NEW()
        
        #* Without Swelling
        # SwR, S_sc = obj.solve_parameters_fsolve_NEW_noSwelling()
        
        # print('S_sc:', S_sc)
        # print('SwellingRatio:', SwR)
        S_sc_exp_values.append(S_sc)
        SwellingRatio_values.append(SwR)
        print('')
    
    # Print results at all pressures
    print('S_sc_exp_values:', *S_sc_exp_values)
    print('SwellingRatio_values:', *SwellingRatio_values)
    print('')
    
    def get_properties(T, p, S_sc):

        obj = S.DetailedSolPol(base_obj, T, p)
        
        # Get omega_cr
        omega_cr = obj.omega_cr
        
        # Get rho_pol_cr
        rho_p_c = obj.rho_pol_cr  # [g/m^3]
        
        # Calculate S_am
        S_am = S_sc / (1-omega_cr)
        
        # Calculate x
        omega_p = 1/(S_am+1)     # [g/g]
        omega_s = 1 - omega_p   # [g/g]
        x_s = (omega_s/obj.MW_sol) / (omega_s/obj.MW_sol + omega_p/obj.MW_pol)   #[mol/mol]
        x_p = 1 - x_s   #[mol/mol]           
        x = hstack([x_s, x_p])   # [mol/mol]
        
        # Calculate Vs and Vp
        V_s, V_p =  obj.Vs_Vp_pmv1(obj.T, obj.P, S_am)    #* Default
        # V_s, V_p =  obj.Vs_Vp_pmv2(obj.T, obj.P)  #* TEST
        
        def rho_tot(S_sc):
            # Calculate S_am_exp
            S_am = S_sc / (1-omega_cr)            
            
            # Calculte rho_tot
            rho_p_am = 1/(S_am*V_s + V_p)  # [g/m^3]
            rho_p_tol = 1/((1-omega_cr)/rho_p_am + omega_cr/rho_p_c)  # [g/m^3]
            rho_tot = (1 + S_sc) * rho_p_tol  # [g/m^3]
            
            return rho_tot
        
        # Calculate mixture Density
        # rho = obj.SinglePhaseDensity(x, obj.T, obj.P, 'L')   # [mol/m^3]
        rho_t = rho_tot(S_sc)   # [mol/m^3]        
        
        return x[0], rho_t, V_s, V_p
    
    rho_TPS = []
    x_s = []
    Vs = []
    Vp = []
    rhoT00_rhoTPS = []
    for i, p in enumerate(pressures):        
        result = get_properties(T, p, S_sc_exp_values[i])
        x_s.append(result[0])
        rho_TPS.append(result[1])
        Vs.append(result[2])
        Vp.append(result[3])
        rhoT00_rhoTPS.append((SwellingRatio_values[i]+1) / (S_sc_exp_values[i]+1))        
    
    print('pressures:', pressures.values.tolist())
    print('rho_TPS:', rho_TPS)
    print('x_s:', x_s)
    print('Vs:', Vs)
    print('Vp:', Vp)
    print('rhoT00_rhoTPS:', rhoT00_rhoTPS)
        
    # Plotting SinglePhaseDensity against SwellingRatio
    plt.figure(figsize=(5, 2*7))  # (width, height)        
    
    plt.subplot(7, 1, 1)
    plt.plot(pressures, S_sc_exp_values, color='black', marker='x')
    plt.xlabel('Pressure [Pa]')
    plt.ylabel(r'$S_{sc}^{exp}$ [g/g]')
    
    plt.subplot(7, 1, 2)
    plt.plot(pressures, SwellingRatio_values, color='black', marker='x')
    plt.xlabel('Pressure [Pa]')
    plt.ylabel(r'SwellingRatio [$m^{3}$/$m^{3}$]')
    
    plt.subplot(7, 1, 3)
    plt.plot(pressures, rho_TPS, color='black', marker='x')
    plt.xlabel('Pressure [Pa]')
    plt.ylabel(r'$\rho_{tot} ( T,p,S_{sc} ) $ [$mol$/$m^{3}$]')
    
    plt.subplot(7, 1, 4)
    plt.plot(pressures, x_s, color='black', marker='x')
    plt.xlabel('Pressure [Pa]')
    plt.ylabel(r'$x_{s}$')
    
    plt.subplot(7, 1, 5)
    plt.plot(pressures, Vs, color='black', marker='x')
    plt.xlabel('Pressure [Pa]')
    plt.ylabel(r'$V_{s}$ [$m^{3}$/g]')
    
    plt.subplot(7, 1, 6)
    plt.plot(pressures, Vp, color='black', marker='x')
    plt.xlabel('Pressure [Pa]')
    plt.ylabel(r'$V_{p}$ [$m^{3}$/g]')
    
    plt.subplot(7, 1, 7)
    plt.plot(pressures, rhoT00_rhoTPS, color='black', marker='x')
    plt.xlabel('Pressure [Pa]')
    plt.ylabel(r'$\frac{\rho_{tot} (T,0,0)} {\rho_{tot} (T,P,S_{sc}) }$')
    plt.show()

def Test_solve_parameters_fsolve_NEW(base_obj, T, p_selected_list = None, display_plot=True, save_plot_dir=None, export_csv_dir=None):
    data = S.SolPolExpData(base_obj.sol, base_obj.pol)    
    _df = {}
    _df=data.get_sorption_data(T)
    
    if p_selected_list == None:
        # Use all pressure values
        pressures = _df["P[bar]"] * 1e5 # [Pa]
        
    elif p_selected_list != None:
        # Use selected pressure values    
        # mask = abs(_df["P[bar]"]*1e5 - p_selected) <= (p_selected*0.01)
        # pressures = _df[mask]["P[bar]"] * 1e5 
        
        #* Test
        masks = []
        for p in p_selected_list:
            mask = abs(_df["P[bar]"]*1e5 - p) <= (p*0.01)
            masks.append(mask)
            
        # Combine all masks
        masks_combined = logical_or.reduce(masks)
        pressures = _df[masks_combined]["P[bar]"] * 1e5 
    
    pressure_values = []
    S_sc_exp_values = []
    SwellingRatio_values = []
    
    for p in pressures:
        print('p:', p)
        obj = S.DetailedSolPol(base_obj, T, p)
        
        #* Using swelling
        try:
            SwR, S_sc = obj.solve_parameters_fsolve_NEW()
        except:
            SwR, S_sc = [None], [None]
        
        #* Without Swelling
        # SwR, S_sc = obj.solve_parameters_fsolve_NEW_noSwelling()
        
        #* Using swelling with constraint pressure
        # SwR, S_sc = obj.solve_parameters_fsolve_NEW_pc(pc=78e6)
        
        S_sc_exp_values.extend(S_sc)
        SwellingRatio_values.extend(SwR)
        pressure_values.extend([p]*len(S_sc))
        print('')
    
    # Print results at all pressures
    print('pressure_values:', *pressure_values)
    print('S_sc_exp_values:', *S_sc_exp_values)
    print('SwellingRatio_values:', *SwellingRatio_values)
    print('')
    
    def get_properties(T, p, S_sc):

        obj = S.DetailedSolPol(base_obj, T, p)
        
        # Get omega_cr
        omega_cr = obj.omega_cr
        
        # Calculate S_am
        S_am = S_sc / (1-omega_cr)
        
        # Calculate x
        omega_p = 1/(S_am+1)     # [g/g]
        omega_s = 1 - omega_p   # [g/g]
        x_s = (omega_s/obj.MW_sol) / (omega_s/obj.MW_sol + omega_p/obj.MW_pol)   #[mol/mol]
        x_p = 1 - x_s   #[mol/mol]           
        x = hstack([x_s, x_p])   # [mol/mol]
        
        # Calculate Vs and Vp
        V_s, V_p =  obj.Vs_Vp_pmv1(obj.T, obj.P, S_am)  #* Default
        # V_s, V_p =  obj.Vs_Vp_pmv2()    #* TEST
        
        # Calculate mixture Density
        rho_t = obj.rho_tot(T, p, S_sc)   # [mol/m^3]        
        
        return x[0], rho_t, V_s, V_p
    
    rho_TPS = []
    x_s = []
    Vs = []
    Vp = []
    rhoT00_rhoTPS = []
    
    if len(pressure_values) == len(S_sc_exp_values) == len(SwellingRatio_values):
        for i, p in enumerate(pressure_values):
            try: 
                result = get_properties(T, p, S_sc_exp_values[i])
            except:
                result = [None, None, None, None]
                
            x_s.append(result[0])
            rho_TPS.append(result[1])
            Vs.append(result[2])
            Vp.append(result[3])
            
            try:
                rhoT00_rhoTPS.append((SwellingRatio_values[i]+1) / (S_sc_exp_values[i]+1))        
            except:
                rhoT00_rhoTPS.append(None)
    
    print('pressures:', pressure_values)
    print('rho_TPS:', rho_TPS)
    print('x_s:', x_s)
    print('Vs:', Vs)
    print('Vp:', Vp)
    print('rhoT00_rhoTPS:', rhoT00_rhoTPS)
    
    # Create a dictionary with the data
    data = {
        'Pressure [Pa]': pressure_values,
        'S_sc [g/g_total]': S_sc_exp_values,
        'SwellingRatio [m3/m3]': SwellingRatio_values,
        'x_s [mol/mol]': x_s,
        'rho_TPS [g/m3]': rho_TPS,
        'Vs': Vs,
        'Vp': Vp,
        'rhoT00_rhoTPS': rhoT00_rhoTPS
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    if export_csv_dir != None:
        # Export the DataFrame to a CSV file
        df.to_csv(export_csv_dir, index=False)
        print(f"Data exported to: {export_csv_dir}")
    
    # Convert pressures to numpy arrray
    pressure_values = array(pressure_values)    
    
    # Plot results
    n_plots = 6
    # fig, ax = plt.subplots(n_plots, 1, figsize=(5, 2*n_plots))  # (width, height)    
    fig, ax = plt.subplots(n_plots, 1, figsize=(3, 1.8*n_plots))  # (width, height)    
    
    # Solubility vs. Pressure
    ax[0].plot(pressure_values*1e-5, S_sc_exp_values, color='black', marker='.', linestyle='None')
    ax[0].set_xlabel('Pressure [bar]')
    ax[0].set_ylabel(r'$S_{sc}$ [g/g]')
    ax[0].set_title(f'T = {T-273} °C')    
    S.update_subplot_ticks(ax[0], x_lo=0, y_lo=0)

    # Swelling Ratio vs. Pressure
    ax[1].plot(pressure_values*1e-5, SwellingRatio_values, color='black', marker='.', linestyle='None')
    ax[1].set_xlabel('Pressure [bar]')
    ax[1].set_ylabel(r'SwR [$m^{3}$/$m^{3}$]')
    S.update_subplot_ticks(ax[1], x_lo=0, y_lo=0)

    # x_s vs. Pressure
    ax[2].plot(pressure_values*1e-5, x_s, color='black', marker='.', linestyle='None')
    ax[2].set_xlabel('Pressure [bar]')
    ax[2].set_ylabel(r'$x_{s}$ [mol/mol]')
    S.update_subplot_ticks(ax[2])
    
    # rho_TPS vs. Pressure
    ax[3].plot(pressure_values*1e-5, rho_TPS, color='black', marker='.', linestyle='None')
    ax[3].set_xlabel('Pressure [bar]')
    ax[3].set_ylabel(r'$\rho_{tot} (T,p,S_{sc}) $ [g/$m^{3}$]')
    S.update_subplot_ticks(ax[3])

    # Vs vs. Pressure
    ax[4].plot(pressure_values*1e-5, Vs, color='black', marker='.', linestyle='None')
    ax[4].set_xlabel('Pressure [bar]')
    ax[4].set_ylabel(r'$\bar{V}_{s} (T,p,S_{sc})$ [$m^{3}$/g]')
    S.update_subplot_ticks(ax[4])

    # Vp vs. Pressure
    ax[5].plot(pressure_values*1e-5, Vp, color='black', marker='.', linestyle='None')
    ax[5].set_xlabel('Pressure [bar]')
    ax[5].set_ylabel(r'$\bar{V}_{p} (T,p,S_{sc})$ [$m^{3}$/g]')
    S.update_subplot_ticks(ax[5])

    # rhoT00_rhoTPS vs. Pressure
    # plt.subplot(n_plots, 1, 7)
    # plt.plot(pressure_values*1e-6, rhoT00_rhoTPS, color='black', marker='.', linestyle='None')    
    # plt.xlabel('Pressure [MPa]')
    # plt.ylabel(r'$\frac{\rho_{tot} (T,0,0)} {\rho_{tot} (T,P,S_{sc}) }$')
    
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    if display_plot == True:
        plt.show()

def Test_solve_parameters_fsolve_NEW_integration(base_obj, T, display_plot=True, save_plot_dir=None):
    data = S.SolPolExpData(base_obj.sol, base_obj.pol)
    
    # Finding largest pressure value in experimental data
    _df = {}
    _df=data.get_sorption_data(T)
    pressures = _df["P[bar]"] * 1e5 # [Pa]
        
    ###*
    #* Choose single pressure only
    # p_selected = 10028380.0 # [Pa]
    # mask = abs(_df["P[bar]"]*1e5 - p_selected) <= (p_selected*0.01)
    # pressures = _df[mask]["P[bar]"] * 1e5 
    ###*
    
    pressure_values = []
    S_sc_exp_values = []
    SwellingRatio_values = []
    
    for p in pressures:
        print('p:', p)
        obj = S.DetailedSolPol(base_obj, T, p)
        
        #* Using swelling with integration for Vs and Vp
        SwR, S_sc = obj.solve_parameters_fsolve_NEW_integration()
        
        S_sc_exp_values.extend(S_sc)
        SwellingRatio_values.extend(SwR)
        pressure_values.extend([p]*len(S_sc))
        print('')
    
    # Print results at all pressures
    print('pressure_values:', *pressure_values)
    print('S_sc_exp_values:', *S_sc_exp_values)
    print('SwellingRatio_values:', *SwellingRatio_values)
    print('')
    
    def get_properties(T, p, S_sc):

        obj = S.DetailedSolPol(base_obj, T, p)
        
        # Get omega_cr
        omega_cr = obj.omega_cr
        
        # Get rho_pol_cr
        rho_p_c = obj.rho_pol_cr  # [g/m^3]
        
        # Calculate S_am
        S_am = S_sc / (1-omega_cr)
        
        # Calculate x
        omega_p = 1/(S_am+1)     # [g/g]
        omega_s = 1 - omega_p   # [g/g]
        x_s = (omega_s/obj.MW_sol) / (omega_s/obj.MW_sol + omega_p/obj.MW_pol)   #[mol/mol]
        x_p = 1 - x_s   #[mol/mol]           
        x = hstack([x_s, x_p])   # [mol/mol]
        
        # Calculate Vs and Vp
        V_s, V_p =  obj.Vs_Vp_pmv1(obj.T, obj.P, S_am)  #* Default
        # V_s, V_p =  obj.Vs_Vp_pmv2()    #* TEST
        
        def rho_tot(Ssc):
            # Calculate S_am_exp
            S_am = Ssc / (1-omega_cr)
            
            # Calculte rho_tot
            V_p_am = obj.get_V_p_am(T, p, S_am)  # [m3/g]
            rho_p_am = 1/(V_p_am)  # [g/m^3]
            rho_p_tol = 1/((1-omega_cr)/rho_p_am + omega_cr/rho_p_c)  # [g/m^3]
            rho_tot = (1 + Ssc) * rho_p_tol  # [g/m^3]
            
            return rho_tot
        
        # Calculate mixture Density
        rho_t = rho_tot(S_sc)   # [mol/m^3]        
        
        return x[0], rho_t, V_s, V_p
    
    rho_TPS = []
    x_s = []
    Vs = []
    Vp = []
    rhoT00_rhoTPS = []
    
    if len(pressure_values) == len(S_sc_exp_values) == len(SwellingRatio_values):
        for i, p in enumerate(pressure_values):        
            result = get_properties(T, p, S_sc_exp_values[i])
            x_s.append(result[0])
            rho_TPS.append(result[1])
            Vs.append(result[2])
            Vp.append(result[3])
            rhoT00_rhoTPS.append((SwellingRatio_values[i]+1) / (S_sc_exp_values[i]+1))        
    
    print('pressures:', pressure_values)
    print('rho_TPS:', rho_TPS)
    print('x_s:', x_s)
    print('Vs:', Vs)
    print('Vp:', Vp)
    print('rhoT00_rhoTPS:', rhoT00_rhoTPS)
        
    # Convert pressure_values to numpy array to perform mathematical operations
    pressure_values = array(pressure_values)
    
    # Plotting SinglePhaseDensity against SwellingRatio
    plt.figure(figsize=(5, 2*7))  # (width, height)    
    # plt.suptitle(f'{T-273} °C', x = 0.6)
    
    plt.subplot(7, 1, 1)
    plt.plot(pressure_values*1e-6, S_sc_exp_values, color='black', marker='.', linestyle='None')
    plt.xlabel('Pressure [MPa]')
    plt.ylabel(r'$S_{sc}^{exp}$ [g/g]')    

    plt.subplot(7, 1, 2)
    plt.plot(pressure_values*1e-6, SwellingRatio_values, color='black', marker='.', linestyle='None')
    plt.xlabel('Pressure [MPa]')
    plt.ylabel(r'SwellingRatio [$m^{3}$/$m^{3}$]')

    plt.subplot(7, 1, 3)
    plt.plot(pressure_values*1e-6, rho_TPS, color='black', marker='.', linestyle='None')
    plt.xlabel('Pressure [MPa]')
    plt.ylabel(r'$\rho_{tot} ( T,p,S_{sc} ) $ [$mol$/$m^{3}$]')

    plt.subplot(7, 1, 4)
    plt.plot(pressure_values*1e-6, x_s, color='black', marker='.', linestyle='None')
    plt.xlabel('Pressure [MPa]')
    plt.ylabel(r'$x_{s}$')

    plt.subplot(7, 1, 5)
    plt.plot(pressure_values*1e-6, Vs, color='black', marker='.', linestyle='None')
    plt.xlabel('Pressure [MPa]')
    plt.ylabel(r'$V_{s}$ [$m^{3}$/g]')

    plt.subplot(7, 1, 6)
    plt.plot(pressure_values*1e-6, Vp, color='black', marker='.', linestyle='None')
    plt.xlabel('Pressure [MPa]')
    plt.ylabel(r'$V_{p}$ [$m^{3}$/g]')

    plt.subplot(7, 1, 7)
    plt.plot(pressure_values*1e-6, rhoT00_rhoTPS, color='black', marker='.', linestyle='None')    
    plt.xlabel('Pressure [MPa]')
    plt.ylabel(r'$\frac{\rho_{tot} (T,0,0)} {\rho_{tot} (T,P,S_{sc}) }$')
    
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    if display_plot == True:
        plt.show()


def Test_SscVsSwR_buoyancyCorrection(base_obj, T):
    data = S.SolPolExpData(base_obj.sol, base_obj.pol)    
    df = data.get_sorption_data(T)
    
    SwR_values = linspace(0, 0.1, 10)
    print('SwR_values:', SwR_values)
        
    def S_sc_exp(p, SwR):
        mask = abs(df["P[bar]"]*1e5 - p) <= (p*0.01)
        try:
            m_net_exp = df[mask]["MP1*[g]"].values[0] - data.m_met_filled  # [g]
            rho_f_exp = df[mask]["ρ[g/cc]"].values[0]  # [g/cc]
            V_b_exp = data.Vbasket  # [cc]
            V_t0_exp = data.Vs  # [cc]
            m_ptot_exp = data.ms    # [g]
            
        except Exception as e:
            print("Error: ")
            print(e)
        
        S_sc = (m_net_exp + rho_f_exp * (V_b_exp + V_t0_exp * (1 + SwR))) / m_ptot_exp
        
        return S_sc
        
    pressure_values = df["P[bar]"] * 1e5 # [Pa]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for p in pressure_values:
        S_sc_values = [S_sc_exp(p, SwR) for SwR in SwR_values]
        print('S_sc_values:', S_sc_values)
        ax.plot(SwR_values, S_sc_values, linestyle='solid', marker='None', label=f'{p*1e-6} MPa')    
    
    ax.set_xlabel(r'Swelling Ratio')
    ax.set_ylabel(r'$S_{sc}^{exp}$ [g/g]')
    ax.set_title(f'T = {T-273} °C')
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    plt.show()
    
def Test_VsVp_vs_pressure(base_obj, T):
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

def Test_VsVp_vs_Sam_multiT(base_obj, T_list, p):
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

def Test_VsVp_vs_Sam_multiP(base_obj, T, p_list):
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

def Test_rhotot_vs_Sam_multiP(base_obj, T, p_list):
    S_sc_values = linspace(0., 0.1, 20)    # [g/g]
    rho_tot_values = {}
    def rho_tot(S_sc, omega_cr, rho_p_cr, Vs, Vp):
        # Calculate S_am_exp
        S_am = S_sc / (1-omega_cr)
        
        # Calculte rho_tot
        rho_p_am = 1/(S_am*Vs + Vp)  # [g/m^3]
        rho_p_tol = 1/((1-omega_cr)/rho_p_am + omega_cr/rho_p_cr)  # [g/m^3]
        rho_tot = (1 + S_sc) * rho_p_tol  # [g/m^3]
        
        return rho_tot
    
    for p in p_list:
        print('p:', p)
        obj = S.DetailedSolPol(base_obj, T, p)
        rho_tot_values[p] = []
        for S_sc in S_sc_values:
            Vs, Vp = obj.Vs_Vp_pmv1(obj.T, obj.P, S_sc)
            rho_t = rho_tot(S_sc, obj.omega_cr, obj.rho_pol_cr, Vs, Vp)
            rho_tot_values[p].append(rho_t)
    
        print('rho_tot_values:', rho_tot_values[p])
    plt.figure()
    plt.subplot(1, 1, 1)
    for p in p_list:
        plt.plot(S_sc_values, rho_tot_values[p], linestyle='None', marker='x', label=f'{p*1e-6} MPa')
    
    plt.xlabel(r'$S_{am}$ [g/g]')
    plt.ylabel(r'$\rho_{tot}$ [g/$m^{3}$]')
    plt.title(f'T = {T-273} °C')
    plt.legend()
    plt.show()
    
def Test_get_V_p_am(base_obj, T: float, p: float):
    S_sc_values = linspace(0., 0.1, 5)    # [g/g]
    V_p_am_values = []
    for S_sc in S_sc_values:
        obj = S.DetailedSolPol(base_obj, T, p)
        V_p_am_values.append(obj.get_V_p_am(obj.T, obj.P, S_sc))
    print(V_p_am_values)
    plt.plot(S_sc_values, V_p_am_values, linestyle='None', marker='x')
    plt.show()

if __name__ == '__main__':
    src_dir = os.path.dirname(__file__)
    src_dir = r"\\?\%s" % src_dir  # extended path (for very long path length)

    start_time = time.time()
    now = datetime.now()  # current time
    time_ID = now.strftime("%y%m%d_%H%M")  # YYMMDD_HHMM
    
    # Create new directory to store results
    result_folder_dir = f'{src_dir}\\Anals'
    
    # Define base object
    base_obj = S.BaseSolPol('CO2', 'HDPE')
    
    #* Test_solve_parameters_fsolve_NEW
    # Test_solve_parameters_fsolve_NEW(base_obj,
    #                                 #  T=35+273,  p_selected_list=[20139060],    # 35 °C
    #                                  T=50+273, p_selected_list=[10103760],    # 50 °C
    #                                  display_plot=True, 
    #                                 #  save_plot_dir=f'{result_folder_dir}\\{base_obj.sol}_{base_obj.pol}_{35}C_{time_ID}.png'
    #                                  )
    
    # rho_type = 'rhoSAFT'
    # for T in [25+273, 35+273, 50+273]:
    #     Test_solve_parameters_fsolve_NEW(base_obj, T, display_plot=False,
    #                                      save_plot_dir=f'{result_folder_dir}\\{base_obj.sol}_{base_obj.pol}_{T-273}C_{rho_type}_allData_{time_ID}.png',
    #                                      export_csv_dir=f'{result_folder_dir}\\{base_obj.sol}_{base_obj.pol}_{T-273}C_{rho_type}_allData_{time_ID}.csv')
    
    #* Test solution method
    # Test_solve_parameters_rootEvaluation(base_obj, 35+273, 20139060)
    # Test_solve_parameters_rootEvaluation(base_obj, 50+273, 20087660)
    
    #* Test_SscVsSwR_buoyancyCorrection
    # Test_SscVsSwR_buoyancyCorrection(base_obj, 50+273)
    
    #* Test phase 'L' and 'V' in SinglePhaseDensity for sol-pol mixture
    # obj = S.DetailedSolPol(base_obj, 35+273, 20139060)
    # x_s  = 0.9584692354336282
    # x = x = hstack([x_s, 1-x_s])   # [mol/mol]
    
    # rho_L = obj.SinglePhaseDensity(x, obj.T, obj.P, 'L')   # [mol/m^3]
    # rho_V = obj.SinglePhaseDensity(x, obj.T, obj.P, 'V')   # [mol/m^3]
    
    # print(f'T = {obj.T} K, \t p = {obj.P} Pa' )
    # print('rho_L:', rho_L)
    # print('rho_V:', rho_V)
    
    #* Test Vs and Vp vs. pressure
    # Test_VsVp_vs_pressure(base_obj, T=35+273)
    
    #* Test Vs and Vp vs. S_am at different temperatures
    # Test_VsVp_vs_Sam_multiT(base_obj, T=[25+273, 35+273, 50+273], p=1e6)
    
    #* Test Vs and Vp vs. S_am at different pressures
    # Test_VsVp_vs_Sam_multiP(base_obj, T=35+273, p_list=linspace(1e6, 15e6, 10))
    
    #* Test rho_tot and Vp vs. S_am at different pressures
    # Test_rhotot_vs_Sam_multiP(base_obj, T=35+273, p_list=linspace(1e6, 20e6, 5))

    #* Test_solve_parameters_fsolve_NEW_integration
    # Test_solve_parameters_fsolve_NEW_integration(base_obj, 35+273, display_plot=True)
    
    #* Test get_V_p_am
    # Test_get_V_p_am(base_obj, T=35+273, p=1e6)
    
    #* Find omega_cr
    # for T in [25+273, 35+273, 50+273]:
    #     omega_cr = S.find_omega_cr(base_obj, T)
    #     print(f'T = {T-273} °C, omega_cr = ', omega_cr)
    
    #* Test rho_tot
    obj = S.DetailedSolPol(base_obj, 25+273, 1)
    for T in [25+273, ]:
        rho_p_am = obj.rho_tot(T, 40e6, 0) # [g/m^3]
        print(f'T = {T-273} °C, rho_p_am = {rho_p_am*1e-6} g/cm^3')
        print(f'T = {T-273} °C, V = {1/(rho_p_am*1e-6)} cm^3/g')
    