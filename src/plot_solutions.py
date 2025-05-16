import solubility_master as S
from numpy import *
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import time
import pandas as pd

# Change subplot styling
def plot_solubility_solutions_rootEvaluation(base_obj, T: float, p: float, var: str):
    print('Experimental p:', p)
    
    obj = S.DetailedSolPol(base_obj, T, p,)
    
    if var == 'SwR':
        obj.solve_solubility_plot_SwR()
        
    elif var == 'Ssc':
        obj.solve_solubility_plot_Ssc() 

def get_solubility_solutions(base_obj: S.BaseSolPol, 
                                T: float, 
                                x0: list = linspace(0, 0.1, 10),
                                solver_xtol: float = 1e-10,
                                p_selected_list: list = None, 
                                display_plot: bool = True, 
                                save_plot_dir: str = None, 
                                export_csv_dir: str = None):
    data = S.SolPolExpData(base_obj.sol, base_obj.pol)    
    _df = {}
    _df=data.get_sorption_data(T)
    
    if p_selected_list == None:
        # Use all pressure values
        pressures = _df["P[bar]"] * 1e5 # [Pa]
        
    else:
        # Select specific pressure values
        masks = []
        for p in p_selected_list:
            mask = abs(_df["P[bar]"]*1e5 - p) <= (p*0.01)   # allow for 1% error
            masks.append(mask)
            
        # Combine all masks
        masks_combined = logical_or.reduce(masks)
        pressures = _df[masks_combined]["P[bar]"] * 1e5 
    
    # Store the results
    pressure_values = []
    S_sc_exp_values = []
    SwellingRatio_values = []
    
    for p in pressures:
        print('p:', p)
        obj = S.DetailedSolPol(base_obj, T, p)
        
        try:
            SwR, S_sc = obj.solve_solubility(rhoCO2_type='SW', x0_list=x0, solver_xtol=solver_xtol, debug=True)
        except:
            SwR, S_sc = [None], [None]

        # Add results to lists
        S_sc_exp_values.extend(S_sc)
        SwellingRatio_values.extend(SwR)
        pressure_values.extend([p]*len(S_sc))
        print('')
    
    # Print results at all pressures
    print('pressure_values:', *pressure_values)
    print('S_sc_exp_values:', *S_sc_exp_values)
    print('SwellingRatio_values:', *SwellingRatio_values)
    print('')
    
    def get_properties(T: float, p: float, S_sc: float):

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
        V_s, V_p =  obj.Vs_Vp_pmv1(obj.T, obj.P, S_am)
        
        # Calculate mixture Density
        rho_t = obj.rho_tot(T, p, S_sc)   # [mol/m^3]        
        
        return x[0], rho_t, V_s, V_p
    
    rhoTPS = []
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
            rhoTPS.append(result[1])
            Vs.append(result[2])
            Vp.append(result[3])
            
            try:
                rhoT00_rhoTPS.append((SwellingRatio_values[i]+1) / (S_sc_exp_values[i]+1))        
            except:
                rhoT00_rhoTPS.append(None)
    
    print('pressures:', pressure_values)
    print('rho_TPS:', rhoTPS)
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
        'Vs [m3/g]': Vs,
        'Vp [m3/g]': Vp,
        'rhoTPS [g/m3]': rhoTPS,
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
    ax[3].plot(pressure_values*1e-5, rhoTPS, color='black', marker='.', linestyle='None')
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
    
    #* Find solutions
    get_solubility_solutions(base_obj,x0=linspace(0, 0.2, 5),
                                    #  T=35+273,  p_selected_list=[20139060],    # 35 °C
                                    #  T=50+273, p_selected_list=[10103760],    # 50 °C 10 MPa
                                     T=50+273, p_selected_list=[200.8766*1e5],    # 50 °C 200 MPa
                                     display_plot=True, 
                                    #  save_plot_dir=f'{result_folder_dir}\\{base_obj.sol}_{base_obj.pol}_{35}C_{time_ID}.png'
                                     )
    
    # TODO why LHS-RHS so large for 50 C, 20 MPa?
    # rho_type = 'rhoSAFT'
    # for T in [25+273, 35+273, 50+273]:
    #     get_solubility_solutions(base_obj, T, display_plot=False,
    #                                      save_plot_dir=f'{result_folder_dir}\\{base_obj.sol}_{base_obj.pol}_{T-273}C_{rho_type}_allData_{time_ID}.png',
    #                                      export_csv_dir=f'{result_folder_dir}\\{base_obj.sol}_{base_obj.pol}_{T-273}C_{rho_type}_allData_{time_ID}.csv')
    
    #* Root evaluation
    # plot_solubility_solutions_rootEvaluation(base_obj, 35+273, 20139060, 'SwR')
    # plot_solubility_solutions_rootEvaluation(base_obj, 50+273, 20087660, 'Ssc')
    
    #* Find omega_cr
    # for T in [25+273, 35+273, 50+273]:
    #     omega_cr = S.find_omega_cr(base_obj, T)
    #     print(f'T = {T-273} °C, omega_cr = ', omega_cr)
    
    #* Test rho_p_am
    # obj = S.DetailedSolPol(base_obj, 25+273, 1)
    # for T in [25+273, 35+273, 50+273]:
    #     rho_p_am = obj.rho_tot(T, 40e6, 0, 0) # [g/m^3]
    #     print(f'T = {T-273} °C, rho_p_am = {rho_p_am*1e-6} g/cm^3, V = {1/(rho_p_am*1e-6)} cm^3/g')
    
    #* Test rho_tot
    # obj = S.DetailedSolPol(base_obj, 25+273, 1)
    # for T in [25+273, 35+273, 50+273]:
    #     rho_p_am = obj.rho_tot(T, 40e6, 0) # [g/m^3]
    #     print(f'T = {T-273} °C, rho_p_am = {rho_p_am*1e-6} g/cm^3, V = {1/(rho_p_am*1e-6)} cm^3/g')
    