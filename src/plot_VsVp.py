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


def plot_polymer_compressibility(base_obj, T=323.15, save_fig=False, export_data=False, output_dir=None):
    """
    Plot polymer specific volume vs pressure using different methods. Assuming almost pure polymer.
    Also exports data to Excel file if requested.
    
    Parameters:
    -----------
    base_obj : BaseSolPol object
        Base solubility-polymer object
    T : float
        Temperature in K (default 323.15 K = 50°C)
    save_fig : bool
        Whether to save the figure
    export_data : bool
        Whether to export data to Excel file
    output_dir : str
        Directory for saved outputs (defaults to current directory)
    """
    import pandas as pd
    import os
    from datetime import datetime
    
    # Output directory handling
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create timestamp
    now = datetime.now()
    time_str = now.strftime("%y%m%d_%H%M")  # YYMMDD_HHMM
    
    # CO₂ Widom line values (K, Pa)
    widom_T_P = {35+273: 80.4e5, 
                 50+273: 103e5}  
        
    # P_values = logspace(0, 8.5, 30)  # 1 Pa to 300 MPa
    P_values = linspace(0, 500e5, 50)  # 0 to 500 bar in Pa
    obj = S.DetailedSolPol(base_obj, T, 1e5)

    # Calculate V_p using different methods
    V_p_saft = []
    V_p_tait = []
    
    for P in P_values:
        # SAFT calculation (unbounded)
        x = hstack([0.000000001, 1-0.000000001])  # Almost pure polymer
        V_p_s = obj.V_pol(x, T, P)
        V_p_saft.append(V_p_s*1e6)  # Convert to cm³/g
        
        # Tait equation reference
        # Get reference specific volume at P=1 bar
        P_ref = 1e5  # Reference pressure [Pa]
        V_p_ref = obj.V_pol(x, T, P_ref)  # At reference pressure [m³/g]
        
        # Apply Tait equation with constant parameters on the reference volume
        # B = 350e6  # Characteristic pressure [Pa]
        # C = 0.0849   # Tait constant
        # V_p_t = V_p_ref * (1 - C * log(1 + P/B))  # Tait equation
        # V_p_tait.append(V_p_t*1e6)  # Convert to cm³/g
        
        # Apply Tait equation with temperature dependence on the reference volume
        # Bicerano, Prediction of Polymer Properties, 3rd edition
        C = 0.0849   # Tait constant
        b1=235.0e6  # Characteristic pressure [Pa]
        b2=2.1e-3
        B_T = b1 * exp(-b2 * T) # [Pa]
        log_term = log((B_T + P) / (B_T + P_ref))
        
        # C = 0.0894   # Tait constant
        # b1=197.1e6  # Characteristic pressure [Pa]
        # b2=5.11e-3
        # B_T = b1 * exp(-b2 * T) # [Pa]
        # log_term = log( 1+  P / B_T)
        V_p_t = V_p_ref * (1 - C * log_term)  # [m³/g]
        V_p_tait.append(V_p_t*1e6)  # [cm³/g]
    
    # Plot
    fig = plt.figure(figsize=(6, 4))
    plt.plot(P_values/1e5, V_p_saft, 'r-', label='SAFT-γ Mie')
    plt.plot(P_values/1e5, V_p_tait, 'b--', label='Tait Equation')
    
    # Add Widom line if available for this temperature
    if T in widom_T_P.keys():
        widom_P = widom_T_P[T]*1e-5  # Convert to bar
        plt.axvline(x=widom_P, color='gray', linestyle=':', label='CO₂ Widom Line')

    # Add literature reference points if available
    plt.xlabel('Pressure (bar)')
    plt.ylabel('Polymer Specific Volume (cm³/g)')
    plt.title(f'HDPE Specific Volume at {T-273.15:.0f} °C')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Export data to Excel if requested
    if export_data:
        # Create DataFrame with the results
        data_df = pd.DataFrame({
            'Pressure (Pa)': P_values,
            'Pressure (bar)': P_values/1e5,
            'V_p SAFT (cm³/g)': V_p_saft,
            'V_p Tait (cm³/g)': V_p_tait
        })
        
        # Add metadata
        metadata_df = pd.DataFrame({
            'Parameter': ['Temperature (°C)', 'Temperature (K)', 'Polymer', 'Solvent',
                         'Tait C constant', 'Tait b1 (Pa)', 'Tait b2'],
            'Value': [T-273.15, T, base_obj.pol, base_obj.sol, 
                     C, b1, b2]
        })
        
        # Add Widom line info if available
        if T in widom_T_P.keys():
            widom_df = pd.DataFrame({
                'Parameter': ['Widom Line Pressure (Pa)', 'Widom Line Pressure (bar)'],
                'Value': [widom_T_P[T], widom_T_P[T]*1e-5]
            })
            metadata_df = pd.concat([metadata_df, widom_df], ignore_index=True)
        
        # Create Excel filename
        excel_filename = f'polymer_compressibility_{base_obj.pol}_{T-273.15:.0f}C_{time_str}.xlsx'
        excel_path = os.path.join(output_dir, excel_filename)
        
        # Save to Excel
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            data_df.to_excel(writer, sheet_name='Compressibility Data', index=False)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        print(f"Data exported to {excel_path}")
    
    # Save figure if requested
    if save_fig:
        # Create filename
        fig_filename = f'polymer_compressibility_{base_obj.pol}_{T-273.15:.0f}C_{time_str}.png'
        fig_path = os.path.join(output_dir, fig_filename)
        
        # Save figure
        plt.savefig(fig_path, dpi=400, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
    
    plt.show()
    return fig

def plot_density_ratio(base_obj, T=323.15):
    """Plot density ratio vs pressure to show when problems occur"""
    P_values = logspace(0, 8.5, 30)  # 1 Pa to 300 MPa
    
    # Calculate density ratios
    rhoT00_values = []
    rhoTPS_unlimited_values = []
    rhoTPS_limited_values = []
    ratio_unlimited_values = []
    ratio_limited_values = []
    
    for P in P_values:
        obj = S.DetailedSolPol(base_obj, T, P)
        
        # Store original Vs_Vp_pmv1 function
        original_func = obj.Vs_Vp_pmv1
        
        # Get reference density at P=1 Pa
        rhoT00 = obj.rho_tot(T, 1, 0)
        rhoT00_values.append(rhoT00*1e-6)  # g/cm³
        
        # Calculate density without Tait limits by setting pressure_limit=False
        obj.Vs_Vp_pmv1 = lambda T, P, S_a: original_func(T, P, S_a, pressure_limit=False)
        rhoTPS_unlimited = obj.rho_tot(T, P, 0)
        rhoTPS_unlimited_values.append(rhoTPS_unlimited)
        ratio_unlimited_values.append(rhoT00/rhoTPS_unlimited)
        
        # Calculate density with Tait limits by setting pressure_limit=True
        obj.Vs_Vp_pmv1 = lambda T, P, S_a: original_func(T, P, S_a, pressure_limit=True)
        rhoTPS_limited = obj.rho_tot(T, P, 0)
        ratio_limited_values.append(rhoT00/rhoTPS_limited)
    
    # Plot
    plt.figure(figsize=(6, 5))
    plt.subplot(1,1,1)
    plt.semilogx(P_values/1e5, ratio_unlimited_values, 'r-', label='Original')
    plt.semilogx(P_values/1e5, ratio_limited_values, 'b--', label='With Tait Limit')
    plt.axhline(y=1.0, color='k', linestyle=':')
    plt.axvline(x=73.8, color='gray', linestyle=':', label='CO₂ Widom Line')
    plt.ylabel('ρ(T,0,0)/ρ(T,P,0)')
    plt.legend()
    plt.show()
    

def tait_specific_volume(T, P, 
                         a0=0.882, 
                         a1=6.51e-4, 
                         a2=0.0, 
                         b1=235.0, 
                         b2=2.1e-3, 
                         C=0.0894, 
                         P_ref=0.1013):
    """
    Calculate specific volume v(T,P) using the Tait equation for amorphous polyethylene.
    
    Parameters:
    - T : float or array-like, temperature in Kelvin
    - P : float or array-like, pressure in MPa
    - a0, a1, a2 : coefficients for temperature dependence of v0(T)
    - b1, b2     : coefficients for pressure dependence B(T)
    - C          : universal constant in Tait equation (dimensionless)
    - P_ref      : reference pressure in MPa (default 0.1013 MPa = 1 atm)
    
    Returns:
    - v : specific volume in cm³/g
    """
    T = asarray(T)
    P = asarray(P)

    v0 = a0 + a1 * T + a2 * T**2
    B_T = b1 * exp(-b2 * T)
    
    log_term = log((B_T + P) / (B_T + P_ref))
    v = v0 * (1 - C * log_term)
    return v

if __name__ == "__main__":
    mix = S.BaseSolPol("CO2","HDPE")
    for T in array([25, 35, 50]) + 273:
        # Compare values between pmv1, pmv2 and pmv3
        # plot_VsVp_pmv(mix, T, display_fig=False, save_fig=True)
        
        # Compare compressibility between SAFT and Tait
        plot_polymer_compressibility(
            mix, 
            T, 
            save_fig=False,
            export_data=False
        )
        
        # Plot polymer density ratio vs pressure
        # plot_density_ratio(mix, T)

    #* Test Vs and Vp vs. S_am at different pressures
    # plot_VsVp_vs_Sam_multiP(mix, T=35+273, p_list=linspace(1e6, 15e6, 10))
    