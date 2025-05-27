import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from solubility_master import DetailedSolPol, SolPolExpData

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from solubility_master import DetailedSolPol, SolPolExpData

def plot_raw_mass_sensitivity(base_obj, T: float, p: float, m_raw_range: list = None, m_raw_variation=0.05,
                             rhoCO2_type='SW', save_fig=False, output_dir=None, 
                             fig_format='pdf', dpi=300, export_data=False):
    """
    Plot sensitivity analysis of solubility (S_sc) to variations in raw mass measurements.
    
    Parameters:
    -----------
    base_obj : BaseSolPol object
        Base solubility-polymer object
    T : float
        Temperature in K
    p : float
        Pressure in Pa
    m_raw_range : list
        List of raw mass values to analyze
    m_raw_variation : float
        Variation in m_raw to analyze when m_raw_range is not provided
    rhoCO2_type : str
        Type of CO2 density data to use ('EXP', 'SW', or 'SAFT')
    save_fig : bool
        Whether to save the figure
    output_dir : str
        Directory to save the figure (defaults to ./figures)
    fig_format : str
        Format for saved figure ('pdf', 'png', etc.)
    dpi : int
        Resolution for saved figure
    export_data : bool
        Whether to export the results to Excel file
        
    Returns:
    --------
    fig : matplotlib figure
        Figure object containing the plot
    """
    print(f"Analyzing sensitivity at T = {T-273.15:.1f}°C, P = {p/1e5:.1f} bar")
    
    # Create detailed solubility object
    sol_pol = DetailedSolPol(base_obj, T, p, pmv_method="1")
    
    # Get experimental data
    data = SolPolExpData(base_obj.sol, base_obj.pol)
    _df = data.get_sorption_data(T)
    
    # Check if data exists
    if _df is None:
        raise ValueError(f"No data found for T={T} K")
    
    # Get data point closest to the specified pressure
    mask = abs(_df["P[bar]"]*1e5 - p) <= (p*0.01)
    if not mask.any():
        closest_idx = (_df["P[bar]"]*1e5 - p).abs().idxmin()
        print(f"Warning: No exact match for P={p/1e5} bar. Using closest point P={_df.loc[closest_idx, 'P[bar]']} bar")
        mask = _df.index == closest_idx
    
    # Get baseline m_raw from data if not provided with range
    baseline_m_raw = _df[mask]["MP1*[g]"].values[0] - data.m_met_filled
    
    # If m_raw_range not provided, create a range around baseline
    if m_raw_range is None:
        m_raw_min = baseline_m_raw * (1 - m_raw_variation)
        m_raw_max = baseline_m_raw * (1 + m_raw_variation)
        m_raw_range = np.linspace(m_raw_min, m_raw_max, 20)

    print(f"Baseline m_raw: {baseline_m_raw:.6f} g")

    # Calculate solubility for each m_raw value
    SwR_values = []
    S_sc_values = []
    
    for m_raw in m_raw_range:
        try:
            # Use the solve_solubility method with custom_m_raw parameter
            SwR_vals, S_sc_vals = sol_pol.solve_solubility(
                rhoCO2_type=rhoCO2_type,
                x0_list=np.linspace(0.01, 0.3, 6),
                solver_xtol=1.0e-10,
                custom_m_raw=m_raw,
                debug=False
            )
            
            if SwR_vals[0] is not None:
                SwR_values.append(SwR_vals[0])
                S_sc_values.append(S_sc_vals[0])
            else:
                print(f"No solution found for m_raw={m_raw:.6f} g")
                SwR_values.append(None)
                S_sc_values.append(None)
        except Exception as e:
            print(f"Failed at m_raw={m_raw:.6f} g: {str(e)}")
            SwR_values.append(None)
            S_sc_values.append(None)
    
    # Calculate relative change in percent
    rel_m_raw = 100 * (np.array(m_raw_range) - baseline_m_raw) / baseline_m_raw
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot data
    ax.plot(m_raw_range, S_sc_values, 'o-', color='C0')
    
    # Add vertical line at baseline
    ax.axvline(x=baseline_m_raw, color='red', linestyle='--', 
              label=f'Baseline={baseline_m_raw:.6f}g')
    
    # Formatting
    ax.set_xlabel('Raw mass (g)')
    ax.set_ylabel('Solubility (g/g)')
    ax.set_title(f'Sensitivity of CO2 solubility to raw mass\nT = {T-273.15:.1f}°C, P = {p/1e5:.1f} bar')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Find index of baseline value or closest to it
    valid_indices = [i for i, val in enumerate(S_sc_values) if val is not None]
    if valid_indices:
        baseline_idx = min(valid_indices, key=lambda i: abs(m_raw_range[i] - baseline_m_raw))
        
        # Add baseline information if available
        if S_sc_values[baseline_idx] is not None:
            baseline_S_sc = S_sc_values[baseline_idx]
            text = (f'Baseline S_sc: {baseline_S_sc:.6f} g/g')
            ax.text(0.95, 0.05, text, transform=ax.transAxes, 
                    horizontalalignment='right', verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Calculate and display sensitivity if possible
    valid_values = [(m, s) for m, s in zip(m_raw_range, S_sc_values) if s is not None]
    if len(valid_values) >= 3:
        # Calculate local sensitivity near the baseline using available points
        valid_m_raw = [m for m, s in valid_values]
        valid_S_sc = [s for m, s in valid_values]
        
        # Simple linear regression for sensitivity
        coeffs = np.polyfit(valid_m_raw, valid_S_sc, 1)
        sensitivity = coeffs[0]  # slope
        
        # Calculate elasticity (percent change in output per percent change in input)
        baseline_S_sc = np.interp(baseline_m_raw, valid_m_raw, valid_S_sc)
        elasticity = sensitivity * (baseline_m_raw / baseline_S_sc)  # % change in S per % change in m_raw
        
        sens_text = (f'Sensitivity: {sensitivity:.4f} (g/g)/(g)\n'
                     f'Elasticity: {elasticity:.2f}')
        ax.text(0.95, 0.95, sens_text, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Create a second x-axis showing percent change
    ax2 = ax.twiny()
    ax2.set_xlim([rel_m_raw.min(), rel_m_raw.max()])
    ax2.set_xlabel('Change in raw mass (%)')
    
    # Save figure if requested
    if save_fig:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), 'figures')
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        filename = f'sensitivity_m_raw_{base_obj.sol}_{base_obj.pol}_{T-273.15:.0f}C_{p/1e5:.0f}bar.{fig_format}'
        filepath = os.path.join(output_dir, filename)
        
        # Save figure
        fig.savefig(filepath, format=fig_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {filepath}")
    
    # Export data to Excel if requested
    if export_data:
        # Create DataFrame with the results
        results_df = pd.DataFrame({
            'm_raw (g)': m_raw_range,
            'm_raw change (%)': rel_m_raw,
            'SwellingRatio': SwR_values,
            'Solubility (g/g)': S_sc_values
        })
        
        # Add metadata
        # metadata_df = pd.DataFrame({
        #     'Parameter': ['Temperature (°C)', 'Pressure (bar)', 'Baseline m_raw (g)', 
        #                   'CO2 density type', 'Polymer', 'Solvent'],
        #     'Value': [T-273.15, p/1e5, baseline_m_raw, rhoCO2_type, 
        #              base_obj.pol, base_obj.sol]
        # })
        
        # Calculate sensitivity metrics if possible
        # if len(valid_values) >= 3:
        #     metrics_df = pd.DataFrame({
        #         'Metric': ['Sensitivity (g/g)/(g)', 'Elasticity'],
        #         'Value': [sensitivity, elasticity]
        #     })
        # else:
        #     metrics_df = pd.DataFrame({
        #         'Metric': ['Sensitivity (g/g)/(g)', 'Elasticity'],
        #         'Value': [None, None]
        #     })
        
        # Prepare directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), 'results')
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create Excel filename
        excel_filename = f'sensitivity_m_raw_{base_obj.sol}_{base_obj.pol}_{T-273:.0f}C_{p/1e5:.0f}bar.xlsx'
        excel_path = os.path.join(output_dir, excel_filename)
        
        # Save to Excel
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Sensitivity Data', index=False)
            # metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            # metrics_df.to_excel(writer, sheet_name='Sensitivity Metrics', index=False)
        
        print(f"Data exported to {excel_path}")
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage when the script is run directly
    from solubility_master import BaseSolPol
    
    # Create base object
    base_obj = BaseSolPol('CO2', 'HDPE')
    
    # Example at 50°C and 200 bar
    T = 50+273  # K
    P = 200.8766*1e5  # Pa    
    
    # Call function
    fig = plot_raw_mass_sensitivity(
        base_obj=base_obj,
        T=T,
        p=P,
        m_raw_variation=0.05,  # 5% variation
        save_fig=False,
        export_data=True  # Export results to Excel
    )
    plt.show()