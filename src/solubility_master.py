"""
Louis Nguyen
Department of Cheimcal Engineering, Imperial College London
sn621@ic.ac.uk
15 Feb 2024
"""
# Turn off numba warning
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

# warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
# warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore")

from sgtpy_NETGP import component, mixture, saftgammamie, database
# from sgtpy import component, mixture, saftgammamie, database

import math
import os
import time
from datetime import datetime
import addcopyfighandler
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import pandas as pd
from scipy.optimize import fsolve, minimize_scalar, root, minimize


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

custom_colours = ["black","green","blue","red","purple","orange","brown","plum","indigo","olive","grey"]
custom_markers = ["o", "x", "^", "*", "s", "D", "."]

def update_x0_sol_list(previous_x0_sol:float, no_step:int=50, x0_sol_default_range=(1e-6, 0.99999999)):
    if (previous_x0_sol is None) or (previous_x0_sol < 0.) or (previous_x0_sol > 1.):
        new_x0_list =  linspace(x0_sol_default_range[0], x0_sol_default_range[1], no_step).tolist()
    else:
        new_x0_list = linspace(previous_x0_sol, x0_sol_default_range[1], no_step).tolist()
    return new_x0_list

# Change subplot styling
def update_subplot_ticks(ax, x_lo=None, y_lo=None, x_up=None, y_up=None):
    """Update x and y ticks of subplot ax to cover all data. Put ticks to inside.

    Args:
        ax: plot object.
    """
    # Adjust lower x and y ticks to start from 0
    if x_lo != None:
        ax.set_xlim(left=x_lo)
    if y_lo != None:
        ax.set_ylim(bottom=y_lo)
    if x_up != None:
        ax.set_xlim(right=x_up)
    if y_up != None:
        ax.set_ylim(top=y_up)
    
    # Get the largest and smallest x ticks and y ticks
    max_y_tick = max(ax.get_yticks())
    max_x_tick = max(ax.get_xticks())
    min_y_tick = min(ax.get_yticks())
    min_x_tick = min(ax.get_xticks())
    
    # Get the length of major ticks on the x-axis
    ax_x_major_tick_length = ax.get_xticks()[1] - ax.get_xticks()[0]
    ax_y_major_tick_length = ax.get_yticks()[1] - ax.get_yticks()[0]
    
    # Adjust upper x and y ticks to cover all data
    if x_up == None:        
        ax.set_xlim(right=max_x_tick + ax_x_major_tick_length)
    if y_up == None:
        ax.set_ylim(top=max_y_tick + ax_y_major_tick_length)
    if x_lo == None:
        ax.set_xlim(left=min_x_tick - ax_x_major_tick_length)
    if y_lo == None:
        ax.set_ylim(bottom=min_y_tick - ax_y_major_tick_length)
    
    # Put ticks to inside
    ax.tick_params(direction="in")

class BaseSolPol:
    """Class to store information about the sol-pol mixture
    
    Parameters
    sol[string] :   solute name
    pol[string] :   polymer name
    n_monomer[int]  :   number of repeating units in polymer
    MW_sol[float]   : MW of solute
    MW_pol[float]   : MW of polymer
    eos_sol     : SAFTgMie object of solute
    eos_sol     : SAFTgMie object of polymer
    eos_mix     : SAFTgMie object of mixture

    """
    
    
    def __init__(self, sol: str, pol: str, n_monomer: int = 1000):
        self.sol = sol
        self.pol = pol
        self.n_monomer = n_monomer
        
        MW_monomer = {
            "HDPE": 28,            
            "PE": 28,
        }
        MW_pol = MW_monomer[pol] * n_monomer
        self.MW_pol = MW_pol
        
        MWsolute = {"CO2": 44}  # [g/mol]
        MW_sol = MWsolute[sol]
        self.MW_sol = MW_sol

    
    @property
    def sol_obj(self):
        if self.sol != None:
            if self.sol == "CO2":
                _sol_obj = component(GC={"CO2": 1})
        return _sol_obj
    
    @property
    def pol_obj(self):
        if self.pol != None:
            if self.pol in {"PE","HDPE"} :
                _pol_obj = component(GC={"CH2": 2*self.n_monomer})
        return _pol_obj
    
    @property
    def mix_obj(self):
        if self.sol != None and self.pol != None:
            _mix_obj = self.sol_obj + self.pol_obj
        return _mix_obj
    
    @property
    def eos_sol(self):
        # Create Create SAFT-g Mie EOS object of pure solute
        sol_obj = self.sol_obj
        sol_obj.saftgammamie()
        _eos_sol = saftgammamie(sol_obj)
        return _eos_sol
    
    @property
    def eos_pol(self):
        # Create Create SAFT-g Mie EOS object of pure polymer
        pol_obj = self.pol_obj
        pol_obj.saftgammamie()
        _eos_pol = saftgammamie(pol_obj, compute_critical=False)
        return _eos_pol
    
    @property
    def eos_mix(self):
        # Create SAFT-g Mie EOS object of Mixture
        mix_obj = self.mix_obj
        mix_obj.saftgammamie()
        _eos_mix = saftgammamie(mix_obj, compute_critical=False)            
        return _eos_mix    
    
    def modify_kl(self, eps:float, lr='CR'):
        """Function to modify cross interaction (eps_kl and lambda_repulsive_kl).

        Args:
            eps (float): epsilon_kl.
            lr (str, optional): lambda_repulsive_kl. Defaults to 'CR'.
        """
        
        if self.sol == "CO2" and (self.pol == "HDPE" or self.pol == "PE"):
            database.new_interaction_mie("CO2", "CH2", eps, lr, overwrite=True)
            
class DetailedSolPol(BaseSolPol):
    def __init__(self, baseObj, T: float, P: float, pmv_method:str = "1", **kwargs):
        super().__init__(baseObj.sol, baseObj.pol, baseObj.n_monomer)
        self.T = T  # [K]
        self.P = P  # [Pa]
        self.pmv_method = pmv_method
        self.options = kwargs
        #*Expected keys: x0_sol: float, x0_sol_range: list, auto_iterate_x0: bool
        
        self._omega_cr = None
        self._rho_pol_cr = None
        
    @property
    def omega_cr(self):
        if self._omega_cr is None:
            self._omega_cr = self.get_omega_cr(self.T)
        return self._omega_cr
    @property
    def rho_pol_cr(self):
        if self._rho_pol_cr is None:
            self._rho_pol_cr = self.get_rho_pol_cr(self.T)
        return self._rho_pol_cr

    def solve_solubility(self, rhoCO2_data: str = 'SW', x0_list=linspace(0, 0.3, 6)): # numerical solving
        # Import data object
        data = SolPolExpData(self.sol, self.pol)
        
        # Import sorption data at specified T
        _df = data.get_sorption_data(self.T)
        
        # Raise error if data, _df, is not found
        if _df is None:
            raise Exception(f"Data not found at T = {self.T} K.")            
        
        # Calculate corrected swelling using SAFT and exp data
        mask = abs(_df["P[bar]"]*1e5 - self.P) <= (self.P*0.01)
        
        try:
            m_net_exp = _df[mask]["MP1*[g]"].values[0] - data.m_met_filled
            
            if rhoCO2_data == 'EXP':
                # Use EXP values of rho_f
                rho_f = _df[mask]["ρ[g/cc]"].values[0]
            elif rhoCO2_data == 'SW':
                # Use Span Wagner EoS values of rho_f
                rho_f = _df[mask]["ρSW[g/cc]"].values[0]
            elif rhoCO2_data == 'SAFT':
                # Use SAFT EoS values of rho_f
                rho_f = _df[mask]["ρSAFT[g/cc]"].values[0]
                
            print('rhoCO2 data use: ', rhoCO2_data)
            
            V_b_exp = data.Vbasket
            V_t0_exp = data.Vs
            m_ptot_exp = data.ms
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        # Get omega_cr
        omega_cr = self.omega_cr
        print(f'omega_cr: {omega_cr}')
                
        # Get rho_pol_cr
        rho_p_c = self.rho_pol_cr  # [g/m^3]
        print(f'rho_p_cr = {rho_p_c} g/m^3')
        
        def S_sc_exp(SwR):
            # Calculate V_t0_exp based on calculated rho_tot(T,0,0)
            rhoT00 = self.rho_tot(self.T, 1, 0)*1e-6  # [g/cm^3]
            V_t0_model = m_ptot_exp/rhoT00  # [cm^3]
            
            # Compare rhoT00 and V_t0_exp
            # print(f'V_t0_exp = {V_t0_exp} cm^3')
            # print(f'V_t0_model = {V_t0_model} cm^3')
            
            #* Choose V_t0 values
            # V_t0 = V_t0_exp  # [cm^3], exp value
            V_t0 = V_t0_model  # [cm^3], calculated value
            
            # Calculate S_sc_exp
            S_sc_exp = (m_net_exp + rho_f * (V_b_exp + V_t0 * (1 + SwR))) / m_ptot_exp
            
            return S_sc_exp
            
        def equation(SwellingRatio):
            # Calculate S_sc_exp
            S_sc = S_sc_exp(SwellingRatio)
            
            # Calculate rho_tot(T,0,0)
            rhoT00 = self.rho_tot(self.T, 1, 0)  # [g/m^3], use P = 1 Pa
            
            # Calculate rho_tot(T,P,S)
            rhoTPS = self.rho_tot(self.T, self.P, S_sc)  # [g/m^3]
            
            # Define the equations in terms of these variables and instance variables
            LHS = SwellingRatio
            RHS = (rhoT00 / rhoTPS) * (1 + S_sc) - 1
            
            return LHS-RHS    # Solve for LHS-RHS = 0
        
        
        # *Initial guesses
        initial_guesses = x0_list        
        print("Initial guesses:", *initial_guesses)
        
        solutions = []

        for x0 in initial_guesses:
            print(f"Initial guess: {x0}")
            try:
                solution = fsolve(equation, x0, xtol=1.0e-10)
                print('equation(solution[0]):', equation(solution[0]))
                if isclose([0], [equation(solution[0])], atol=1e-4 if self.P < 80e5 else 1e-3):
                    solutions.append(solution[0])
                    print(f"SwellingRatio = {solution[0]}")
            except:
                print(f"Initial guess {x0} failed.")
                continue
            
        # Create a new matrix of unique solutions
        unique_solutions = []

        for solution in solutions:
            # Only accept non-zero solution, accounting for negative values approaching zero at very low pressures (p < 10 Pa)
            if (self.P < 10 and solution > -1e-4) or (self.P >= 10 and solution > 0):
                
                # Initialisation: if unique_solutions is empty, add the first solution
                if not unique_solutions:
                    unique_solutions.append(solution)
                    
                else:
                    # Check if the solution is already in the unique_solutions list
                    if not any([isclose(solution, unique_solution, rtol=5e-2) for unique_solution in unique_solutions]):
                        unique_solutions.append(solution)
                        print('Solution appended:', solution)

        # Print unique solutions
        print('Unique solutions: ', *unique_solutions)
        
        # Extract S_sc_exp and SwellingRatio from solutions
        SwellingRatio_values = [solution for solution in unique_solutions]
        S_sc_exp_values = [S_sc_exp(SwR) for SwR in SwellingRatio_values]
        print('')
        print('SwellingRatio:', SwellingRatio_values)
        print('S_sc:', S_sc_exp_values)
        
        if len(unique_solutions) == 1:
            print('Single solution found.')
        elif len(unique_solutions) > 1:
            print('Multiple solutions found.')
        elif len(unique_solutions) == 0:
            print('No solution found.')
            SwellingRatio_values, S_sc_exp_values = [None], [None]
        
        return SwellingRatio_values, S_sc_exp_values
        
    def solve_solubility_plot_SwR(self, rhoCO2_data: str = 'SW', SwR_list=linspace(0, 0.1, 10)):
        # Import data object
        data = SolPolExpData(self.sol, self.pol)
        
        # Import sorption data at specified T
        _df = data.get_sorption_data(self.T)
        
        # Raise error if data, _df, is not found
        if _df is None:
            raise Exception(f"Data not found at T = {self.T} K.")            
        
        # Calculate corrected swelling using SAFT and exp data
        mask = abs(_df["P[bar]"]*1e5 - self.P) <= (self.P*0.01)
        
        try:
            m_net_exp = _df[mask]["MP1*[g]"].values[0] - data.m_met_filled  # [g]
            
            if rhoCO2_data == 'EXP':
                # Use EXP values of rho_f
                rho_f = _df[mask]["ρ[g/cc]"].values[0]
            elif rhoCO2_data == 'SW':
                # Use Span Wagner EoS values of rho_f
                rho_f = _df[mask]["ρSW[g/cc]"].values[0]
            elif rhoCO2_data == 'SAFT':
                # Use SAFT EoS values of rho_f
                rho_f = _df[mask]["ρSAFT[g/cc]"].values[0]
                
            print('rhoCO2 data use: ', rhoCO2_data)
            
            V_b_exp = data.Vbasket  # [cc]
            V_t0_exp = data.Vs  # [cc]
            m_ptot_exp = data.ms    # [g]
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        # Get omega_cr
        omega_cr = self.omega_cr        
        
        # Calculate V_t0_exp based on calculated rho_tot(T,0,0)
        rhoT00 = self.rho_tot(self.T, 1, 0)*1e-6  # [g/cm^3]
        V_t0_model = m_ptot_exp/rhoT00  # [cm^3]
        
        #* Choose V_t0 values
        V_t0 = V_t0_exp  # [cm^3], exp value
        # V_t0 = V_t0_model  # [cm^3], calculated value
        
        def equations(SwellingRatio):            
            # Calculate S_sc_exp
            S_sc_exp = (m_net_exp + rho_f * (V_b_exp + V_t0 * (1 + SwellingRatio))) / m_ptot_exp
            
            # Calculate rho_tot(T,P,S)
            rhoT00 = self.rho_tot(self.T, 1, 0)  # [g/m^3]
            rhoTPS = self.rho_tot(self.T, self.P, S_sc_exp)  # [g/m^3]
            
            # Define the equations in terms of these variables and instance variables
            LHS = SwellingRatio
            RHS = (rhoT00 / rhoTPS) * (1 + S_sc_exp) - 1
            
            return LHS-RHS, rhoT00, rhoTPS, S_sc_exp
        
        #* SwellingRatio values to evaluate
        SwellingRatio_values = SwR_list
        
        # Plot LHS-RHS vs SwellingRatio
        eq_values = []
        for SwellingRatio in SwellingRatio_values:
            try:
                eq_values.append(equations(SwellingRatio)[0])
            except:
                eq_values.append(None)
        print('eq_values:', eq_values)
        
        plt.figure(figsize=(3, 1.8*4))  # (width, height)    
        plt.subplot(4, 1, 1)
        plt.plot(SwellingRatio_values, eq_values)
        plt.axhline(y=0, color='black', linestyle='--')  # Add solid horizontal line at y=0
        plt.xlabel('SwellingRatio')
        plt.ylabel('LHS-RHS')
        plt.title('Equations vs SwellingRatio')
        
        # Plot ratio (rho_p_tol):(rho_tot) vs SwellingRatio
        ratio_values = []
        for SwellingRatio in SwellingRatio_values:
            try:
                ratio_values.append(equations(SwellingRatio)[1] / equations(SwellingRatio)[2])
            except:
                ratio_values.append(None)
        print('ratio_values:', ratio_values)
        
        plt.subplot(4, 1, 2)
        plt.plot(SwellingRatio_values, ratio_values)
        plt.xlabel('SwellingRatio')
        plt.ylabel(r'$\rho_{tot}$(T,0,0) : $\rho_{tot}(T,P,S_{sc}$)')
        
        # Plot S_sc_exp vs SwellingRatio
        S_sc_exp_values = []
        for SwellingRatio in SwellingRatio_values:
            try:
                S_sc_exp_values.append(equations(SwellingRatio)[3])
            except:
                S_sc_exp_values.append(None)
        print('S_sc_exp_values:', S_sc_exp_values)
        
        plt.subplot(4, 1, 3)
        plt.plot(SwellingRatio_values, S_sc_exp_values)
        plt.xlabel('SwellingRatio')
        plt.ylabel(r'$S_{sc}$')
        
        # Plot (rho_tot(T,0,0))):(rho_tot(T,P,S))*(1+S_sc_exp)
        lumped_values = []
        for SwellingRatio in SwellingRatio_values:
            try:
                lumped_values.append((equations(SwellingRatio)[1] / equations(SwellingRatio)[2] * (1 + equations(SwellingRatio)[3])))
            except:
                lumped_values.append(None)
        print('lumped_values:', lumped_values)
        
        plt.subplot(4, 1, 4)
        plt.plot(SwellingRatio_values, lumped_values)
        plt.xlabel('SwellingRatio')
        plt.ylabel(r'$\frac{\rho_{tot} \left( T,0,0 \right) }{\rho_{tot} \left( T,P,S_{sc} \right)}(1+S_{sc})$')
        
        plt.tight_layout()
        plt.show()
    
    def solve_solubility_plot_Ssc(self, rhoCO2_data: str = 'SW', Ssc_list=linspace(0, 0.3, 30)):
        # Import data object
        data = SolPolExpData(self.sol, self.pol)
        
        # Import sorption data at specified T
        _df = data.get_sorption_data(self.T)
        
        # Raise error if data, _df, is not found
        if _df is None:
            raise Exception(f"Data not found at T = {self.T} K.")            
        
        # Calculate corrected swelling using SAFT and exp data
        mask = abs(_df["P[bar]"]*1e5 - self.P) <= (self.P*0.01)
        
        try:
            m_net_exp = _df[mask]["MP1*[g]"].values[0] - data.m_met_filled  # [g]
            
            if rhoCO2_data == 'EXP':
                # Use EXP values of rho_f
                rho_f = _df[mask]["ρ[g/cc]"].values[0]
            elif rhoCO2_data == 'SW':
                # Use Span Wagner EoS values of rho_f
                rho_f = _df[mask]["ρSW[g/cc]"].values[0]
            elif rhoCO2_data == 'SAFT':
                # Use SAFT EoS values of rho_f
                rho_f = _df[mask]["ρSAFT[g/cc]"].values[0]
                
            print('rhoCO2 data use: ', rhoCO2_data)
            
            V_b_exp = data.Vbasket  # [cc]
            V_t0_exp = data.Vs  # [cc]
            m_ptot_exp = data.ms    # [g]
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        # Calculate V_t0_exp based on calculated rho_tot(T,0,0)
        rhoT00 = self.rho_tot(self.T, 1, 0)*1e-6  # [g/cm^3]
        V_t0_model = m_ptot_exp/rhoT00  # [cm^3]
        
        #* Choose V_t0 values
        # V_t0 = V_t0_exp  # [cm^3], exp value
        V_t0 = V_t0_model  # [cm^3], calculated value
        
        def equations(S_sc):            
            # Calculate rho_tot(T,P,S)
            # rhoT00 = self.rho_tot(self.T, 1, 0)*1e-6  # [g/cm^3]
            rhoTPS = self.rho_tot(self.T, self.P, S_sc)*1e-6  # [g/cm^3]
            
            # Numerator and denominator of RHS
            numerator1 = m_net_exp
            numerator2 = rho_f * (V_b_exp + m_ptot_exp / rhoTPS)
            numerator = numerator1 + numerator2
            denominator = m_ptot_exp - rho_f * m_ptot_exp / rhoTPS
            
            # Define the equations in terms of these variables and instance variables
            LHS = S_sc
            RHS = numerator / denominator
            
            return LHS-RHS, numerator1, numerator2, numerator, denominator, RHS, m_ptot_exp / rhoTPS
        
        #* S_sc values to evaluate
        S_sc_values = linspace(0, 0.3, 30)
        
        eq_values = []
        numerator1_values = []
        numerator2_values = []
        numerator_values = []
        denominator_values = []
        RHS_values = []
        basket_sample_values = []
        for S_sc in S_sc_values:
            try:
                result = equations(S_sc)
            except:
                result = [None, None, None, None, None, None, None]
            
            eq_values.append(result[0])
            numerator1_values.append(result[1])
            numerator2_values.append(result[2])
            numerator_values.append(result[3])
            denominator_values.append(result[4])
            RHS_values.append(result[5])
            basket_sample_values.append(V_b_exp-result[6])
            
        print('eq_values:', eq_values)
        print('numerator1_values:', numerator1_values)
        print('numerator2_values:', numerator2_values)
        print('numerator_values:', numerator_values)
        print('denominator_values:', denominator_values)
        print('RHS_values:', RHS_values)
        print('basket_sample_values:', basket_sample_values)
        
        # Plot LHS-RHS vs S_sc
        plt.figure(figsize=(3, 1.8*7))  # (width, height)    
        plt.subplot(7, 1, 1)
        plt.plot(S_sc_values, eq_values)
        plt.axhline(y=0, color='black', linestyle='--')  # Add solid horizontal line at y=0
        plt.xlabel(r'$S_{sc}$ [g/g]')
        plt.ylabel('LHS-RHS')
        plt.title(f'T = {self.T-273} °C, P = {self.P} Pa')
        
        # Plot numerator vs S_sc        
        plt.subplot(7, 1, 2)
        plt.plot(S_sc_values, numerator1_values)
        plt.xlabel(r'$S_{sc}$ [g/g]')
        plt.ylabel(r'$m_{raw} (T,P)$')
        
        # Plot numerator vs S_sc        
        plt.subplot(7, 1, 3)
        plt.plot(S_sc_values, numerator2_values)
        plt.xlabel(r'$S_{sc}$ [g/g]')
        plt.ylabel(r'$\rho_{CO2} (T,P) \left[ V_{basket} + \frac{m_{p}}{\rho_{tot}(T,P,S_{sc})} \right]$')
        
        # Plot (V_b - m_p/rho_tot)) vs S_sc        
        plt.subplot(7, 1, 4)
        plt.plot(S_sc_values, basket_sample_values)
        plt.xlabel(r'$S_{sc}$ [g/g]')
        plt.ylabel(r'$V_{basket}-\frac{m_{p}}{\rho_{tot}(T,P,S_{sc})}$ [$cm^{3}$]')
        
        # Plot numerator vs S_sc        
        plt.subplot(7, 1, 5)
        plt.plot(S_sc_values, numerator_values)
        plt.xlabel(r'$S_{sc}$ [g/g]')
        plt.ylabel('Numerator')
        
        # Plot denominator vs S_sc
        plt.subplot(7, 1, 6)
        plt.plot(S_sc_values, denominator_values)
        plt.xlabel(r'$S_{sc}$ [g/g]')
        plt.ylabel('Denominator')
        
        # Plot RHS vs S_sc
        plt.subplot(7, 1, 7)
        plt.plot(S_sc_values, RHS_values)
        plt.xlabel(r'$S_{sc}$ [g/g]')
        plt.ylabel('RHS')
        
        plt.tight_layout()
        plt.show()
        
    ### INDEPENDENT functions
    
    def SinglePhaseDensity(self, x:array , T: float, P: float, phase:str = None):
        """Function to calculate single phase density of mixture without specifying phase.
        Unit = [mol/m^3]

        Args:
            x (array_like): molar fraction.
            T (float): Temperature [K].
            P (float): Pressure [Pa].
        """
        
        
        rhoL = self.eos_mix.density(x, T, P, "L")   # [mol/m^3]
        rhoV = self.eos_mix.density(x, T, P, "V")   # [mol/m^3]
        
        if phase == 'L':
            rho = rhoL
            # print("L phase")
            
        elif phase == 'V':
            rho = rhoV
            # print("V phase")
        
        return rho

    def V_sol(self, x: array, T: float, P:float, eps:float = 1.0e-10):
        """Function to calculate partial volume of solute in mixture. 
        Assuming constant in condensed phase.
        Unit = [m3/g].

        Args:
            x (array_like): molar fraction.
            T (float): Temperature [K].
            P (float): Pressure [Pa].
            eps (float): step length in numerical differentiation.
        """
        
        n_sol = 1*x[0]  # [mol]
        n_pol = 1*x[1]  # [mol]
        n = n_sol+n_pol # [mol]
        n_up = n_sol+eps+n_pol  # [mol]
        n_lo = n_sol-eps+n_pol  # [mol]
        
        x_up = hstack([(n_sol+eps)/n_up, n_pol/n_up])  #[mol/m^3]
        x_lo = hstack([(n_sol-eps)/n_lo, n_pol/n_lo])  #[mol/m^3]        
        
        # Eq. 6
        if n_sol > eps:
            dV_dns = (n_up/self.SinglePhaseDensity(x_up, T, P, 'L')-n_lo/self.SinglePhaseDensity(x_lo, T, P, 'L'))/(2*eps)    # [m^3/mol]
        else: #case where n_sol = 0
            dV_dns = (n_up/self.SinglePhaseDensity(x_up, T, P, 'L')-n/self.SinglePhaseDensity(x, T, P, 'L'))/(eps) # [m^3/mol]
        
        return dV_dns/self.MW_sol   # [m^3/g]

    def V_pol(self, x: array, T: float, P:float, eps:float = 1.0e-10):
        """Function to calculate partial volume of polymer in mixture. 
        Assuming constant in condensed phase.
        Unit = [m3/g].

        Args:
            x (array_like): molar fraction.
            T (float): Temperature [K].
            P (float): Pressure [Pa].
            eps (float): step length in numerical differentiation.
        """
        n_sol = 1*x[0]  # [mol]
        n_pol = 1*x[1]  # [mol]
        n = n_sol+n_pol # [mol]
        n_up = n_sol+eps+n_pol  # [mol]
        n_lo = n_sol-eps+n_pol  # [mol]
        
        x_up = hstack([n_sol/n_up,(n_pol+eps)/n_up])  #[mol/m^3]
        x_lo = hstack([n_sol/n_lo,(n_pol-eps)/n_lo])  #[mol/m^3]        
        
        # Eq. 6
        if n_pol > eps:
            dV_dnp = (n_up/self.SinglePhaseDensity(x_up, T, P, 'L')-n_lo/self.SinglePhaseDensity(x_lo, T, P, 'L'))/(2*eps)    # [m^3/mol]
        else: #case where n_sol = 0
            dV_dnp = (n_up/self.SinglePhaseDensity(x_up, T, P, 'L')-n/self.SinglePhaseDensity(x, T, P, 'L'))/(eps) # [m^3/mol]
        
        return dV_dnp / self.MW_pol # [m^3/g]
    

    def Vs_Vp_pmv1(self, T: float, P: float, S_a: float):
        """Function to calculate partial volume of solute in mixture, using pmv method 1.
        pmv method 1: using solubility composition.
        Unit = [m3/g]

        """
        
        omega_p = 1/(S_a+1)     # [g/g]
        omega_s = 1 - omega_p   # [g/g]
        x_s = (omega_s/self.MW_sol) / (omega_s/self.MW_sol + omega_p/self.MW_pol)   #[mol/mol]
        x_p = 1 - x_s   #[mol/mol]           
        x = hstack([x_s, x_p])   # [mol/mol]        
        V_s = self.V_sol(x, T, P)  # [m^3/g]        
        V_p = self.V_pol(x, T, P)  # [m^3/g]
        return V_s, V_p
    
    def Vs_Vp_pmv2(self):
        """Function to calculate partial volume of solute in mixture, using pmv method 2.
        pmv method 2: assuming Vs and Vp same as specific volume at __infinitely dilution__.
        Unit = [m3/g]

        """
        T = self.T
        P = self.P
        V_s = self.V_sol(hstack([0., 1.]), T, P)  # [m^3/g]        
        V_p = self.V_pol(hstack([0., 1.]), T, P)  # [m^3/g]
        return V_s, V_p
    
    def Vs_Vp_pmv3(self):
        """Function to calculate partial volume of solute in mixture, using pmv method 3.
        pmv method 3: assuming Vs and Vp at __infinitely dilution__, unchanged at atmospheric pressure.
        Unit = [m3/g]
        
        """
        T = self.T        
        V_s = self.V_sol(hstack([0., 1.]), T, 1e5)  # [m^3/g]        
        V_p = self.V_pol(hstack([0., 1.]), T, 1e5)  # [m^3/g]
        return V_s, V_p

    def get_omega_cr(self, T: float):
        """Function to get omega_cr as a function of T. Reads data from excel sheet called /data_CO2-HDPE_trimmed.xlsx.
        Ref: [1]: Polymer, 59, 2015, 270-277.
        
        """
        
        data = SolPolExpData(self.sol, self.pol)
        
        try:
            # Read exp file
            df = pd.read_excel(data.file, sheet_name="omega_cr")
        
        except Exception as e:
            print("Error: ")
            print(e)
            return None        
        
        if self.pol == "HDPE":
            # print(df.dtypes)  # Chek data types
            
            # Use literature values
            # omega_c = df[df["T (°C)"] == (T-273)]["omega_cr_HDPE"].values[0]
            
            #* Use converged values based on convergence between exp data and SAFT values of V_t0
            omega_c = df[df["T (°C)"] == (T-273)]["rho_cr_HDPE_converged (g/cm3)"].values[0]
            
        else:
            omega_c = 0
            
        return omega_c
    
    def get_rho_pol_cr(self, T: float):
        """Function to get density of crystalline domain of polymer.
            
        """
        
        data = SolPolExpData(self.sol, self.pol)
        try:
            # Read exp file
            df = pd.read_excel(data.file, sheet_name="rho_cr")
        
        except Exception as e:
            print("Error: ")
            print(e)
            return None        
        
        if self.pol == "HDPE":
            # print(df.dtypes)  # Chek data types
            _rho_pol_cr = df[df["T (°C)"] == (T-273)]["rho_cr_HDPE (g/cm3)"].values[0]   # [g/cm^3]
            
        else:
            _rho_pol_cr = 1000   # [g/cm^3]
        
        rho_pol_cr = _rho_pol_cr *1e6   # [g/m^3]
        return rho_pol_cr
    
    ### DEPENDENT functions
    def rho_tot(self, T, P, S_sc, omega_cr = None):
        if omega_cr == None:
            omega_cr = self.omega_cr
        
        # Get rho_pol_cr
        rho_p_c = self.rho_pol_cr  # [g/m^3]
        
        # Calculate S_am_exp
        S_am = S_sc / (1-omega_cr)
        
        # Calculate pmv V_s and V_p
        Vs, Vp = self.Vs_Vp_pmv1(T, P, S_am)   # [m3/g]     #* Default
        # Vs, Vp = self.Vs_Vp_pmv2()   # [m3/g]         #* TEST
        
        # Calculte rho_tot
        rho_p_am = 1/(S_am*Vs + Vp)  # [g/m^3]
        rho_p_tol = 1/((1-omega_cr)/rho_p_am + omega_cr/rho_p_c)  # [g/m^3]
        rho_tot = (1 + S_sc) * rho_p_tol  # [g/m^3]
        
        # print(f'_rho_p_c = {rho_p_c} g/m^3')
        # print(f"_rho_tot = {rho_tot} g/m^3")
        return rho_tot
    
        
class SolPolExpData():
    def __init__(self, sol: str, pol: str, data_file="../data/data_CO2-HDPE.xlsx"):
        self.sol = sol
        self.pol = pol
        # mixture = SolPolMixture(sol, pol)
        try:
            # Read exp file
            path = os.path.dirname(__file__)
            filepath = path + f"/{data_file}"
            file = pd.ExcelFile(filepath, engine="openpyxl")
            df_aux = pd.read_excel(file, sheet_name="aux data")
    
        except Exception as e:
            print("Error: ")
            print(e)
        
        self.path = path
        self.file = file
        self.ms = df_aux["ms [g]"].values[0]    # [g]
        # self.MW = mixture.MW_sol    # [g/mol]
        self.Vsk = df_aux["Vsk [cc]"].values[0]    # [cm^3]
        self.Vs = df_aux["Vs [cc]"].values[0]    # [cm^3]
        self.Vbasket = df_aux["Vbasket [cc]"].values[0]    # [cm^3]
        self.rho_pol = df_aux["ρ_pol [g/cc]"].values[0]    # [g/cm^3]
        
        self.m_met_empty = df_aux["m_met_empty [g]"].values[0]    # [g]
        self.V_met_empty = df_aux["V_met_empty [cc]"].values[0]    # [cm^3]
        self.m_met_filled = df_aux["m_met_filled [g]"].values[0]    # [g]
        self.V_met_filled = df_aux["V_met_filled [cc]"].values[0]    # [cm^3]
        self.data_dict = {}
    
    def get_sorption_data(self, T: float):
        df = pd.read_excel(self.file, sheet_name=f"{T-273}C")
        return df

def find_omega_cr(base_obj, T):
    obj = DetailedSolPol(base_obj, T, 1)
    
    # Get experimental data
    data = SolPolExpData(obj.sol, obj.pol)
    V_t0_exp = data.Vs   # [cm^3]
    m_ptot_exp = data.ms    # [g]
    
    def eq(_omega_cr):
        # Calculate V_t0_exp based on calculated rho_tot(T,0,0)
        rhoT00 = obj.rho_tot(obj.T, 1, 0, _omega_cr)*1e-6  # [g/cm^3]
        V_t0_model = m_ptot_exp/rhoT00  # [cm^3]
        
        return V_t0_exp - V_t0_model
    
    omega_cr = fsolve(eq, 0.5)[0]
    # print('omega_cr:', omega_cr)
    
    return omega_cr

def fit_epskl(base_obj, T:float, x0: float = 200, epskl_bounds:tuple = (100, 500), **kwargs):
    """Function to fit eps_kl to fit SAFT prediction to corected experimental data.

    Args:
        base_obj (_type_): _description_
        T (float): Temperature [K].
        x0 (float, optional): Starting value of eps_kl. Defaults to 200.
        epskl_bounds (tuple, optional): Upper and lower bounds for eps_kl. Defaults to (100, 500).

    Returns:
        epskl, fobj: tuple
    """
    
    data = SolPolExpData(base_obj.sol, base_obj.pol)
    _df=data.get_sorption_data(T)
    
    if "pmv_method" in kwargs:
        pmv_method = kwargs["pmv_method"]
    else:
        pmv_method = "1"
        
    def fobj(var):
        eps = var
        print("eps = ", eps)
        base_obj.modify_kl(eps)
        
        P_list = _df["P[bar]"].values * 1e5    # [Pa]
        
        objects = []
        solubility_SAFT = []
        solubility_exp_corrected = []
        for j, _p in enumerate(P_list):
            # Create objects at each T and P
            if j == 0:
                obj = DetailedSolPol(base_obj, T, _p, pmv_method=pmv_method)
            else:
                x0_list = update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                obj = DetailedSolPol(base_obj, T, _p, x0_sol_range = x0_list, pmv_method=pmv_method)
            
            # Solubility prediction from SAFT
            try:
                _S_SAFT = obj.S_sc_EOS
            except:
                _S_SAFT = None            
            
            # Calculate corrected swelling using SAFT and exp data
            mask = abs(_df["P[bar]"]*1e5 - _p) <= (_p*0.01)
            try:
                _SwR = obj.SwellingRatio
                _S_exp_corrected = (_df[mask]["MP1*[g]"]-data.m_met_filled+_df[mask]["ρ[g/cc]"]*(data.Vs*(1+_SwR)+data.Vbasket)) / data.ms
                _S_exp_corrected = _S_exp_corrected.values[0]
            except:
                _S_exp_corrected = None
            
            objects.append(obj)
            solubility_SAFT.append(_S_SAFT)
            solubility_exp_corrected.append(_S_exp_corrected)
        
        solubility_SAFT = array(solubility_SAFT)
        solubility_exp_corrected = array(solubility_exp_corrected)
        return sum((solubility_SAFT - solubility_exp_corrected)**2)
    
    result = minimize_scalar(fobj, x0, method='bounded', bounds=epskl_bounds)
    
    print("Optimised value of eps: ", result.x)
    print("Objective function value at optimised: ", fobj(result.x))
    return result.x, fobj(result.x)

if __name__ == "__main__":    
    mix = BaseSolPol("CO2","HDPE")
    # data = SolPolExpData(mix.sol, mix.pol)
    # print(data.get_sorption_data(25+273))
    # obj=DetailedSolPol(mix,25+273,1e6)
    # print("S_am = ", obj.S_am)
    # print("x_am = ", obj.x_am)
    # print("omaga_am = ", obj.omega_am)
    # mix.pmv_method = "2"
    
    # plot_isotherm_EOSvExp(mix,[25+273, 35+273, 50+273], export_data=False, display_fig=False, save_fig=True)
    # for T in [25+273, 35+273, 50+273]:
    #     plot_isotherm_pmv(mix, [T], export_data=False, display_fig=False, save_fig=True)
    #     plot_VsVp_pmv(mix, T, display_fig=False, save_fig=True)
    
    #* eps_kl_EOS
    
    # for T in [25+273, 35+273, 50+273]:
    #     plot_isotherm_epskl_EOS(mix, T_list=[T], P_list=linspace(1, 200e5, 5), \
    #         eps_list=[276.45, 276.45*0.95, 276.45*1.05], \
    #             export_data=False, display_fig=False, save_fig=True)
    
    #* eps_kl_EOSvsExp
    # for T in [25+273, 35+273, 50+273]:
    #     plot_isotherm_epskl_EOSvExp(mix, T_list=[T],
    #                     eps_list=[276.45, 276.45*0.95, 276.45*1.05], 
    #                     export_data=False,  display_fig=False, save_fig=True)
    
    #* fit_epskl    
    # eps25_pmv1, fobj25_pmv1 = fit_epskl(mix, T=25+273, x0=200, epskl_bounds=(200, 300), pmv_method = "1")
    # eps35_pmv1, fobj35_pmv1 = fit_epskl(mix, T=35+273, x0=200, epskl_bounds=(200, 300), pmv_method = "1")
    # eps50_pmv1, fobj50_pmv1 = fit_epskl(mix, T=50+273, x0=200, epskl_bounds=(200, 300), pmv_method = "1")
    
    # eps25_pmv2, fobj25_pmv2 = fit_epskl(mix, T=25+273, x0=200, epskl_bounds=(200, 300), pmv_method = "2")
    # eps35_pmv2, fobj35_pmv2 = fit_epskl(mix, T=35+273, x0=200, epskl_bounds=(200, 300), pmv_method = "2")
    # eps50_pmv2, fobj50_pmv2 = fit_epskl(mix, T=50+273, x0=200, epskl_bounds=(200, 300), pmv_method = "2")
    
    # eps25_pmv3, fobj25_pmv3 = fit_epskl(mix, T=25+273, x0=200, epskl_bounds=(200, 300), pmv_method = "3")
    # eps35_pmv3, fobj35_pmv3 = fit_epskl(mix, T=35+273, x0=200, epskl_bounds=(200, 300), pmv_method = "3")
    # eps50_pmv3, fobj50_pmv3 = fit_epskl(mix, T=50+273, x0=200, epskl_bounds=(200, 300), pmv_method = "3")
    # print("pmv 1")
    # print(f"25°C, eps = {eps25_pmv1}, fobj = {fobj25_pmv1}")
    # print(f"35°C, eps = {eps35_pmv1}, fobj = {fobj35_pmv1}")
    # print(f"50°C, eps = {eps50_pmv1}, fobj = {fobj50_pmv1}")    
    # print("pmv 2")
    # print(f"25°C, eps = {eps25_pmv2}, fobj = {fobj25_pmv2}")
    # print(f"35°C, eps = {eps35_pmv2}, fobj = {fobj35_pmv2}")
    # print(f"50°C, eps = {eps50_pmv2}, fobj = {fobj50_pmv2}")
    # print("pmv 3")
    # print(f"25°C, eps = {eps25_pmv3}, fobj = {fobj25_pmv3}")
    # print(f"35°C, eps = {eps35_pmv3}, fobj = {fobj35_pmv3}")
    # print(f"50°C, eps = {eps50_pmv3}, fobj = {fobj50_pmv3}")
    # mix.modify_kl(259.78)
    # plot_isotherm_EOSvExp(mix, T_list=[25+273], export_data=False, display_fig=False, save_fig=True)
    # mix.modify_kl(244.23)
    # plot_isotherm_EOSvExp(mix, T_list=[35+273], export_data=False, display_fig=False, save_fig=True)
    # mix.modify_kl(251.05)
    # plot_isotherm_EOSvExp(mix, T_list=[50+273], export_data=False, display_fig=False, save_fig=True)
    
    
    