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
from matplotlib.lines import Line2D
from colour import Color
from numpy import *
import pandas as pd
from scipy.optimize import fsolve, minimize_scalar, root, minimize
from scipy import integrate
from sympy import symbols, Eq, solve, nsolve


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
        
        self._S_am_EOS = None
        self._S_sc_EOS = None
        self._x_am_EOS = None
        self._omega_am_EOS = None
        self._S_sc_exp_uncorr = None
        self._S_sc_exp = None
        self._S_am_exp = None
        self._x_am_exp = None
        self._omega_am_exp = None
        self._rho_pol = None
        self._omega_cr = None
        self._muad_sol_ext = None
        self._rho_pol_cr = None
        self._rho_am = None
        self._rho_mix = None
        self._rho_mix_0 = None
        self._SwellingRatio = None
                
    @property
    def S_sc_exp(self):
        if self._S_sc_exp is None or not self.check_condition():
            self._S_sc_exp = self.get_S_sc_exp()
        return self._S_sc_exp
    @property
    def SwellingRatio(self):
        if self._SwellingRatio is None or not self.check_condition():
            self._SwellingRatio = self.get_SwellingRatio()
        return self._SwellingRatio
    
    
    @property
    def S_am_EOS(self):
        if self._S_am_EOS is None:
            self._S_am_EOS = self.get_S_am_EOS(self.T, self.P)
        return self._S_am_EOS
    @property
    def S_sc_EOS(self):
        if self._S_sc_EOS is None:
            self._S_sc_EOS = self.get_S_sc_EOS()
        return self._S_sc_EOS
    @property
    def x_am_EOS(self):
        if self._x_am_EOS is None:
            self._x_am_EOS = self.get_x_am_EOS()
        return self._x_am_EOS
    @property
    def omega_am_EOS(self):
        if self._omega_am_EOS is None:
            self._omega_am_EOS = self.get_omega_am_EOS()
        return self._omega_am_EOS
    @property
    def S_am_exp(self):
        if self._S_am_exp is None:
            self._S_am_exp = self.get_S_am_exp()
        return self._S_am_exp
    @property
    def x_am_exp(self):
        if self._x_am_exp is None:
            self._x_am_exp = self.get_x_am_exp()
        return self._x_am_exp
    @property
    def omega_am_exp(self):
        if self._omega_am_exp is None:
            self._omega_am_exp = self.get_omega_am_exp()
        return self._omega_am_exp
    @property
    def S_sc_exp_uncorr(self):
        if self._S_sc_exp is None:
            self._S_sc_exp = self.get_S_sc_exp_uncorr()
        return self._S_sc_exp_uncorr
    @property
    def rho_am(self):
        if self._rho_am is None:
            self._rho_am = self.get_rho_am()
        return self._rho_am
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
    @property
    def muad_sol_ext(self):
        if self._muad_sol_ext is None:
            self._muad_sol_ext = self.get_muad_sol_ext(self.T, self.P)
        return self._muad_sol_ext
    @property
    def rho_mix(self):
        if self._rho_mix is None:
            self._rho_mix = self.get_rho_mix()
        return self._rho_mix
    @property
    def rho_mix_0(self):
        if self._rho_mix_0 is None:
            self._rho_mix_0 = self.get_rho_mix_0()
        return self._rho_mix_0

    ### Auxillary fucntions to check solubiltiy and swelling consitency
    def check_conditions(self):
        
        ### Check solubility and SwR consistency
        return True
    
    def _solve_parameters_minimize(self): # numerical solving
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
            rho_f_exp = _df[mask]["ρ[g/cc]"].values[0]
            V_b_exp = data.Vbasket
            V_t0_exp = data.Vs
            m_ptot_exp = data.ms
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        # Get omega_cr
        omega_cr = self.omega_cr
        
        # Get rho_pol_cr
        rho_p_c = self.rho_pol_cr  # [g/m^3]
        
        def equations(vars):
            S_sc_exp, SwellingRatio = vars
            
            # Calculate S_am_exp
            S_am_exp = S_sc_exp / (1-self.omega_cr)  # [g/g]
            
            # Calculate pmv V_s and V_p
            Vs, Vp = self.Vs_Vp_pmv1(self.T, self.P, S_am_exp)   # [m3/g]
            
            # Define auxiliary variables
            rho_p_am = 1/(S_am_exp*Vs + Vp)  # [g/m^3]
            rho_p_tol = 1/((1-omega_cr)/rho_p_am + omega_cr/rho_p_c)  # [g/m^3]
            rho_tot = (1 + S_sc_exp) * rho_p_tol  # [g/m^3]
        
            # Define the equations in terms of these variables and instance variables
            eq1 = -S_sc_exp + (m_net_exp + rho_f_exp * (V_b_exp + V_t0_exp * (1 + SwellingRatio))) / m_ptot_exp
            eq2 = -SwellingRatio + (rho_p_tol / rho_tot) * (1 + S_sc_exp) - 1
            
            return [eq1, eq2]
        
        # Objective function (sum of squares of equations)
        def objective(vars):
            eqs = equations(vars)
            return sum(eq**2 for eq in eqs)
        
        # Initial guess        
        # initial_guesses = [[0.01, 0.], [0.01, 0.01], [0.01, 0.02], [0.01, 0.03],] # [S_sc_exp_guess, SwellingRatio_guess]
        initial_guesses = [[S_sc_exp, SwellingRatio] for S_sc_exp in arange(0, 0.11, 0.01) for SwellingRatio in arange(0, 0.11, 0.01)]
        
        print("Initial guesses:", *initial_guesses)
        
        solutions = []
        objective_values = []
        
        # Bounds
        # bounds = [(0, None), (0, None)] # Non-negative values
        bounds = [(0.001, None), (0.001, None)] # Positive values

        for x0 in initial_guesses:
            print(f"Initial guess: {x0}")
            solution = minimize(objective, x0, method='L-BFGS-B', bounds=bounds) # Use 'L-BFGS-B' or 'TNC
            solutions.append(solution.x)
            objective_values.append(solution.fun)
            
            print(f"S_sc_exp = {solution.x[0]}, SwellingRatio = {solution.x[1]}")
            print("Objective:", solution.fun)
            print("")
            
        # Extract S_sc_exp and SwellingRatio from solutions
        S_sc_exp_values = [solution[0] for solution in solutions]
        SwellingRatio_values = [solution[1] for solution in solutions]

        # Plot S_sc_exp
        plt.figure(figsize=(5, 6))
        plt.subplot(3, 1, 1)
        plt.plot(S_sc_exp_values)
        plt.xlabel('Iteration')
        plt.ylabel('S_sc_exp')
        plt.title('S_sc_exp vs Iteration')

        # Plot SwellingRatio
        plt.subplot(3, 1, 2)
        plt.plot(SwellingRatio_values)
        plt.xlabel('Iteration')
        plt.ylabel('SwellingRatio')
        plt.title('SwellingRatio vs Iteration')
        
        # Plot objective function
        plt.subplot(3, 1, 3)
        plt.plot(objective_values)
        plt.xlabel('Iteration')
        plt.ylabel('Objective')
        plt.title('Objective Function vs Iteration')

        plt.tight_layout()
        plt.show()
        
        print(solutions)
        
        
        return solutions
        
    def _solve_parameters_analytical(self): # analytically solving
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
            rho_f_exp = _df[mask]["ρ[g/cc]"].values[0]
            V_b_exp = data.Vbasket
            V_t0_exp = data.Vs
            m_ptot_exp = data.ms
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        # Get omega_cr
        omega_cr = self.omega_cr
        
        # Get rho_pol_cr
        rho_p_c = self.rho_pol_cr  # [g/m^3]
        
        # Define the symbols
        S_sc_exp, SwellingRatio = symbols('S_sc_exp SwellingRatio')
        
        # Define additional variables
        S_am_exp = S_sc_exp / (1-self.omega_cr)  # [g/g]
        
        #* Use pmv2 so that Vs and Vp are constant
        Vs, Vp = self.Vs_Vp_pmv1(self.T, self.P, S_am_exp)   # [m3/g]
        # Vs, Vp = self.Vs_Vp_pmv2()   # [m3/g]
        
        rho_p_am = 1/(S_am_exp*Vs + Vp)  # [g/m^3]
        rho_p_tol = 1/((1-omega_cr)/rho_p_am + omega_cr/rho_p_c)  # [g/m^3]
        rho_tot = (1 + S_sc_exp) * rho_p_tol  # [g/m^3]
        
        # Define simulation equations
        eq1 = Eq(S_sc_exp , (m_net_exp + rho_f_exp * (V_b_exp + V_t0_exp * (1 + SwellingRatio))) / m_ptot_exp)
        eq2 = Eq(SwellingRatio , (rho_p_tol / rho_tot) * (1 + S_sc_exp) - 1)
        
        # Solve the equations
        try:
            solution = solve((eq1, eq2), (S_sc_exp, SwellingRatio))
            print(f"S_sc_exp = {solution[S_sc_exp]}, SwellingRatio = {solution[SwellingRatio]}")
        except Exception as e:
            print("Error: ")
            print(e)        
        
        return solution[S_sc_exp], solution[SwellingRatio]

    def _solve_parameters_nsolve(self): # numerical solving
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
            rho_f_exp = _df[mask]["ρ[g/cc]"].values[0]
            V_b_exp = data.Vbasket
            V_t0_exp = data.Vs
            m_ptot_exp = data.ms
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        # Get omega_cr
        omega_cr = self.omega_cr
        
        # Get rho_pol_cr
        rho_p_c = self.rho_pol_cr  # [g/m^3]
        
        # Define the symbols
        S_sc_exp, SwellingRatio = symbols('S_sc_exp SwellingRatio')
        
        # Define additional variables
        S_am_exp = S_sc_exp / (1-self.omega_cr)  # [g/g]
        
        #* Use pmv2 so that Vs and Vp are constant
        # Vs, Vp = self.Vs_Vp_pmv1(self.T, self.P, S_am_exp)   # [m3/g]
        Vs, Vp = self.Vs_Vp_pmv2()   # [m3/g]
        
        rho_p_am = 1/(S_am_exp*Vs + Vp)  # [g/m^3]
        rho_p_tol = 1/((1-omega_cr)/rho_p_am + omega_cr/rho_p_c)  # [g/m^3]
        rho_tot = (1 + S_sc_exp) * rho_p_tol  # [g/m^3]
        
        # Define simulation equations
        eq1 = Eq(S_sc_exp , (m_net_exp + rho_f_exp * (V_b_exp + V_t0_exp * (1 + SwellingRatio))) / m_ptot_exp)
        eq2 = Eq(SwellingRatio , (rho_p_tol / rho_tot) * (1 + S_sc_exp) - 1)
        
        # Initial guess        
        # initial_guesses = [[0.01, 0.0], [0.01, 0.01], [0.01, 0.02], [0.01, 0.03]] # [S_sc_exp_guess, SwellingRatio_guess]
        initial_guesses = [[S_sc_exp, SwellingRatio] \
            for S_sc_exp in arange(0, 10., 0.1) \
            for SwellingRatio in arange(0, 1., 0.01)]
        
        solutions = []
        
        # Iterate over the initial guesses
        for guess in initial_guesses:
            try:
                # Solve the equations
                solution = nsolve((eq1, eq2), (S_sc_exp, SwellingRatio), guess)
                print(f"Initial guess: {guess}")
                print(f"S_sc_exp = {solution[0]}, SwellingRatio = {solution[1]}")
                print("")
                solutions.append(solution)

            except Exception as e:
                print(f"Error with initial guess {guess}:")
                print(e)
        
        # Extract S_sc_exp and SwellingRatio from solutions
        S_sc_exp_values = [solution[0] for solution in solutions]
        SwellingRatio_values = [solution[1] for solution in solutions]

        # Plot S_sc_exp
        plt.figure(figsize=(5, 6*2/3))  # (width, height)
        plt.subplot(2, 1, 1)
        plt.plot(S_sc_exp_values)
        plt.xlabel('Iteration')
        plt.ylabel('S_sc_exp')
        plt.title('S_sc_exp vs Iteration')

        # Plot SwellingRatio
        plt.subplot(2, 1, 2)
        plt.plot(SwellingRatio_values)
        plt.xlabel('Iteration')
        plt.ylabel('SwellingRatio')
        plt.title('SwellingRatio vs Iteration')

        plt.tight_layout()
        plt.show()
        
        # return solution[S_sc_exp], solution[SwellingRatio]
        
    def _solve_parameters_plots(self):
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
            rho_f_exp = _df[mask]["ρ[g/cc]"].values[0]  # [g/cc]
            V_b_exp = data.Vbasket  # [cc]
            V_t0_exp = data.Vs  # [cc]
            m_ptot_exp = data.ms    # [g]
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        # Get omega_cr
        omega_cr = self.omega_cr
        
        # Get rho_pol_cr
        rho_p_c = self.rho_pol_cr  # [g/m^3]
        
        def equations(SwellingRatio):
            
            # Calculate S_sc_exp
            S_sc_exp = (m_net_exp + rho_f_exp * (V_b_exp + V_t0_exp * (1 + SwellingRatio))) / m_ptot_exp
            
            # Calculate S_am_exp
            S_am_exp = S_sc_exp / (1-self.omega_cr)  # [g/g]
            
            # Calculate pmv V_s and V_p
            Vs, Vp = self.Vs_Vp_pmv1(self.T, self.P, S_am_exp)   # [m3/g]
            
            # Define auxiliary variables
            rho_p_am = 1/(S_am_exp*Vs + Vp)  # [g/m^3]
            rho_p_tol = 1/((1-omega_cr)/rho_p_am + omega_cr/rho_p_c)  # [g/m^3]
            rho_tot = (1 + S_sc_exp) * rho_p_tol  # [g/m^3]
        
            # Define the equations in terms of these variables and instance variables
            LHS = SwellingRatio
            RHS = (rho_p_tol / rho_tot) * (1 + S_sc_exp) - 1
            
            return LHS-RHS, rho_p_tol, rho_tot, S_sc_exp
        
        # Evaluate SwellingRatio between 0 and 10
        SwellingRatio_values = linspace(0, 0.1, 20)
        
        # Plot LHS-RHS vs SwellingRatio
        eq_ab_values = [abs(equations(SwellingRatio)[0]) for SwellingRatio in SwellingRatio_values]
        
        plt.figure(figsize=(5, 8))  # (width, height)        
        plt.subplot(4, 1, 1)
        plt.plot(SwellingRatio_values, eq_ab_values)
        plt.axhline(y=0, color='black', linestyle='--')  # Add solid horizontal line at y=0
        plt.xlabel('SwellingRatio')
        plt.ylabel('LHS-RHS')
        plt.title('Equations vs SwellingRatio')
        
        # Plot ratio (rho_p_tol):(rho_tot) vs SwellingRatio        
        ratio_values = [abs(equations(SwellingRatio)[1] / equations(SwellingRatio)[2]) for SwellingRatio in SwellingRatio_values]
        
        plt.subplot(4, 1, 2)
        plt.plot(SwellingRatio_values, ratio_values)
        plt.xlabel('SwellingRatio')
        plt.ylabel(r'$\rho_{p,tot}$ : $\rho_{tot}$')
        plt.title(r'$\rho_{p,tot}$:$\rho_{tot}$ vs SwellingRatio')
        
        # Plot S_sc_exp vs SwellingRatio
        S_sc_exp_values = [equations(SwellingRatio)[3] for SwellingRatio in SwellingRatio_values]
        print('S_sc_exp_values:', S_sc_exp_values)
        
        plt.subplot(4, 1, 3)
        plt.plot(SwellingRatio_values, S_sc_exp_values)
        plt.xlabel('SwellingRatio')
        plt.ylabel(r'$S_{sc}^{exp}$')
        plt.title('S_sc_exp vs SwellingRatio')
        
        # Plot (rho_p_tol):(rho_tot)*(1+S_sc_exp)
        lumped_values = [(equations(SwellingRatio)[1] / equations(SwellingRatio)[2] * (1 + equations(SwellingRatio)[3])) for SwellingRatio in SwellingRatio_values]
        print('lumped_values:', lumped_values)
        plt.subplot(4, 1, 4)
        plt.plot(SwellingRatio_values, lumped_values)
        plt.xlabel('SwellingRatio')
        plt.ylabel(r'$\frac{\rho_{p,tot}}{\rho_{tot}}$(1+$S_{sc}^{exp})$')
        plt.title(r'$\frac{\rho_{p,tot}}{\rho_{tot}}$(1+$S_{sc}^{exp})$ vs SwellingRatio')
        
        plt.tight_layout()
        plt.show()
    
    def solve_parameters_fsolve_NEW(self): # numerical solving
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
            
            #* Use EXP values of rho_f
            # rho_f = _df[mask]["ρ[g/cc]"].values[0]
            
            #* Use Span Wagner EoS values of rho_f
            # rho_f = _df[mask]["ρSW[g/cc]"].values[0]
            
            #* Use SAFT EoS values of rho_f
            rho_f = _df[mask]["ρSAFT[g/cc]"].values[0]
            
            V_b_exp = data.Vbasket
            V_t0_exp = data.Vs
            m_ptot_exp = data.ms
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        ## TEST
        print(f'rho_f: {rho_f} g/cm^3')
        ## /TEST
        
        # Get omega_cr
        omega_cr = self.omega_cr
        print(f'omega_cr: {omega_cr}')
                
        # Get rho_pol_cr
        rho_p_c = self.rho_pol_cr  # [g/m^3]
        print(f'rho_p_cr = {rho_p_c} g/m^3')
        
        def S_sc_exp(SwR):
            # Calculate V_t0_exp based on calculated rho_tot(T,0,0)
            rhoT00 = self.rho_tot(self.T, 1, 0)*1e-6  # [g/cm^3]  #* Default
            V_t0_model = m_ptot_exp/rhoT00  # [cm^3]
            
            # rhoTP0 = self.rho_tot(self.T, self.P, 0)*1e-6  # [g/cm^3]    #* TEST
            # V_t0_model = m_ptot_exp/rhoTP0  # [cm^3]
            
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
        initial_guesses = [i for i in linspace(0, 0.3, 6)]
        
        print("Initial guesses:", *initial_guesses)
        
        solutions = []

        for x0 in initial_guesses:
            print(f"Initial guess: {x0}")
            try:
                solution = fsolve(equation, x0, xtol=1.0e-10)
                print('equation(solution[0]):', equation(solution[0]))
                # if isclose([0], [equation(solution[0])], atol=1e-2): # isclose() requires input arrays #*Default
                if isclose([0], [equation(solution[0])], atol=1e-4 if self.P < 80e5 else 1e-3): #* TEST
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
                    # if not any([isclose(solution, unique_solution, rtol=1e-2) for unique_solution in unique_solutions]):    #* Default
                    if not any([isclose(solution, unique_solution, rtol=5e-2) for unique_solution in unique_solutions]):    #* TEST
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
        
    def solve_parameters_fsolve_NEW_noSwelling(self): # numerical solving
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
            rho_f_exp = _df[mask]["ρ[g/cc]"].values[0]
            V_b_exp = data.Vbasket
            V_t0_exp = data.Vs
            m_ptot_exp = data.ms
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        # Get omega_cr
        omega_cr = self.omega_cr
        
        # Get rho_pol_cr
        rho_p_c = self.rho_pol_cr  # [g/m^3]
        
        def rho_tot(T, P, S_sc):
            # Calculate S_am_exp
            S_am = S_sc / (1-omega_cr)
            
            # Calculate pmv V_s and V_p
            Vs, Vp = self.Vs_Vp_pmv1(T, P, S_am)   # [m3/g]
            
            # Calculte rho_tot
            rho_p_am = 1/(S_am*Vs + Vp)  # [g/m^3]
            rho_p_tol = 1/((1-omega_cr)/rho_p_am + omega_cr/rho_p_c)  # [g/m^3]
            rho_tot = (1 + S_sc) * rho_p_tol  # [g/m^3]
            
            return rho_tot
        
        def S_sc_exp(SwR):
            # Calculate S_sc_exp
            S_sc_exp = (m_net_exp + rho_f_exp * (V_b_exp + V_t0_exp * (1 + SwR))) / m_ptot_exp
            return S_sc_exp

        # Extract S_sc_exp and SwellingRatio from solutions
        SwellingRatio_values = [0]  # No swelling
        S_sc_exp_values = [S_sc_exp(SwR) for SwR in SwellingRatio_values]
        
        return SwellingRatio_values[0], S_sc_exp_values[0]
        
    def solve_parameters_fsolve_NEW_pc(self, pc: float): # numerical solving
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
            rho_f_exp = _df[mask]["ρ[g/cc]"].values[0]
            V_b_exp = data.Vbasket
            V_t0_exp = data.Vs
            m_ptot_exp = data.ms
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        # Get omega_cr
        omega_cr = self.omega_cr
        
        # Get rho_pol_cr
        rho_p_c = self.rho_pol_cr  # [g/m^3]
        
        # Effective pressure
        p_eff = self.P + pc # [Pa]
        
        def rho_tot(T, P, S_sc):
            # Calculate S_am_exp
            S_am = S_sc / (1-omega_cr)
            
            # Calculate pmv V_s and V_p
            Vs, Vp = self.Vs_Vp_pmv1(T, P, S_am)   # [m3/g]     #* Default
            
            # Calculte rho_tot
            rho_p_am = 1/(S_am*Vs + Vp)  # [g/m^3]
            rho_p_tol = 1/((1-omega_cr)/rho_p_am + omega_cr/rho_p_c)  # [g/m^3]
            rho_tot = (1 + S_sc) * rho_p_tol  # [g/m^3]
            
            return rho_tot
        
        def S_sc_exp(SwR):
            # Calculate S_sc_exp
            S_sc_exp = (m_net_exp + rho_f_exp * (V_b_exp + V_t0_exp * (1 + SwR))) / m_ptot_exp
            return S_sc_exp
            
        def equation(SwellingRatio):
            # Calculate S_sc_exp
            S_sc = S_sc_exp(SwellingRatio)
            
            # Calculate rho_tot(T,P,S)
            rhoT00 = rho_tot(self.T, 1, 0)  # [g/m^3], use P = 1 Pa #TODO: Check if P = 1 Pa is correct
            rhoTPS = rho_tot(self.T, p_eff, S_sc)  # [g/m^3]
            
            # Define the equations in terms of these variables and instance variables
            LHS = SwellingRatio
            RHS = (rhoT00 / rhoTPS) * (1 + S_sc) - 1
            
            return LHS-RHS    # Solve for LHS-RHS = 0
        
        
        # *Initial guesses
        # initial_guesses = [i for i in linspace(0, 0.5, 11)] #* Full range
        initial_guesses = [i for i in linspace(0, 0., 2)]  # *TEST
        
        print("Initial guesses:", *initial_guesses)
        
        solutions = []

        for x0 in initial_guesses:
            print(f"Initial guess: {x0}")
            try:
                solution = fsolve(equation, x0)
                print('equation(solution[0]):', equation(solution[0]))
                if isclose([0], [equation(solution[0])], atol=1e-2): # isclose() requires input arrays
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
                    if not any([isclose(solution, unique_solution, rtol=1e-2) for unique_solution in unique_solutions]):
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

        # # Plot S_sc_exp vs. iterations
        # plt.figure(figsize=(5, 4))  # (width, height)
        # plt.subplot(2, 1, 1)
        # plt.plot(S_sc_exp_values)
        # plt.xlabel('Iteration')
        # plt.ylabel('S_sc_exp')
        # plt.title('S_sc_exp vs Iteration')

        # # Plot SwellingRatio
        # plt.subplot(2, 1, 2)
        # plt.plot(SwellingRatio_values)
        # plt.xlabel('Iteration')
        # plt.ylabel('SwellingRatio')
        # plt.title('SwellingRatio vs Iteration')

        # plt.tight_layout()
        # plt.show()
        
        if len(unique_solutions) == 1:
            print('Single solution found.')
            return SwellingRatio_values, S_sc_exp_values
        elif len(unique_solutions) > 1:
            print('Multiple solutions found.')
            return SwellingRatio_values, S_sc_exp_values
        elif len(unique_solutions) == 0:
            print('No solution found.')
            return None, None
        
    def solve_parameters_plots_NEW(self):
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
            
            #* Use EXP values of rho_f
            # rho_f = _df[mask]["ρ[g/cc]"].values[0]
            
            #* Use Span Wagner EoS values of rho_f
            rho_f = _df[mask]["ρSW[g/cc]"].values[0]
            
            #* Use SAFT EoS values of rho_f
            # rho_f = _df[mask]["ρSAFT[g/cc]"].values[0]
            
            V_b_exp = data.Vbasket  # [cc]
            V_t0_exp = data.Vs  # [cc]
            m_ptot_exp = data.ms    # [g]
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        # Get omega_cr
        omega_cr = self.omega_cr
        
        # Get rho_pol_cr
        rho_p_c = self.rho_pol_cr  # [g/m^3]
        
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
        
        def get_SinglePhaseDensity(SwellingRatio):
            # Calculate S_sc
            S_sc = (m_net_exp + rho_f * (V_b_exp + V_t0 * (1 + SwellingRatio))) / m_ptot_exp
            
            # Calculate S_am
            S_am = S_sc / (1-omega_cr)
            
            # Calculate x
            omega_p = 1/(S_am+1)     # [g/g]
            omega_s = 1 - omega_p   # [g/g]
            x_s = (omega_s/self.MW_sol) / (omega_s/self.MW_sol + omega_p/self.MW_pol)   #[mol/mol]
            x_p = 1 - x_s   #[mol/mol]           
            x = hstack([x_s, x_p])   # [mol/mol]
            
            # Calculate mixture Density
            rho = self.SinglePhaseDensity(x, self.T, self.P, 'L')   # [mol/m^3]
            
            return rho
        
        #* Evaluate SwellingRatio between 0 and 0.5
        SwellingRatio_values = linspace(0, 0.1, 10)
        
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
        plt.title(r'$\rho_{tot}$(T,0,0):$\rho_{tot}(T,P,S_{sc})$ vs SwellingRatio')
        
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
        plt.title('S_sc vs SwellingRatio')
        
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
        plt.title(r'$\frac{\rho_{tot} \left( T,0,0 \right) }{\rho_{tot} \left( T,P,S_{sc} \right)}(1+S_{sc})$ vs SwellingRatio')
        
        # Plot SinglePhaseDenisty vs SwellingRatio
        # rho_values = []
        # for SwellingRatio in SwellingRatio_values:
        #     try:
        #         rho_values.append(get_SinglePhaseDensity(SwellingRatio))
        #     except:
        #         rho_values.append(None)
        # print('rho_values:', rho_values)
        
        # plt.subplot(5, 1, 5)
        # plt.plot(SwellingRatio_values, rho_values)
        # plt.xlabel('SwellingRatio')
        # plt.ylabel(r'$\rho_{tot}$')
        # plt.title(r'$\rho_{tot}$ vs SwellingRatio')
        
        plt.tight_layout()
        plt.show()
    
    def solve_parameters_plots_NEW2(self):
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
            
            #* Use EXP values of rho_f
            # rho_f = _df[mask]["ρ[g/cc]"].values[0]
            
            #* Use Span Wagner EoS values of rho_f
            rho_f = _df[mask]["ρSW[g/cc]"].values[0]
            
            #* Use SAFT EoS values of rho_f
            # rho_f = _df[mask]["ρSAFT[g/cc]"].values[0]
            
            V_b_exp = data.Vbasket  # [cc]
            V_t0_exp = data.Vs  # [cc]
            m_ptot_exp = data.ms    # [g]
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        # Get omega_cr
        omega_cr = self.omega_cr
        
        # Get rho_pol_cr
        rho_p_c = self.rho_pol_cr  # [g/m^3]
        
        # Calculate V_t0_exp based on calculated rho_tot(T,0,0)
        rhoT00 = self.rho_tot(self.T, 1, 0)*1e-6  # [g/cm^3]
        V_t0_model = m_ptot_exp/rhoT00  # [cm^3]
        
        #* Choose V_t0 values
        # V_t0 = V_t0_exp  # [cm^3], exp value
        V_t0 = V_t0_model  # [cm^3], calculated value
        
        def equations(S_sc):            
            # Calculate rho_tot(T,P,S)
            rhoT00 = self.rho_tot(self.T, 1, 0)*1e-6  # [g/cm^3]
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
        
        #* Evaluate SwellingRatio between 0 and 0.5
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
    
    def solve_parameters_fsolve_NEW_integration(self): # numerical solving
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
            rho_f_exp = _df[mask]["ρ[g/cc]"].values[0]
            V_b_exp = data.Vbasket
            V_t0_exp = data.Vs
            m_ptot_exp = data.ms
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        ## TEST
        print(f'rho_f_exp: {rho_f_exp} g/cm^3')
        ## /TEST
        
        # Get omega_cr
        omega_cr = self.omega_cr
        
        # Get rho_pol_cr
        rho_p_c = self.rho_pol_cr  # [g/m^3]
        
        def rho_tot(T, P, S_sc):
            # Calculate S_am_exp
            S_am = S_sc / (1-omega_cr)
            
            # Calculte rho_tot
            V_p_am = self.get_V_p_am(T, P, S_am)  # [m3/g]
            rho_p_am = 1/(V_p_am)  # [g/m^3]
            rho_p_tol = 1/((1-omega_cr)/rho_p_am + omega_cr/rho_p_c)  # [g/m^3]
            rho_tot = (1 + S_sc) * rho_p_tol  # [g/m^3]
            
            return rho_tot
        
        def S_sc_exp(SwR):
            # Calculate S_sc_exp
            S_sc_exp = (m_net_exp + rho_f_exp * (V_b_exp + V_t0_exp * (1 + SwR))) / m_ptot_exp
            return S_sc_exp
            
        def equation(SwellingRatio):
            # Calculate S_sc_exp
            S_sc = S_sc_exp(SwellingRatio)
            
            # Calculate rho_tot(T,0,0)
            rhoT00 = rho_tot(self.T, 1, 0)  # [g/m^3], use P = 1 Pa #TODO: Check if P = 1 Pa is correct
            
            # Calculate rho_tot(T,P,S)
            rhoTPS = rho_tot(self.T, self.P, S_sc)  # [g/m^3]
            
            # Define the equations in terms of these variables and instance variables
            LHS = SwellingRatio
            RHS = (rhoT00 / rhoTPS) * (1 + S_sc) - 1
            
            return LHS-RHS    # Solve for LHS-RHS = 0
        
        # *Initial guesses
        # initial_guesses = [i for i in linspace(0, 0.5, 11)] #* Full range
        initial_guesses = [i for i in linspace(0, 0.3, 3)]  # *TEST
        
        print("Initial guesses:", *initial_guesses)
        
        solutions = []

        for x0 in initial_guesses:
            print(f"Initial guess: {x0}")
            try:
                solution = fsolve(equation, x0)
                print('equation(solution[0]):', equation(solution[0]))
                if isclose([0], [equation(solution[0])], atol=1e-2): # isclose() requires input arrays
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
                    if not any([isclose(solution, unique_solution, rtol=1e-2) for unique_solution in unique_solutions]):
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

        # # Plot S_sc_exp vs. iterations
        # plt.figure(figsize=(5, 4))  # (width, height)
        # plt.subplot(2, 1, 1)
        # plt.plot(S_sc_exp_values)
        # plt.xlabel('Iteration')
        # plt.ylabel('S_sc_exp')
        # plt.title('S_sc_exp vs Iteration')

        # # Plot SwellingRatio
        # plt.subplot(2, 1, 2)
        # plt.plot(SwellingRatio_values)
        # plt.xlabel('Iteration')
        # plt.ylabel('SwellingRatio')
        # plt.title('SwellingRatio vs Iteration')

        # plt.tight_layout()
        # plt.show()
        
        if len(unique_solutions) == 1:
            print('Single solution found.')
            return SwellingRatio_values, S_sc_exp_values
        elif len(unique_solutions) > 1:
            print('Multiple solutions found.')
            return SwellingRatio_values, S_sc_exp_values
        elif len(unique_solutions) == 0:
            print('No solution found.')
            return None, None
        
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
    
    def get_muad_sol_ext(self, T:float, P:float):        
        eos_sol = self.eos_sol
        psat, vlsat, vvsat = eos_sol.psat(T)
        
        # Saturation Pressure (Pa), saturated liquid volume (m3/mol), saturated vapor volume (m3/mol).
        if P >= psat:  # L phase
            rho_1 = eos_sol.density(T, P, "L")  # [mol/m^3]
        else:  # V phase
            rho_1 = eos_sol.density(T, P, "V")  # [mol/m^3]

        muad_sol_ext = eos_sol.muad(rho_1, T) / (8.314 * T)  # [adim]
        return muad_sol_ext

    def V_sol(self, x: array, T: float, P:float, eps:float = 1.0e-10):   #* default: eps=1.0e-5
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

    def V_pol(self, x: array, T: float, P:float, eps:float = 1.0e-10):   #* default: eps=1.0e-5
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
        # T = self.T
        # P = self.P        
        
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
    
    def get_V_p_am(self, T:  float, P: float, S_a: float):
        def func_Vs(S):
            Vs, _Vp = self.Vs_Vp_pmv1(T, P, S)   # [m3/g]
            return (Vs)  #[m^3/g]
        # def func_Vp(S):
        #     Vs, Vp = self.Vs_Vp_pmv1(T, P, S)   # [m3/g]
        #     return (Vp)  #[m^3/g]
        # Assuming Vp does not change with S_a
        _, Vp = self.Vs_Vp_pmv2()   # [m3/g]
        
        # Use integrate.quad()
        # int_result_Vs, error_Vs = integrate.quad(func_Vs, 0, S_a)
        # int_result_Vp, error_Vp = integrate.quad(func_Vp, 0, 1)
        
        
        # Use trapz()
        N = 5
        x1 = linspace(0, S_a, N+1)
        y1 = [func_Vs(s) for s in x1]
        y1 = array(y1)
        
        # x2 = linspace(0, 1, N+1)
        # y2 = [func_Vp(s) for s in x2]
        # y2 = array(y2)
        
        int_result_Vs = trapz(y1, x1)
        # int_result_Vp = trapz(y2, x2)
        
        return int_result_Vs + Vp
        

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
            
            # Use converged values based on convergence between exp data and SAFT values of V_t0
            omega_c = df[df["T (°C)"] == (T-273)]["rho_cr_HDPE_converged (g/cm3)"].values[0]
            
        else:
            omega_c = 0
        # self.omega_cr = omega_cr
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
    
    def get_S_am_EOS(self, T: float, P: float):
        """Solve solubility in amorphous rubbery polymer at equilibrium.
        
        Args:
            T (float): Temperature [K].
            P (float): Pressure [Pa].
        
        """
        
        MW_sol = self.MW_sol
        MW_pol = self.MW_pol
        
        # Chemical potential of external gas (EQ)
        muad_s_ext = self.muad_sol_ext
        eos_mix = self.eos_mix
                
        # sol-pol mixture (EQ)
        def func(_x_1):
            _x = hstack([_x_1, 1 - _x_1])  # [mol/mol-mix]
            _rhol = self.SinglePhaseDensity(_x, T, P)   
            
            _rho_i = _x * _rhol  # [mol/m^3-mix]
            _muad_m = eos_mix.muad(_rho_i, T)  # dimensionless [mu/RT]
            
            # Eq. 1
            _muad_s_m = _muad_m[0]      # dimensionless chemical potential of solute in sol-pol mixture
            return [_muad_s_m - muad_s_ext]
        
        x0_sol_single = self.options.get("x0_sol", None)
        x0_sol_range = self.options.get("x0_sol_range", linspace(1e-6, 0.99999999, 50).tolist())
        
        if x0_sol_single is None:
            x0_list = x0_sol_range
        else:
            x0_list = [x0_sol_single]
        
        print("")
        print(f"T = {T} K, P = {P} Pa")
        for i, x0 in enumerate(x0_list):            
            # print(f"i = {i}") 
            try:
                solution = fsolve(func, x0=float(x0_list[i]))
                residue = func(_x_1=solution)
                x_sol = solution[0]  # [mol-sol/mol-mix]
                residue_float = [float(i) for i in residue]                
                
                if isclose(residue_float, [0.0]).all() == True:    
                    # print(x_sol)
                    if x_sol > 1:
                        raise Exception("x > 1")
                    elif x_sol < 0:
                        raise Exception("x < 0")                        
                    else:
                        omega_sol = (x_sol * MW_sol) / (x_sol * MW_sol + (1 - x_sol) * MW_pol)
                        solubility_gg = omega_sol / (1-omega_sol)
                        return solubility_gg
            except Exception as e:
                print(f"Step {i+1}/{len(x0_list)} (x0={x0_list[i]}): ", e)
            
            
        print("Failed to find solution within max iteractions number")
        print("")
        return None
    
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
    
    def get_rho_am(self):
        """Function to get density of amorphous domain of polymer.
        Unit = [g/m^3]
        
        """
        
        # Use exp
        S_a = self.S_am_exp  # [g/g]

        #* METHOD 1: Evaluate Vs and Vp at each composition, most robust
        if self.pmv_method == "1":
            V_s, V_p =  self.Vs_Vp_pmv1(self.T, self.P, S_a)            
        
        #* METHOD 2: Assuming Vs and Vp same as specific volume at __infinitely dilution__ 
        if self.pmv_method == "2":
            V_s, V_p =  self.Vs_Vp_pmv2()
        
        #* METHOD 3: Assuming Vs and Vp at __infinitely dilution__, unchanged at atmospheric pressure, least robust
        if self.pmv_method == "3":
            V_s, V_p =  self.Vs_Vp_pmv3()

        # Eq. 8         
        rho_am = 1 / (S_a*V_s + V_p) # [g/m^3]
        
        return rho_am

    def get_omega_am_EOS(self):
        S_a = self.S_am_EOS
        
        # Eq. 3
        omega_p = 1/(S_a+1)     # [g/g]
        
        omega_s = 1 - omega_p   # [g/g]
        return hstack([omega_s, omega_p])
    
    def get_x_am_EOS(self):
        w = self.omega_am_EOS
        omega_s = w[0]
        omega_p = w[1]
        x_s = (omega_s/self.MW_sol) / (omega_s/self.MW_sol + omega_p/self.MW_pol)   #[mol/mol]
        return hstack([x_s, 1-x_s])
    
    def get_S_sc_EOS(self):
        """Function to calculate overall solubility of sol in pol, adjsuted for omega_cr.

        """
        
        # Get omega_cr
        omega_c = self.omega_cr
        
        # Get solubility in amorphous domain
        S_a = self.S_am_EOS    #[g/g]
        
        # Get solubility in semi-crystalline polymer
        # Eq. 4
        S_sc = S_a * (1-omega_c)  # [g/g]
        
        return S_sc
        
    def get_rho_mix(self):
        """Function to get total density of mixture.
        Unit = [g/m^3]

        """    
        omega_c = self.omega_cr
        rho_pol_am = self.rho_am  # [g/m^3]
        rho_pol_cr = self.rho_pol_cr  # [g/m^3]
        
        # Use S_sc from exp
        S = self.S_sc_exp # [g_sol/g_pol]
        
        # Eq. 14
        rho_mix = (1 + S) / ((1-omega_c)/rho_pol_am + omega_c/rho_pol_cr) # [g/m^3]
        
        return rho_mix

    def get_rho_mix_0(self):
        """Function to calculate dry polymer density. This is equal to overall dry polymer  density.

        """
        omega_c = self.omega_cr  # [g/g]
        rho_pol_cr = self.rho_pol_cr  # [g/m^3]
        rho_pol_am = self.SinglePhaseDensity(array([0., 1.]), self.T, P=1)*self.MW_pol # [g/m^3] #!check P=1
        
        # Eq. 16
        rho_pol = 1 / ((1-omega_c)/rho_pol_am + omega_c/rho_pol_cr) # [g/m63]
        
        return rho_pol

    def get_SwellingRatio(self):
        """Function to get swelling ratio.
        
        """
        # Eq. 21
        SR = (self.rho_mix_0 / self.rho_mix * (1 + self.S_sc_EOS)) - 1
        
        return SR

    def get_S_sc_exp_uncorr(self):
        """Function to get solubility in semi-crystalline polymer, uncorrected for swelling.
        
        """
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
            rho_f_exp = _df[mask]["ρ[g/cc]"].values[0]
            V_b_exp = data.Vbasket
            V_t0_exp = data.Vs
            m_ptot_exp = data.ms
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        # Eq. 17
        S_sc_exp_uncorr = (m_net_exp + rho_f_exp * (V_b_exp + V_t0_exp)) / m_ptot_exp # [g/g]
        
        return S_sc_exp_uncorr        
    
    def get_S_sc_exp(self):
        """Function to get solubility in semi-crystalline polymer, corrected for swelling.
        
        """
        SwR = self.SwellingRatio
        
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
            rho_f_exp = _df[mask]["ρ[g/cc]"].values[0]
            V_b_exp = data.Vbasket
            V_t0_exp = data.Vs
            m_ptot_exp = data.ms
            
        except Exception as e:
            print("Error: ")
            print(e)
            return None
        
        # Eq. 18
        S_sc_exp_corr = (m_net_exp + rho_f_exp * (V_b_exp + V_t0_exp * (1 + SwR))) / m_ptot_exp # [g/g]
        
        return S_sc_exp_corr

    def get_S_am_exp(self):
        
        # Get omega_cr
        omega_c = self.omega_cr
        
        # Get S_sc from exp
        S_sc = self.S_sc_exp # [g_sol/g_pol]
        
        # Get S_am from exp
        # Eq. 4
        S_a = S_sc / (1-omega_c)  # [g/g]
        
        return S_a
    
    def get_omega_am_exp(self):
        S_a = self.S_am_exp
        
        # Eq. 3
        omega_p = 1/(S_a+1)     # [g/g]
        
        omega_s = 1 - omega_p   # [g/g]
        return hstack([omega_s, omega_p])
        
    def get_x_am_exp(self):
        w = self.omega_am_exp
        omega_s = w[0]
        omega_p = w[1]
        
        x_s = (omega_s/self.MW_sol) / (omega_s/self.MW_sol + omega_p/self.MW_pol)   #[mol/mol]
        
        return hstack([x_s, 1-x_s])
        
class SolPolExpData():
    def __init__(self, sol: str, pol: str, data_file="data_CO2-HDPE.xlsx"):
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
    data = SolPolExpData(base_obj.sol, base_obj.pol)
    
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
                    obj = DetailedSolPol(base_obj, T, _p, pmv_method=k)                
                else:
                    x0_list = update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                    obj = DetailedSolPol(base_obj, T, _p, pmv_method=k, x0_sol_range = x0_list,)
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
                    obj = DetailedSolPol(base_obj, T, _p, pmv_method=k)                
                else:
                    x0_list = update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                    obj = DetailedSolPol(base_obj, T, _p, pmv_method=k, x0_sol_range = x0_list,)
                objects.append(obj)
                S_SAFT_pmv_pexp.append(obj.S_sc_EOS)
            _df[f'S_SAFT_pmv{k}[g/g]'] = S_SAFT_pmv_pexp            
            
            # Calculates sorption from SAFT predictions only
            P_SAFT[T] = linspace(_df["P[bar]"].values[0],_df["P[bar]"].values[-1], 30) * 1e5   # [Pa]
            objects = []
            S_SAFT_pmv[k] = []
            for j, _p in enumerate(P_SAFT[T]):
                if j == 0:
                    obj = DetailedSolPol(base_obj, T, _p, pmv_method=k)                
                else:
                    x0_list = update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                    obj = DetailedSolPol(base_obj, T, _p, pmv_method=k, x0_sol_range = x0_list,)
                objects.append(obj)
                S_SAFT_pmv[k].append(obj.S_sc_EOS)
        
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
        ax.plot(df[mask1]["P[bar]"], df[mask1]['S_NoCorrection[g/g]'],color=custom_colours[0], linestyle="None", marker="o",label=f"{T-273}°C exp - not corrected")
        ax.plot(df[mask1]["P[bar]"], df[mask1]['S_Corrected_pmv1[g/g]'],color=custom_colours[1], linestyle="None", marker="x",label=f"{T-273}°C exp - corrected with pmv1")
        ax.plot(df[mask1]["P[bar]"], df[mask1]['S_Corrected_pmv2[g/g]'],color=custom_colours[2], linestyle="None", marker="x",label=f"{T-273}°C exp - corrected with pmv2")
        ax.plot(df[mask1]["P[bar]"], df[mask1]['S_Corrected_pmv3[g/g]'],color=custom_colours[3], linestyle="None", marker="x",label=f"{T-273}°C exp - corrected with pmv3")
        mask2 = (df_SAFT['T [K]'] == T)
        ax.plot(df_SAFT[mask2]["P [Pa]"]*1e-5, df_SAFT[mask2]["S_sc_pmv1 [g/g]"],color=custom_colours[1], linestyle="solid",marker="None",label=f"{T-273}°C SAFT pmv1")
        ax.plot(df_SAFT[mask2]["P [Pa]"]*1e-5, df_SAFT[mask2]["S_sc_pmv2 [g/g]"],color=custom_colours[2], linestyle="solid",marker="None",label=f"{T-273}°C SAFT pmv2")
        ax.plot(df_SAFT[mask2]["P [Pa]"]*1e-5, df_SAFT[mask2]["S_sc_pmv3 [g/g]"],color=custom_colours[3], linestyle="solid",marker="None",label=f"{T-273}°C SAFT pmv3")
    ax.set_xlabel("P [bar]")
    ax.set_ylabel(r"$S_{sc}$ [$g_{sol}$/$g_{pol \: sc}$]")
    ax.set_ylim(top=0.15)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    if display_fig == True:
        plt.show()
    if save_fig == True:
        save_fig_path = f"{data.path}/Anals/IsothermPmv_{time_str}.png"
        plt.savefig(save_fig_path, dpi=1200)
        print(f"Plot successfully exported to {save_fig_path}.")

def plot_VsVp_pmv(base_obj, T: float, display_fig:bool=True, save_fig:bool=False):
    """Function to plot partial molar volume isotherms at single temperature.

    Args:
        spm (class object): class object representing the sol-pol mixture.
        T (float): temperature [K].  
    """
    now = datetime.now()  # current time
    time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM    
    
    data = SolPolExpData(base_obj.sol, base_obj.pol)    
    _df=data.get_sorption_data(T)
    P_list= linspace(1, _df["P[bar]"].values[-1]*1e5, 50) #[Pa]
    objects = []
    
    # pmv 1
    Vs_pmv1 = []
    Vp_pmv1 = []
    for j, _p in enumerate(P_list):
        if j == 0:
            obj = DetailedSolPol(base_obj, T, _p,)
        else:
            x0_list = update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
            obj = DetailedSolPol(base_obj, T, _p, x0_sol_range = x0_list,)
        Vspmv1, Vppmv1 = obj.Vs_Vp_pmv1()
        
        objects.append(obj)
        Vs_pmv1.append(Vspmv1)
        Vp_pmv1.append(Vppmv1)
    
    # pmv 2
    Vs_pmv2, Vp_pmv2 = zip(*[DetailedSolPol(base_obj, T, _p).Vs_Vp_pmv2() for _p in P_list])
    
    # pmv 3
    Vs_pmv3, Vp_pmv3 = zip(*[DetailedSolPol(base_obj, T, _p).Vs_Vp_pmv3() for _p in P_list])
    
    # Vs and Vp 
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(P_list*1e-5, Vs_pmv1, color=custom_colours[1], linestyle="solid", marker="None")
    ax1.plot(P_list*1e-5, Vs_pmv2, color=custom_colours[2], linestyle="solid", marker="None")
    ax1.plot(P_list*1e-5, Vs_pmv3, color=custom_colours[3], linestyle="solid", marker="None")
    ax1.plot(P_list*1e-5, Vp_pmv1, color=custom_colours[1], linestyle="dashed", marker="None")
    ax1.plot(P_list*1e-5, Vp_pmv2, color=custom_colours[2], linestyle="dashed", marker="None")
    ax1.plot(P_list*1e-5, Vp_pmv3, color=custom_colours[3], linestyle="dashed", marker="None")
    ax1.set_xlabel("P [bar]")
    ax1.set_ylabel(r"$\bar{V}$ [$m^{3}/g$]")
    ax1.tick_params(direction="in")
    # Legends
    legend_colours = [Line2D([0], [0], linestyle="None", marker=".", color=custom_colours[i+1],
                             label=f"{T-273}°C pmv{pmv}") for i, pmv in enumerate(["1", "2", "3"])]
    legend_linestyles = [Line2D([0], [0], linestyle=line, marker="None", color="black",
                             label=f"{label}") for line, label in zip(["solid", "dashed"],["sol","pol"])]
    legend = legend_colours + legend_linestyles
    ax1.legend(handles=legend)
    
    # Vs plot
    # fig = plt.figure()
    # ax2 = fig.add_subplot()
    # ax2.plot(P_list*1e-5, Vs_pmv1, color=custom_colours[1], linestyle="solid", marker="None", label=f"{T-273}°C pmv1")
    # ax2.plot(P_list*1e-5, Vs_pmv2, color=custom_colours[2], linestyle="solid", marker="None", label=f"{T-273}°C pmv2")
    # ax2.plot(P_list*1e-5, Vs_pmv3, color=custom_colours[3], linestyle="solid", marker="None", label=f"{T-273}°C pmv3")
    # ax2.set_xlabel("P [bar]")
    # ax2.set_ylabel(r"$V_{sol}$ [$m^{3}/g$]")
    # ax2.tick_params(direction="in")
    # ax2.legend().set_visible(True)
    
    # # Vp plot
    # fig = plt.figure()
    # ax3 = fig.add_subplot()
    # ax3.plot(P_list*1e-5, Vp_pmv1, color=custom_colours[1], linestyle="solid", marker="None", label=f"{T-273}°C pmv1")
    # ax3.plot(P_list*1e-5, Vp_pmv2, color=custom_colours[2], linestyle="solid", marker="None", label=f"{T-273}°C pmv2")
    # ax3.plot(P_list*1e-5, Vp_pmv3, color=custom_colours[3], linestyle="solid", marker="None", label=f"{T-273}°C pmv3")
    # ax3.set_xlabel("P [bar]")
    # ax3.set_ylabel(r"$V_{pol}$ [$m^{3}/g$]")
    # ax3.tick_params(direction="in")
    # ax3.legend().set_visible(True)
    
    if display_fig == True:
        plt.show()
    if save_fig == True:
        save_fig_path = f"{data.path}/Anals/PlotVsVp_{time_str}.png"
        plt.savefig(save_fig_path, dpi=1200)
        print(f"Plot successfully exported to {save_fig_path}.")

def plot_isotherm_epskl_EOS(base_obj, T_list: list, P_list: list, eps_list: list, export_data:bool=False, display_fig:bool=True, save_fig:bool=False):
    """Function to plot solubility isotherms for multiple eps_kl values at specified T, P.

    Args:
        spm (class object): class object representing the sol-pol mixture.
        T_list (list): temperature liít [K].
        P_list (list): pressure list [Pa].
        eps_list (list): epsilon_kl list.
        export_data (bool, optional): export data. Defaults to False.
    """
    data = SolPolExpData(base_obj.sol, base_obj.pol)
    now = datetime.now()  # current time
    time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM    
    
    solubility = []
    
    df = pd.DataFrame(columns=['T [K]', 'eps_kl', 'P [Pa]', 'S_sc [g/g]'])
    for T in T_list:
        for eps in eps_list:
            base_obj.modify_kl(eps)
            
            objects = []
            solubility= []            
            for j, _p in enumerate(P_list):
                if j == 0:
                    obj = DetailedSolPol(base_obj, T, _p,)
                else:
                    x0_list = update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                    obj = DetailedSolPol(base_obj, T, _p, x0_sol_range = x0_list,)
                
                try:                    
                    _S = obj.S_sc_EOS
                except:
                    _S = None
                
                objects.append(obj)
                solubility.append(_S)
            _df=pd.DataFrame({'T [K]': [T for _p in P_list],
                              'eps_kl': [eps for _p in P_list],
                              'P [Pa]':P_list,
                              'S_sc [g/g]':solubility})
            df = pd.concat([df,_df], ignore_index=True)
    print(df)
    
    if export_data == True:
        now = datetime.now()  # current time
        time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM
        path = os.path.dirname(__file__)
        export_path = f"{path}/PlotIsothermEps_{time_str}.xlsx"
        with pd.ExcelWriter(export_path) as writer:
            df.to_excel(writer, index=False)
        print("Data successfully exported to: ", export_path)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, T in enumerate(T_list):
        for j, eps in enumerate(eps_list):
            mask = (df['T [K]'] == T) & (df['eps_kl'] == eps)
            ax.plot(df[mask]['P [Pa]']*1e-5, df[mask]['S_sc [g/g]'], color=custom_colours[i+1], linestyle="solid", marker=custom_markers[j], label=f"{T-273}°C eps={eps}")
    ax.set_xlabel("P [bar]")
    ax.set_ylabel(r"$S_{sc}$ [$g_{sol}$/$g_{pol \: sc}$]")
    ax.set_ylim(top=1.70)
    # ax.set_yscale('log')
    ax.tick_params(direction="in")
    
    # Legends
    legend_markers = [Line2D([0], [0], linestyle="None", marker=custom_markers[i], color="black",
                             label=f"eps = {eps}") for i, eps in enumerate(eps_list)]
    legend_colours = [Line2D([0], [0], marker="None", color=custom_colours[i+1],
                             label=f"T = {T-273}°C") for i, T in enumerate(T_list)]
    legend = legend_colours + legend_markers
    ax.legend(handles=legend)
    if display_fig == True:
        plt.show()
    if save_fig == True:
        save_fig_path = f"{data.path}/Anals/IsothermEpsEOSvExp_{time_str}.png"
        plt.savefig(save_fig_path, dpi=1200)
        print(f"Plot successfully exported to {save_fig_path}.")    

def plot_isotherm_epskl_EOSvExp(base_obj, T_list: list, eps_list:list, export_data:bool=False, display_fig:bool=True, save_fig:bool=False):
    """Functio to plot solubility isotherms for chosen eps_kl values at different 

    Args:
        base_obj (_type_): _description_
        T_list (list): _description_
        eps_list (list): _description_
        export_data (bool, optional): _description_. Defaults to False.
    """
    
    data = SolPolExpData(base_obj.sol, base_obj.pol)
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
                    obj = DetailedSolPol(base_obj, T, _p,)
                else:
                    x0_list = update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                    obj = DetailedSolPol(base_obj, T, _p, x0_sol_range = x0_list,)
                
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
        
    fig = plt.figure(figsize=[8.0, 3.5])
    ax1 = fig.add_subplot(121)  # SAFT only
    ax2 = fig.add_subplot(122)  # corrected exp
    for i, T in enumerate(T_list):
        for j, eps in enumerate(eps_list):
            
            mask = (df['T [K]'] == T) & (df['eps_kl'] == eps)
            ax1.plot(df[mask]['P [Pa]']*1e-5, df[mask]['S_sc_SAFT [g/g]'], color=custom_colours[i+1], linestyle="solid", marker=custom_markers[j], label=f"{T-273}°C eps={eps}")
            ax2.plot(df[mask]['P [Pa]']*1e-5, df[mask]['S_sc_exp_corrected [g/g]'], color=custom_colours[i+1], linestyle="solid", marker=custom_markers[j], label=f"{T-273}°C eps={eps}")
    for ax in ax1, ax2:
        ax.set_xlabel("P [bar]")
        ax.set_ylabel(r"$S_{sc}$ [$g_{sol}$/$g_{pol \: sc}$]")
        # ax.set_yscale('log')
        ax.tick_params(direction="in")
    # ax1.set_title("SAFT prediction")
    # ax2.set_title("Experimental with swelling correction")
    # ax1.set_ylim(top=1.)
    # ax2.set_ylim(top=2.)
    
    # Legends setting
    legend_markers = [Line2D([0], [0], linestyle="None", marker=custom_markers[i], color="black",
                             label=f"eps = {eps}") for i, eps in enumerate(eps_list)]
    legend_colours = [Line2D([0], [0], marker="None", color=custom_colours[i+1],
                             label=f"T = {T-273}°C") for i, T in enumerate(T_list)]
    legend = legend_colours + legend_markers
    ax2.legend(handles=legend)
    
    if save_fig == True:
        save_fig_path = f"{data.path}/Anals/IsothermEpsEOSvExp_{time_str}.png"
        plt.savefig(save_fig_path, dpi=1200)
        print(f"Plot successfully exported to {save_fig_path}.")
    if display_fig == True:
        plt.show()    
    
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
    
    #* SW total
    # p = linspace(1,10e5,5)
    # SwR_25C = [mix.SwellingRatio(25+273, _p) for _p in p]
    # SwR_35C = [mix.SwellingRatio(35+273, _p) for _p in p]
    # SwR_50C = [mix.SwellingRatio(50+273, _p) for _p in p]
    # print("Swelling Ratio at 25C = ", *SwR_25C)
    # print("Swelling Ratio at 35C = ", *SwR_35C)
    # print("Swelling Ratio at 50C = ", *SwR_50C)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(p*1e-5,SwR_25C,label="25C")
    # ax.plot(p*1e-5,SwR_35C,label="35C")
    # ax.plot(p*1e-5,SwR_50C,label="50C")
    # ax.set_xlabel("p [bar]")
    # ax.set_ylabel("Swelling Ratio (cm3/cm3)")
    # ax.legend().set_visible(True)
    # plt.show()
    
    #* SW total with different pmv method
    # T=25+273    # [K]
    # p = linspace(1,10e5,5)
    # mix.pmv_method = "1"    
    # SwR_pmv1 = [mix.SwellingRatio(T, _p) for _p in p]
    # mix.pmv_method = "2"
    # SwR_pmv2 = [mix.SwellingRatio(T, _p) for _p in p]
    # mix.pmv_method = "3"
    # SwR_pmv3 = [mix.SwellingRatio(T, _p) for _p in p]
    # print(f"Swelling Ratio pmv1 at {T-273}C = ", *SwR_pmv1)
    # print(f"Swelling Ratio pmv2 at {T-273}C = ", *SwR_pmv2)
    # print(f"Swelling Ratio pmv3 at {T-273}C = ", *SwR_pmv3)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(p*1e-5,SwR_pmv1,label="pmv1")
    # ax.plot(p*1e-5,SwR_pmv2,label="pmv2")
    # ax.plot(p*1e-5,SwR_pmv3,label="pmv3")
    # ax.set_xlabel("p [bar]")
    # ax.set_ylabel("Swelling Ratio (cm3/cm3)")
    # ax.legend().set_visible(True)
    # plt.show()
    
    #* SW am
    # p = linspace(1,200e5,10)    # [Pa]
    # SwRa_25C = [mix.SwellingRatio_am(25+273, _p) for _p in p]
    # SwRa_35C = [mix.SwellingRatio_am(35+273, _p) for _p in p]
    # SwRa_50C = [mix.SwellingRatio_am(50+273, _p) for _p in p]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(p*1e-5,SwRa_25C,label="25C")
    # ax.plot(p*1e-5,SwRa_35C,label="35C")
    # ax.plot(p*1e-5,SwRa_50C,label="50C")
    # ax.set_xlabel("p [bar]")
    # ax.set_ylabel("V_am(p) / V_am(0) [cm3/cm3]")
    # ax.legend().set_visible(True)
    # plt.show()  
    
    #* Vm
    # p = linspace(1, 10e5, 5)  #[Pa]  
    # Vm_25C = [1/mix.rho_mix(25+273,_p) for _p in p]
    # Vm_35C = [1/mix.rho_mix(35+273,_p) for _p in p]
    # Vm_50C = [1/mix.rho_mix(50+273,_p) for _p in p]
    # print("Vm at 25C = ", *Vm_25C)
    # print("Vm at 35C = ", *Vm_35C)
    # print("Vm at 50C = ", *Vm_50C)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(p*1e-5,Vm_25C,label="25C")
    # ax.plot(p*1e-5,Vm_35C,label="35C")
    # ax.plot(p*1e-5,Vm_50C,label="50C")
    # ax.set_xlabel("p [bar]")
    # ax.set_ylabel("V_m [m3/g]")
    # ax.legend().set_visible(True)
    # plt.show()  
    
    # * Vam
    # p = linspace(1, 100e5, 5)  #[Pa]  
    # Va_25C = [1/mix.rho_am(25+273,_p) for _p in p]
    # Va_35C = [1/mix.rho_am(35+273,_p) for _p in p]
    # Va_50C = [1/mix.rho_am(50+273,_p) for _p in p]
    # print("Va at 25C = ", *Va_25C)
    # print("Va at 35C = ", *Va_35C)
    # print("Va at 50C = ", *Va_50C)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(p*1e-5,Va_25C,label="25C")
    # ax.plot(p*1e-5,Va_35C,label="35C")
    # ax.plot(p*1e-5,Va_50C,label="50C")
    # ax.set_xlabel("p [bar]")
    # ax.set_ylabel("V_am [m3/g]")
    # ax.legend().set_visible(True)
    # plt.show()
    
    
    # * Ssc
    # p = linspace(1, 10e5, 5)  #[Pa]  
    # Ssc_25C = [mix.S_sc(25+273,_p) for _p in p]
    # Ssc_35C = [mix.S_sc(35+273,_p) for _p in p]
    # Ssc_50C = [mix.S_sc(50+273,_p) for _p in p]
    
    # print("Ssc at 25C = ", *Ssc_25C)
    # print("Ssc at 35C = ", *Ssc_35C)
    # print("Ssc at 50C = ", *Ssc_50C)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(p*1e-5,Ssc_25C,label="25C")
    # ax.plot(p*1e-5,Ssc_35C,label="35C")
    # ax.plot(p*1e-5,Ssc_50C,label="50C")
    # ax.set_xlabel("p [bar]")
    # ax.set_ylabel("S_sc [g/g]")
    # ax.legend().set_visible(True)
    # plt.show()
    
    #* Solve SwR and S_sc_exp
    