"""
Louis Nguyen
Department of Cheimcal Engineering, Imperial College London
sn621@ic.ac.uk
15 Feb 2024
"""
# Turn off numba warning
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from colour import Color
from numpy import *
import pandas as pd
from scipy.optimize import curve_fit, fsolve, minimize_scalar



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
custom_markers = ["o", "x", "^", "*", "s", "D"]

def update_x0_sol_list(previous_x0_sol:float, no_step:int=10, x0_sol_default_range=(0.9, 0.9999)):
    if (previous_x0_sol is None) or (previous_x0_sol < 0.) or (previous_x0_sol > 1.):
        new_x0_list =  linspace(x0_sol_default_range[0], x0_sol_default_range[1], no_step).tolist()
    else:
        new_x0_list = linspace(previous_x0_sol, x0_sol_default_range[1], no_step).tolist()
    return new_x0_list

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
        
        self._S_am = None
        self._S_sc = None
        self._x_am = None
        self._omega_am = None
        self._rho_pol = None
        self._omega_cr = None
        self._muad_sol_ext = None
        self._rho_pol_cr = None
        self._rho_am = None
        self._rho_mix = None
        self._rho_mix_0 = None
        self._SwellingRatio = None
        
    @property
    def S_am(self):
        if self._S_am is None:
            self._S_am = self.get_S_am(self.T, self.P)
        return self._S_am
    @property
    def S_sc(self):
        if self._S_sc is None:
            self._S_sc = self.get_S_sc()
        return self._S_sc
    @property
    def x_am(self):
        if self._x_am is None:
            self._x_am = self.get_x_am()
        return self._x_am
    @property
    def omega_am(self):
        if self._omega_am is None:
            self._omega_am = self.get_omega_am()
        return self._omega_am
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
    @property
    def SwellingRatio(self):
        if self._SwellingRatio is None:
            self._SwellingRatio = self.get_SwellingRatio()
        return self._SwellingRatio
    
    ### INDEPENDENT functions
    def SinglePhaseDensity(self, x:array , T: float, P: float):
        """Function to calculate single phase density of mixture without specifying phase.
        Unit = [mol/m^3]

        Args:
            x (array_like): molar fraction.
            T (float): Temperature [K].
            P (float): Pressure [Pa].
        """
        rhoL = self.eos_mix.density(x, T, P, "L")   # [mol/m^3]
        rhoV = self.eos_mix.density(x, T, P, "V")   # [mol/m^3]
        
        if isclose(P, self.eos_mix.pressure(x,rhoL,T), P*0.01):
            rho = rhoL
            
        elif isclose(P, self.eos_mix.pressure(x,rhoV,T), P*0.01):
            rho = rhoV
        
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

    def V_sol(self, x: array, T: float, P:float, eps:float = 1.0e-5): 
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
        
        if n_sol > eps:
            dV_dns = (n_up/self.SinglePhaseDensity(x_up, T, P)-n_lo/self.SinglePhaseDensity(x_lo, T, P))/(2*eps)    # [m^3/mol]
        else: #case where n_sol = 0
            dV_dns = (n_up/self.SinglePhaseDensity(x_up, T, P)-n/self.SinglePhaseDensity(x, T, P))/(eps) # [m^3/mol]
        
        return dV_dns/self.MW_sol   # [m^3/g]

    def V_pol(self, x: array, T: float, P:float, eps:float = 1.0e-5): 
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
        
        if n_pol > eps:
            dV_dnp = (n_up/self.SinglePhaseDensity(x_up, T, P)-n_lo/self.SinglePhaseDensity(x_lo, T, P))/(2*eps)    # [m^3/mol]
        else: #case where n_sol = 0
            dV_dnp = (n_up/self.SinglePhaseDensity(x_up, T, P)-n/self.SinglePhaseDensity(x, T, P))/(eps) # [m^3/mol]
        
        return dV_dnp / self.MW_pol # [m^3/g]
    
    def Vs_Vp_pmv1(self):
        """Function to calculate partial volume of solute in mixture, using pmv method 1.
        pmv method 1: using solubility composition.
        Unit = [m3/g]

        """
        T = self.T
        P = self.P
        S_a = self.S_am  # [g/g]    
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
        """Function to get omega_cr as a function of T. Reads data from excel sheet called /data_example.xlsx.
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
            omega_c = df[df["T (°C)"] == (T-273)]["omega_cr_HDPE"].values[0]
            
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
    
    def get_S_am(self, T: float, P: float):
        """Solve solubility in amorphous rubbery polymer at equilibrium.
        
        Args:
            T (float): Temperature [K].
            P (float): Pressure [Pa].
        
        """
        MW_sol = self.MW_sol
        MW_pol = self.MW_pol
        # chemical potential of external gas (EQ)
        muad_s_ext = self.muad_sol_ext
        eos_mix = self.eos_mix
                
        # sol-pol mixture (EQ)
        def func(_x_1):
            _x = hstack([_x_1, 1 - _x_1])  # [mol/mol-mix]
            _rhol = eos_mix.density(_x, T, P, "L")
            _rho_i = _x * _rhol  # [mol/m^3-mix]
            _muad_m = eos_mix.muad(_rho_i, T)  # dimensionless [mu/RT]
            _muad_s_m = _muad_m[0]      # dimensionless chemical potential of solute in sol-pol mixture
            return [_muad_s_m - muad_s_ext]
        
        x0_sol_single = self.options.get("x0_sol", None)
        x0_sol_range = self.options.get("x0_sol_range", linspace(0.90, 0.999, 10).tolist())
        # auto_iterate_x0 = self.options.get("auto_iterate_x0", False)
        
        if x0_sol_single is None:
            x0_list = x0_sol_range
        else:
            x0_list = [x0_sol_single]
        
        print("\n")
        print(f"T = {T} K, P = {P} Pa")
        for i, x0 in enumerate(x0_list):            
            try:
                solution = fsolve(func, x0=float(x0_list[i]))
                residue = func(_x_1=solution)
                residue_float = [float(i) for i in residue]
                if isclose(residue_float, [0.0]).all() == True:
                    x_sol = solution[0]  # [mol-sol/mol-mix]
                    omega_sol = (x_sol * MW_sol) / (x_sol * MW_sol + (1 - x_sol) * MW_pol)
                    solubility_gg = omega_sol / (1-omega_sol)
                    return solubility_gg
            except Exception as e:
                print(f"Step {i+1}/{len(x0_list)+1} (x0={x0_list[i]}): ", e)
            
            if i >= (len(x0_list)):
                print("Failed to find solution within max iteractions number")
                print("")
                return None
    
    ### DEPENDENT functions
    def get_rho_am(self):
        """Function to get density of amorphous domain of polymer.
        Unit = [g/m^3]
        
        """
        
        S_a = self.S_am  # [g/g]

        #* METHOD 1: Evaluate Vs and Vp at each composition, most robust
        if self.pmv_method == "1":
            V_s, V_p =  self.Vs_Vp_pmv1()            
        
        #* METHOD 2: Assuming Vs and Vp same as specific volume at __infinitely dilution__ 
        if self.pmv_method == "2":
            V_s, V_p =  self.Vs_Vp_pmv2()
        
        #* METHOD 3: Assuming Vs and Vp at __infinitely dilution__, unchanged at atmospheric pressure, least robust
        if self.pmv_method == "3":
            V_s, V_p =  self.Vs_Vp_pmv3()

        rho_am = 1 / (S_a*V_s + V_p) # [g/m^3]
        return rho_am

    def get_omega_am(self):
        S_a = self.S_am
        omega_p = 1/(S_a+1)     # [g/g]
        omega_s = 1 - omega_p   # [g/g]
        return hstack([omega_s, omega_p])
    
    def get_x_am(self):
        w = self.omega_am
        omega_s = w[0]
        omega_p = w[1]
        x_s = (omega_s/self.MW_sol) / (omega_s/self.MW_sol + omega_p/self.MW_pol)   #[mol/mol]
        return hstack([x_s, 1-x_s])
    
    def get_S_sc(self):
        """Function to calculate overall solubility of sol in pol, adjsuted for omega_cr.

        """
        # Get omega_cr
        omega_c = self.omega_cr
        # Get solubility in amorphous domain
        S_a = self.S_am    #[g/g]
        # Get solubility in semi-crystalline polymer
        S_sc = S_a * (1-omega_c)  # [g/g]
        return S_sc

    def get_rho_mix(self):
        """Function to get total density of mixture.
        Unit = [g/m^3]

        """    
        omega_c = self.omega_cr
        rho_pol_am = self.rho_am  # [g/m^3]
        rho_pol_cr = self.rho_pol_cr  # [g/m^3]
        S = self.S_sc # [g_sol/g_pol]
        rho_mix = (1 + S) / ((1-omega_c)/rho_pol_am + omega_c/rho_pol_cr) # [g/m^3]
        return rho_mix

    def get_rho_mix_0(self):
        """Function to calculate dry polymer density. This is equal to overall dry polymer  density.

        """
        omega_c = self.omega_cr  # [g/g]
        rho_pol_cr = self.rho_pol_cr  # [g/m^3]
        rho_pol_am = self.SinglePhaseDensity(array([0., 1.]), self.T, P=1)*self.MW_pol # [g/m^3]
        rho_pol = 1 / ((1-omega_c)/rho_pol_am + omega_c/rho_pol_cr) # [g/m63]
        return rho_pol

    def get_SwellingRatio(self):
        """Function to get swelling ratio.
        """
        
        SR = (self.rho_mix_0 / self.rho_mix * (1 + self.S_sc)) - 1
        return SR


class SolPolExpData():
    def __init__(self, sol: str, pol: str, data_file="data_example.xlsx"):
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

def fit_epskl(base_obj, T:float, x0: float = 200, epskl_bounds:tuple = (100, 500)):
    """Function to fit eps_kl to fit SAFT prediction to corected experimental data.

    Args:
        base_obj (_type_): _description_
        T (float): Temperature [K].
        x0 (float, optional): Starting value of eps_kl. Defaults to 200.
        epskl_bounds (tuple, optional): Upper and lower bounds for eps_kl. Defaults to (100, 500).

    Returns:
        _type_: _description_
    """
    
    data = SolPolExpData(base_obj.sol, base_obj.pol)
    _df=data.get_sorption_data(T)
    
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
                obj = DetailedSolPol(base_obj, T, _p,)
            else:
                x0_list = update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                obj = DetailedSolPol(base_obj, T, _p, x0_sol_range = x0_list,)
            
            # Solubility prediction from SAFT
            try:
                _S_SAFT = obj.S_sc
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

def plot_isotherm_EOSvExp(base_obj, T_list:list[float], export_data:bool = False):
    """Function to plot sorption of EOS and experimental data (not corrected for swelling).

    Args:
        spm (class object): class object representing the sol-pol mixture.
        T (float): temperature [K].
        P (float): pressure [Pa].
        export_data (bool): export data.
    """
    data = SolPolExpData(base_obj.sol, base_obj.pol)
    
    df={}    
    P_SAFT={}
    S_SAFT={}
    
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
                obj = DetailedSolPol(base_obj, T, _p,)
            else:
                x0_list = update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                obj = DetailedSolPol(base_obj, T, _p, x0_sol_range = x0_list,)
            objects.append(obj)
            SwR_SAFT.append(obj.SwellingRatio)

        _df['SwR_SAFT[cc/cc]'] = SwR_SAFT
        # Sorption with swelling correction
        S_exp_SW = (_df["MP1*[g]"]-data.m_met_filled+_df["ρ[g/cc]"]*(data.Vs*(1+_df['SwR_SAFT[cc/cc]'])+data.Vbasket)) / data.ms
        _df['S_exp_SW[g/g]'] = S_exp_SW
        
        # Calculate S_sc for continuous SAFT line        
        objects = []
        S_SAFT[T] = []
        P_SAFT[T] = linspace(_df["P[bar]"].values[0],_df["P[bar]"].values[-1], 30) * 1e5   # [Pa]
        for j, _p in enumerate(P_SAFT[T]):
            if j == 0:
                obj = DetailedSolPol(base_obj, T, _p,)
            else:
                x0_list = update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                obj = DetailedSolPol(base_obj, T, _p, x0_sol_range = x0_list,)
            objects.append(obj)
            S_SAFT[T].append(obj.S_sc)
            
        # Calculate S_sc at experimental pressure points
        objects = []
        S_SAFT_pexp = []
        for j, _p in enumerate(_df["P[bar]"]*1e5):
            if j == 0:
                obj = DetailedSolPol(base_obj, T, _p,)
            else:
                x0_list = update_x0_sol_list(previous_x0_sol=objects[j-1].x_am[0])
                obj = DetailedSolPol(base_obj, T, _p, x0_sol_range = x0_list,)
            objects.append(obj)
            S_SAFT_pexp.append(obj.S_sc)
        _df['S_SAFT[g/g]'] = S_SAFT_pexp
        
        df[T] = _df
        print("")
        print("T = ", T)
        print(df[T])
        print("")
    
    if export_data == True:
        now = datetime.now()  # current time
        time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM
        export_path = f"{data.path}/PlotIsothermEOSvExp_{time_str}.xlsx"
        with pd.ExcelWriter(export_path) as writer:
            for T in T_list:
                df[T].to_excel(writer, sheet_name=f"{T-273}C", index=False)
        print("Data successfully exported to: ", export_path)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, T in enumerate(T_list):
        ax.plot(df[T]["P[bar]"],df[T]['S_exp_woSW[g/g]'],color=custom_colours[i], linestyle="None", marker="o",label=f"{T-273}°C exp - swelling uncorrected")
        ax.plot(df[T]["P[bar]"],df[T]['S_exp_SW[g/g]'],color=custom_colours[i], linestyle="None", marker="x",label=f"{T-273}°C exp - swelling corrected")
        ax.plot(P_SAFT[T]*1e-5,S_SAFT[T],color=custom_colours[i], linestyle="solid",marker="None",label=f"{T-273}°C SAFT")
    ax.set_xlabel("P [bar]")
    ax.set_ylabel("S [g/g]")
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    plt.show()

def plot_isotherm_pmv(base_obj, T_list:list[float], export_data:bool = False):
    """Function to plot sorption of EOS and experimental data to compare different partial molar volume approaches.

    Args:
        spm (class object): class object representing the sol-pol mixture.
        T (float): temperature [K].
        P (float): pressure [Pa].
        export_data (bool): export data.
    """
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
                S_SAFT_pmv_pexp.append(obj.S_sc)
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
        ax.plot(df[mask1]["P[bar]"], df[mask1]['S_NoCorrection[g/g]'],color=custom_colours[0], linestyle="None", marker="o",label=f"{T-273}°C exp - not corrected")
        ax.plot(df[mask1]["P[bar]"], df[mask1]['S_Corrected_pmv1[g/g]'],color=custom_colours[1], linestyle="None", marker="x",label=f"{T-273}°C exp - corrected with pmv1")
        ax.plot(df[mask1]["P[bar]"], df[mask1]['S_Corrected_pmv2[g/g]'],color=custom_colours[2], linestyle="None", marker="x",label=f"{T-273}°C exp - corrected with pmv2")
        ax.plot(df[mask1]["P[bar]"], df[mask1]['S_Corrected_pmv3[g/g]'],color=custom_colours[3], linestyle="None", marker="x",label=f"{T-273}°C exp - corrected with pmv3")
        mask2 = (df_SAFT['T [K]'] == T)
        ax.plot(df_SAFT[mask2]["P [Pa]"]*1e-5, df_SAFT[mask2]["S_sc_pmv1 [g/g]"],color=custom_colours[1], linestyle="solid",marker="None",label=f"{T-273}°C SAFT pmv1")
        ax.plot(df_SAFT[mask2]["P [Pa]"]*1e-5, df_SAFT[mask2]["S_sc_pmv2 [g/g]"],color=custom_colours[2], linestyle="solid",marker="None",label=f"{T-273}°C SAFT pmv2")
        ax.plot(df_SAFT[mask2]["P [Pa]"]*1e-5, df_SAFT[mask2]["S_sc_pmv3 [g/g]"],color=custom_colours[3], linestyle="solid",marker="None",label=f"{T-273}°C SAFT pmv3")
    ax.set_xlabel("P [bar]")
    ax.set_ylabel("S [g/g]")
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    plt.show()

def plot_VsVp_pmv(base_obj, T: float):
    """Function to plot partial molar volume isotherms at single temperature.

    Args:
        spm (class object): class object representing the sol-pol mixture.
        T (float): temperature [K].  
    """
    data = SolPolExpData(base_obj.sol, base_obj.pol)    
    _df=data.get_sorption_data(T)
    p = linspace(1, _df["P[bar]"].values[-1]*1e5, 10) #[Pa]
    Vs_pmv1, Vp_pmv1 = zip(*[DetailedSolPol(base_obj, T, _p).Vs_Vp_pmv1() for _p in p])
    Vs_pmv2, Vp_pmv2 = zip(*[DetailedSolPol(base_obj, T, _p).Vs_Vp_pmv2() for _p in p])
    Vs_pmv3, Vp_pmv3 = zip(*[DetailedSolPol(base_obj, T, _p).Vs_Vp_pmv3() for _p in p])
        
    fig = plt.figure(figsize=[4.0, 7])
    ax1 = fig.add_subplot(211)
    ax1.plot(p*1e-5, Vs_pmv1, color=custom_colours[1], linestyle="solid", marker="None", label=f"{T-273}°C pmv1")
    ax1.plot(p*1e-5, Vs_pmv2, color=custom_colours[2], linestyle="solid", marker="None", label=f"{T-273}°C pmv2")
    ax1.plot(p*1e-5, Vs_pmv3, color=custom_colours[3], linestyle="solid", marker="None", label=f"{T-273}°C pmv3")
    ax1.set_xlabel("P [bar]")
    ax1.set_ylabel(r"$V_{sol}$ [$m^{3}/g$]")
    ax1.tick_params(direction="in")
    ax1.legend().set_visible(True)
    
    ax2 = fig.add_subplot(212)
    ax2.plot(p*1e-5, Vp_pmv1, color=custom_colours[1], linestyle="solid", marker="None", label=f"{T-273}°C pmv1")
    ax2.plot(p*1e-5, Vp_pmv2, color=custom_colours[2], linestyle="solid", marker="None", label=f"{T-273}°C pmv2")
    ax2.plot(p*1e-5, Vp_pmv3, color=custom_colours[3], linestyle="solid", marker="None", label=f"{T-273}°C pmv3")
    ax2.set_xlabel("P [bar]")
    ax2.set_ylabel(r"$V_{pol}$ [$m^{3}/g$]")
    ax2.tick_params(direction="in")
    ax2.legend().set_visible(True)
    plt.show()

def plot_isotherm_epskl_EOS(base_obj, T_list: list, P_list: list, eps_list: list, export_data:bool=False):
    """Function to plot solubility isotherms for multiple eps_kl values at specified T, P.

    Args:
        spm (class object): class object representing the sol-pol mixture.
        T_list (list): temperature liít [K].
        P_list (list): pressure list [Pa].
        eps_list (list): epsilon_kl list.
        export_data (bool, optional): export data. Defaults to False.
    """
    
    solubility = []
    _df = {}
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
                    _S = obj.S_sc
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
    ax.set_ylabel("S [g/g]")
    ax.set_yscale('log')
    ax.tick_params(direction="in")
    
    # Legends
    legend_markers = [Line2D([0], [0], linestyle="None", marker=custom_markers[i], color="black",
                             label=f"eps = {eps}") for i, eps in enumerate(eps_list)]
    legend_colours = [Line2D([0], [0], marker="None", color=custom_colours[i+1],
                             label=f"T = {T-273}°C") for i, T in enumerate(T_list)]
    legend = legend_colours + legend_markers
    ax.legend(handles=legend)
    
    plt.show()

def plot_isotherm_epskl_EOSvExp(base_obj, T_list: list, eps_list:list, export_data:bool=False):
    """Functio to plot solubility isotherms for chosen eps_kl values at different 

    Args:
        base_obj (_type_): _description_
        T_list (list): _description_
        eps_list (list): _description_
        export_data (bool, optional): _description_. Defaults to False.
    """
    
    data = SolPolExpData(base_obj.sol, base_obj.pol)
    
    solubility_SAFT = []
    _df = {}
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
                    _S_SAFT = obj.S_sc
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
        now = datetime.now()  # current time
        time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM
        path = os.path.dirname(__file__)
        export_path = f"{path}/PlotIsothermEpsEOSvExp_{time_str}.xlsx"
        with pd.ExcelWriter(export_path) as writer:
            df.to_excel(writer, index=False)
        print("Data successfully exported to: ", export_path)
        
    fig = plt.figure(figsize=[8.0, 3.5])
    ax1 = fig.add_subplot(121)  # SAFT only
    ax2 = fig.add_subplot(122)  # corrected exp
    for i, T in enumerate(T_list):
        for j, eps in enumerate(eps_list):
            #TODO try with pmv method 3
            mask = (df['T [K]'] == T) & (df['eps_kl'] == eps)
            ax1.plot(df[mask]['P [Pa]']*1e-5, df[mask]['S_sc_SAFT [g/g]'], color=custom_colours[i+1], linestyle="solid", marker=custom_markers[j], label=f"{T-273}°C eps={eps}")
            ax2.plot(df[mask]['P [Pa]']*1e-5, df[mask]['S_sc_exp_corrected [g/g]'], color=custom_colours[i+1], linestyle="solid", marker=custom_markers[j], label=f"{T-273}°C eps={eps}")
    for ax in ax1, ax2:
        ax.set_xlabel("P [bar]")
        ax.set_ylabel("S [g/g]")
        ax.set_yscale('log')
        ax.tick_params(direction="in")
    ax1.set_title("SAFT prediction")
    ax2.set_title("Experimental with swelling correction")
    # ax1.set_ylim(top=1.)
    # ax2.set_ylim(top=2.)
    
    # Legends setting
    legend_markers = [Line2D([0], [0], linestyle="None", marker=custom_markers[i], color="black",
                             label=f"eps = {eps}") for i, eps in enumerate(eps_list)]
    legend_colours = [Line2D([0], [0], marker="None", color=custom_colours[i+1],
                             label=f"T = {T-273}°C") for i, T in enumerate(T_list)]
    legend = legend_colours + legend_markers
    ax2.legend(handles=legend)
    plt.show()

if __name__ == "__main__":    
    mix = BaseSolPol("CO2","HDPE")
    # mix.pmv_method = "2"
    
    # plot_isotherm_EOSvExp(mix,[25+273, 35+273, 50+273],export_data="True")
    # plot_isotherm_pmv(mix, [50+273,], export_data=False)
    # plot_VsVp_pmv(mix, 50+273)    
    
    #* eps_kl_EOS
    # plot_isotherm_epskl_EOS(mix, T_list=[25+273,], P_list=linspace(1, 200e5, 5), 
    #                   eps_list=[276.45, 276.45*0.95, 276.45*1.05], 
    #                   export_data=False)
    
    #* eps_kl_EOSvsExp
    # plot_isotherm_epskl_EOSvExp(mix, T_list=[25+273, 35+273, 50+273],
    #                   eps_list=[276.45, 276.45*0.95, 276.45*1.05], 
    #                   export_data=False)
    
    #* fit_epskl
    fit_epskl(mix, T=25+273, x0=200, epskl_bounds=(50, 500))
    
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