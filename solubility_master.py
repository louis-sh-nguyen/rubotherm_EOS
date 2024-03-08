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

from sgtpy_NETGP import component, mixture, saftgammamie
# from sgtpy import component, mixture, saftgammamie

import math
import os
import time
from datetime import datetime
import addcopyfighandler
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from colour import Color
from numpy import *
import pandas as pd
from scipy.optimize import curve_fit, fsolve
from scipy.integrate import quad


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

custom_colours = [
    "black",
    "green",
    "blue",
    "red",
    "purple",
    "orange",
    "brown",
    "plum",
    "indigo",
    "olive",
    "grey",
]

class SolPolMixture:
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
        self.pmv_method = "1"
        
        MW_monomer = {
            "HDPE": 28,            
            "PE": 28,
        }
        MW_pol = MW_monomer[pol] * n_monomer
        self.MW_pol = MW_pol
        
        MWsolute = {"CO2": 44}  # [g/mol]
        MW_sol = MWsolute[sol]
        self.MW_sol = MW_sol
        
        if pol != None:
            if pol in {"PE","HDPE"} :
                pol_obj = component(GC={"CH2": 2*n_monomer})
            # Create Create SAFT-g Mie EOS object of pure polymer
            pol_obj.saftgammamie()
            eos_pol = saftgammamie(pol_obj, compute_critical=False)
            self.eos_pol = eos_pol
        
        if sol != None:
            if sol == "CO2":
                sol_obj = component(GC={"CO2": 1})
            # Create Create SAFT-g Mie EOS object of pure solute
            sol_obj.saftgammamie()
            eos_sol = saftgammamie(sol_obj)
            self.eos_sol = eos_sol
        
        if sol != None and pol != None:
            # mixture object
            mix_obj = mixture(sol_obj, pol_obj)
            # Create SAFT-g Mie EOS object of Mixture
            mix_obj.saftgammamie()
            eos_mix = saftgammamie(mix_obj, compute_critical=False)
            self.eos_mix = eos_mix
    
    # @property
    # def eos_sol(self):
    #     if self.sol != None:
    #         if self.sol == "CO2":
    #             sol_obj = component(GC={"CO2": 1})
    #         # Create Create SAFT-g Mie EOS object of pure solute
    #         sol_obj.saftgammamie()
    #         eos_sol = saftgammamie(sol_obj)            
    #     return eos_sol
    # @property
    # def eos_pol(self):
    #     if self.pol != None:
    #         if self.pol in {"PE","HDPE"} :
    #             pol_obj = component(GC={"CH2": 2*self.n_monomer})
    #         # Create Create SAFT-g Mie EOS object of pure polymer
    #         pol_obj.saftgammamie()
    #         eos_pol = saftgammamie(pol_obj, compute_critical=False)            
    #     return self.eos_pol    
    # @property
    # def eos_mix(self):
    #     if self.sol != None and self.pol != None:
    #         # mixture object
    #         mix_obj = mixture(self.sol_obj, self.pol_obj)
    #         # Create SAFT-g Mie EOS object of Mixture
    #         mix_obj.saftgammamie()
    #         eos_mix = saftgammamie(mix_obj, compute_critical=False)
    #         self.eos_mix = eos_mix
    #     return self._eos_mix
    
    def muad_sol_ext(self, T: float, P: float):
        eos_sol = self.eos_sol
        psat, vlsat, vvsat = eos_sol.psat(T)
        # Saturation Pressure (Pa), saturated liquid volume (m3/mol), saturated vapor volume (m3/mol).
        if P >= psat:  # L phase
            rho_1 = eos_sol.density(T, P, "L")  # [mol/m^3]
        else:  # V phase
            rho_1 = eos_sol.density(T, P, "V")  # [mol/m^3]

        muad_sol_ext = eos_sol.muad(rho_1, T) / (8.314 * T)  # dimensionless
        return muad_sol_ext

    def omega_cr(self, T: float):
        """Function to get omega_cr as a function of T.
        Ref: [1]: Polymer, 59, 2015, 270-277.

        Args:            
            T (float): Temperature [K].
            P (float): Pressure [Pa].
        """
        try:
            # Read exp file
            path = os.path.dirname(__file__)
            filepath = path + "/data_example.xlsx"
            file = pd.ExcelFile(filepath, engine="openpyxl")
            df = pd.read_excel(file, sheet_name="omega_cr")
        
        except Exception as e:
            print("Error: ")
            print(e)
            return None        
        
        if self.pol == "HDPE":
            # print(df.dtypes)  # Chek data types
            omega_cr = df[df["T (°C)"] == (T-273)]["omega_cr_HDPE"].values[0]
            
        else:
            omega_cr = 0
        # self.omega_cr = omega_cr
        return omega_cr

    def rho_pol_cr(self, T: float):
        """Function to get density of crystalline domain of polymer.

        Args:
            T (float): Temperature [K].
            P (float): Pressure [Pa].
        """
        try:
            # Read exp file
            path = os.path.dirname(__file__)
            filepath = path + "/data_example.xlsx"
            file = pd.ExcelFile(filepath, engine="openpyxl")
            df = pd.read_excel(file, sheet_name="rho_cr")
        
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
    
    def SinglePhaseDensity(self, x , T: float, P: float):
        """Function to calculate single phase density of mixture without specifying phase.
        Unit = mol/m^3

        Args:
            x (array_like): molar fraction arrray.
            T (float): Temperature [K].
            P (float): Pressure [Pa].
        """
        rhoL = self.eos_mix.density(x, T, P, "L")   # [mol/m^3]
        rhoV = self.eos_mix.density(x, T, P, "V")   # [mol/m^3]
        
        if isclose(P, self.eos_mix.pressure(x,rhoL,T), P*0.01):
            rho = rhoL
            print("Use rhoL")
        elif isclose(P, self.eos_mix.pressure(x,rhoV,T), P*0.01):
            rho = rhoV
            print("Use rhoV")
        
        return rho

    def V_sol(self, x, T: float, P:float, eps:float = 1.0e-5): 
        """Function to calculate partial volume of solute in mixture. 
        Assuming constant in condensed phase.
        Unit = [m3/g].

        Args:
            T (float): Temperature [K].
            P (float): Pressure [Pa].
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

    def V_pol(self, x, T: float, P:float, eps:float = 1.0e-5): 
        """Function to calculate partial volume of polymer in mixture. 
        Assuming constant in condensed phase.
        Unit = [m3/g].

        Args:
            T (float): Temperature [K].
            P (float): Pressure [Pa].
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
    
    def rho_am(self, T: float, P: float):
        """Function to get density of amorphous domain of polymer.
        Unit = [g/m^3]

        Args:
            T (float): Temperature [K].
            P (float): Pressure [Pa].
            pmv_method (str): method to calculate partial molar volume. Default: "1".
        """
        
        S_a = self.S_am(T, P)  # [g/g]        
        omega_p = 1/(S_a+1)     # [g/g]
        omega_s = 1 - omega_p   # [g/g]
        x_s = (omega_s/self.MW_sol) / (omega_s/self.MW_sol + omega_p/self.MW_pol)   #[mol/mol]
        x_p = 1 - x_s   #[mol/mol]           
        x = hstack([x_s, x_p])   # [mol/mol]        
        print("x = ", *x)
        
        #* METHOD 1: Evaluate Vs and Vp at each composition, most robust
        if self.pmv_method == "1":
            V_s = self.V_sol(x, T, P)  # [m^3/g]        
            V_p = self.V_pol(x, T, P)  # [m^3/g]
        
        #* METHOD 2: Assuming Vs and Vp same as specific volume at __infinitely dilution__ 
        if self.pmv_method == "2":
            V_s = self.V_sol(hstack([0., 1.]), T, P)  # [m^3/g]        
            V_p = self.V_pol(hstack([0., 1.]), T, P)  # [m^3/g]
        
        #* METHOD 3: Assuming Vs and Vp at __infinitely dilution__, unchanged at atmospheric pressure, least robust
        if self.pmv_method == "3":
            V_s = self.V_sol(hstack([0., 1.]), T, 1e5)  # [m^3/g]        
            V_p = self.V_pol(hstack([0., 1.]), T, 1e5)  # [m^3/g]
        
        print(f"Vs at {T}K and {P} Pa = ", V_s)
        print(f"Vp at {T}K and {P} Pa = ", V_p)
        print(f"Sa at {T}K and {P} Pa = ", S_a)
        rho_am = 1 / (S_a*V_s + V_p) # [g/m^3]
        return rho_am

    def S_am(self, T: float, P: float):
        """Solve solubility in amorphous rubbery polymer at equilibrium.

        Args:
            T (float): temperature [K].
            P (float): pressure [Pa].
        """
        MW_sol = self.MW_sol
        MW_pol = self.MW_pol
        # chemical potential of external gas (EQ)
        muad_s_ext = self.muad_sol_ext(T, P)
        
        # sol-pol mixture (EQ)
        def func(_x_1):
            _x = hstack([_x_1, 1 - _x_1])  # [mol/mol-mix]
            _rhol = self.eos_mix.density(_x, T, P, "L")  # *
            _rho_i = _x * _rhol  # [mol/m^3-mix]
            _muad_m = self.eos_mix.muad(_rho_i, T)  # dimensionless [mu/RT]
            _muad_s_m = _muad_m[0]      # dimensionless chemical potential of solute in sol-pol mixture
            return [_muad_s_m - muad_s_ext]
        
        x0 = linspace(9.90e-1, 9.99e-1, 10) 
        i = 0
        while i < (len(x0)):
            try:
                solution = fsolve(func, x0=float(x0[i]))
                residue = func(_x_1=solution)
                residue_float = [float(i) for i in residue]
                if isclose(residue_float, [0.0]).all() == True:
                    x_sol = solution[0]  # [mol-sol/mol-mix]
                    omega_sol = (x_sol * MW_sol) / (x_sol * MW_sol + (1 - x_sol) * MW_pol)
                    solubility_gg = omega_sol / (1-omega_sol)
                    return solubility_gg
                else:
                    i += 1
            except Exception as e:
                print(e)
                i += 1
                print("\tMoving to step i = ", i)
        
        if i >= (len(x0)):
            print("\nNo solution found for T = %g K\tP = %g" % (T - 273, P))
            print("")                

    def S_sc(self, T: float, P: float):
        """Function to calculate overall solubility of sol in pol, adjsuted for omega_cr.

        Args:
            T (float): temperature [K].
            P (float): pressure [Pa].
        """
        # Get omega_cr
        omega_c = self.omega_cr(T)
        # print("omega_cr = ", omega_cr)
        
        # Get solubility in amorphous domain
        S_a = self.S_am(T, P)    #[g/g]
        # print("S_am = ", S_am, " g/g_am")
        
        # Get solubility in semi-crystalline polymer
        S_sc = S_a * (1-omega_c)  # [g/g]
        # print("S = ", S_sc, " g/g_sc")
        return S_sc

    def rho_mix(self, T:float, P: float):
        """Function to get total density of mixture.

        Args:
            spm (class object): class object representing the sol-pol mixture.
            T (float): temperature [K].
            P (float): pressure [Pa].
        """    
        omega_c = self.omega_cr(T)     
        print(f"omega_cr  {T}K = ", omega_c)
        rho_pol_am = self.rho_am(T, P)  # [g/m^3]
        print(f"rho_pol_am at {T}K and {P} Pa = ", rho_pol_am*1e-6, "g/cm^3")
        rho_pol_cr = self.rho_pol_cr(T)  # [g/m^3]
        print(f"rho_pol_cr {T}K and {P} Pa = ", rho_pol_cr*1e-6, "g/cm^3")
        S = self.S_sc(T, P) # [g_sol/g_pol]
        print("S = ", S)
        
        rho_mix = (1 + S) / ((1-omega_c)/rho_pol_am + omega_c/rho_pol_cr) # [g/m^3]
        return rho_mix

    def SwellingRatio(self, T: float, P: float):
        """Function to get swelling ratio.

        Args:
            T (float): temperature [K].
            P (float): pressure [Pa].
        """
        
        SR = (self.rho_mix(T,1) / self.rho_mix(T,P) * (1 + self.S_sc(T,P))) - 1
        return SR


    def SwellingRatio_am(self, T: float, P: float):
        """Function to get swelling ratio of amorphous domain in polymer.

        Args:
            T (float): temperature [K].
            P (float): pressure [Pa].
        """
        
        SR_a = (self.rho_am(T, 1) / self.rho_am(T,P) * (1 + self.S_am(T,P))) - 1
        return SR_a
    
class SolPolExpData():
    def __init__(self, sol: str, pol: str):
        self.sol = sol
        self.pol = pol
        # mixture = SolPolMixture(sol, pol)
        try:
            # Read exp file
            path = os.path.dirname(__file__)
            filepath = path + "/data_example.xlsx"
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

def plot_isotherm_EOSvExp(spm, T_list:list[float], export_data:bool = False):
    """Function to plot sorption of EOS and experimental data (not corrected for swelling).

    Args:
        spm (class object): class object representing the sol-pol mixture.
        T (float): temperature [K].
        P (float): pressure [Pa].
        export_data (bool): export data.
    """
    data = SolPolExpData(spm.sol, spm.pol)
    
    df={}
    
    P_SAFT={}
    S_SAFT={}
    for i, T in enumerate(T_list):
        _df=data.get_sorption_data(T)
        # Sorption without swelling correction
        S_exp_woSW = (_df["MP1*[g]"]-data.m_met_filled+_df["ρ[g/cc]"]*(data.Vs+data.Vbasket)) / data.ms
        _df['S_exp_woSW[g/g]']=S_exp_woSW
        # Calculate swelling ratio from SAFT
        SwR_SAFT = [spm.SwellingRatio(T,_p) for _p in _df["P[bar]"]*1e5]
        _df['SwR_SAFT[cc/cc]'] = SwR_SAFT
        # Sorption with swelling correction
        S_exp_SW = (_df["MP1*[g]"]-data.m_met_filled+_df["ρ[g/cc]"]*(data.Vs*(1+_df['SwR_SAFT[cc/cc]'])+data.Vbasket)) / data.ms
        _df['S_exp_SW[g/g]'] = S_exp_SW
        
        P_SAFT[T] = linspace(_df["P[bar]"].values[0],_df["P[bar]"].values[-1], 30) * 1e5   # [Pa]
        S_SAFT[T] = [spm.S_sc(T,_p) for _p in P_SAFT[T]]
        _S_SAFT = [spm.S_sc(T,_p) for _p in _df["P[bar]"]*1e5]
        _df['S_SAFT[g/g]'] = _S_SAFT
        df[T] = _df
        print("\n \n")
        print("T = ", T)
        print(df[T])
        # print("P [bar] = ",*_df["P[bar]"], sep="\n")
        # print("S_exp_woSW = ", *S_exp_woSW, sep="\n")
        # print("S_exp_SW = ", *S_exp_SW, sep="\n")        
        # print("SwR_SAFT = ", *SwR_SAFT, sep="\n")
        # print("S_SAFT = ", *_S_SAFT, sep="\n")        
        print("")
    
    if export_data == True:
        now = datetime.now()  # current time
        time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM
        export_path = f"{data.path}/exportedData_{time_str}.xlsx"
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


def plot_isotherm_pmv(spm, T_list:list[float], export_data:bool = False):
    """Function to plot sorption of EOS and experimental data to compare different partial molar volume approaches.

    Args:
        spm (class object): class object representing the sol-pol mixture.
        T (float): temperature [K].
        P (float): pressure [Pa].
        export_data (bool): export data.
    """
    data = SolPolExpData(spm.sol, spm.pol)
    
    df={}    
    P_SAFT={}

    S_SAFT_pmv1={}
    S_SAFT_pmv2={}
    S_SAFT_pmv3={}
    for i, T in enumerate(T_list):
        _df=data.get_sorption_data(T)
        # Sorption without swelling correction
        S_exp_woSW = (_df["MP1*[g]"]-data.m_met_filled+_df["ρ[g/cc]"]*(data.Vs+data.Vbasket)) / data.ms
        _df['S_exp_woSW[g/g]']=S_exp_woSW
        # Calculate swelling ratio from SAFT using pmv 1
        spm.pmv_method = "1"
        SwR_SAFT_pmv1 = [spm.SwellingRatio(T,_p) for _p in _df["P[bar]"]*1e5]
        _df['SwR_SAFT_pmv1[cc/cc]'] = SwR_SAFT_pmv1        
        S_exp_SW_pmv1 = (_df["MP1*[g]"]-data.m_met_filled+_df["ρ[g/cc]"]*(data.Vs*(1+_df['SwR_SAFT_pmv1[cc/cc]'])+data.Vbasket)) / data.ms
        _df['S_exp_SW_pmv1[g/g]'] = S_exp_SW_pmv1
        P_SAFT[T] = linspace(_df["P[bar]"].values[0],_df["P[bar]"].values[-1], 30) * 1e5   # [Pa]
        S_SAFT_pmv1[T] = [spm.S_sc(T,_p) for _p in P_SAFT[T]]
        _S_SAFT_pmv1 = [spm.S_sc(T,_p) for _p in _df["P[bar]"]*1e5]
        _df['S_SAFT_pmv1[g/g]'] = _S_SAFT_pmv1
        
        # Calculate swelling ratio from SAFT using pmv 2
        spm.pmv_method = "2"
        SwR_SAFT_pmv2 = [spm.SwellingRatio(T,_p) for _p in _df["P[bar]"]*1e5]
        _df['SwR_SAFT_pmv2[cc/cc]'] = SwR_SAFT_pmv2
        S_exp_SW_pmv2 = (_df["MP1*[g]"]-data.m_met_filled+_df["ρ[g/cc]"]*(data.Vs*(1+_df['SwR_SAFT_pmv2[cc/cc]'])+data.Vbasket)) / data.ms
        _df['S_exp_SW_pmv2[g/g]'] = S_exp_SW_pmv2                
        S_SAFT_pmv2[T] = [spm.S_sc(T,_p) for _p in P_SAFT[T]]
        _S_SAFT_pmv2 = [spm.S_sc(T,_p) for _p in _df["P[bar]"]*1e5]
        _df['S_SAFT_pmv2[g/g]'] = _S_SAFT_pmv2
        
        # Calculate swelling ratio from SAFT using pmv 3
        spm.pmv_method = "3"
        SwR_SAFT_pmv3 = [spm.SwellingRatio(T,_p) for _p in _df["P[bar]"]*1e5]
        _df['SwR_SAFT_pmv3[cc/cc]'] = SwR_SAFT_pmv3
        S_exp_SW_pmv3 = (_df["MP1*[g]"]-data.m_met_filled+_df["ρ[g/cc]"]*(data.Vs*(1+_df['SwR_SAFT_pmv3[cc/cc]'])+data.Vbasket)) / data.ms
        _df['S_exp_SW_pmv3[g/g]'] = S_exp_SW_pmv3                
        S_SAFT_pmv3[T] = [spm.S_sc(T,_p) for _p in P_SAFT[T]]
        _S_SAFT_pmv3 = [spm.S_sc(T,_p) for _p in _df["P[bar]"]*1e5]
        _df['S_SAFT_pmv3[g/g]'] = _S_SAFT_pmv3
        df[T] = _df
        print("\n \n")
        print("T = ", T)
        print(df[T])    
        print("")
    
    if export_data == True:
        now = datetime.now()  # current time
        time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM
        export_path = f"{data.path}/exportedData_{time_str}.xlsx"
        with pd.ExcelWriter(export_path) as writer:
            for T in T_list:
                df[T].to_excel(writer, sheet_name=f"{T-273}C", index=False)
        print("Data successfully exported to: ", export_path)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, T in enumerate(T_list):
        ax.plot(df[T]["P[bar]"],df[T]['S_exp_woSW[g/g]'],color=custom_colours[0], linestyle="None", marker="o",label=f"{T-273}°C exp - swelling uncorrected")
        ax.plot(df[T]["P[bar]"],df[T]['S_exp_SW_pmv1[g/g]'],color=custom_colours[1], linestyle="None", marker="x",label=f"{T-273}°C exp - swelling pmv1")
        ax.plot(df[T]["P[bar]"],df[T]['S_exp_SW_pmv2[g/g]'],color=custom_colours[2], linestyle="None", marker="x",label=f"{T-273}°C exp - swelling pmv2")
        ax.plot(df[T]["P[bar]"],df[T]['S_exp_SW_pmv3[g/g]'],color=custom_colours[3], linestyle="None", marker="x",label=f"{T-273}°C exp - swelling pmv3")
        ax.plot(P_SAFT[T]*1e-5,S_SAFT_pmv1[T],color=custom_colours[1], linestyle="solid",marker="None",label=f"{T-273}°C SAFT pmv1")
        ax.plot(P_SAFT[T]*1e-5,S_SAFT_pmv2[T],color=custom_colours[2], linestyle="solid",marker="None",label=f"{T-273}°C SAFT pmv2")
        ax.plot(P_SAFT[T]*1e-5,S_SAFT_pmv3[T],color=custom_colours[3], linestyle="solid",marker="None",label=f"{T-273}°C SAFT pmv3")
    ax.set_xlabel("P [bar]")
    ax.set_ylabel("S [g/g]")
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    plt.show()


    
if __name__ == "__main__":
    
    # data = SolPolExpData("CO2","HDPE")
    # print(data.V_met_filled)
    
    mixture = SolPolMixture("CO2","HDPE")
    # mixture.pmv_method="3"
    # rho = mixture.SinglePhaseDensity(hstack([1.0, 0]), 25+273, 1e5)
    # print("rho =", rho)
    # print("SR = ",mixture.SwellingRatio(35+273,10e5))
    # plot_isotherm_EOSvExp(mixture, [50+273,], export_data=False)
    # plot_isotherm_EOSvExp(mixture,[25+273, 35+273, 50+273],export_data="True")
    plot_isotherm_pmv(mixture, [50+273,], export_data=True)
    # S = get_S_sc(mixture, 35+273, 1e5)
    # rho_m = mixture.rho_mix(35+273, 1e5)
    # print(f"rho_mix = {rho_m*1e-6} g/cm^3")
    # print("Vs = ", mixture.V_sol(hstack([0, 1]), 50+273, 1e5), " m^3/g")
    # print("Vs = ", mixture.V_sol(hstack([0, 1]), 50+273, 1e6), " m^3/g")
    # print("Vp = ", mixture.V_pol(hstack([0, 1]), 50+273,1e5), " m^3/g")
    # print("Vp = ", mixture.V_pol(hstack([0, 1]), 50+273,1e6), " m^3/g")
    # print("rho_am = ", mixture.rho_am(35+273,1e5), " g/m^3")
    # print("SwR = ", mixture.SwellingRatio(25+273,1e5), " cm^3/cm^3")
    
    #* SW total
    # p = linspace(1,10e5,5)
    # SwR_25C = [mixture.SwellingRatio(25+273, _p) for _p in p]
    # SwR_35C = [mixture.SwellingRatio(35+273, _p) for _p in p]
    # SwR_50C = [mixture.SwellingRatio(50+273, _p) for _p in p]
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
    
    #* SW am
    # p = linspace(1,200e5,10)    # [Pa]
    # SwRa_25C = [mixture.SwellingRatio_am(25+273, _p) for _p in p]
    # SwRa_35C = [mixture.SwellingRatio_am(35+273, _p) for _p in p]
    # SwRa_50C = [mixture.SwellingRatio_am(50+273, _p) for _p in p]
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
    # Vm_25C = [1/mixture.rho_mix(25+273,_p) for _p in p]
    # Vm_35C = [1/mixture.rho_mix(35+273,_p) for _p in p]
    # Vm_50C = [1/mixture.rho_mix(50+273,_p) for _p in p]
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
    # Va_25C = [1/mixture.rho_am(25+273,_p) for _p in p]
    # Va_35C = [1/mixture.rho_am(35+273,_p) for _p in p]
    # Va_50C = [1/mixture.rho_am(50+273,_p) for _p in p]
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
    p = linspace(1, 10e5, 5)  #[Pa]  
    Ssc_25C = [mixture.S_sc(25+273,_p) for _p in p]
    Ssc_35C = [mixture.S_sc(35+273,_p) for _p in p]
    Ssc_50C = [mixture.S_sc(50+273,_p) for _p in p]
    
    print("Ssc at 25C = ", *Ssc_25C)
    print("Ssc at 35C = ", *Ssc_35C)
    print("Ssc at 50C = ", *Ssc_50C)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(p*1e-5,Ssc_25C,label="25C")
    ax.plot(p*1e-5,Ssc_35C,label="35C")
    ax.plot(p*1e-5,Ssc_50C,label="50C")
    ax.set_xlabel("p [bar]")
    ax.set_ylabel("S_sc [g/g]")
    ax.legend().set_visible(True)
    plt.show()    