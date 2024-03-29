"""
Script to verify amorhpous volume calculated from m_s*V_s+M_p*V_p and from SAFT predictions are equal.
26 March 2024
Louis Nguyen
sn621@ic.ac.uk
"""
import solubility_master as S
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

mix = S.BaseSolPol("CO2","HDPE")
T_list = [25+273, 35+273, 50+273]
# T_list = [25+273, ]
# P_list = linspace(1, 100, 10)     # [Pa]
P_list = linspace(1, 200e5, 50)     # [Pa]


V_fromPmv = {}
V_fromSAFT = {}
Vmolar_fromPmv = {}
Vmolar_fromSAFT = {}
x_sol = {}
x_pol = {}
V_sol = {}
V_pol = {}
V_term1 = {}
V_term2 = {}
V_pmv = {}
V_SAFT = {}
objects = {}

df = pd.DataFrame(columns=['T [K]', 'P [Pa]', 'x_sol [mol/mol]', 'x_pol [mol/mol]', 'V_sol [m3/g]', 'V_pol [m3/g]', 
                           'term1 [m3/mol]', 'term2 [m3/mol]','Vmolar_fromPmv [m3/mol]', 'Vmolar_fromSAFT [m3/mol]', 
                           'V_fromPmv [m3/g_pol_am]', 'V_fromSAFT [m3/g_pol_am]'])

for i, T in enumerate(T_list):
    V_fromPmv[T] = []
    V_fromSAFT[T] = []
    Vmolar_fromPmv[T] = []
    Vmolar_fromSAFT[T] = []
    x_sol[T] = []
    x_pol[T] = []
    V_sol[T] = []
    V_pol[T] = []
    V_term1[T] = []
    V_term2[T] = []
    objects[T] = []
    
    for j, P in enumerate(P_list):
        if j == 0:
            obj = S.DetailedSolPol(mix, T, P,)
        else:
            x0_list = S.update_x0_sol_list(previous_x0_sol=objects[T][j-1].x_am[0])
            obj = S.DetailedSolPol(mix, T, P, x0_sol_range = x0_list,)
        
        # get V_s and V_p from 
        V_s, V_p = obj.Vs_Vp_pmv1()     # [m3/g]
        
        # Vmix from constituents V_sol and V_pol
        term1 = obj.x_am[0]*V_s*obj.MW_sol
        term2 = obj.x_am[1]*V_p*obj.MW_pol
        V1_molar =  (term1 + term2)   # [m3/mol]
        V1 =  (term1 + term2) /(obj.x_am[1] * obj.MW_pol)   # [m3/g_pol_am]
        
        # Vmix from SAFT calculation
        V2_molar = 1/obj.SinglePhaseDensity(obj.x_am,obj.T,obj.P)    # [m3/g_pol_am]
        V2 = 1/obj.SinglePhaseDensity(obj.x_am,obj.T,obj.P) / (obj.x_am[1] * obj.MW_pol)   # [m3/g_pol_am]
        
        x_sol[T].append(obj.x_am[0])
        x_pol[T].append(obj.x_am[1])
        V_sol[T].append(V_s)
        V_pol[T].append(V_p)
        V_fromPmv[T].append(V1)
        V_fromSAFT[T].append(V2)
        Vmolar_fromPmv[T].append(V1_molar)
        Vmolar_fromSAFT[T].append(V2_molar)
        V_term1[T].append(term1)
        V_term2[T].append(term2)
        objects[T].append(obj)
        
    _df = pd.DataFrame({'T [K]': [T for _p in P_list],
                        'P [Pa]': P_list,
                        'x_sol [mol/mol]': x_sol[T],
                        'x_pol [mol/mol]': x_pol[T],
                        'V_sol [m3/g]': V_sol[T],
                        'V_pol [m3/g]': V_pol[T],
                        'term1 [m3/mol]': V_term1[T],
                        'term2 [m3/mol]': V_term2[T],
                        'Vmolar_fromPmv [m3/mol]': Vmolar_fromPmv[T],
                        'Vmolar_fromSAFT [m3/mol]': Vmolar_fromSAFT[T],
                        'V_fromPmv [m3/g_pol_am]': V_fromPmv[T], 
                        'V_fromSAFT [m3/g_pol_am]': V_fromSAFT[T], 
                        })
    df = pd.concat([df, _df], ignore_index=True)
    
print(df)
data = S.SolPolExpData(mix.sol, mix.pol)
now = datetime.now()  # current time
time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM
export_path = f"{data.path}/verifyV_{time_str}.xlsx"
with pd.ExcelWriter(export_path) as writer:
    df.to_excel(writer, index=False)
print("Data successfully exported to: ", export_path)

#* V am per g po am comparison
fig = plt.figure()
ax = fig.add_subplot(111)
for i, T in enumerate(T_list):
    ax.plot(P_list*1e-5, array(V_fromPmv[T]), color=S.custom_colours[i], linestyle="solid", label = f"{T-273}°C from Vs and Vp")
    ax.plot(P_list*1e-5, array(V_fromSAFT[T]), color=S.custom_colours[i], linestyle="dashed", label = f"{T-273}°C from SAFT prediction")
ax.set(xlabel='P [bar]', ylabel=r"$V_{am}$ [$m^{3}$/$g_{pol am}$]",
       title=r"Comparison of total amorphous volume $V_{am}$")
ax.legend().set_visible(True)

#* V am per mol comparison
fig = plt.figure()
ax = fig.add_subplot(111)
for i, T in enumerate(T_list):
    ax.plot(P_list*1e-5, Vmolar_fromPmv[T], color=S.custom_colours[i], linestyle="solid", label = f"{T-273}°C from Vs and Vp")
    ax.plot(P_list*1e-5, Vmolar_fromSAFT[T], color=S.custom_colours[i], linestyle="dashed", label = f"{T-273}°C from SAFT prediction")
ax.set(xlabel='P [bar]', ylabel=r"$\hat{V}_{am}$ [$m^{3}$/mol]",
       title=r"Comparison of moar amorphous volume $\hat{V}_{am}$")
ax.set_yscale('log')
ax.legend().set_visible(True)

# *  x plot
fig = plt.figure()
ax = fig.add_subplot(111)
for i, T in enumerate(T_list):
    ax.plot(P_list*1e-5, x_sol[T], color=S.custom_colours[i], linestyle="solid", label = f"{T-273}°C x_sol")
    ax.plot(P_list*1e-5, x_pol[T], color=S.custom_colours[i], linestyle="dashed", label = f"{T-273}°C x_pol")
ax.set(xlabel='P [bar]', ylabel=r"$x$ [mol/mol]", title=r"Comparison of molar fraction $x_{sol}$ and $x_{pol}$")
ax.legend().set_visible(True)

# * xsol*Vsol + xpol*Vpol
fig = plt.figure()
ax = fig.add_subplot(111)
for i, T in enumerate(T_list):
    ax.plot(P_list*1e-5, V_term1[T], color=S.custom_colours[i], linestyle="solid", label = f"{T-273}°C - x_sol*V_sol*MW_sol")
    ax.plot(P_list*1e-5, V_term2[T], color=S.custom_colours[i], linestyle="dashed", label = f"{T-273}°C - x_pol*V_pol*MW_pol")
ax.set(xlabel='P [bar]', ylabel= "V [m3/mol]",
       title="Term 1 vs. Term 2")
# ax.set_yscale('log')
ax.legend().set_visible(True)
plt.show()
