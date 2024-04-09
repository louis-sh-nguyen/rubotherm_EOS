"""
Script to plot and compare density of CO2 between SAFT and Rubotherm experiments.
25 March 2024
Louis Nguyen
sn621@ic.ac.uk
"""

import solubility_master as S
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

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

now = datetime.now()  # current time
time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM    
    
mix = S.BaseSolPol("CO2","HDPE")
T_list = [25+273, 35+273, 50+273]
data = S.SolPolExpData(mix.sol, mix.pol)
df = {}
P_list = {}
rhoCO2_SAFT = {}
for i, T in enumerate(T_list):    
    df[T]=data.get_sorption_data(T)
    max_i = df[T]["P[bar]"].max()
    if i == 0:
        pbar_max = max_i
    else:
        pbar_max = max(pbar_max, max_i)

for i, T in enumerate(T_list):
    # Using fixed range
    # P_list[T] = linspace(1, pbar_max*1e5, 100)  # [Pa]    
    
    # Using matching pressure range with exp data
    P_list[T] = linspace(1, df[T]["P[bar]"].max()*1e5, 100)  # [Pa]
    
    # SAFT prediciton
    _rhoCO2_SAFT = [S.DetailedSolPol(mix, T, P).SinglePhaseDensity(array([1., 0.]),T,P) for P in P_list[T]] # [mol/m3]
    rhoCO2_SAFT[T] = array(_rhoCO2_SAFT) * mix.MW_sol  * 1e-6  # [g/cm3]


# Plot figure
fig = plt.figure()
ax = fig.add_subplot(111)
for i, T in enumerate(T_list):
    
    # Exp measurement
    ax.plot(df[T]["P[bar]"], df[T]["ρ[g/cc]"], \
        color=S.custom_colours[i], linestyle="None", marker=S.custom_markers[-1], label=f"{T-273}°C - exp")
    
    # SAFT predictions
    ax.plot(P_list[T]*1e-5, rhoCO2_SAFT[T], \
        color=S.custom_colours[i], linestyle="solid", marker="None", label=f"{T-273}°C - SAFT")

ax.set_xlabel("P [bar]")
ax.set_ylabel(r"$\rho_{CO2}$ [$cm^{3}$/g]")
# ax.set_title("Comparison of CO2 density")
ax.tick_params(direction="in")
ax.legend().set_visible(True)
plt.show()

# Save figure
save_fig_path = f"{data.path}/PlotRhoCO2_{time_str}.png"
plt.savefig(save_fig_path, dpi=1200)
print(f"Plot successfully exported to {save_fig_path}.")