#! /usr/bin/env python3
'''===============================================================================================================
 - Author: M. Guesmi
 - Adress: Chair of process engineering TU Dresden, George-Bähr-Straße 3b, 01069 Dresden, Germany
 - Summary: Two-Phase Flow and Heat Transfer in Plate and Frame Heat Exchanger: Flow pattern map 
          > •	https://doi.org/10.1016/j.applthermaleng.2024.122542	
            •  	https://doi.org/10.1016/j.ces.2024.119905	
            •	https://doi.org/10.1016/j.ijmultiphaseflow.2024.104871
=================================================================================================================='''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PHE_functions import *
from matplotlib.patches import Rectangle
from PHE_flow_pattern_module import *

# Enable LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})

# -------------------------------------------------
# PHE Geometry Parameters (all dimensions in mm)
# -------------------------------------------------
depth = 2.0                                           # Plate depth (mm)
thickness = 0.5                                       # Plate thickness (mm)
corrugation = 30.0                                    # Corrugation angle (degrees)
wave_length = 10.*np.sin(np.deg2rad(corrugation))     # Wavelength of corrugation (mm)
width = 278.2                                         # Plate width (mm)
length = 326.1                                        # Plate length (mm)
roughness = 0.0                                       # Surface roughness (assumed smooth)
Lambda = 15.0                                         # Thermal conductivity (W/mK)

# -------------------------------------------------
# PHE Geometry and Hydraulic Diameter
# -------------------------------------------------
# Initialize the PHE geometry
phe = PHE_geometry(depth, wave_length, thickness, corrugation, width, length, roughness, Lambda)

# Calculate Hydraulic Diameter
d_H = phe.hydraulic_diameter()

# -------------------------------------------------
# Operating Parameters
# -------------------------------------------------
T = 20.0  # Temperature (°C)
p = 1.0   # Pressure (Pa)

# -------------------------------------------------
# Fluid Properties for Air-Water System
# -------------------------------------------------
phase1 = 'water'
phase2 = 'oxygen'
liquid = Fluid(T=T, P=p, media=phase1)     # Liquid phase properties
gas = Fluid(T=T, P=p, media=phase2)        # Gas phase properties

# -------------------------------------------------
# Drift Flux and Transition Parameters
# -------------------------------------------------
M = 0.3           # Transition parameter for coarse/fine bubbly flow
void_fraction_model = "Asano"   # Asano Drift flux model 
#void_fraction_model = "Margat" # Maragt Drift flux model 
#void_fraction_model = "Homogeneous"  # Homogeneous model 
f_d_model = 'Passoni'
f_d_model = 'Guesmi'

# -------------------------------------------------
# Load and Process Experimental Data
# -------------------------------------------------
filename = "results_Testmatrix.xlsx"
df = pd.read_excel(filename, sheet_name=0)
df.columns = df.columns.str.strip()

# Geometry parameters
B_p = 0.08        # Plate width [m]
a = 0.001         # Amplitude of the sine wave [m]
b_p = 2 * a       # Plate depth [m]
A_f = B_p * b_p   # Flow cross-section [m²]

# Uncertainties of the geometry
e_B_p = 0.0005    # [m]
e_b_p = 0.00005   # [m]
lambda_ = 0.00866 # [m]
phi_deg = 60
phi_rad = np.deg2rad(phi_deg)

# Approximation error (e_approx)
e_approx = (lambda_ * b_p) / (2 * np.pi * np.cos(phi_rad))

# Error of the area A_f
e_A_f = np.sqrt((b_p * e_B_p)**2 + (B_p * e_b_p)**2 + e_approx**2)

# Convert volume flows [m³/s]
df["V_dot_liquid"] = df["V_punkt_L [m^3/h]"] / 3600
df["V_dot_gas"] = df["V_punkt_G [ml/min]"] * 1e-6 / 60

# Superficial velocities [m/s]
df["U_s_liquid"] = df["V_dot_liquid"] / A_f
df["U_s_gas"] = df["V_dot_gas"] / A_f

# Device-specific error specifications
Q_l_max = 150 / 60 / 1000  # [m³/s] → 150 L/min
Q_g_max = 10 / 60 / 1000   # [m³/s] → 10 L/min

e_Q_l = 0.008 * df["V_dot_liquid"] + 0.002 * df["V_dot_liquid"] + 0.001 * Q_l_max
e_Q_g = (0.01 + 0.0025) * Q_g_max

# Calculate flow errors
df["e_Q_l"] = e_Q_l
df["e_Q_g"] = e_Q_g

# Errors of the superficial velocities
df["e_U_s_liquid"] = df["U_s_liquid"] * np.sqrt((e_A_f / A_f)**2 + (df["e_Q_l"] / df["V_dot_liquid"])**2)
df["e_U_s_gas"] = df["U_s_gas"] * np.sqrt((e_A_f / A_f)**2 + (df["e_Q_g"] / df["V_dot_gas"])**2)

# Filter data - only include points where U_s_gas > 0
nonzero_mask = df["U_s_gas"] > 0

# -------------------------------------------------
# Create Combined Plot
# -------------------------------------------------

plt.figure(figsize=(12, 9))

# Add a title to the plot
plt.title('Void Fraction Isolines and Transition Line with Experimental Data\nUsing Drift Flux Model', fontsize=16)


# Add isolines (already done with your function `plot_isolines`)
plot_isolines(void_fraction_model=void_fraction_model,f_d_model=f_d_model, d_H=d_H, liquid=liquid, gas=gas, M=M, theta=T, add_legend=True)
# plot transition coarse fine bubbly flow for M=0.5
plot_transition_fine_coarse(void_fraction_model=void_fraction_model,f_d_model=f_d_model, d_H=d_H, liquid=liquid, gas=gas, M=M, theta=T, alpha_threshold=0.1)


# Add experimental data points with error bars (only where U_s_gas > 0)
plt.errorbar(
    df.loc[nonzero_mask, "U_s_gas"],
    df.loc[nonzero_mask, "U_s_liquid"],
    xerr=df.loc[nonzero_mask, "e_U_s_gas"],
    yerr=df.loc[nonzero_mask, "e_U_s_liquid"],
    fmt='o', markersize=4, ecolor='blue', capsize=2, color='red',
    label='Experimental Data Points'
)

# Function to add a text box with label
def add_region_label(x, y, text, ax=None):
    if ax is None:
        ax = plt.gca()
    props = dict(boxstyle='round', facecolor='cyan', alpha=0.5)
    ax.text(x, y, text, fontsize=14, bbox=props, ha='center', va='center')

# Add region labels - you can adjust the x, y coordinates as needed
add_region_label(0.5e-2, 0.5, 'Fine Bubbly')  # Upper left
add_region_label(.5e-1, 0.1, 'Taylor-like Bubbly')  # Upper right
add_region_label(5e-3, 5e-2, 'Coarse Bubbly')  # Lower left
add_region_label(4e-1, 2e-1, 'Heterogeneous Flow')  # Center

# Set log-log scale and labels
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Superficial Gas Velocity, $U_{SG}$ (m/s)', fontsize=14)
plt.ylabel('Superficial Liquid Velocity, $U_{SL}$ (m/s)', fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Adjust limits to accommodate both datasets
plt.ylim(1e-2, 2)
plt.xlim(1e-3, 2)

# Show the plot
plt.tight_layout()
plt.savefig("Flow_pattern_map_with_our_test_points.pdf", format="pdf", dpi=300)
plt.show()

