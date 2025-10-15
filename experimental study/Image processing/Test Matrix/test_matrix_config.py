import numpy as np
import pandas as pd
#import openpyxl
from PHE_functions import *
from CoolProp.CoolProp import PropsSI
from tabulate import tabulate

# Input values
T = 40.0  # Temperature in °C
p = 1.0   # Pressure in bar

S = 1
results = []
g = 9.81  # m/s^2

# Ranges for epsilon and Re
epsilon_range = [0.10, 0.25]  # 10% and 25%
Re_range = np.arange(1000, 5401, 600)  # Step of 600 instead of 200

# PHE geometry parameters in mm and W/mK
corrugation = 60.0             # in degrees (if relevant)
amplitude = 1.0                # in mm
wave_length_projected = 10.0   # in mm
wave_length = wave_length_projected * np.sin(np.deg2rad(corrugation))  # in mm
#wave_length = 8.66            # in mm
thickness = 0.5                # in mm
width = 80.0                   # in mm
length = 200.0                 # in mm
roughness = 0.0                # Surface roughness
thermal_conductivity = 15.0    # Thermal conductivity of material in W/mK

# Instantiate PHE class
phe = PHE_geometry(amplitude, wave_length, thickness, corrugation, width, length, roughness, thermal_conductivity)

# Calculate hydraulic diameter
d_h = phe.hydraulic_diameter()
print(f"Hydraulic diameter: {d_h} m")

# Inlet area
A_in = phe.inlet_area()
print(f"Inlet area PHE: {A_in} m^2")

# Critical Reynolds number according to doi.org/10.1016/j.ijheatmasstransfer.2020.120370
corrugation_rad = np.radians(corrugation)
Re_c1 = np.e**((5.62 - 1.13 * np.sin(corrugation_rad) ** 2.5) / (0.83 + 0.43 * np.sin(corrugation_rad) ** 5))
Re_c2 = 954 * cos(corrugation_rad) ** 4 + 53
print(f"Critical Reynolds number variant 1: {Re_c1}")
print(f"Critical Reynolds number variant 2: {Re_c2}")

# Calculation
# Phases and fluid properties
water = Fluid(T=T, P=p, media='water')
oxygen = Fluid(T=T, P=p, media='oxygen')

print(f"Water density: {water.rho} kg/m^3")
print(f"Oxygen density: {oxygen.rho} kg/m^3")
print(f"Water kinematic viscosity: {water.nuu} m^2/s")
print(f"Oxygen kinematic viscosity: {oxygen.nuu} m^2/s")

for epsilon in epsilon_range:
    for Re in Re_range:
        
        # Calculate mixture properties
        rho_mixture, mu_mixture, nu_mixture, Cp_mixture, lambda_mixture, Pr_mixture, x = mixture_properties(water, oxygen, epsilon)
        
        # Flow velocity and mass flow rates in PHE
        U_mixture = (Re * nu_mixture) / d_h
        m_dot_tp = rho_mixture * A_in * U_mixture
        #m_dot_L = (1.0 - x) * m_dot_tp
        m_dot_G = x * m_dot_tp
        m_dot_L = m_dot_tp - m_dot_G
        Fr = U_mixture ** 2 / (g * 2 * amplitude / 1000)
        V_dot_tp = m_dot_tp * 3600 / rho_mixture
        
        V_dot_G = m_dot_G / PropsSI('D', 'T', 294.15, 'P', 101325, 'Oxygen') * 10**6 * 60  # rho_G from manufacturer O2-nozzle
        V_dot_L = m_dot_L / water.rho * 3600
        
        results.append({
            "T[Celcius deg]": T,
            "p[bar]": p,
            "epsilon": epsilon,
            "Re": Re,
            "V_dot_L [m^3/h]": V_dot_L,
            "V_dot_G [ml/min]": V_dot_G,
            "V_dot_tp [m^3/h]": V_dot_tp,
            "flow quality": x,
            "m_dot_L [kg/s]": m_dot_L,
            "m_dot_G [kg/s]": m_dot_G,
            "U_mixture [m/s]": U_mixture,
            "rho_mixture [kg/m^3]": rho_mixture,
            "nu_mixture [m^2/s]": nu_mixture,
            "Fr": Fr
        })

df_results = pd.DataFrame(results)

# Save as CSV with temperature in filename
df_results.to_csv(f"results_test_matrix_60_T{T}.csv", index=False)

# Print the results as a table in the console
table_output = tabulate(df_results, headers="keys", tablefmt="grid")
#print(table_output)

# Write the table to a text file with temperature in filename
with open(f"flow_boundary_conditions_T{T}.txt", "w") as file:
    file.write(table_output)
    file.write("\n")
    
