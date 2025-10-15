#! /usr/bin/env python3
'''===============================================================================================================
 - Author: M. Guesmi
 - Adress: Chair of process engineering TU Dresden, George-Bähr-Straße 3b, 01069 Dresden, Germany
 - Summary: Two-Phase bubbly Flow in a plate HE: semi-analytical for pressure drop and Heat Transfer
    > predict the occuring flow form 
    > estimate the pressure drop based on aaplicable correlation available in the literature
    > estimate the heat transfer based on Nusselt correlations available in literature 
=================================================================================================================='''
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, exp, log, log10, sqrt, arccos, arcsin, radians
from numpy import pi, e
import sys, os
import csv # print data to file
import CoolProp.CoolProp as cp
from PHE_functions import *
import matplotlib.font_manager as fm



# -------------------------------
# Globale rcParams konfigurieren
# -------------------------------

plt.rcParams.update({
    "text.usetex": False,
    "font.size": 11,
})


# input parameters of the plate geometry 
'''
    δ: plate depth/ corrugation amplitude in mm
    λ: chevron pitch - wavelength in mm
    β: corrugation inclination (chevron) in ° 
    e: plate thickness in mm
    B: plate width  in mm
    L: plate length in mm
'''
amplitude      = 2
wave_length    = 4 * amplitude
thickness      = 0.5
chevron_angle  = 60.0
roughness      = 0.
width          = 100
length         = 200

# flow input parameters 
U_0        = 0.5       # > flow velocity in m/s
T_E        = 20.       # > Inlet Temperature in °C
P          = 1.        # > operating pressure in bar 
epsilon    = 0.00      # 5 % Gas fraction
mixture    = ["water", "oxygen"]
Liquid     = Fluid(T=T_E, P=P, media=mixture[0]) 
Gas        = Fluid(T=T_E, P=P, media=mixture[1])
# 
Pr         = Liquid.Pr

# constant mass flow rate 
assumption = True
# else constant volume flow rate
#assumption = False
# Specify the range and step size
x_min = 30.
x_max = 60.
step_size = 15  # Adjust the step size as needed

# tolerance value error band in %
err_band = 0.

# Define the range of Reynolds numbers
Re_range = np.linspace(100, 5000, 50)
#print(Re_range)

# Define the beta values
betas = [30, 60]

# List of 7 different markers
M = ['o', 'v', '^', '<', '>', 's', 'p', 'h', 'D', 'o', 'v', '^', '<', '>', 's']
M = ['s' ,'^' ,'v' ,'D' ,'+' ,'o' ,'h' ,'D' ,'o', '*', 'x', 's','^' ,'v' ,'D' ,'+']  # '*''x'

# List of 7 different colors
C = ['C1','C2','C3','C4','C5','C6','C7','C8','C9', 'C10', 'C11', 'C12','C13', 'C14', 'C15', 'C16']
C = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'peru', 'olive', 'navy', 'aqua', 'gold']

# plot directory 
dir_name = 'plots'

# Check if the directory already exists
if not os.path.exists(dir_name):
    # If the directory doesn't exist, create it
    os.makedirs(dir_name)
    
# Pressure drop coeff. subdirectory
dir_name = 'plots/pressure_drop'

# Check if the directory already exists
if not os.path.exists(dir_name):
    # If the directory doesn't exist, create it
    os.makedirs(dir_name)


# Loop over the beta values
for j, beta in enumerate(betas):
    
    # Create a subfolder for this beta
    subfolder =  f'{dir_name}/beta_{beta}'
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # Create a list to store the pressure drop coefficients for each method
    pressure_drop_coeffs = [[] for _ in range(14)]  # assuming there are 7 methods

    # Loop over the Reynolds numbers
    for Re in Re_range:
        # Create an instance of the class
        estimator = pressure_coefficient_Estimator(beta, Re, amplitude, wave_length)
        # Get the list of applicable methods for this beta and Re
        methods = estimator.applicable_methods()
        
        # Loop over the applicable methods
        for method in methods:
            # Druckverlustkoeffizient berechnen und zur Liste hinzufügen
            coeff = estimator.select_method(method)
            if coeff is not None:  # Nur gültige Koeffizienten berücksichtigen
                pressure_drop_coeffs[method].append(coeff)

    # Get the labels using list comprehension
    labels = [estimator.get_correlation_name(index) for index in range(14)]

    # Plot the pressure drop coefficients for each method separately
    for i, coeffs in enumerate(pressure_drop_coeffs):
        if coeffs:  # check if the list is not empty
            plt.figure(figsize=(10, 6))  # Create a new figure for each method
            plt.plot(Re_range[:len(coeffs)], coeffs, linestyle='None', marker=M[i], color=C[i], markersize=3, label=labels[i])
            plt.xlabel(f'$Re_h$')
            plt.ylabel(f'Druckverlustbeiwert $\\xi$')
            plt.title(f'Druckverlustbeiwert $\\xi = f(Re)$ nach {labels[i]} für $\\beta$ = {beta}°')
            plt.grid(True)
            plt.legend()
            plt.axis('tight')
            plt.savefig(f'{subfolder}/{labels[i]}.pdf')  # Save the figure to the subfolder
            plt.savefig(f'{subfolder}/{labels[i]}.png')
            #plt.close()  # Close the figure to free up memory

    # Create a new figure for this beta
    plt.figure(figsize=(8, 5))

    # Plot all methods together
    for i, coeffs in enumerate(pressure_drop_coeffs):
        if coeffs:  # check if the list is not empty
            plt.plot(Re_range[:len(coeffs)], coeffs, linestyle='-', marker=M[i], color=C[i], markersize=3, label=labels[i])

            # Add error bands for ±20% around each curve in grey

            # Länge der gültigen Koeffizienten und Reynolds-Zahlen bestimmen
            valid_len = min(len(Re_range), len(coeffs))
            plt.fill_between(Re_range[:valid_len], (1.0 - err_band / 100.0) * np.array(coeffs[:valid_len]), (1.0 + err_band / 100.0) * np.array(coeffs[:valid_len]), color='grey', alpha=0.3)

    # Set the labels and title
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(f'Reynolds-Zahl $Re_h$')
    plt.ylabel(f'Druckverlustbeiwert $\\xi_h$')
    #plt.title(f'Druckverlustbeiwert $\\xi = f(Re)$ für $\\beta$ = {beta}°')
    plt.grid(True)
    plt.legend()
    plt.axis('tight')

    plt.savefig(f'{subfolder}/all_methods.pdf')  # Save the figure to the subfolder
    plt.savefig(f'{subfolder}/all_methods.svg')
    #plt.close()  # Close the figure to free up memory

# show plots
plt.show()