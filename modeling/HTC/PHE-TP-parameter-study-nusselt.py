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
    "mathtext.fontset": "custom"
    #"mathtext.bf": font_name_cambriamathbd,
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
amplitude      = 2.4
wave_length    = 2 * amplitude
thickness      = 0.5
chevron_angle  = 60.0
roughness      = 0.
width          = 278.2
length         = 326.1

# flow input parameters 
U_0        = 0.5       # > flow velocity in m/s
T_E        = 20.       # > Inlet Temperature in °C
P          = 1.        # > operating pressure in bar 
epsilon    = 0.05      # 5 % Gas fraction
mixture    = ["water", "oxygen"]
Liquid     = Fluid(T=T_E, P=P, media=mixture[0]) 
Gas        = Fluid(T=T_E, P=P, media=mixture[1])
# Prandtl number
Pr         = Liquid.Pr

# Initialize xsi 
xsi = 1.

# constant mass flow rate 
assumption = True
# else constant volume flow rate
#assumption = False
# Specify the range and step size
x_min = 30.
x_max = 60.
step_size = 15  # Adjust the step size as needed



# Define the range of Reynolds numbers
Re_range = np.linspace(100, 5000, 50)
#print(Re_range)

# Define the beta values
betas = [30, 60]

# List of 7 different markers
M = ['o', 'v', '^', '<', '>', 's', 'p']
M = ['s' ,'^' ,'v' ,'D' ,'+' ,'o' ,'h' ,'D' ,'o', '*', 'x'  ]  # '*''x'

# List of 7 different colors
C = ['C1','C2','C3','C4','C5','C6','C7','C8','C9', 'C10', 'C11' ]
C = ['b', 'g', 'r', 'c', 'm', 'y', 'k']



# plot directory 
dir_name = 'plots'

# Check if the directory already exists
if not os.path.exists(dir_name):
    # If the directory doesn't exist, create it
    os.makedirs(dir_name)
# Nusselt subdirectory
dir_name = 'plots/nusselt'
# Loop over the beta values
for j, beta in enumerate(betas):
    
    # Create a subfolder for this beta
    subfolder = f'{dir_name}/beta_{beta}'
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # Create lists to store the Nusselt number for each method
    Nusselt_coeffs = [[] for _ in range(5)]  
    dp_coeffs = [[] for _ in range(5)]  

    # Loop over the Reynolds numbers
    for Re in Re_range:
        # Create an instance of the class
        dp_estimator = pressure_coefficient_Estimator(beta, Re, amplitude, wave_length)
        Nu_estimator = Nusselt_estimator(xsi, Re, beta, Pr)

        # Get the list of applicable methods for this beta and Re
        methods = Nu_estimator.applicable_methods()
        
        # Loop over the applicable methods
        for method in methods:
            if (method == 4):
                Nusselt_coeffs[method].append(Nu_estimator.select_method(method))
            else:
                xsi = dp_estimator.select_method(method)
                Nu_estimator = Nusselt_estimator(xsi, Re, beta, Pr)
                Nusselt_coeffs[method].append(Nu_estimator.select_method(method))

    # Get the labels using list comprehension
    labels = [Nu_estimator.get_correlation_name(index) for index in range(5)]

    # Plot the Nusselt number for each method separately
    for i, coeffs in enumerate(Nusselt_coeffs):
        if coeffs:  # check if the list is not empty
            plt.figure(figsize=(10, 6))  # Create a new figure for each method
            plt.plot(Re_range[:len(coeffs)], coeffs, linestyle='None', marker=M[i], color=C[i], markersize=3, label=labels[i])
            plt.xlabel('Re')
            plt.ylabel('Nusselt number')
            plt.title(f'Nusselt vs Re for $\\beta={beta}$ using {labels[i]}')
            plt.grid(True)
            plt.legend()
            plt.axis('tight')
            plt.savefig(f'{subfolder}/{labels[i]}.pdf')  # Save the figure to the subfolder
            plt.savefig(f'{subfolder}/{labels[i]}.png')
            plt.close()  # Close the figure to free up memory

    # Create a new figure for this beta
    plt.figure(figsize=(8, 5))

    # Plot all methods together
    for i, coeffs in enumerate(Nusselt_coeffs):
        if coeffs:  # check if the list is not empty
            plt.plot(Re_range[:len(coeffs)], coeffs, linestyle='-', marker=M[i], color=C[i], markersize=3, label=labels[i])

    # Set the labels and title
    plt.xlabel('Reynolds-Zahl $Re_h$')
    plt.ylabel('Nusselt-Zahl $Nu_h$ ')
    #plt.title(f'Nusselt vs Re for $\\beta={beta}$')
    plt.grid(True)
    plt.legend()
    plt.axis('tight')
    plt.savefig(f'{subfolder}/all_methods.pdf')  # Save the figure to the subfolder
    plt.savefig(f'{subfolder}/all_methods.svg')
    #plt.close()  # Close the figure to free up memory

# show plots
plt.show()
