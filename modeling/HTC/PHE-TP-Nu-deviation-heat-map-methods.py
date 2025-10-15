#! /usr/bin/env python3
'''===============================================================================================================
 - Author: M. Guesmi
 - Adress: Chair of process engineering TU Dresden, George-Bähr-Straße 3b, 01069 Dresden, Germany
 - Summary: Two-Phase Flow and Heat Transfer
    > cell model to estimate the heat transfer of two phase bubbly flow
    > pipe flow --> semi-analytical solution based on empirical correlations 
=================================================================================================================='''

from tkinter import font
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PHE_functions import *
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm

# -------------------------------
# Schriftarten hinzufügen
# -------------------------------


# -------------------------------
# Globale rcParams konfigurieren
# -------------------------------
# from matplotlib import rc
plt.rcParams.update({
    "font.family": "serif",    # use serif/main font for text elements
    "text.usetex": True,       # use inline math for ticks
    "pgf.rcfonts": False,      # don't setup fonts from rc parameters
    "figure.dpi": 300         # Higher DPI for better quality figures
})

# Input parameters for the plate geometry
amplitude = 2
wave_length = 4 * amplitude
thickness = 1
roughness = 0.0
width = 100
n = 100
length = n * wave_length

# Flow input parameters
U_0 = 0.5  # flow velocity in m/s
T_E = 20.0  # Inlet Temperature in °C
P = 1.0  # operating pressure in bar
epsilon = 0.00  # 5% Gas fraction
mixture = ["water", "oxygen"]
Liquid = Fluid(T=T_E, P=P, media=mixture[0])
Gas = Fluid(T=T_E, P=P, media=mixture[1])
Pr = Liquid.Pr
xsi = pressure_coefficient_Estimator
# Constant mass flow rate assumption
assumption = True

# Therefore, the common Re_h range that satisfies all given conditions for applicable indices is 1000 < 𝑅𝑒_ℎ < 3000
Re_range = np.linspace(1000, 5000, 50)

betas = [30, 60]

# Marker and color styles
M = ['s', '^', 'v', 'D', '+', 'o', 'h', 'D', 'o', '*', 'x']
C = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

# Plot directory setup
plot_dir = './plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

nusselt_dir = os.path.join(plot_dir, 'nusselt')
if not os.path.exists(nusselt_dir):
    os.makedirs(nusselt_dir)

# Tolerance value for error band in %
err_band = 20.0

# Function to create a custom colormap with discrete color mapping
def create_custom_colormap():
    colors = ['green', 'red']
    cmap = ListedColormap(colors)
    return cmap

# ParaView-style graded colormaps for scientific visualization
def create_paraview_colormap(style='cool_warm'):
    """
    Create ParaView-style graded colormaps for scientific visualization.
    
    Parameters:
    -----------
    style : str
        'cool_warm' - Blue to white to red (excellent for diverging data)
        'plasma' - Purple to pink to yellow (perceptually uniform)
        'turbo' - Blue to green to yellow to red (high contrast)
        'jet_improved' - Improved version of jet colormap
        'scientific' - Blue to cyan to yellow to red (scientific standard)
        'viridis_enhanced' - Enhanced viridis with better contrast
    """
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    if style == 'cool_warm':
        # ParaView's Cool to Warm colormap - excellent for deviation data
        colors = [
            '#3B4CC0',  # Cool blue
            '#5977E3',  # Light blue
            '#7B9FF0',  # Lighter blue
            '#A5C7F7',  # Very light blue
            '#D1E5FD',  # Almost white blue
            '#F7F7F7',  # White/neutral
            '#FDE5D1',  # Very light red
            '#F7C7A5',  # Light red
            '#F09F7B',  # Lighter red
            '#E37759',  # Red
            '#C04B3B'   # Deep red
        ]
        
    elif style == 'plasma':
        # Plasma colormap - perceptually uniform
        colors = [
            '#0C0786',  # Deep purple
            '#40039A',  # Purple
            '#6A00A7',  # Magenta-purple
            '#8F0DA4',  # Magenta
            '#B12A90',  # Pink-magenta
            '#CC4778',  # Pink
            '#E16462',  # Coral
            '#F1834A',  # Orange
            '#FCA636',  # Yellow-orange
            '#FCCE25',  # Yellow
            '#F0F921'   # Bright yellow
        ]
        
    elif style == 'turbo':
        # Google's Turbo colormap - high contrast
        colors = [
            '#30123B',  # Dark blue
            '#455CC9',  # Blue
            '#3F8CCA',  # Light blue
            '#2FB47C',  # Green
            '#A4D649',  # Light green
            '#FDE725',  # Yellow
            '#FB8022',  # Orange
            '#E65D2F',  # Red-orange
            '#C73E1D',  # Red
            '#A01A7D',  # Purple-red
            '#7A0403'   # Dark red
        ]
        
    elif style == 'scientific':
        # Scientific blue-cyan-yellow-red colormap
        colors = [
            '#0000FF',  # Pure blue
            '#0080FF',  # Light blue
            '#00FFFF',  # Cyan
            '#80FF80',  # Light green
            '#FFFF00',  # Yellow
            '#FF8000',  # Orange
            '#FF0000'   # Red
        ]
        
    elif style == 'viridis_enhanced':
        # Enhanced viridis with better contrast
        colors = [
            '#440154',  # Dark purple
            '#482777',  # Purple
            '#3F4A8A',  # Blue-purple
            '#31678E',  # Blue
            '#26838F',  # Teal
            '#1F9D8A',  # Green-teal
            '#6CCE5A',  # Green
            '#B6DE2B',  # Yellow-green
            '#FEE825'   # Yellow
        ]
        
    else:  # Default to cool_warm
        colors = [
            '#3B4CC0', '#5977E3', '#7B9FF0', '#A5C7F7', '#D1E5FD',
            '#F7F7F7', '#FDE5D1', '#F7C7A5', '#F09F7B', '#E37759', '#C04B3B'
        ]
    
    # Create the colormap
    n_bins = 256  # Smooth gradations
    cmap = LinearSegmentedColormap.from_list(f'paraview_{style}', colors, N=n_bins)
    
    return cmap

# Loop over beta angles
for beta in betas:
    subfolder = os.path.join(nusselt_dir, f'beta_{beta}')
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # Initialize pressure drop coefficients for this beta based on applicable methods
    # Erstelle eine Instanz der pressure_coefficient_Estimator-Klasse
    instanz = pressure_coefficient_Estimator(beta, Re_range[0], amplitude, wave_length)

    # Rufe die Methode friction_coeff_I über die Instanz auf
    xsi = instanz.friction_coeff_I()
    estimator = Nusselt_estimator(xsi, Re_range[0], beta, Pr)  # Using Re_range[0] for initial estimation
    methods = estimator.applicable_methods()
    labels = [estimator.get_correlation_name(method_index) for method_index in methods]

    print(methods)
    num_methods = len(methods)
    pressure_drop_coeffs = [[] for _ in range(num_methods)]
    # Loop over Reynolds numbers
    for Re in Re_range:
        estimator = Nusselt_estimator(xsi, Re_range[0], beta, Pr)
        methods = estimator.applicable_methods()
        for method_index in methods:
            coeff = estimator.select_method(method_index)
            if coeff is not None:
                index_in_methods = methods.index(method_index)
                pressure_drop_coeffs[index_in_methods].append(coeff)

    # Calculate mean absolute deviations between each pair of methods for this beta
    mean_absolute_deviations = np.zeros((num_methods, num_methods))

    for i in range(num_methods):
        for j in range(i + 1, num_methods):
            if pressure_drop_coeffs[i] and pressure_drop_coeffs[j]:
                min_length = min(len(pressure_drop_coeffs[i]), len(pressure_drop_coeffs[j]))
                absolute_deviations = [
                    abs((pressure_drop_coeffs[i][k] - pressure_drop_coeffs[j][k]) / pressure_drop_coeffs[i][k])
                    for k in range(min_length)
                ]
                mean_absolute_deviations[i][j] = np.mean(absolute_deviations) * 100

    # Print the mean absolute deviations for this beta
    print(f"Mean Absolute Deviations between Methods for Beta {beta}:")
    for i in range(num_methods):
        for j in range(i + 1, num_methods):
            print(f"Mean Absolute Deviation between Method {i + 1} and Method {j + 1}: {mean_absolute_deviations[i][j]:.2f}")

    # Create a ParaView-style colormap for better scientific visualization
    paraview_cmap = create_paraview_colormap('cool_warm')  # Excellent for deviation data

    # Create a mask to hide the lower triangle (including the diagonal)
    mask = np.tril(np.ones_like(mean_absolute_deviations, dtype=bool), k=-1)

    # Plot the mean absolute deviations with ParaView colormap for this beta
    plt.figure(figsize=(12, 7))

    # Format the annotations to use commas as decimal separators
    annotations = np.array(
        [["{:.2f}".format(value).replace(".", ",") for value in row] for row in mean_absolute_deviations])

    # Plot the heatmap with ParaView-style colormap
    ax = sns.heatmap(
        mean_absolute_deviations,
        annot=annotations,
        fmt="",
        xticklabels=labels,
        yticklabels=labels,
        cmap=paraview_cmap,  # Use ParaView-style colormap
        mask=mask,
        vmin=0,
        vmax=100,
        square=False,
        cbar_kws={
            'label': r'Mittlere Prozentuale Abweichung (\%)',
            'shrink': 0.8,  # Better colorbar proportions
            'aspect': 20    # Better aspect ratio
        },
        annot_kws={"size": 11, "weight": "bold"}  # Bold annotations for better readability
    )

    # Drehe die X-Achsenbeschriftungen um 45 Grad nach links
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")

    #plt.title(f'Mittlerer absoluter prozentualer Fehler für $β = {beta}$')
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(os.path.join(subfolder, f'mean_absolute_deviations_custom_beta_{beta}.svg'))
    plt.show()
