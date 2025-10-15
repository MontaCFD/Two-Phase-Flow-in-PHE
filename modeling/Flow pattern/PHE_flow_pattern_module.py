#! /usr/bin/env python3
'''===============================================================================================================
 - Author: M. Guesmi
 - Adress: Chair of process engineering TU Dresden, George-Bähr-Straße 3b, 01069 Dresden, Germany
 - Summary: Two-Phase Flow and Heat Transfer in Plate and Frame Heat Exchanger
            > geometrical and flow characteristics in PHE 
            > thermodyanamic properties and averaging of mixture properties 
            > function to classify flow pattern in PHE

=================================================================================================================='''
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, exp, log, log10, sqrt, arccos, arcsin, radians
from numpy import pi, e
import sys, os
import csv # print data to file
import CoolProp.CoolProp as cp
from scipy.optimize import fsolve
from scipy.optimize import bisect

# language selector 
deutsch = False
# class thermodynamic properties
class Fluid(object):
    """
          This class contains the thermodynamics properties of a fluid
          Args:
                  T (in °C) Temperature 0 bis 99
                  p (in bar) pressure
          Returns:
                  roh (in Kg/m3) water density
                  c_p (in J/Kg.K) Specific heat capacity
                  lambda (in W/mK) water thermal conductivity
                  nu (in m2/s) water kinematic viscosity
                  Pr (-) water prandtl number

      """
    def __init__(self, T, P, media):
        """
            This function initializes all the thermodynamic properties of a fluid
            at a given temperature and pressure.

            Parameters
            ----------
            T : float
                Temperature in degrees Celsius
            P : float
                Pressure in bar
            media : str
                fluid name (e.g. 'water', 'air', 'oxygen', etc.)
        """
        # Store input parameters
        self.T = T
        self.P = P
        self.media = media

        # Calculate thermodynamic properties using CoolProp's PropsSI function
        # Input parameters: 'P', P*10**5, 'T', T + 273.15, media
        self.k   = cp.PropsSI('L', 'P', P*10**5, 'T', T + 273.15, media)        # thermal conductivity
        self.Cp  = cp.PropsSI('C', 'P', P*10**5, 'T', T + 273.15, media)        # specific heat capacity
        self.rho = cp.PropsSI('D', 'P', P*10**5, 'T', T + 273.15, media)        # density
        self.muu = cp.PropsSI('V', 'P', P*10**5, 'T', T + 273.15, media)        # dynamic viscosity
        self.nuu = self.muu / self.rho                                          # kinematic viscosity
        self.Pr  = cp.PropsSI('Prandtl', 'P', P*10**5, 'T', T + 273.15, media)  # Prandtl number





# area enlargement factor 
def area_enlargement(X):
    """
        Calculates the area enlargement factor using the three-point integration formula.

        Parameters
        ----------
        X : float
            X = π.λ/δ

        Returns
        -------
        φ(X) : float
            The area enlargement factor, calculated as (1 + sqrt(1 + X^2) + 4*sqrt(1 + X^2/2)) / 6

        Notes
        -----
        The area enlargement factor is a dimensionless quantity that describes the ratio of the
        effective heat transfer area to the projected area of the plate. The three-point integration
        formula is an approximation of the area enlargement factor, which is calculated as the
        average of the area enlargement factors of the three points at which the plate is in contact
        with the fluid.

    """
    return     (1. / 6.) * (1.0 + sqrt(1.0 + X ** 2) + 4.0 * sqrt(1.0 + 0.5 * X ** 2.0))


# > class for plate geometry
class PHE_geometry(object):
    """
    Class to calculate the hydraulic diameter and heat transfer area of a plate heat exchanger.

    Attributes
    ----------
    δ : float
        Plate depth/ corrugation depth in mm
    λ : float
        Chevron pitch - wavelength in mm
    β : float
        Corrugation inclination (chevron) in mm
    e : float
        Plate thickness in mm
    B : float
        Plate width in mm
    L : float
        Plate length in mm
    """

    def __init__(self, depth: float, wave_length: float, thickness: float, corrugation: float, width: float, length:float, roughness: float, Lambda: float) -> None:
        self.depth      = depth
        self.wave_length    = wave_length
        self.corrugation    = corrugation
        self.thickness      = thickness / 1000.
        self.width          = width / 1000.  # mm to m
        self.length         = length / 1000. # mm to m
        self.roughness      = roughness
        self.Lambda         = Lambda
        
    def hydraulic_diameter(self) -> float:
        """
            Calculates the hydraulic diameter of the PHE according to the non-circular definition
            see: https://doi.org/10.1016/j.rser.2018.11.017
        """
        # corrugation parameter 
        X = pi * self.depth / self.wave_length 
        # area enlargement factor using three-point integration formula 
        phi = area_enlargement(X)
        #print('phi=', phi)
        # hydraulic diameter according to non-circular definition in m 
        D_h = 2.0 * self.depth / phi * 1 / 1000.
        #print("D_h = ",D_h)
        return D_h


    def heat_area(self)-> float :
        """
            Returns the heat transfer area in m2
        """
        # calculate the heat transfer area in m^2
        heat_area = self.width  * self.length 
        return  heat_area

    def conductivity(self) -> float:
        """
            Returns the thermal conductivity of steel in W/mK
        """
        Lambda = self.Lambda
        return Lambda

    def inlet_area(self) -> float:
        """
        Returns the inlet area in m^2.
        """
        d_H = self.hydraulic_diameter()
        A_inlet = 2 * self.depth * self.width     #check again
        return A_inlet

#  function homogeneous two phase flow model 
def flow_quality(epsilon, rho_L, rho_G, S) :
    """
    Calculates the flow quality x for a two-phase flow

    Parameters
    ----------
    epsilon : float
        The void fraction of the two-phase flow
    rho_L : float
        The density of the liquid phase in kg/m3
    rho_G : float
        The density of the gas phase in kg/m3
    S : float
        The slip ratio between the gas and liquid phases

    Returns
    -------
    x : float
        The flow quality x, which is a dimensionless value between 0 and 1
    """
    if (epsilon == 0):
        # if the void fraction is zero, then the flow quality is also zero
        x = 0.
    else:
        # calculate the flow quality x
        x = 1.0 / (1.0 + (1.0 - epsilon) / epsilon * 1 / S * rho_L / rho_G)
    return x

# averaging the thermodynamic properties of two phase consídering the mixture a pseudo-Fluid 
def mixture_properties(Fluid1, Fluid2, epsilon):
    """
    Calculates the averaged thermodynamic properties of a two-phase flow.

    Parameters
    ----------
    Fluid1 : class
        The thermodynamic properties of the first fluid (e.g. liquid)
    Fluid2 : class
        The thermodynamic properties of the second fluid (e.g. gas)
    epsilon : float
        The void fraction of the two-phase flow

    Returns
    -------
    rho_M : float
        The averaged density of the two-phase flow in kg/m3
    mu_M : float
        The averaged dynamic viscosity of the two-phase flow in kg/m/s
    nu_M : float
        The averaged kinematic viscosity of the two-phase flow in m2/s
    Cp_M : float
        The averaged isobaric heat capacity of the two-phase flow in J/kgK
    lamda_M : float
        The averaged thermal conductivity of the two-phase flow in W/mK
    Pr_M : float
        The averaged Prandtl number of the two-phase flow

    """
    # theromdynamic properties of each phase
    rho_L    = Fluid1.rho 
    mu_L     = Fluid1.muu
    nu_L     = Fluid1.nuu
    Cp_L     = Fluid1.Cp 
    lamda_L  = Fluid1.k 
    Pr_L     = Fluid1.Pr
    rho_G    = Fluid2.rho 
    mu_G     = Fluid2.muu
    nu_G     = Fluid2.nuu
    Cp_G     = Fluid2.Cp 
    lamda_G  = Fluid2.k
    Pr_G     = Fluid2.Pr
    # Averaging
    if (epsilon > 0.):
        x = flow_quality(epsilon=epsilon, rho_L=Fluid1.rho, rho_G=Fluid2.rho, S=1.0)
        #print('x=', x)
        # mixture thermodynamic properties
        rho_M    = epsilon * rho_G    + (1.0 - epsilon) * rho_L 
        mu_M     = epsilon * mu_G     + (1.0 - epsilon) * mu_L 
        #nu_M     = x       * nu_G     + (1.0 - x)       * nu_L
        nu_M     = mu_M / rho_M
        Cp_M     = x       * Cp_G     + (1.0 - x)       * Cp_L 
        lamda_M  = x       * lamda_G  + (1.0 - x)       * lamda_L
        #lamda_M  = lamda_L
        #lamda_M  = epsilon * lamda_G  + (1.0 - epsilon) * lamda_L
        Pr_M     = mu_M * Cp_M / lamda_M 
        return rho_M, mu_M, nu_M, Cp_M, lamda_M, Pr_M, x
    else: 
        x = 0.0
        return rho_L, mu_L, nu_L, Cp_L, lamda_L, Pr_L, x


# Reynolds number function 
def ReynoldsNumber(U, d_h, nu):
    """
    Calculate the Reynolds number (Re) based on the flow velocity U (m/s), 
    hydraulic diameter d_h (m) and the kinematic viscosity nu (m^2/s)
    
    Parameters
    ----------
    U : float
        Flow velocity in m/s
    d_h : float
        Hydraulic diameter in m
    nu : float
        Kinematic viscosity in m^2/s
        
    Returns
    -------
    Re : float
        Reynolds number
    """
    Re = U * d_h / nu
    return Re






    

def get_void_fraction_model_constants(model, J=None):
    """
    Get the constants C_0 and U_gj for the specified model.

    Parameters:
    -----------
    model : str, optional
        Drift flux model name ("Margat", "Homogeneous", "Asano").
    J : float, optional
        Mixture velocity (J) for the Asano model.

    Returns:
    --------
    tuple
        (C_0, U_gj): Distribution parameter and drift velocity.
    """
    constants = {
        "Homogeneous": (1.0, 0.0),  # No slip between phases
        "Margat": (0.8831, 0.4296),
        "Asano": (1.1653, 1.229)  # Special handling for Asano (Asano_high)
    }

    if model not in constants:
        raise ValueError(f"Unknown model: {model}")

    if model == "Asano":
        # Determine constants based on J
        if J is None or J < 3.0:
            constants[model] = (1.4795, 0.2337)  # Asano_low
  

    return constants[model]


def U_SL_isoline(void_fraction_model, U_SG, alpha, tol=1e-6, max_iter=100):
    """
    Calculate \( U_{SL} \) directly given \( U_{SG} \), a target \( \alpha \), and model-specific constants.

    Parameters:
    -----------
    U_SG : float
        Superficial gas velocity (m/s).
    alpha : float
        Target void fraction (dimensionless).
    model : str, optional
        Drift flux model name ("Margat", "Asano", "Homogeneous"), default is "Margat".
    tol : float, optional
        Tolerance for iterative refinement, default is 1e-6.
    max_iter : int, optional
        Maximum iterations for convergence, default is 100.

    Returns:
    --------
    U_SL : float
        Superficial liquid velocity (m/s).
    """
    # one can directly use this formula
    # U_SL = U_SG * (1 / (alpha * C_0) - 1) - U_gj / C_0
    
    # Retrieve constants based on the model
    if void_fraction_model == "Asano":
        C_0_low, U_gj_low   = 1.4795, 0.2337  # Constants for J < 3
        C_0_high, U_gj_high = 1.1653, 1.229   # Constants for J >= 3

        # Initial estimate of J using low constants
        C_0, U_gj = C_0_low, U_gj_low
        J = (U_SG / alpha - U_gj) / C_0
        J = U_SG / alpha
        for _ in range(max_iter):
            # Check if J satisfies the condition for low/high constants
            if J < 3:
                C_0, U_gj = C_0_low, U_gj_low
            else:
                C_0, U_gj = C_0_high, U_gj_high
            
            #### Test Passoni paper 
            # we believe that the plot of isolines alpha = 0.5 and alpha = 0.56 are not correct
            #if alpha < 0.51:
                #C_0, U_gj = C_0_low, U_gj_low
            #else:
                #C_0, U_gj = C_0_high, U_gj_high
            ###

            # Update J based on current constants
            new_J = (U_SG / alpha - U_gj) / C_0

            # Check for convergence
            if abs(new_J - J) < tol:
                J = new_J
                break
            J = new_J
        else:
            raise ValueError("Iteration did not converge for Asano model.")

        # Calculate alpha_H and U_SL
        alpha_H = U_SG / J
        U_SL = (1.0 - alpha_H) * J

    else:
        # For other models, directly fetch constants
        C_0, U_gj = get_void_fraction_model_constants(void_fraction_model)
        J = (U_SG / alpha - U_gj) / C_0
        alpha_H = U_SG / J
        U_SL = (1.0 - alpha_H) * J

    return U_SL


def U_SL_fine_coarse_Bubbly(void_fraction_model,f_d_model, U_SG, d_H, liquid, gas, M=0.5, theta=25):
    """
    Computes U_SL as a solution of the given equation for the coarse-fine bubbly transition.
    
    Parameters:
        U_SG (float): Superficial velocity of the gas phase (m/s).
        d_H (float): Hydraulic diameter (m).
        liquid (Fluid): Liquid phase object with attributes `rho` (density) and `nu` (kinematic viscosity).
        gas (Fluid): Gas phase object with attributes `rho` (density) and `nu` (kinematic viscosity).
        M (float): Maximum bubble diameter factor (default is 0.5).
        theta (float): Temperature for sigma estimation (°C).
        
    Returns:
        float: Computed U_SL (m/s).
    """
    # Constants for darcy factor correlation f_D = A * Re^(-n) + B
    # will be substitute by the constants determined from investigation of water oxygen in PHE
    if (f_d_model == 'Passoni'):
        A = 217.94
        B = 3.99
        n = 0.74
        # C constant for Two-phase pressure gradient multiplier phi_TP = 1 + C / X + 1 / X^2
        C = 8.77   # Passoni 
        D = 1.0
    if (f_d_model == 'Guesmi'):
        A = 217.94
        B = 3.99
        n = 0.74
        # C constant for Two-phase pressure gradient multiplier phi_TP = 1 + C / X + 1 / X^2
        C = 8.77   # Passoni 
        D = 2.73   # Asano 
    # constants of drift flux model for void fraction 
    C_0 =  1.1653
    U_GJ = 1.2239
    C_0, U_GJ = get_void_fraction_model_constants(void_fraction_model, J=None)
    # Surface tension estimation (linear relationship with theta)
    sigma_0 = 75.6  # mN/m
    sigma = (sigma_0 - 0.14 * theta) * 1e-3  # Convert to N/m

    # Define the equation to solve for U_SL
    def equation(U_SL):
        # Reynolds numbers
        Re_SL = (U_SL * d_H) / liquid.nuu
        Re_SG = (U_SG * d_H) / gas.nuu
        
        # Friction factors
        f_D_L = A * Re_SL**(-n) + B
        f_D_G = A * Re_SG**(-n) + B
        
        # Lockhart-Martinelli parameter (X)
        X = sqrt((f_D_L / f_D_G) * (liquid.rho / gas.rho) * (U_SL / U_SG)**2)
        
        # Two-phase pressure gradient multiplier
        Phi_TP_squared = (1 + C * (1 / X) + 1 / X**2)
        
        # Single-phase liquid pressure gradient
        dP_dz_L = f_D_L * (liquid.rho * U_SL**2) / (2 * d_H)
        
        # Two-phase pressure gradient
        dP_dz_TP = Phi_TP_squared * dP_dz_L
        
        # Gas fraction (\( \alpha \))
        alpha = U_SG / (C_0 * (U_SL + U_SG) + U_GJ)
        
        # Turbulent dissipation rate (corrected)
        epsilon = dP_dz_TP * (U_SG + U_SL) / (liquid.rho * (1 - alpha) + gas.rho * alpha)
        
        # Calculate K
        K = 0.725 + 4.15 * sqrt(alpha)
        
        # Maximum bubble diameter
        d_max = K * (sigma / liquid.rho)**(3/5) * epsilon**(-2/5)
        
        # Return the target equation
        return (d_max - M * d_H)
    # solve equation for U_SL 
    U_SL_solution = fsolve(equation, U_SG * 0.5)  # Initial guess: half of U_SG
    return U_SL_solution[0]

def is_d_max_greater(void_fraction_model, f_d_model , U_SL, U_SG, d_H, liquid, gas, M=0.5, theta=25):
    """
    Determines if d_max > M * d_H for given U_SL and U_SG.
    
    Parameters:
        U_SL (float): Superficial velocity of the liquid phase (m/s).
        U_SG (float): Superficial velocity of the gas phase (m/s).
        d_H (float): Hydraulic diameter (m).
        liquid (Fluid): Liquid phase object with attributes `rho` (density) and `nu` (kinematic viscosity).
        gas (Fluid): Gas phase object with attributes `rho` (density) and `nu` (kinematic viscosity).
        M (float): Maximum bubble diameter factor (default is 0.5).
        theta (float): Temperature for sigma estimation (°C).
        
    Returns:
        bool: True if d_max > M * d_H, otherwise False.
    """
    # Constants for darcy factor correlation f_D = A * Re^(-n) + B
    # will be substitute by the constants determined from investigation of water oxygen in PHE
    if (f_d_model == 'Passoni'):
        A = 217.94
        B = 3.99
        n = 0.74
        # C constant for Two-phase pressure gradient multiplier phi_TP = 1 + C / X + 1 / X^2
        C = 8.77   # Passoni 
        D = 1.0
    if (f_d_model == 'Guesmi'):
        A = 217.94
        B = 3.99
        n = 0.74
        # C constant for Two-phase pressure gradient multiplier phi_TP = 1 + C / X + 1 / X^2
        C = 8.77   # Passoni 
        D = 2.73   # Asano 
    # constants of drift flux model for void fraction 
    C_0 =  1.1653
    U_GJ = 1.2239
    C_0, U_GJ = get_void_fraction_model_constants(void_fraction_model, J=None)
    # Surface tension estimation (linear relationship with theta)
    sigma_0 = 75.6  # mN/m
    sigma = (sigma_0 - 0.14 * theta) * 1e-3  # Convert to N/m
    
    # Reynolds numbers
    Re_SL = (U_SL * d_H) / liquid.nuu
    Re_SG = (U_SG * d_H) / gas.nuu
    
    # Friction factors
    f_D_L = A * Re_SL**(-n) + B
    f_D_G = A * Re_SG**(-n) + B
    
    # Lockhart-Martinelli parameter (X)
    X = sqrt((f_D_L / f_D_G) * (liquid.rho / gas.rho) * (U_SL / U_SG)**2)
    
    # Two-phase pressure gradient multiplier
    Phi_TP_squared = (1 + C * (1 / X) + D / X**2)
    
    # Single-phase liquid pressure gradient
    dP_dz_L = f_D_L * (liquid.rho * U_SL**2) / (2 * d_H)
    
    # Two-phase pressure gradient
    dP_dz_TP = Phi_TP_squared * dP_dz_L
    
    # Gas fraction (\( \alpha \))
    alpha = U_SG / (C_0 * (U_SL + U_SG) + U_GJ)
    
    # Turbulent dissipation rate
    epsilon = dP_dz_TP * (U_SG + U_SL) / (liquid.rho * (1 - alpha) + gas.rho * alpha)
    
    # Calculate K
    K = 0.725 + 4.15 * sqrt(alpha)
    
    # Maximum bubble diameter
    d_max = K * (sigma / liquid.rho)**(3/5) * epsilon**(-2/5)
    
    # Compare d_max with M * d_H and return result
    return d_max >= M * d_H
    
# function to compute gas fraction using drift flux model 
def void_fraction(model, U_SG, U_SL):
    J = U_SG + U_SL
    C_0, U_gj = get_void_fraction_model_constants(model, J)
    return U_SG / (C_0 * J + U_gj) 

# function to determine flow pattern 
def classify_flow_pattern(void_fraction_model,f_d_model, U_SG, U_SL, liquid, gas, d_H, M, theta):
    """
    Classify the flow pattern based on superficial velocities and transition criteria.

    Parameters:
    -----------
    U_SG : float
        Superficial gas velocity (m/s).
    U_SL : float
        Superficial liquid velocity (m/s).
    liquid : Fluid
        Liquid phase properties (e.g., water).
    gas : Fluid
        Gas phase properties (e.g., air).
    d_H : float
        Hydraulic diameter (m).
    M : float
        Transition parameter for coarse/fine bubbly flow.
    theta : float
        Angle for the flow condition (degrees).

    Returns:
    --------
    str
        Identified flow pattern.
    """
    # Calculate void fraction
    try:
        alpha = void_fraction(void_fraction_model, U_SG, U_SL)
    except Exception as e:
        raise ValueError(f"Error calculating void fraction: {e}")

    # Determine the flow pattern based on the table logic
    if alpha > 0.56 :
        return "Film Flow"
    elif 0.5 < alpha <= 0.56:
        # Partial Film Flow to Film Flow Transition
        return "Partial Film Flow"
    elif 0.25 < alpha <= 0.50:
        # Heterogeneous Flow to Partial Film Flow Transition
        return "Heterogeneous Flow"
    elif alpha <= 0.10: 
        fine_coarse=is_d_max_greater(void_fraction_model,f_d_model, U_SL, U_SG, d_H, liquid, gas, M, theta)
        # Coarse Bubbly to Fine Bubbly Transition
        if not fine_coarse:
            return "Fine Bubbly"
        elif alpha <= 0.06:
            return "Coarse Bubbly"

    else: 
        return "Taylor-like Bubbly"




# plot functions for transition lines 
def plot_isolines(void_fraction_model,f_d_model, d_H, liquid, gas, M, theta, add_legend=True):
    """
    Plot isolines of void fraction for a given drift flux model.

    Parameters:
    -----------
    model : str, optional
        Drift flux model name ("Margat", "Asano_low", "Asano_high", "Homogeneous", "Passoni"), default is "Margat".
    add_legend : bool, optional
        Whether to add a legend to the plot.
    """
    # Get model constants
    C_0, U_gj = get_void_fraction_model_constants(void_fraction_model, J=None)
    #print(C_0, U_gj)
    # Define color map for models
    model_colors = {
        "Margat": "blue",
        "Asano": "green",
        "Homogeneous": "purple",
    }

    if void_fraction_model not in model_colors:
        raise ValueError(f"Unknown model: {void_fraction_model}")

    color = model_colors[void_fraction_model]  # Get color for the selected model

    # Define U_SG range and alpha values
    U_SG_values = np.logspace(-3, 1, 1000)
    alpha_values = [0.06, 0.1, 0.25, 0.5, 0.56]

    # Store results for isolines
    isolines = {}
    for alpha in alpha_values:
        U_SL_list = []
        U_SG_list = []
        if alpha in [0.06, 0.1]:
            for U_SG in U_SG_values:
                try:
                    # compute U_SL 
                    U_SL = U_SL_isoline(void_fraction_model, U_SG, alpha, tol=1e-6, max_iter=100)
                    if (U_SL > 0.0):
                        # Check the is_d_max_greater condition
                        condition = is_d_max_greater(void_fraction_model,f_d_model, U_SL, U_SG, d_H, liquid, gas, M=M, theta=theta)
                        #print(f"alpha={alpha}, U_SG={U_SG}, condition={condition}")
                        if (alpha == 0.06 and condition) or (alpha == 0.1 and  not condition):
                            # Store U_SG and U_SL if conditions are met
                            U_SG_list.append(U_SG)
                            U_SL_list.append(U_SL)
                        else:
                            #print(f"Warning: Negative U_SL computed for alpha={alpha}, U_SG={U_SG}")
                            # Break the loop if the condition is not satisfied only for alpha = 0.06
                            if (alpha == 0.06):
                                break
                except Exception as e:
                    # If U_SL computation fails, skip this U_SG
                    print(f"U_SL_isoline failed for alpha={alpha}, U_SG={U_SG}: {e}")
                    continue
        
        else:
            for U_SG in U_SG_values:
                try:
                    # compute U_SL 
                    U_SL = U_SL_isoline(void_fraction_model, U_SG, alpha, tol=1e-6, max_iter=100)
                    if (U_SL >= 0.):
                        U_SG_list.append(U_SG)
                        U_SL_list.append(U_SL)
                    else:
                        True  # just to keep print text here
                        #print(f"Warning: Negative U_SL computed for alpha={alpha}, U_SG={U_SG}")
                    
                except:
                    # If U_SL computation fails, skip this U_SG
                    print(f"Error for alpha={alpha}, U_SG={U_SG}, U_SL={U_SL}")
                    continue
            
        # Store the isoline for this alpha
        isolines[alpha] = (U_SG_list, U_SL_list)
        print(len(U_SG_list), len(U_SL_list))
    # Plot isolines
    k = -1
    for alpha, isoline in isolines.items():
        U_SG_list, U_SL_list = isoline
        k = k + 1
        if (k==0):
            if (deutsch):
                plt.plot(U_SG_list, U_SL_list, ls="--", color=color, label=f'Isolinien mit {void_fraction_model.capitalize()} Modell' if add_legend else None)
            else:
                plt.plot(U_SG_list, U_SL_list, ls="--", color=color, label=f'Isolines with {void_fraction_model.capitalize()} model' if add_legend else None)
        else:
            plt.plot(U_SG_list, U_SL_list,ls="--",  color=color)

      
        # Add the value of alpha as a label in the middle of the isoline
        if (alpha==0.1):
            midpoint_value = 0.6
        else: 
            midpoint_value = 0.03

         
        U_SL_array = np.array(U_SL_list)

        # Compute the closest index to the midpoint
        closest_index = np.argmin(np.abs(U_SL_array - midpoint_value))
        U_SG_midpoint = U_SG_list[closest_index]
        U_SL_midpoint = U_SL_list[closest_index]

        # Add label at the midpoint
        plt.text(U_SG_midpoint, U_SL_midpoint, r'$\alpha = {:.2f}$'.format(alpha),
                fontsize=12, color=color, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))
    
    
    
def plot_transition_fine_coarse(void_fraction_model,f_d_model, d_H, liquid, gas, M, theta, alpha_threshold):
    """
    Plots the transition line between fine and coarse bubbly flow.

    Parameters:
    -----------
    model: string 
        drift flux model for void fraction ("Margat" or "Asano") 
    d_H : float
        Hydraulic diameter (m).
    liquid : Fluid
        Liquid fluid object.
    gas : Fluid
        Gas fluid object.
    M : float
        Transition parameter.
    theta : float
        Inclination angle (degrees).
    alpha_threshold : float, optional
        Void fraction threshold for the transition (default is 0.1).

    Returns:
    --------
    None
    """
    # Define the range for U_SG (superficial gas velocity)
    U_SG_range = np.logspace(-3, 1, 1000)

    # Initialize lists for U_SG and U_SL
    U_SG_fine_coarse = []
    U_SL_fine_coarse = []

    # Compute the transition line
    for U_SG in U_SG_range:
        try:
            # Compute U_SL for the current U_SG
            U_SL = U_SL_fine_coarse_Bubbly(void_fraction_model, f_d_model, U_SG, d_H, liquid, gas, M=M, theta=theta)

            # Calculate J (mixture velocity) and retrieve model constants
            J = U_SG + U_SL
            C_0, U_gj = get_void_fraction_model_constants(void_fraction_model, J=J)

            # Compute the void fraction alpha
            alpha = U_SG / (C_0 * (U_SG + U_SL) + U_gj)

            # Check alpha condition
            if alpha <= alpha_threshold:
                U_SG_fine_coarse.append(U_SG)
                U_SL_fine_coarse.append(U_SL)
            else:
                # Stop if alpha exceeds the threshold
                break
        except Exception as e:
            # Handle errors and append NaN
            U_SL_fine_coarse.append(np.nan)
            U_SG_fine_coarse.append(U_SG)

    # Plot the transition line
    if (deutsch ==True):
        plt.plot(U_SG_fine_coarse, U_SL_fine_coarse, color='orange', linestyle='--', linewidth=2, label=rf"Transition feine - grobe Blasen (M = {M})")
    else:
        plt.plot(U_SG_fine_coarse, U_SL_fine_coarse, color='orange', linestyle='--', linewidth=2, label=rf"Transition fine coarse bubbly (M = {M})")
