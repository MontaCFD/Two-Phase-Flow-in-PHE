#! /usr/bin/env python3
'''===============================================================================================================
 - Author: M. Guesmi
 - Adress: Chair of process engineering TU Dresden, George-Bähr-Straße 3b, 01069 Dresden, Germany
 - Summary: Two-Phase Flow and Heat Transfer
    > cell model to estimate the heat transfer of two phase bubbly flow
    > pipe flow --> semi-analytical solution based on empirical correlations 
=================================================================================================================='''
import numpy as np
from numpy import sin, cos, tan, exp, log, log10, sqrt, arccos, arcsin, radians
from numpy import pi, e
import sys, os
import csv # print data to file
import CoolProp.CoolProp as cp
from scipy.optimize import fsolve

#TODO 
'''
    > class for the geometry of the plate. ---> Done :) 
    > Two phase class 
    > function to predict flow form 
    > function to determine friction factor ---> Done :) 
    > function for heat transfer ---> Done :)
    > TODO: correlations for pressure drop coeff. and Nu from the same source should have the same index (done)
    > TODO: call a function that list the different correlations, so an overview can be output in the terminal (done)
    > TODO: function to determine list of correlations depending on input scenario --> done
'''

# > function to define an array with startpoint, endpoint and step
def linspace(start, stop, step=1.):
    """
    A function to generate a sequence of numbers over a specified range
    with a specified step.

    Parameters
    ----------
    start : float
        The start of the sequence.
    stop : float
        The end of the sequence.
    step : float, optional
        The step size. Defaults to 1.

    Returns
    -------
        A sequence of numbers from start to stop with the specified step.

    Notes
    -----
    This function is inclusive of the stop value, i.e. the sequence will
    include the stop value.

    Examples
    --------
    >>> linspace(1, 3, 0.5)
    array([1. , 1.5, 2. , 2.5, 3. ])
    """
    return np.linspace(start, stop, int((stop - start) / step + 1))

# > Define the parametrized system of energy equations
def energy_equations(vars, *params):
    
    """
    A function to calculate the energy equations for a plate heat exchanger.

    Parameters
    ----------
    vars : array_like
        The variables for which the energy equations are to be solved. The
        variables are the temperatures of the hot and cold sides, i.e. T_OH
        and T_OC.
    *params : tuple
        The parameters are:
            - m_h (kg/s): mass flow rate of the hot side
            - Cp_h (J/(kg·K)): heat capacity of the hot side
            - T_IH (K): inlet temperature of the hot side
            - m_c (kg/s): mass flow rate of the cold side
            - Cp_c (J/(kg·K)): heat capacity of the cold side
            - T_IC (K): inlet temperature of the cold side
            - k (W/(m²·K)): heat transfer coefficient
            - A (m²): heat transfer area

    Returns
    -------
        The energy equations for the plate heat exchanger.
        - The first element is the energy equation for the hot side. 
        - the second element is the energy equation for the cold side. 


    """
    T_OH, T_OC = vars
    m_h, Cp_h,T_IH, m_c, Cp_c, T_IC,  k, A  = params
    eq1 = m_h * Cp_h * (T_OH - T_IH) * log((T_IH - T_OC) / (T_OH - T_IC)) + k * A * ((T_IH - T_OC) - (T_OH - T_IC)) 
    eq2 = m_c * Cp_c * (T_OC - T_IC) * log((T_IH - T_OC) / (T_OH - T_IC)) - k * A * ((T_IH - T_OC) - (T_OH - T_IC))
    
    return [eq1, eq2]


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

class Forced(Fluid):
    pass

class Free(Fluid):
    pass

# area enlargement factor 
def area_enlargement(X):
    """
        Calculates the area enlargement factor using the three-point integration formula.

        Parameters
        ----------
        X : float
            X = 2π.λ/δ

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

#print(area_enlargement(pi/2))

# > class for plate geometry
class PHE_geometry(object):
    """
    Class to calculate the hydraulic diameter and heat transfer area of a plate heat exchanger.

    Attributes
    ----------
    δ : float
        Plate depth/ corrugation amplitude in mm
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

    def __init__(self, amplitude: float, wave_length: float, thickness: float, corrugation: float, width: float, length:float, roughness: float, Lambda: float) -> None:
        self.amplitude      = amplitude / 1000 #mm to m
        self.depth          = 2 * amplitude / 1000 #mm to m
        self.wave_length    = wave_length / 1000 #mm to m
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
        D_h = 2.0 * self.depth / phi
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
        A_inlet =self.depth * self.width
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




class pressure_coefficient_Estimator:
    '''
    class that contains different correlations to estimate friction coefficient PHE
    '''
    def __init__(self, beta, Re_h, amplitude, wave_length):
        """
        Initialize the estimator for friction coefficient xsi of a PHE

        Parameters
        ----------
        β : float
            Chevron angle in degrees
        δ : float
            Plate depth / corrugation amplitude in mm
        λ : float
            Chevron pitch - wavelength in mm
        Re_h : float
            Reynolds number based on hydraulic diameter
        """
        self.beta = beta
        self.Re_h = Re_h
        self.amplitude = amplitude
        self.wave_length = wave_length

    # function to print all correlations
    def list_correlation_methods(self):
        print("Available correlation methods for xsi:")
        print("1. Martin Holger (1995): https://doi.org/10.1016/0255-2701(95)04129-X")
        print("2. Al zahrani, Islam und Saha: (2019): https://doi.org/10.1016/j.egypro.2019.02.211")
        print("3. Muley and Manglik (1999): https://sci-hub.hkvisa.net/10.1115/1.2825923")
        print("4. Lee and Lee (2015): https://doi.org/10.1016/j.applthermaleng.2015.05.080")
        print("5. Focke, Zachariades and Olivier (1985): https://doi.org/10.1016/0017-9310(85)90249-2")
        print("6. Zhu and Haglind (2020): https://doi.org/10.1016/j.ijheatmasstransfer.2020.120370")
        print("7. Buscher (2021): https://doi.org/10.1016/j.expthermflusci.2021.110380")
        print("8. Heavner (1993): https://www.tandfonline.com/doi/epdf/10.1080/01457630304056?src=getftr")
        print("9. Wanniarachchi (1995): https://link.springer.com/article/10.1007/s12206-017-0454-0")
        print("9. Kim and Park (2016): https://link.springer.com/article/10.1007/s12206-017-0454-0")


    # Ansatz I: Friction coefficient from a theoretical approach for single phase flow (paper of Martin Holger)
    def friction_coeff_I(self):
        '''
        This function calculates the Darcy friction coefficient
                Args:
                        chevron angle beta
                        Reynolds number Re_h
                Returns:
                        Friction coefficient
        '''
        beta = self.beta 
        Re_h = self.Re_h
        # constants
        B0 = 64.
        B1 = 597.
        C1 = 3.85
        K1 = 39.
        n = 0.289

        if (Re_h < 2000.) and 0 <= beta <= 80:
            xsi_0 = B0 / Re_h
            xsi_1 = B1 / Re_h + C1
        elif 0 <= beta <= 80:
            xsi_0 = 1. / (1.8 * log10(Re_h) - 1.5) ** 2.0
            xsi_1 = K1 / (Re_h ** n)

        beta_radians = radians(beta)
        a = 3.8
        b = 0.18
        c = 0.36
        xsi = 1. / (
                cos(beta_radians) / sqrt(b * tan(beta_radians)
                                         + c * sin(beta_radians)
                                         + xsi_0 / cos(beta_radians))
                + (1 - cos(beta_radians)) / (sqrt(a * xsi_1))
        ) ** 2.0

        return xsi

    # Ansatz II: Al Zahrani et al.
    def friction_coeff_II(self):
        '''
        Al Zahrani et al. Correlation for Fanning friction coefficient
        > A thermo-hydraulic characteristics investigation in corrugated plate heat exchanger (beta = 60°/60°, 500 < Re < 3000, 0.72 < Pr < 7.5)
        > https://doi.org/10.1016/j.egypro.2019.02.211
        '''
        beta = self.beta 
        Re_h = self.Re_h
        
        if beta == 60 and 500 < Re_h < 3000:
            f = 1/4 * 2.15 * Re_h**(-0.1342)
            return  f
        
    
    
    # Ansatz III: Muley and Manglik (1999) 
    def friction_coeff_III(self):
        '''
        This function calculates the Fanning friction coefficient using a correlation developed by Muley and Manglik (1999)
        > Experimental Study of Turbulent Flow Heat Transfer and Pressure Drop in a Plate Heat Exchanger with Chevron Plates
        > https://sci-hub.hkvisa.net/10.1115/1.2825923
        '''
        beta = self.beta 
        Re_h = self.Re_h

        # corrugation parameter
        X = pi * self.amplitude / self.wave_length
        # area enlargement factor using three-point integration formula
        phi = area_enlargement(X)

        if 30 <= beta <= 60 and Re_h >= 1000:
            f = 1/4 * (2.917 - 0.1277 * beta + 2.016 * 10**(-3) * beta ** 2) * (5.474 - 19.02 * phi + 18.93 * phi**2 - 5.341 * phi**3) * Re_h**(-(0.2 + 0.0577*sin(pi * beta/45 + 2.1)))
            return  f

    # Ansatz IV: Lee and Lee (2015)  
    def friction_coeff_IV(self):
        '''
        This function calculates the Fanning friction coefficient based on correlation developed by Lee et al. (2015)
        > Friction and Colburn factor correlations and shape optimization of chevron-type Plate Heat exchangers
        > https://doi.org/10.1016/j.applthermaleng.2015.05.080
        > β = 15, 30, 45, 60, 75; for each β the values p/h = 2.0, 2.8, 3.6, 4.4 
            Args:
                chevron angle beta
                Reynolds number Re_h
            Returns:
                Friction coefficient
        '''
        beta = self.beta 
        Re_h = self.Re_h
        
        z = 2.0  # = p/h --> Anpassen   ##TODO:verify
        #print('test')
        if 15 <= beta <= 75 and 2.0 <= z < 4.4 and 200 < Re_h < 10000:
            f = 1/4 * 5.9666 * ( pi / 180 * beta) ** (- 0.0767 * log(Re_h) + 1.5858) * z ** (-(1.283 * 10**(-5) * Re_h + 0.8886)) * Re_h ** (-0.1718)
            #print( f)
            return f
        
    # Ansatz V: Focke, Zachariades and Olivier (1985)
    def friction_coeff_V(self):
        '''
        This function calculates the Darcy friction coefficient base on correlation developed by Focke et al. (1985)
        > The effect of the corrugation inclination angle on the thermohydraulic performance of plate heat exchangers
        > https://doi.org/10.1016/0017-9310(85)90249-2
            Args:
                chevron angle beta
                Reynolds number Re_h
            Returns:
                Friction coefficient
        '''
        beta = self.beta 
        Re_h = self.Re_h
        
        if beta == 0 and Re_h < 2300.:
            f = 1/4 * 114.4 / Re_h
            return f

        if beta == 0 and 2300 < Re_h < 56000:
            f = 1/4 * 0.552 * Re_h**(-0.263)
            return f

        if beta == 30 and 260 < Re_h < 3000:
            f = 1/4 * (0.37 + 230/Re_h)
            return f

        if beta == 30 and 3000 <= Re_h < 50000:
            f = 1/4 * 3.59 * Re_h**(-0.263)
            return f

        if beta == 45 and 150 < Re_h < 1800:
            f = 1/4 * (1.21 + 367/Re_h)
            return f

        if beta == 45 and 1800 <= Re_h < 30000:
            f = 1/4 * 5.84 * Re_h**(-0.177)
            return f

        if beta == 60 and 90 < Re_h < 400:
            f = 1/4 * (5.03 + 755/Re_h)
            return f

        if beta == 60 and 400 <= Re_h < 16000:
            f = 1/4 * (26.8*Re_h**(-0.209))
            return f

        if beta == 72 and 110 < Re_h < 500:
            f = 1/4 * (19.0 + 764/Re_h)
            return f

        if beta == 72 and 500 <= Re_h < 12000:
            f = 1/4 * 132 * Re_h**(-0.296)
            return f

        if beta == 80 and 130 < Re_h < 3700:
            f = 1/4 * 140 * Re_h**(-0.28)
            return f

        if beta == 90 and 200 < Re_h < 3000:
            f = 1/4 * (5.63 + 1280/Re_h)
            return f

        if beta == 90 and 3000 <= Re_h < 16000:
            f = 1/4 * 63.8*Re_h**(-0.289)
            return f

    # Ansatz VI: Zhu and Haglind 2020
    def friction_coeff_VI(self):
        '''
        This function calculates the Darcy friction coefficient based on LES-derived correlation by Zhu and Haglind 
        > paper https://doi.org/10.1016/j.ijheatmasstransfer.2020.120370
                   Args:
                            chevron angle beta
                            Reynolds number Re_h
                   Returns:
                            Friction coefficient
        '''
        beta = self.beta 
        Re_h = self.Re_h
        xsi = None
        
        # constants from 
        def colebrook(Re, beta):
            beta_rad = radians(beta)
            a1 = 1.48 * sin(beta_rad) ** 4.85 * cos(beta_rad) ** 0.45
            a2 = 60.0 * sin(2 * beta_rad) ** 3 * cos(beta_rad) ** 5 + 16.0
            # error
            err = 1.0
            # convergence criterium
            tol = 10 ** (-5)
            # start value for a good choice
            x0 = -2.0 * log10(a1)
            # fixed point iteration method to determine the factor
            while (err > tol):
                x = (- 2.0 * log10(a1 + a2 / (Re * sqrt(x0)))) ** (-2)
                err = abs(x - x0)
                x0 = x
            return x

        # constants C and n
        def coeff(beta):
            beta_rad = radians(beta)
            x = sin(beta_rad)
            C = exp(1.13 * x ** 2.5 + 4.13)
            n = 0.43 * x ** 5.0 - 0.92
            return C, n

        beta_radians = radians(beta)
        Re_crit = 954.0 * (cos(beta_radians)) ** 4.0 + 53.0

        if (Re_h < Re_crit) and (18 <= beta <= 72) and (10 <= Re_h <= 6000):
            C, n = coeff(beta)
            xsi = 1/4 * C * Re_h ** n
        elif 18 <= beta <= 72 and 10 <= Re_h <= 6000:
            xsi = 1/4 * colebrook(Re_h, beta)
        return xsi

    
    # Ansatz VII: Friction for two phase flow for Re > 100 (Paper von Susanne Buscher (2021))
    def friction_coeff_VII(self):
        '''
        This function calculates the Darcy friction coefficient based on correlation of S. Buschner 
        > https://doi.org/10.1016/j.expthermflusci.2021.110380
        > β = 75°, 100 < Re < 2000
            Args:
                chevron angle beta
                Reynolds number Re_h
            Returns:
                Friction coefficient
        '''
        beta = self.beta
        Re_h = self.Re_h
        # constants
        
        C = 19.9
        n = - 0.225

        xsi = None
        if Re_h >= 100. and beta == 75:
                xsi = C * Re_h ** n

        return xsi

    def friction_coeff_VIII(self):
        '''
        This function calculates the Darcy friction coefficient based on correlation of Heavner et al. (1993)
        > https://www.tandfonline.com/doi/epdf/10.1080/01457630304056?src=getftr
        > 400 < Re < 10000
            Args:
                chevron angle beta
                Reynolds number Re_h
            Returns:
                Friction coefficient
        '''
        beta = self.beta
        Re_h = self.Re_h

        # corrugation parameter
        X = pi * self.amplitude / self.wave_length
        # area enlargement factor using three-point integration formula
        phi = area_enlargement(X)

        xsi = None

        if beta == 67 and 400 < Re_h < 10000:

            C = 0.490
            p = 0.1814

            xsi = 1/4 * C * phi ** (p+1) * Re_h ** (-p)

            return xsi

        if beta == 56 and 400 < Re_h < 10000:
            C = 0.545
            p = 0.1555

            xsi = 1/4 * C * phi ** (p+1) * Re_h ** (-p)

            return xsi

        if beta == 45 and 400 < Re_h < 10000:
            C = 0.687
            p = 0.1405

            xsi = 1/4 * C * phi ** (p+1) * Re_h ** (-p)

            return xsi

        if beta == 33.5 and 400 < Re_h < 10000:
            C = 1.441
            p = 0.1353

            xsi = 1/4 * C * phi ** (p+1) * Re_h ** (-p)

            return xsi

        if beta == 22.5 and 400 < Re_h < 10000:
            C = 1.458
            p = 0.0838

            xsi = 1/4 * C * phi ** (p+1) * Re_h ** (-p)

            return xsi

    def friction_coeff_IX(self):
        '''
        This function calculates the Darcy friction coefficient based on correlation of Wanniarachchi et al. (1995)
        > https://link.springer.com/article/10.1007/s12206-017-0454-0
        > 1 < Re < 10000, 23 <= beta <= 67.5
            Args:
                chevron angle beta
                Reynolds number Re_h
            Returns:
                Friction coefficient
        '''
        beta = self.beta
        Re_h = self.Re_h

        if 20 <= beta <= 62 and 1 <= Re_h <= 10000:

            # corrugation parameter
            X = pi * self.amplitude / self.wave_length
            # area enlargement factor using three-point integration formula
            phi = area_enlargement(X)

            teta = 90 - beta
            p = 0.00423 * teta + 0.0000223 * teta**2

            c1 = 1774*teta**(-1.026)*phi**2*Re_h**(-1)
            c2 = 46.6*teta**(-1.08)*phi**(1+p)*Re_h**(-p)

            xsi = 1/4 * (c1**3 + c2**3)**(1/3)

            return xsi

    def friction_coeff_X(self):
        '''
        This function calculates the Darcy friction coefficient based on correlation of Kim and Park (2016)
        > https://link.springer.com/article/10.1007/s12206-017-0454-0
        > 950 < Re < 1400, beta = 65
        Args:
            chevron angle beta
            Reynolds number Re_h
        Returns:
            Friction coefficient
        '''

        beta = self.beta
        Re_h = self.Re_h

        if 950 < Re_h < 1400 and beta == 65:

            # corrugation parameter
            X = pi * self.amplitude / self.wave_length
            # area enlargement factor using three-point integration formula
            phi = area_enlargement(X)

            xsi = 1/4 * phi**4 * (0.6796*phi*Re_h**(-0.0551)+0.2)

            return xsi

    def select_method(self, index):
        '''
        This function selects the ansatz function based on the given index
                Args:
                        index: an integer representing the ansatz function to be used
                Returns:
                        Friction coefficient calculated by the selected ansatz function
        '''
        if index == 0:
            return self.friction_coeff_I()
        elif index == 1:
            return self.friction_coeff_II()
        elif index == 2:
            return self.friction_coeff_III()
        elif index == 3:
            return self.friction_coeff_IV()
        elif index == 4:
            return self.friction_coeff_V()
        elif index == 5:
            return self.friction_coeff_VI()
        elif index == 6:
            return self.friction_coeff_VII()
        elif index == 7:
            return self.friction_coeff_VIII()
        elif index == 8:
            return self.friction_coeff_IX()
        elif index == 9:
            return self.friction_coeff_X()
        else:
            raise ValueError("Invalid index! Please a value between 0 and 8")

    # list of indexes of the applicable methods
    def applicable_methods(self):
        '''
        This function checks the conditions for each correlation method and returns a list of indexes of the applicable methods
                Args:
                        beta: chevron angle
                        Pr: Prandtl number
                        Re_h: Reynolds number
                Returns:
                        List of indexes of the applicable methods
        '''
        Re_h = self.Re_h
        beta = self.beta

        applicable_indices = []

        # Check conditions for each method and add index to list if conditions are met
        if  0 <= beta <= 80:
            applicable_indices.append(0)
        if beta == 60 and 500 < Re_h < 3000:
            applicable_indices.append(1)
        if 30 <= beta <= 60 and Re_h >= 1000:
            applicable_indices.append(2)
        if 15 <= beta <= 75 and 200 < Re_h and Re_h < 10000:
            applicable_indices.append(3)
        if beta in [0, 15, 30, 45, 60, 72, 80, 90] and Re_h > 100:
            applicable_indices.append(4)
        if 18 <= beta and beta <= 72 and 10 <= Re_h <= 6000:
            applicable_indices.append(5)
        if Re_h >= 100. and beta == 75:
            applicable_indices.append(6)
        if beta in [22.5, 33.5, 45, 56, 67] and 400 <= Re_h <= 10000:
            applicable_indices.append(7)
        if 20 <= beta <= 62 and 1 <= Re_h <= 10000:
            applicable_indices.append(8)
        if beta == 65 and 950 < Re_h < 1400:
            applicable_indices.append(9)

        return applicable_indices
    
    def get_correlation_name(self, index: int) -> str:
        """
        Returns the name of the correlation method based on the given index.

        Args:
            index (int): Index of the correlation method

        Returns:
            str: Name of the correlation method
        """
        correlations = {
            0: "Martin Holger (1995)",
            1: "Al zahrani, Islam and Saha (2019)",
            2: "Muley and Manglik (1999)",
            3: "Lee and Lee (2015)",
            4: "Focke, Zachariades and Olivier (1985)",
            5: "Zhu and Haglind (2020)",
            6: "Buscher (2021)",
            7: "Heavner (1993)",
            8: "Wanniarachchi (1995)",
            9: "Kim and Park (2016)"
        }
        # Return the name of the correlation method or an error message if the index is out of range
        return correlations.get(index, "Invalid index! Please enter a value between 0 and 8")

# class for the nusselt correlations developed in the literature
class Nusselt_estimator:
    def __init__(self, xsi, Re_h, beta, Pr):
        """
        Constructor for Nusselt_estimator class

        Parameters:
            xsi (float): dimensionless distance along the plate
            Re_h (float): Reynolds number
            beta (float): chevron angle in degrees
            Pr (float): Prandtl number
        """
        self.xsi  = xsi  # dimensionless distance along the plate
        self.Re_h = Re_h # Reynolds number
        self.beta = beta # chevron angle in degrees
        self.Pr   = Pr   # Prandtl number

    
    # function to print all correlations
    def list_correlation_methods(self):
        print("Available correlation methods for Nu:")
        print("1. Martin Holger (1995): https://doi.org/10.1016/0255-2701(95)04129-X")
        print("2. Al zahrani, Islam und Saha: (2019): https://doi.org/10.1016/j.egypro.2019.02.211")
        print("3. Muley and Manglik (1999): https://sci-hub.hkvisa.net/10.1115/1.2825923")
        print("4. Lee and Lee (2015): https://doi.org/10.1016/j.applthermaleng.2015.05.080")
        print("5. Asif, Aftab, Syed, Ali and Muizz: https://doi.org/10.18280/ijht.350127")
    
            
    # Nusselt correlation based on Leveque equation Martin Holger 
    def nusselt_I(self):
        '''
        > This function gives an estimation of the Nusselt number developed by Martin Holger 1995 based on Leveque equation 
        > Paper: https://doi.org/10.1016/0255-2701(95)04129-X 
            Args:
                    chevron angle beta 
                    Reynolds number Re_h
            Returns:
                    Friction coefficient 
        '''
        xsi  = self.xsi   
        Re_h = self.Re_h 
        beta = self.beta 
        Pr   = self.Pr  
        #TODO 
        # those coefficients are from which source?
        cq = 0.122
        q  = 0.374 
        # Martin Holger 
        cq = 0.122
        q  = 0.374
        Nu = cq * (Pr ** (1/3)) * (xsi * Re_h ** 2.0 * sin(radians(2 * beta)) ) ** q #eq.(28) 
        return Nu
    
    

    # Al Zahrani et al.
    def nusselt_II(self):
        '''
        > This function estimates the Nusselt number. The correlation is developed by Al zahrani, Islam und Saha
        > Paper: A thermo-hydraulic characteristics investigation in corrugated plate heat exchanger https://doi.org/10.1016/j.egypro.2019.02.211
            Args:
                    chevron angle beta
                    Reynolds number Re_h
            Returns:
                    Nusselt number
        '''
        mu   = 1   #(Annahme, da keine genauere Info)
        mu_w = 1 #(Annahme, da keine genauere Info)
        
        xsi  = self.xsi   
        Re_h = self.Re_h 
        beta = self.beta 
        Pr   = self.Pr 
        
        if beta == 60 and 500 < Re_h < 3000 and 0.72 < Pr < 7.5:
            Nu = 0.238 * Re_h ** (0.6417) * Pr ** (1/3) * (mu/mu_w) ** (0.14)
            return Nu

    

    # Muley and Manglik (1999)

    def nusselt_III(self):
        '''
            > Experimental Study of Turbulent Flow Heat Transfer and Pressure Drop in a Plate Heat Exchanger with Chevron Plates
            > https://sci-hub.hkvisa.net/10.1115/1.2825923
        '''
        my = 1 #Annahme
        my_w = 1 #Annahme
        
        xsi  = self.xsi   
        Re_h = self.Re_h 
        beta = self.beta 
        Pr   = self.Pr 
        
        if 30 <= beta <= 60 and Re_h >= 1000:
            Nu = (0.2668 - 0.006967 * beta + 7.244 * 10 ** (-5) * beta ** 2) * Re_h ** (0.728+0.0543 * sin(pi * beta/45 + 3.7)) * Pr**(1/3) * (my/my_w)**0.14
            return Nu

    # Lee and Lee (2015) 
    def nusselt_IV(self):
        '''
            Friction and Colburn factor correlations and shape optimization of chevron-type Plate Heat exchangers
            > https://doi.org/10.1016/j.applthermaleng.2015.05.080
            > beta in paper 60°; p/h = 2.8
            > Pr(Water) = 4.07 used here
        '''
        xsi  = self.xsi   
        Re_h = self.Re_h 
        beta = self.beta 
        Pr   = self.Pr

        if beta in [30, 45, 60] and 2 < Pr < 50 and 200 < Re_h < 10000: # beta == 60 
            Nu = 0.1312 * (Re_h**0.78) * Pr ** (1/3)
            return Nu

    # Asif et al.
    def nusselt_V(self):
        '''
        Simulation of corrugated plate heat exchanger for heat and flow analysis
        > https://doi.org/10.18280/ijht.350127
        
        Args:
            xsi, Re_h, beta, Pr
        Returns:
            Nusselt number calculated by the selected method
        '''
        mu   = 1
        mu_w = 1

        xsi  = self.xsi   
        Re_h = self.Re_h 
        beta = self.beta 
        Pr   = self.Pr 
        
        if beta == 30 and 500 <= Re_h <= 2500 and 3.5 <= Pr <= 7.5:
            Nu = 0.093 * Re_h ** (0.7106) * Pr ** (1.3) * (mu/mu_w) ** (0.14)
            return Nu

        if beta == 60 and 500 <= Re_h <= 2500 and 3.5 <= Pr <= 7.5:
            Nu = 0.112 * Re_h ** (0.714) * Pr ** (1.3) * (mu / mu_w) ** (0.14)
            return Nu

    # selector of the Nusselt Ansatz
    def select_method(self, index):
        '''
        This function selects the method based on the given index
                Args:
                    Index: an integer representing the method to be used
                Returns:
                    Nusselt number calculated by the selected method
        '''
        if index == 0:
            return self.nusselt_I()
        elif index == 1:
            return self.nusselt_II()
        elif index == 2:
            return self.nusselt_III()
        elif index == 3:
            return self.nusselt_IV()
        elif index == 4:
            return self.nusselt_V()
        else:
            raise ValueError("Invalid index. Please a value between 0 and 4")
    
    # list of applicable indices.    
    def applicable_methods(self):
        '''
        This function checks the conditions for each correlation and returns a list of applicable indices.
        Args:
            None
        Returns:
            A list of indices of the applicable correlations.
        '''
        Re_h = self.Re_h 
        beta = self.beta 
        Pr   = self.Pr
        
        applicable_indices = []
        
        # Check conditions for each correlation
        if  0 <= beta <= 80:
            applicable_indices.append(0)
        if beta == 60 and 500 < Re_h < 3000:
            applicable_indices.append(1)
        if 30 <= beta <= 60 and Re_h >= 1000:
            applicable_indices.append(2)
        if 15 <= beta <= 75 and 200 < Re_h < 10000:
            applicable_indices.append(3)
        if beta in [30, 60] and 500 <= Re_h <= 2500 and 3.5 <= Pr <= 7.5:
            applicable_indices.append(4)

        return applicable_indices

    def get_correlation_name(self, index):
        correlations = {
            0: "Martin Holger (1995)",
            1: "Al zahrani, Islam und Saha (2019)",
            2: "Muley and Manglik (1999)",
            3: "Lee and Lee (2015)",
            4: "Asif et al. (2017)"
        }
        return correlations.get(index, "Invalid index! Please enter a value between 0 and 4.")

