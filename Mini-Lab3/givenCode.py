import numpy as np
import matplotlib.pyplot as plt

# Calculate black body radiation intensity W m^-2 nm sr^-1
def BlackBody(w,T):
    # parameters:
    # w Wavelenght in meters
    # T Temperature in Kelvin
    # Returns:
    # Spectral radiance in units W m^-2 nm sr^-1
    # Constants
    h = 6.62606957e-34 # joule·s
    c = 2.99792458e8 # Speed of light in m/s
    k = 1.38e-23 # Boltzmann constant in m^2 kg / s^2 / K
    return (2*h*c**2/w**5)*(1/(np.exp((h*c)/(w*k*T))- 1))

def BlackBodySun(w,T,R_sun_earth=149600000):
    # Calculate the irradiance from the sun at average sun earth distance.
    # R_sun_earth distance given in km
    # return spectral irradiance W m^-2 nm^-1
    R_sun=696340 #km
    return (R_sun/R_sun_earth)**2*np.pi*BlackBody(w,T)*1.0e-9

def StefanBoltzmann(T):
    return 5.670374419e-8*T**4
