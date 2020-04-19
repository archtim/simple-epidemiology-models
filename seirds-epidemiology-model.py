# This notebook implements an SEIRDS compartmental model for infectious disease forecasting.
# SEIRDS is a variation of the SEIRS model with a compartment for deceased separate from "recovered". 
# The variable names in these compartmental epidemiology models are very confusing
# so here is a short explanation of them:
# Susceptible: People who are not exposed or infected but that are not immune, so they are susceptible to the virus
# Exposed: People who have been exposed but are not yet contagious. This variable accounts for the incubation period.
# Infected: People who have become infected by the virus and are contagious therefore able to transmit it to others.
# Recovered: People who had the virus and recovered such that they are now immune for some period
# Deceased: People who succumbed to the effects of the virus and passed away. 
# Recovered people remain immune for some period before becoming susceptible again
# This model currently ignores "Vital Dynamics", which accounts for people who are born or die or other causes.
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#The following two functions implement the differential equations for the model

# This function implements the derivative calculations
# It is only called by the SIR_model function - not intended to be called directly
def SEIRDS_derivatives(y, t, N, beta, gamma, delta, sigma):
    S, E, I, R, D = y
    dS = -beta * S * I / N
    dE = (beta * S * I / N) - sigma * E
    dI = (sigma * E) - ((gamma + delta) * I)
    dR = gamma * I
    dD = delta * I
    return dS, dE, dI, dR, dD

# This function is intended to be called to execute the SEIRS model
# N: Total population
# S0: Initial number of susceptible people
# E0: Initial number of exposed people
# I0: Initial number of infected people
# R0: Initial number of people who have recovered or died as a result of the disease
# beta: Average number of contacts per person per time multiplied by the probability of disease transmission
# gamma: Rate of recovery
# sigma: Rate that people who have recovered become susceptible again due to loss of immunity
# ts: Time series
# returns: Returns 4 arrays of len(ts), i.e. one series each for susceptible, exposed, infected and recovered people
def SEIRDS_model(N,I0,E0,R0,D0,beta,gamma,delta,sigma,ts):
    S0 = N - I0 - E0 - R0 - D0
    y0 = S0, E0, I0, R0, D0
    return odeint(SEIRDS_derivatives, y0, ts, args=(N,beta,gamma,delta,sigma))
    
# Change the following values to execute the model for a specific scenario
N = 100000 #Total population
I0 = 100 # Initial number of infected people
R0 = 20 # Initial number of recovered people
E0 = 0 # Initial number of exposed people
D0 = 0 # Initial number of people who have died
beta = 0.24 # Average number of contacts per person per time multiplied by the probability of disease transmission
delta = 0.005 # Rate of death 
gamma = 0.10 # Rate of recovery
sigma = 0.4 # Rate that people who have recovered become susceptible again due to loss of immunity

# Create a numpy array of time points (in days)
# Range from 0 to 120 in increments of 1 (day)
t = np.linspace(0, 160, 160, 160)

# Call the SEIRS model function
ret = SEIRDS_model(N,I0,E0,R0,D0,beta,gamma,delta,sigma,t)

# Transpose the results separate arrays
S, E, I, R, D = ret.T

# Plot the data on four separate curves for S(t), E(t), I(t) and R(t)
plt.figure(figsize=[24,12])
plt.plot(t, S/N, label='Susceptible')
plt.plot(t, E/N, label='Exposed')
plt.plot(t, I/N, label='Infected')
plt.plot(t, R/N, label='Recovered')
plt.plot(t, D/N, label='Deceased')
plt.legend()
plt.title(label='Simple SEIRDS Model', loc='center')
plt.show()
plt.close()
