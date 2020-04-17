# This notebook implements a basic Susceptible, Infected, Recovered (SIR) model for infectious disease
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#The following two functions implement the differential equations for the model

# This function implements the derivative calculations
# It is only called by the SIR_model function - not intended to be called directly
def SIR_derivatives(y, t, N, beta, gamma):
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return dS, dI, dR

# This function is intended to be called to execute the SIR model
# S0: Initial number of susceptible people
# IO: Initial number of infected people
# R0: Initial number of people who have recovered or died as a result of the disease
# N: Total population
# beta: Average number of contacts per person per time multiplied by the probability of disease transmission
# gamma: Rate of recovery or mortality
# ts: Time series
# returns: Returns 3 arrays of len(ts), i.e. one series each for susceptible, infected and recovered people
def SIR_model(N,S0,I0,R0,beta,gamma,ts):
    # Initial conditions array needed for odeint function
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    return odeint(SIR_derivatives, y0, ts, args=(N, beta, gamma))

# Change the following values to execute the model for a specific scenario
N = 1000 #Total population
I0 = 1 # Initial number of infected people
R0 = 0 # Initial number of recovered people
S0 = N - I0 - R0 # S0 is the initial number of susceptible people, i.e. population - infected - recovered

# Effective contact rate, i.e. an infected individual comes into contact with βN other individuals 
# per unit time (of which the fraction that are susceptible to contracting the disease is S/N)
beta = 0.2

# Mean recovery rate, i.e. 1/γ is the mean period of time during which an infected individual can pass it on
gamma = 1/10 

# Create a numpy array of time points (in days)
# Range from 0 to 160 in increments of 1 (day)
t = np.linspace(0, 160, 160, False)

#Execute the SIR model and return the future time series array
y = SIR_model(N,S0,I0,R0,beta,gamma,t)

# Extract the results into three different arrays
S, I, R = y.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.figure(figsize=[24,12])
plt.plot(t, S/N, label='Susceptible')
plt.plot(t, I/N, label='Infected')
plt.plot(t, R/N, label='Recovered')
plt.legend()
plt.title(label='Basic SIR Model', loc='center')
plt.show()
plt.close()
