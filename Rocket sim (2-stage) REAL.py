# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:44:49 2024

@author: pareshdokka
"""

### IMPORTS
import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
from scipy import constants
### Eq of motion
# F = ma = m ..x > 2nd ODE

plt.close('all')
### PARAMETERS

## Earth Parameters
G = 6.6742e-11 
mass_planet = 5.972e24 #kg
radius_planet = 6357000 #m
name = 'Earth'


## Rocket Parameters
mass0 = 6080 # First stage #kg    
max_thrust = 500000
Isp1 = 300 #sec  #ISP is specific impulse. It measures how long it takes for the fuel to be converted into thrust
Isp2 = 300
tMECO = 29 #sec    The time at which main engine is cut off/ Burntime for stage 1
tsep1 = 2  #sec    Length of time for first stage to dettach
mass1 = 1000 # Second stage #kg
t2start = 60  #The time at which second stage begins
t2end = t2start+5.4  #The time at which second stage ends
D = 1.5 #Diameter in meters
S = np.pi*(D/2)**2 # cross section area of rocket
CD = 0.1 #Drag coeffiecient
## Initial conditions for rocket
x0 = radius_planet
z0 = 0
velx0 = 0
velz0 = 0
r0 = radius_planet+1600000
period = 2*np.pi/np.sqrt(G*mass_planet)*r0**(3/2)*1.5   #sec


### AERODYNAMICS MODEL:
class Aerodynamics():
    def __init__(self,name):
        self.name = name
        if name == 'Earth':
            ## going to use earth aeromodel 
            self.beta = 0.1354/1000 # density constant (exponential term)
            self.rhos = 1.225 # kg/m^3 density at sea level
            
    def getDensity(self,altitude):    
        if self.name == 'Earth':
            ## This is the density eq. for an isothermal atmosphere (simple model)
            rho = self.rhos*np.exp(-self.beta*altitude) # 
        return rho
# Create an aeromodel variable, which is an instance of the class Aerodynamics
# It is up here because it is a global
aeroModel = Aerodynamics(name)


### GRAVITATIONAL ACCELERATION MODEL
def gravity(x,z):
    global radius_planet, mass_planet
    
    r = np.sqrt(x**2+z**2)
    
    if r<radius_planet:
        accelx = 0
        accelz = 0
    else:
        accelx = G*mass_planet/r**3 *x
        accelz = G*mass_planet/r**3 *z
    return np.array([accelx,accelz]), r

### PROPULSION MODEL
def propsulsion(t):
    global max_thrust, Isp1,Isp2, tMECO, ve
    
    if t>=0 and t<tMECO:        
        # this is the time during the first stage
        # we are firing the main thruster (engine)
        theta = 25*np.pi/180
        thrustF = max_thrust
        ve = Isp1*9.81   # exaust velocity:the maximum velocity that can be achieved by the exhaust gases of a rocket propulsion system,
        mdot = -thrustF/ve     # change in mass of rocket due to loss of fuel
        
    
    if t>tMECO and t< (tMECO+tsep1):
        # this is the time between when main thruster is shut 
        # and when it is completely detached
        theta = 0
        thrustF = 0
        #masslost = mass0 - mass of first stage
        mass_0st = (mass0-mass1)-(max_thrust/(Isp1*9.81) * tMECO)
        mdot = -mass_0st/tsep1   # change in mass of rocket due to loss of first stage thruster, mass0
       
    if t>(tMECO+tsep1):
        # this is time after 1stage seperation
        theta = 0
        thrustF = 0
        mdot = 0
    
    if t>t2start and t<t2end:
        # this is the time during second stage
        theta = 120*np.pi/180
        thrustF = max_thrust
        ve = Isp2*9.81
        mdot = -thrustF/ve
        
    if t>t2end:
        #this is the time after 2stage
        theta = 0
        thrustF = 0
        mdot = 0
    ## angle of thrust
    
    thrustx = thrustF*np.cos(theta)
    thrustz = thrustF*np.sin(theta)
    
    return np.array([thrustx,thrustz]), mdot    
        
### DERIVATIVES
def Derivatives(state,t):
    global aeroModel, radius_planet
    x = state[0]
    z = state[1]
    velx = state[2]
    velz = state[3]
    mass = state[4]
    # velocity
    xdot = velx
    zdot = velz
    # Total force: GRAVITY, AERODYNM, THRUST
    accel,r = gravity(x,z)  #gravity(x,z) returns accel(x,y),r
    gravityF = -accel*mass
    
     
    ## Aerodynamics - DRAG
    altitude = r - radius_planet
    rho = aeroModel.getDensity(altitude)
    V = np.sqrt(velx**2+velz**2)
    # aeroF = 0.5*rho*CD*S*v^2//S-Cross sec area, CD-drag coeff, rho-density
    aeroF = -0.5*rho*CD*S*abs(V)*np.array([velx,velz])
    
    
    ## Thrust - UPWARD MOTION
    thrustF, mdot = propsulsion(t)
    
    Forces = gravityF + aeroF + thrustF
    # acceleration
    if mass>0:     
        ddot = Forces/mass # accelearation as an array of xddot and zddot
    else:
       ddot = 0
       mdot = 0
      
    # copmute the .state vector or derivative of state vector
    statedot = np.array([xdot,zdot,ddot[0],ddot[1], mdot])

    return statedot
    
    
##################### MAIN BELOW ##############


### MAIN

## Plot the Air density as a function of AGL(above ground level)
test_altitude = np.linspace(0,100000,100)
test_rho = aeroModel.getDensity(test_altitude)
plt.figure()
plt.plot(test_altitude, test_rho, 'b-')
plt.xlabel('altitude (m)')
plt.ylabel('Air density (kg/m^3)')
plt.grid()


# Populate initial condition vector
stateinitial = np.array([x0, z0, velx0, velz0, mass0])

# time window - how long roket is in air for
tout = np.linspace(0,period,1500) # 1000 datapoints between 0s and 30s.


## NUMERICAL INTEGRATION CALL
stateout = sci.odeint(Derivatives, stateinitial, tout)

# x = xdot*t   xdot = xddot*t, => [xdot = -9.81*t, x = -9.81*t^2]
# Stateout stores values of xdot and x a 1000 times in the span of 30s

print(stateout)
# RENAME VARIABLES
xout =  stateout[:,0] # [first_row:last_row, column 0] slices all values in col 0
zout =  stateout[:,1]
altitude = np.sqrt(xout**2+zout**2) - radius_planet
velxout = stateout[:,2]
velzout = stateout[:,3]
velout = np.sqrt(velxout**2+velzout**2)
massout = stateout[:,4]


## PLOT

# ALTITUDE
plt.figure()
plt.plot(tout, altitude)
plt.xlabel('Time(sec)')
plt.ylabel('Altitude(metres)')
plt.grid()

#VELOCITY
plt.figure()
plt.plot(tout,velout)
plt.xlabel('Time(sec)')
plt.ylabel('Total speed (m/s)')
plt.grid()

#MASS
plt.figure()
plt.plot(tout,massout)
plt.xlabel('Time(sec)')
plt.ylabel('Mass(kg)')
plt.grid()

#ALL
plt.figure()
plt.plot(tout,massout, 'r-', label = 'Massout')
plt.plot(tout, velout, 'b-', label = 'Velocity')
plt.plot(tout, altitude, 'g-', label = 'Altitude')
plt.xlabel('Time(sec)')
plt.ylabel('Mass(kg)/Velocity(m/s)/Altitude(m)')
plt.grid()
plt.legend()


#2D ORBIT
plt.figure()
plt.plot(xout,zout,'r-',label='Orbit')
plt.plot(xout[0],zout[0], 'g*')
theta = np.linspace(0,2*np.pi, 1000)
xplanet = radius_planet* np.sin(theta)
yplanet = radius_planet*np.cos(theta)
plt.plot(xplanet, yplanet, 'b-',label = 'Planet')
plt.xlabel('distance(m)')
plt.ylabel('distance(m)')
plt.grid()
plt.legend()



