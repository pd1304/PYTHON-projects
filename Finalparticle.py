# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:46:29 2021

@author: pares
"""
"""
This file contains the Particle class, which holds the properties of a particle that will be continuely updated.
Furthermore, it also includes the different numerical approximations that will be used to simulated a gravitaionally 
interacting system. 
"""
import copy
import numpy as np


class Particle:
    def __init__(self,position=np.array([0, 0, 0], dtype=np.float64),
                 velocity=np.array([0, 0, 0], dtype=np.float64),
                 acceleration=np.array([0, 0, 0], dtype=np.float64),
                 name='Ball', 
                 mass=1.0):
        'This is the inialisation function.'
        self.position = np.array(position, dtype = np.float64)
        self.velocity = np.array(velocity, dtype = np.float64)
        self.acceleration = np.array(acceleration, dtype = np.float64)
        self.name = name
        self.mass = mass
        self.G    = 6.67408e-20
        '''
        We have just defined the postions, velocities and accelerations to be arrays that take in floats as values.
        Furthermore self.name, self.mass and G has also been defined.
        '''
    def updateEuler(self, deltaT):
             self.position = self.position + self.velocity*deltaT
             self.velocity = self.velocity + self.acceleration*deltaT
             
    ''' The code above describes the process of the Euler numerical approximation'''         
    def updateEuler_Cromer(self, deltaT):
            self.velocity = self.velocity + self.acceleration*deltaT
            self.position = self.position + self.velocity*deltaT
    ''' The code above describes the process of the Euler-Cromer numerical approximation'''         

    def updateVerlet(self, deltaT, Planets):
            self.position = self.position + self.velocity*deltaT + 0.5*self.acceleration*(deltaT**2)            
            old_acceleration = self.acceleration
            self.updateGravitationalAcceleration(Planets)    
            self.velocity = self.velocity + 0.5*(self.acceleration + old_acceleration)*deltaT
    ''' The code above describes the process of the Verlet numerical approximation'''         

              
                
    def __str__(self):
        return "Particle: {0}, Mass: {1:.3e}, Position: {2}, Velocity: {3}, Acceleration: {4}".format(self.name, self.mass,self.position, self.velocity, self.acceleration)
        


    def updateGravitationalAcceleration(self, Planets):
        for j in Planets:   
            if self.name != j.name:
                r = np.linalg.norm(self.position-j.position)
                acc = -self.G*j.mass/(r)**2 *(self.position-j.position)/(r)
                self.acceleration = acc
            return
    ''' 
    The code above describes the gravitational acceleration of the bodies and the constant updating of them.Planets is the list
    in which all the planets with their properties is recorded. The for loop goes around the Planets list and calculates the 
    acceleration between two bodies for all bodies in the system.
    '''
    
    def kineticEnergy(self):
        v = self.velocity
        K = 1/2 * self.mass * (v[0]**2+v[1]**2+v[2]**2)
        return K
    ''' The code above describes the kinetic energy of a body '''
    
    def potentialEnergy(self,Planets):
         for j in Planets:
             r2 = np.linalg.norm(self.position - j.position)
             U = -(self.G * self.mass * j.mass)/r2
             U = 0.5*U
             return U
    ''' 
     The code above describes the potential energy of each body due to the rest The reason U = 0.5*U is because,
     when two bodies gravitaionally interact, the energy produced acts on both in same magnitudes, so inorder to get the 
     correct U, it will need to be divided by 2.
    '''
     
    def momentum(self):
       v = np.linalg.norm(self.velocity) 
       p = self.mass*v
       return p
    ''' The code above describes the momentum for a body '''
    def finalEnergy(self, Planets):  
        TE = self.kineticEnergy() + self.potentialEnergy(Planets)

        return TE
    ''' The final energy is the summation of the kinetic and potential energies'''      
        

    