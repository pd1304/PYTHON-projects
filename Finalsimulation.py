00# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:04:07 2021

@author: pares
"""

'''
This file contains the code for the actual simulation of the solar system.  The Final particle is imported and used to so the
bodies are constantly updated  with the timestep given for a given number of iterations and matplotlib is the library that was
imported to plot the trajectories of the bodies on a 2D graph. Planets is the list from another file system_bodies, which contains
all the planets and the Sun in our solar system. 
'''
import copy
import numpy as np
from Finalparticle import Particle
import matplotlib.pyplot as plt 
from system_bodies import System_bodies
from system_bodies import Planets




Energy_List = []
Data = []
time = 0
dt = 3600
n_iter = round(2*365*24*3600/dt)
   
'''
An empty list is made of the Data and Energy. dt is the timestep, so it determines how often the bodies update. n_iter is the 
number of iterations run by the file. The default dt = 3600 (every 1 hour), with an n_iter = 2 years . 
'''      
method = int(input("What approximation would you like to use?\n press 0 for Euler\n press 1 for Euler-Cromer\n press 2 for Verlet\n"))

''' The line above takes the user input to either be 0, 1, or 2. Thid decideds what numerical approximation to use to simulate
     the solar system.
'''
for n in range(n_iter):   
    P = 0
    body_var = []
    for j in Planets:
    
        time += dt   
        j.updateGravitationalAcceleration(Planets)
        j.momentum()
        P = P+j.momentum()
        #This updates the acceleration, momentum for the planets in the list Planets every dt. 
        
        if method == 0:
                j.updateEuler(dt)   
                   
      
        if method == 1:
                    j.updateEuler_Cromer(dt)
        
        if method == 2:
                    j.updateVerlet(dt, Planets)
        
        # The code above basically takes the user input for method, and depending on the input wither updates Euler, Cromer or Verlet. 
    
        body_var.append(copy.deepcopy(j))
        
        # This is the code that appends the update properties of the planets into the list Data.
        
    Tot_Ene = Planets[0].finalEnergy(Planets) + Planets[1].finalEnergy(Planets) + Planets[2].finalEnergy(Planets) 
    + Planets[3].finalEnergy(Planets) + Planets[4].finalEnergy(Planets) + Planets[5].finalEnergy(Planets) 
    + Planets[3].finalEnergy(Planets) + Planets[4].finalEnergy(Planets) + Planets[5].finalEnergy(Planets)
    + Planets[6].finalEnergy(Planets) + Planets[7].finalEnergy(Planets) + Planets[8].finalEnergy(Planets) 
    
    # The Tot_Ene is defined to the the summation of the final energies of all the planets in the list of Planets. 
    if (n)%10 == 0:   
        Data.append([time] + body_var)  
        Energy_List.append([time, copy.deepcopy(Tot_Ene)])

   # The code above appends the updated Tot_Ene into the Energy_List. 



for i in range(0,len(Planets)):
    j_position = np.array([entry[i+1].position for entry in Data])
    plt.plot(j_position[:,0], j_position[:,1], label = Data[0][i+1].name)
    
    
plt.legend()
plt.xlabel("distance (m)")
plt.ylabel("distance (m)")
plt.show()

'''
The code above is what plots the trajectories of the planets. They take the positions of each of the planets from the Data list
and then plot the x and y coordinates through matplotlib. 
'''


''' Unfortunately, none of the energy, momenutm code seem to have been effective in producing anything. After a lot of tinkering, I was unable 
to get them to work. THe code is still there, but it doens't do anything  ''' 




        
