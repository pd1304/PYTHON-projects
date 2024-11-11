# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:12:41 2021

@author: pares
"""

"""
This is the file containing the System_bodies class which stores all the information of the planets in our solar system. 
The information was taken for JPL Ephermis page. 
"""
from Finalparticle import Particle
import numpy as np

"""
Here are the positions and velocities of all the bodies in our Solar systems. The initial positions of
these bodies are the positions, velocities and accelerations on the DECEMBER 1ST 2021. 
"""

class System_bodies:

    sunMass = 1988500E24     
    Sun = Particle(
        position=np.array([-1.267282417445780E+06, 5.396678685776476E+05,  2.520205624705725E+04]),
        velocity=np.array([-6.439216648251454E-03, -1.446064170987709E-02, 2.656070267795371E-04]),
        acceleration=np.array([0, 0, 0]),
        name="Sun",
        mass=sunMass
        )   

    mercuryMass = 3.302E23     
    Mercury = Particle(
        position=np.array([-2.294393076199030E+07, -6.565192403790341E+07, -3.395490575049143E+06]),
        velocity=np.array([3.650284065728554E+01, -1.275360410508055E+01, -4.389654584163237E+00]),
        acceleration=np.array([0, 0, 0]),
        name="Mercury",
        mass=mercuryMass
        )

    venusMass = 48.685E23     
    Venus = Particle(
        position=np.array([7.449585631940070E+07, 7.765573978390406E+07, -3.288285900700271E+06]),
        velocity=np.array([-2.509495329824682E+01, 2.437846872883341E+01, 1.782790255006867E+00]),
        acceleration=np.array([0, 0, 0]),
        name="Venus",
        mass=venusMass
        )

    earthMass = 5.97219E24     
    Earth = Particle(
        position=np.array([5.231887467709020E+07,1.379907686020697E+08, 1.810405597399175E+04]),
        velocity=np.array([-2.825069244864891E+01,1.070442504310021E+01, 8.207575198717620E-04]),
        acceleration=np.array([0, 0, 0]),
        name="Earth",
        mass=earthMass
        )

    marsMass = 6.4171E23   
    Mars = Particle(
        position=np.array([-1.813297166946259E+08,-1.514112512541462E+08 , 1.257560419978045E+06]),
        velocity=np.array([1.652549596240535E+01,-1.645865322450017E+01,-7.498941664745544E-01]),
        acceleration=np.array([0, 0, 0]),
        name="Mars",
        mass=marsMass
        )

    jupiterMass = 189818722E22   
    Jupiter = Particle(
        position=np.array([6.826048462737756E+08 ,-3.019409789737509E+08, -1.401889605310588E+07]),
        velocity=np.array([5.127473183658640E+00,1.256387178453981E+01,-1.667986704735060E-01]),
        acceleration=np.array([0, 0, 0]),
        name="Jupiter",
        mass=jupiterMass
        )

    saturnMass = 5.6834E26  
    Saturn = Particle(
        position=np.array([1.022786588855915E+09,-1.074589060771025E+09,-2.203648301950133E+07]),
        velocity=np.array([6.457278681486529E+00,6.640204640874535E+00,-3.730592022967962E-01]),
        acceleration=np.array([0, 0, 0]),
        name="Saturn",
        mass=saturnMass
        )

    uranusMass = 86.813E24  
    Uranus = Particle(
        position=np.array([2.165135560642323E+09,2.004390354023007E+09,-2.060537440699327E+07]),
        velocity=np.array([-4.676521719585446E+00,4.680150658258555E+00, 7.808375041637405E-02]),
        acceleration=np.array([0, 0, 0]),
        name="Uranus",
        mass=uranusMass
        )


    neptuneMass = 102.409E24
    Neptune = Particle(
        position=np.array([4.429873292359221E+09,-6.259530226625593E+08 ,-8.920069074546283E+07]),
        velocity=np.array([7.237546300967174E-01,5.413478273613189E+00,-1.281515528291954E-01]),
        acceleration=np.array([0, 0, 0]),
        name="Neptune",
        mass=neptuneMass
        )
    

Planets = [System_bodies.Sun, System_bodies.Mercury, System_bodies.Venus, System_bodies.Earth, System_bodies.Mars, System_bodies.Jupiter, 
           System_bodies.Saturn, System_bodies.Uranus, System_bodies.Neptune]

