'''
Description: Runs program to simulate Cart Pole dynamics

Input: none

Output: Animation of Simulated System

Uses: GallowayMath7203MiniProject02Functions.py,
Python Version 3.7.3, Matplotlib Version 3.0.3, Numpy Version 1.16.2

Author: Josh Galloway
Version: 1.0
Date: 11 April 2020
'''
import numpy as np
from os import _exit
from matplotlib.pyplot import show,title
from GallowayMath7203MiniProject02Functions import simulate,buildAnimation

if __name__ == "__main__":
    
    # Initial Conditions
    X0 = [0,0,0.1,0]
    
    print('\n=========================================================')
    print('This Program Runs a Simulation for a Cart Pole System')
    print('x = {:0.2f} initial position of the cart\n'.format(X0[0]) +
          'dx/dt = {:0.2f} intiial velocity of cart\n'.format(X0[1]) +
          'theta = {:0.2f} initial angle of pole with respect to vertical\n'.format(X0[2]) +
          'd theta/dt = {:0.2f} initial angular velocity of pole\n'.format(X0[3]) +
          'g = 9.81 m/s^2 acceleration due to gravity\n' +
          'M = 1.0 kg mass of the cart\n' +
          'm = 0.1 kg mass of the pole\n' +
          'L = 0.5 m length of the pole\n' +
          'F = 0 N force applied to the cart')
    print('=========================================================')
    print()
    print('Close Simulation Display Window or Terminal to End...')

    '''Simulate'''
    t = np.linspace(0,10,10*24)
    x_pend,y_pend,x_cart,X = simulate(X0,t)
        
    '''Build Animation Video'''
    amin = buildAnimation(t,x_pend,y_pend,x_cart)
    title('Cart Pole Simulation No Force,\n Penulum Started Off Center',
          fontweight='bold')
    show()
    _exit(0)
