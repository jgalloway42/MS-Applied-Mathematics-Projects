'''
Description: Runs program to simulate Cart Pole dynamics with
controller designed by Linear Quadratic Regulator

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
from GallowayMath7203MiniProject02Functions import simulateControl,buildAnimation,lqr

if __name__ == "__main__":
    
    # Initial Conditions
    X0 = [-0.75,0,0.1,0]
    M,m,L,g,F = 1,0.1,0.5,-9.81,0
    
    print('\n=========================================================')
    print('This Program Runs a Simulation for a Cart Pole System\n'+
           'with an LQR controller')
    print('x = {:0.2f} initial position of the cart\n'.format(X0[0]) +
          'dx/dt = {:0.2f} intiial velocity of cart\n'.format(X0[1]) +
          'theta = {:0.2f} initial angle of pole with respect to vertical\n'.format(X0[2]) +
          'd theta/dt = {:0.2f} initial angular velocity of pole\n'.format(X0[3]) +
          'g = {:0.2f} m/s^2 acceleration due to gravity\n'.format(g) +
          'M = {:0.2f} kg mass of the cart\n'.format(M) +
          'm = {:0.2f} kg mass of the pole\n'.format(m) +
          'L = {:0.2f} meters length of the pole\n'.format(L) +
          'F = {:0.2f} N force applied to the cart'.format(F))
    print('=========================================================')
	
    '''Build '''
    A = np.matrix([[0,1,0,0],[0,0,m*g/M,0],[0,0,0,1],[0,0,-g*(M+m)/L/M,0]])
    B = np.matrix([[0],[1/M],[0],[-1/L/M]])
    
    SIM = 3  # number of seconds to simulate
    
    # ## Linear Qudratic Regualtor Design
    
    '''Cost matrix for state variables
     x, xdot, theta, theta dot'''
    Q = np.diagflat([100,1,10,1])
    print('\nState Variable Cost Matrix:\n',Q)
    
    ''' Cost matrix for manipulated variable
    '''
    R = np.matrix([0.01])
    print('\nCost matrix for manipulated variable:\n',R)
    
    K, ev = lqr(A,B,Q,R)
    print('\nLQR K matrix:\n',K)
    
    
    # ## Simulate LQR Response
    
    '''now simulate with F = -K*(X - X_desired)'''
    '''Simulate'''
    t = np.linspace(0,SIM,SIM*24)
    x_pend,y_pend,x_cart,X = simulateControl(X0,t,K)
    
    amin = buildAnimation(t,x_pend,y_pend,x_cart)

    title('Cart Pole Simulation,\nLQR Controller',
          fontweight='bold')
            
    print()
    print('Close Simulation Display Window or Terminal to End...')
    
    show()
   
    _exit(0)
