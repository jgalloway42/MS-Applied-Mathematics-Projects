'''
Description: Runs program to simulate Cart Pole dynamics with
controller designed by non-algorithmic pole placement

Input: none

Output: Animation of Simulated System

Uses: GallowayMath7203MiniProject02Functions.py,
Python Version 3.7.3, Matplotlib Version 3.0.3, Numpy Version 1.16.2,
Control Version 0.8.3

Author: Josh Galloway
Version: 1.0
Date: 11 April 2020
'''
import numpy as np
from os import _exit
from control import ctrb,place
from matplotlib.pyplot import show,title
from GallowayMath7203MiniProject02Functions import simulateControl,buildAnimation

if __name__ == "__main__":

    # Initial Condition
    X0 = [-0.75,0,0.1,0]
    
    # System Parameters
    M,m,L,g,F = 1,0.1,0.5,-9.81,0
    
    
    print('\n=========================================================')
    print('This Program Runs a Simulation for a Cart Pole System\n'+
           'with controller designed by non-algorithmic pole placement')
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


    
    '''Get Controllabiltiy Matrix Using Library'''
    CM = ctrb(A,B)
    
    # calculation of controllability matrix by hand
    CM2 = np.hstack((B,A*B,A*A*B,A*A*A*B))
    
    print('\nControllabiltiy Matrix from Library\n',CM)
    
    print('\nControllabiltiy Matrix by hand\n',CM2)
    
    '''Controlability Requires Controlability matrix be of full rank'''
    print('\nControllability Matrix Rank:\n',np.linalg.matrix_rank(CM))
    
    '''Eigenvalues of A, positive is unstable, has one unstable eigen value and one stable'''
    print('\nA matrix eigenvalues:\n',np.linalg.eigvals(A))
    
    '''Design a control law u = -Kx, by pole placement'''
    k1 = [-0.6,-0.8,-1,-1.2]
    print('\nPlace Poles at:\n',k1)
    
    K = place(A,B,k1)
    print('\nResulting K matrix:\n',K)
    
    '''Designed system has response X dot = (A - BK)x'''
    print('\nCheck Designed system has response eig(A - BK):\n',np.linalg.eigvals(A-B*K))
    
    
    # ## Simulate Non-algorithmic Pole Placement Controller
    
    '''now simulate with F = -K*(X - X_desired)'''
    '''Simulate'''
    t = np.linspace(0,15,15*24)
    x_pend,y_pend,x_cart,X = simulateControl(X0,t,K)
    
    amin = buildAnimation(t,x_pend,y_pend,x_cart)
    title('Cart Pole Simulation,\n Non-Algorithmic Controller Design',
          fontweight='bold')
    
    print()
    print('Close Simulation Display Window or Terminal to End...')
    
    show()

    
    _exit(0)
