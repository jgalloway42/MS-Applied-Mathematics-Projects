'''
Description: Runs program to simulate Hodgkin-Hukely Model
with an imput current pulse short enough to create
a single limit cycle

Input: none

Output: Graph ofSimulated System

Uses: HHModel.py, Solvers.py
Python Version 3.7.3, Matplotlib Version 3.0.3, Numpy Version 1.16.2

Author: Josh Galloway
Version: 1.0
Date: 23 Oct 2020
'''
import numpy as np
from os import _exit
import matplotlib.pyplot as plt
from HHModel import *

if __name__ == "__main__":
    
    # Constants
    STEP = 0.01  # delta t step size
    T_END = 30.  # Simulation time end [ms]
    METHOD = 'heunsMethod'
    TITLE = 'Huen\'s Method'
    
    print('='*65)
    print('This Program Runs a Simulation of the Hodgkin-Huxley Model')
    print('of the electrical repsonse of an axon to a current stimulous.')
    print('Simulation Method: {:s}'.format(TITLE))
    print('\u0394 t = {:0.4f} ms, Length = {:0.1f} ms'.format(STEP,T_END))
    print('='*65)
    print()

    '''Build Model Object'''
    hhm = HHModel() # build model object with default parameters
    
    """Solve equations"""
    t = np.arange(0,T_END,STEP)
    Y = hhm.simulate(t,method=METHOD,I_funct=hhm.I_spike)

    '''Plot Coefficients vs Voltage'''
    hhm.plotSolution(TITLE + ' \u0394 t = {:0.4f} ms'.format(STEP),
                     Y,t,I_funct=hhm.I_spike)
    
    print('Close Figure Window to Exit...')
    plt.show()
    _exit(0)
