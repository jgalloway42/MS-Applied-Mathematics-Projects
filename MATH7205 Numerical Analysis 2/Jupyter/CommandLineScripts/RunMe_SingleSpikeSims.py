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
    
    print('='*65)
    print('This Program Runs a Simulation of the Hodgkin-Huxley Model')
    print('of the electrical repsonse of an axon to a current stimulous.')
    print('Simulation Method: Forward Euler, Huen\'s Method, Runga_Kutta')
    print('\u0394 t = {:0.4f} ms, Length = {:0.1f} ms'.format(STEP,T_END))
    print('='*65)
    print()

    '''Build Model Object'''
    hhm = HHModel() # build model object with default parameters
    
    """Solve equations"""
    t = np.arange(0,T_END,STEP)

    Ye = hhm.simulate(t,method='forwardEuler',I_funct=hhm.I_spike)
    Yh = hhm.simulate(t,method='heunsMethod',I_funct=hhm.I_spike)        
    Yrk = hhm.simulate(t,I_funct=hhm.I_spike)

    '''Plot Coefficients vs Voltage'''
    simulations = [Ye,Yh,Yrk]
    titles = ['Forward Euler',
             'Huen\'s Method',
             'Runga-Kutta 4th Order']
    fig, axes = plt.subplots(3, 1, facecolor='#d3d3d3', figsize=(10,10),
                             sharex=True, tight_layout = True)
    axes = axes.ravel()
    
    for i,ax in enumerate(axes):
        ax.plot(t,simulations[i][0,:])
        ax.set_title(titles[i],fontweight='bold')
        ax.grid(True)
        ax.set_ylabel('Membrane Voltage\n[$V_m$]')
        if i == 2:
            ax.set_xlabel('Time [ms]')

    print('Close Figure Window to Exit...')
    plt.show()
    _exit(0)
