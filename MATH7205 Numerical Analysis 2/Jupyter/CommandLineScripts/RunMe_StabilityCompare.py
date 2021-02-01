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
from scipy.interpolate import interp1d
from HHModel import *
import warnings


if __name__ == "__main__":
    warnings.filterwarnings('ignore') # To suppress nan warnings when the 
                                      # simulations fail    
    print('='*65)
    print('This Program Runs Mutliple Simulations of the Hodgkin-Huxley Model')
    print('of the electrical repsonse of an axon to a current stimulous,')
    print('and compares the stability of different integration methods.')
    print('='*65)
    print()
    
    tend = 250
    step_sizes = [0.01,0.05,0.1,0.5,1]
    sims = [lambda t: hhm.simulate(t)[0,:],
           lambda t: hhm.simulate(t,method='forwardEuler')[0,:],
           lambda t: hhm.simulate(t,method='heunsMethod')[0,:]]
    lbls = ['Runga-Kutta','Forward Euler','Huen\'s Method']
    markers = ['-','-.','--']
    
    
    '''Build Model Object'''
    hhm = HHModel() # build model object with default parameters
    
    # use RK4 0.001 as comparison
    print('Running Benchmark Simulation Runga-Kutta, Step Size 0.001 ms...')
    t = np.arange(0,tend,0.001)
    comparison = interp1d(t, hhm.simulate(t)[0,:], kind='cubic')
    fig, axes = plt.subplots(len(step_sizes), 1, facecolor='#d3d3d3',
                             figsize=(10,10),sharex=True, tight_layout = True)
    axes = axes.ravel()
    for i,step_size in enumerate(step_sizes):
        """Solve equations"""
        t = np.arange(0,tend,step_size)
        # get interpolation from comparison simulation
        print()
        comp = comparison(t)
        axes[i].plot(t,comp,':',label='RK4 $\Delta t$ 0.001 ms')
        for j,sim in enumerate(sims):
            print('Running {:s}, Step Size {:0.2f} ms...'.format(lbls[j],
                  step_size))
            y = sim(t)
            axes[i].plot(t,sim(t),markers[j],label=lbls[j])
        
        axes[i].set_title('Simulation with Step Size {:0.2f} ms'.format(step_size),
                         fontweight='bold')
        axes[i].legend(loc='best')
        axes[i].set_ylim(-80,40)
        axes[i].set_ylabel('$V_m$ [mV]')
        if i == len(step_sizes) - 1:
            axes[i].set_xlabel('Time [ms]')
    
    print()
    print('Close Figure Window to Exit...')    
    plt.show()
    _exit(0)
