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

# Root Mean Squared Error
def RMSE(y1,y2):
    n = np.size(y1)
    rmse = (y1 - y2)
    rmse = np.sum(rmse*rmse)/n
    rmse = np.sqrt(rmse)
    return rmse

if __name__ == "__main__":
    warnings.filterwarnings('ignore') # To suppress nan warnings when the 
                                      # simulations fail    
    
    print('='*80)
    print('This Program Runs Mutliple Simulations of the Hodgkin-Huxley Model')
    print('of the electrical repsonse of an axon to a current stimulous,')
    print('and compares the root mean squared error of different integration methods.')
    print('='*80)
    print()
    
    tend = 250
    sims = [lambda t: hhm.simulate(t)[0,:],
           lambda t: hhm.simulate(t,method='forwardEuler')[0,:],
           lambda t: hhm.simulate(t,method='heunsMethod')[0,:]]
    lbls = ['Runga-Kutta','Forward Euler','Huen\'s Method']
    markers = [':o',':X',':^']
    
    
    '''Build Model Object'''
    hhm = HHModel() # build model object with default parameters
    
    # use RK4 0.001 as comparison
    print('Running Benchmark Simulation Runga-Kutta, Step Size 0.001 ms...')
    t = np.arange(0,tend,0.001)
    comparison = interp1d(t, hhm.simulate(t)[0,:], kind='cubic')
    
    
    '''Perform higher resolution test on RMSE'''
    err = [[],[],[]]
    step_sizes = np.linspace(0.005,0.05,10)


    for i,step_size in enumerate(step_sizes):
        """Solve equations"""
        t = np.arange(0,tend,step_size)
        # get interpolation from comparison simulation
        comp = comparison(t)
        print()
        for j,sim in enumerate(sims):
            print('Running {:s}, Step Size {:0.4f} ms...'.format(lbls[j],
                  step_size))
            y = sim(t)
            err[j].append(RMSE(y,comp))

        
    '''Graph Error from Higher Resolution Simulation Runs'''
    fig, ax = plt.subplots(1, 1, facecolor='#d3d3d3', figsize=(7,4),sharex=True,
                            tight_layout = True)
    lbls = ['Runga-Kutta','Forward Euler','Huen\'s Method']
    markers = [':o',':X',':^']
        
    for i in range(len(err)):
        ax.plot(step_sizes,err[i],markers[i],label=lbls[i],markersize=8)
    ax.legend(loc='best')
    ax.set_ylabel('RMS-Error')
    ax.set_xlabel('Step Size $\Delta t$ [ms]')
    _ = ax.set_title('RMSE for Methods vs Step Size',
                fontweight='bold')   
        
    print()
    print('Close Figure Window to Exit...')    
    plt.show()
    _exit(0)
