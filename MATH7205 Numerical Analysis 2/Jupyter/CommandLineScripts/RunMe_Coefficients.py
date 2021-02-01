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
    METHOD = None # RK4 is default method 
    TITLE = 'Runga-Kutta 4th Order'
    
    print('='*65)
    print('This Program Plots the Coeiffcients of the Hodgkin-Huxley Model')
    print('for the Gating Variables.')
    print()

    '''Build Model Object'''
    hhm = HHModel() # build model object with default parameters
    
    '''Plot Coefficients vs Voltage'''
    coeffs = [(hhm.alpha_n, hhm.beta_n),
             (hhm.alpha_m, hhm.beta_m),
             (hhm.alpha_h, hhm.beta_h)]
    titles = ['Potassium Channel',
             'Sodium Activation Channel',
             'Sodium Deactivation Channel']
    vm = np.linspace(-50,150,200)
    fig, axes = plt.subplots(1, 3, facecolor='#d3d3d3', figsize=(15,5))
    fig.suptitle('Voltage Dependent Coefficients',fontweight='bold', fontsize=16)
    axes = axes.ravel()
    
    for i,ax in enumerate(axes):
        lbl = ['\u03B1','\u03B2'] # alpha and beta
        for j,coef in enumerate(coeffs[i]):
            ax.plot(vm,coef(vm),label=lbl[j])
        ax.set_title(titles[i])
        ax.grid(True)
        ax.legend(loc='best')
        ax.set_ylabel('Unitless')
        ax.set_xlabel('Membrane Voltage\n[$V_m$]')
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    print('Close Figure Window to Exit...')    
    plt.show()
    _exit(0)
