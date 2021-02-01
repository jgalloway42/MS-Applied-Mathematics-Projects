# -*- coding: utf-8 -*-
"""
Hodgekin and Huxley Models

@author: Josh.Galloway
"""

import numpy as np
import matplotlib.pyplot as plt
from Solvers import *

class HHModel:
    '''
    Hodgekin-Huxley Action Potential Model

    '''
    # Constants
    Cm = 1.0 # membrane capacitance uF/cm^2
    
    gk_bar = 36.0 # Potassium max conductance mS/cm^2
    gna_bar = 120.0 # Sodium max conductance mS/cm^2
    gl_bar = 0.3 # Leakage max conductance mS/cm^2
 
    Vna = 50. # Sodium Voltage (mV)
    Vk = -77.0 # Potassium Voltage (mV)
    Vl = -54.387 # Leakage Voltage (mV)
    
    X0 = np.array([-65, 0.32, 0.05, 0.6]) # Vm,n,m,h starting values
    # ...these are very fiddly
    
    def __init__(self,  ):
        '''Empty Constructor'''
        pass
        
    """ Alpha and Beta Equations"""
    def alpha_n(self,Vm):
        # potassium channel
         return 0.01*(Vm+55.)/(1-np.exp(-(Vm+55.)/10))

    def beta_n(self,Vm):
        # potassium channel
        return 0.125*np.exp(-(Vm+65.)/80.)
    
    def alpha_m(self,Vm):
        # sodium channel activating molecules
        return 0.1*(Vm+40.)/(1-np.exp(-(Vm+40.)/10))

    def beta_m(self,Vm):
        # sodium channel activating molecules
        return 4.0*np.exp(-(Vm+65.)/18.)

    
    def alpha_h(self,Vm):
        # sodium channel inactivating molecules
        return 0.07*np.exp(-(Vm+65.)/20.)
    
    def beta_h(self,Vm):
        # sodium channel inactivating molecules
        return 1.0/(1. + np.exp(-(Vm+35.)/10.))
    
    '''Default Step Function for Injected Current'''
    def I_inject(self,t):
        # Step function spaning 1 sec of time
        # steps up to values of S1 and S2
        
        S1 = 10.  #microamps/cm^2
        S2 = 35.
        t1,t2,t3,t4 = 100.,200.,300.,400. #milliseconds
        I = S1*(t>t1) - S1*(t>t2) + S2*(t>t3) - S2*(t>t4)
        return I

    def I_spike(self,t):
        # initial step of current for PDE solve
        s1 = 35.
        t1,t2 = 10.,12.5
        I = s1*(t>t1) - s1*(t>t2)
        return I
    
    def get_I(self,t,I_funct):
        # returns vector of default injection current values
        # at for time vector passed in
        I = np.zeros_like(t)
        for i,ts in enumerate(t):
            I[i] = I_funct(ts)
        return I
            
    
    ''' System of Equations '''
    def dXdt(self,t,X,I_funct=None):
        '''
        Desc: Calculates the Derivatives of the HH model
        Inputs: t= time (not used)
                X = State variable vector of floats
                index map 0 = Vm, 1 = n, 2 = m, 3 = h
                I = Injected total current (forcing function) 
        Outputs: dX = Derivative of state vector
        '''
        # Breakout state vector for clarity
        Vm = X[0]
        n = X[1]
        m = X[2]
        h = X[3]
        
        an = self.alpha_n(Vm)
        am = self.alpha_m(Vm)
        ah = self.alpha_h(Vm)
        
        bn = self.beta_n(Vm)
        bm = self.beta_m(Vm)
        bh = self.beta_h(Vm)
        
        Ik = self.gk_bar*n*n*n*n*(Vm - self.Vk)
        Ina = self.gna_bar*m*m*m*h*(Vm - self.Vna)
        Ileak = self.gl_bar*(Vm - self.Vl)
        
        if I_funct is None:
            I = self.I_inject(t)  # passed in forcing function
        else:
            I = I_funct(t)
            
        dVm = (I - Ik - Ina -Ileak)/self.Cm
        dn = an*(1.0 - n) - bn*n
        dm = am*(1.0 - m) - bm*m
        dh = ah*(1.0 - h) - bh*h
        
        Y = np.array([dVm,dn,dm,dh])
        return Y

    def simulate(self,t,method ='rk4',
                 x0 = None,I_funct = None):
        '''
        Desc: Simulates the model
        Inputs: Time vector, method of integration, initial parameters
                and injection current function
        Output: Simulation
        '''
        if x0 is None:
            x0 = self.X0
        args = {I_funct:'I_funct'}
        sol = Solvers()
        if method == 'forwardEuler':
            Y = sol.forwardEuler(t,self.dXdt,x0,args)
        elif method =='heunsMethod':
            Y = sol.heunsMethod(t,self.dXdt,x0,args)
        else:
            Y = sol.rk4(t,self.dXdt,x0,args)
        return Y
        
    ''' Phase-Plane Equations '''
    def vectorField(self,Space,N):
        '''
        Desc: calculates vector field given variable spans and
              returns the meshgrid and values
        Inputs: Space tuple or list
                Index map:
                0 Vm = Membrane Voltage Span
                1 n,2 m,3 h = n,m,h Span [0,1]
        Outputs: XY = Tuple of Meshgrids, and xy Tuple of Values
        '''
        # breakout for clarity
        vmspan = Space[0]
        nspan = Space[1]
        mspan = Space[2]
        hspan = Space[3]
        vm = np.linspace(*vmspan,N)
        n = np.linspace(*nspan,N)
        m = np.linspace(*mspan,N)
        h = np.linspace(*hspan,N)
        
        VM,N,M,H = np.meshgrid(vm,n,m,h)
        
        t = 150 # time at which current in nonzero in default function
        
        dvm,dn,dm,dh = np.zeros(VM.shape), np.zeros(N.shape),\
        np.zeros(M.shape), np.zeros(H.shape)

        ni, nj, nk, nl  = VM.shape
        
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for l in range(nl):
                        xvm = VM[i, j,k,l]
                        xn = N[i, j, k, l]
                        xm = M[i, j, k, l]
                        xh = H[i, j, k, l]
                        dydt = self.dXdt(t,np.array([xvm,xn,xm,xh]))
                        dvm[i,j,k,l] = dydt[0]
                        dn[i,j,k,l] = dydt[1]
                        dm[i,j,k,l] = dydt[2]
                        dh[i,j,k,l] = dydt[3]
        XY = (VM,N,M,H)
        xy = (dvm,dn,dm,dh)
        return XY,xy
    
    def phasePlane(self,Y,t):
        '''
        Desc: calculates phase plane  given variable spans and
              returns the meshgrid and values
        Inputs: Y solution from simulation or list
                Index map:
                0 Vm = Membrane Voltage Span
                1 n,2 m,3 h = n,m,h Span [0,1]
        Outputs: PP derivatives evaluated over provided space
        '''
        # breakout for clarity
        PP = np.zeros_like(Y)
        for i in range(np.size(t)):
            PP[:,i] = self.dXdt(t[i],Y[:,i])
        return PP
        
        
    def plotSimulation(self,t,Y,I,v_ax,nmh_ax,I_ax):
        '''
        Desc: Plots simulation of system of differential equations
        Inputs: t = time vector, Y = solution vector
        v_ax = axis object for Vm, nmh_ax = axis obj for n, m, h 
        Output: Graph
        '''
        v_idx, n_idx, m_idx, h_idx = 0,1,2,3
        v_ax.plot(t,Y[v_idx,:])
        v_ax.set_title('Membrane Voltage vs Time',
                       fontweight='bold')
        v_ax.set_ylabel('$V_m$\n[mV]')
        v_ax.set_xlabel('t [ms]')
        v_ax.grid(True)
        
        I_ax.plot(t,I)
        I_ax.set_title('Injected Current vs Time',
                       fontweight='bold')
        I_ax.set_ylabel('$I_{injected}$\n[$\mu A / cm^2$]')
        I_ax.set_xlabel('t [ms]')
        I_ax.grid(True)
         
        lbl = [' ',
               'Potassium (n)',
               'Sodium Activation (m)',
               'Sodium Inactivation (h)']
        for i in [n_idx, m_idx, h_idx]:
            nmh_ax.plot(t,Y[i,:],label=lbl[i])
            nmh_ax.set_title(
                    'Potassium and Sodium Ion\nChannel Gaiting Value vs Time',
                    fontweight='bold')
            nmh_ax.set_ylabel('Gaiting Value\n[unitless]')
            nmh_ax.set_xlabel('t [ms]')
            nmh_ax.legend(loc='best')
            nmh_ax.grid(True)
        return None
            
        
    def plotSolution(self,super_title,Y,t,I_funct = None):
        '''
        Desc: Helper Function Plots Solution with Default Injection
        Current
        Inputs: Y solution output from simulation
        t time vector from simulation
        Outputs: Graph
        '''
        fig, axes = plt.subplots(3, 1, facecolor='#d3d3d3',
                                      figsize=(15,10), sharex=True)
        
        fig.suptitle(super_title,fontweight='bold', fontsize=16)
        axes = axes.ravel()
        if I_funct is None:
            I_funct = self.I_inject
        I_inject = self.get_I(t,I_funct)
        
        self.plotSimulation(t,Y,I_inject,axes[0],axes[1],axes[2])
        plt.tight_layout()
        fig.subplots_adjust(top=0.88)
        return None
        
    def plotVectorField(self,XY,xy):
        '''
        Desc: Plots Vector Field Calculated by Method in this class
        Inputs: Outputs of vectorField()
        Output: Graph
        '''
        VM,N,M,H = XY
        dvm, dn, dm, dh = xy
        
        pp = [(N[:,:,0,0],VM[:,:,0,0],dn[:,:,0,0],dvm[:,:,0,0]),
              (M[:,0,:,0],VM[:,0,:,0],dm[:,0,:,0],dvm[:,0,:,0]),
              (H[:,0,0,:],VM[:,0,0,:],dh[:,0,0,:],dvm[:,0,0,:])]
        lbls = [('$dn/dt$','$dV_M/dt$'),
               ('$dm/dt$','$dV_M/dt$'),
               ('$dh/dt$','$dV_M/dt$')]
        xlims = [(0,1),(0,1),(0,1)]
        ylims = [(-50,150),(None,None),(None,None)]
        fig, axes = plt.subplots(1, 3, facecolor='#d3d3d3',
                                      figsize=(15,5))
        axes = axes.ravel()
        
        for i,ax in enumerate(axes):
            ttl = lbls[i][1] + ' vs ' + lbls[i][0]
            ax.quiver(*pp[i],color='tab:orange',angles='xy')
            ax.set_title(ttl,fontweight='bold')
            ax.set_ylim(ylims[i])
            ax.set_xlim(xlims[i])
            ax.set_ylabel(lbls[i][1])
            ax.set_xlabel(lbls[i][0])
            ax.grid(True)
        
        plt.tight_layout()
        return None      
    
    def plotPhasePlane(self,Y,PP):
        '''
        Desc: Plots Output of PhasePlane function
        Inputs: Y output of simulation, PP output of phasePlane
        Output: Graphs
        '''
        c = ['b','r','m','tab:orange']
        ttl = ['Membrane Voltage\nPhase Plot',
               'Potassium Gaiting\nPhase Plot',
               'Sodium Activation\nPhase Plot',
               'Sodium Inactivation\nPhase Plot']
        ylbls = ['dVm/dt','dn/dt','dm/dt','dh/dt']
        xlbls = ['Membrane Voltage ($V_m$)',
                 'Potassium Gaiting Value ($n$)',
                 'Sodium Activation Value ($m$)',
                 'Sodium Inactivation Value ($h$)']
        fig, axes = plt.subplots(1, 4, facecolor='#d3d3d3',
                                      figsize=(15,4))
        axes = axes.ravel()
        for i,ax in enumerate(axes):
            ax.plot(Y[i,:],PP[i,:],color=c[i])
            ax.set_title(ttl[i],fontweight='bold')
            ax.set_ylabel(ylbls[i])
            ax.set_xlabel(xlbls[i])
            ax.grid(True)
        plt.tight_layout()
        return None
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        