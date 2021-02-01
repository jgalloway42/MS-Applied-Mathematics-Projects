# -*- coding: utf-8 -*-
"""
Solvers for Numerical Analysis MiniProject 

@author: Josh.Galloway
"""
import numpy as np

class Solvers:
    def __init__(self):
        # no operations
        pass
    
    def rk4(self,t,dydx,x0,args=None):
        '''
        Desc: Runge-Kutta 4th order 
        Inputs: t = evenly spaced time vector
                dydx = function of derivitive
                ...with call dydx(time,X,optional arguments)
                x0= initial conditions vector for state variables
                args = tuple of optional arguments for dydx call
        Outputs: Y(t)
        '''
        N = np.size(t)
        dt = np.abs(t[1] - t[0]) #delta time
        n_vars = np.size(x0)
        y = np.zeros((n_vars,N))
        y[:,0] = x0
        
        if args is None:
            for i in range(1,N):
                yn = y[:,i-1]
                tn = t[i-1]
                k1 = dydx(tn,yn)
                k2 = dydx(tn + 0.5*dt,yn + 0.5*dt*k1)
                k3 = dydx(tn + 0.5*dt,yn + 0.5*dt*k2)
                k4 = dydx(tn + 0.5*dt,yn + 0.5*dt*k3)
                y[:,i] = yn + 1/6*dt*(k1+ 2*k2 + 2*k3 + k4)
        else:
            for i in range(1,N):
                yn = y[:,i-1]
                tn = t[i-1]
                k1 = dydx(tn,yn,*args)
                k2 = dydx(tn + 0.5*dt,yn + 0.5*dt*k1,*args)
                k3 = dydx(tn + 0.5*dt,yn + 0.5*dt*k2,*args)
                k4 = dydx(tn + 0.5*dt,yn + 0.5*dt*k3,*args)
                y[:,i] = yn + 1/6*dt*(k1+ 2*k2 + 2*k3 + k4)
        return y
  
    def forwardEuler(self,t,dydx,x0,args=None):
        '''
        Desc: Forward Euler 
        Inputs: t = evenly spaced time vector
                dydx = function of derivitive
                ...with call dydx(time,X,optional arguments)
                x0= initial conditions vector for state variables
                args = tuple of optional arguments for dydx call
        Outputs: Y(t)
        '''
        N = np.size(t)
        dt = np.abs(t[1] - t[0]) #delta time
        n_vars = np.size(x0)
        y = np.zeros((n_vars,N))
        y[:,0] = x0
        
        if args is None:
            for i in range(1,N):
                yn = y[:,i-1]
                tn = t[i-1]
                y[:,i] = yn + dt*dydx(tn,yn)
        else:
            for i in range(1,N):
                yn = y[:,i-1]
                tn = t[i-1]
                y[:,i] = yn + dt*dydx(tn,yn,*args)
        return y
  
    def heunsMethod(self,t,dydx,x0,args=None):
        '''
        Desc: Heun's Method as presented in lecture 
        Inputs: t = evenly spaced time vector
                dydx = function of derivitive
                ...with call dydx(time,X,optional arguments)
                x0= initial conditions vector for state variables
                args = tuple of optional arguments for dydx call
        Outputs: Y(t)
        '''
        N = np.size(t)
        dt = np.abs(t[1] - t[0]) #delta time
        n_vars = np.size(x0)
        y = np.zeros((n_vars,N))
        y[:,0] = x0
        
        if args is None:
            for i in range(1,N):
                yn = y[:,i-1]
                tn = t[i-1]
                tnp1 = t[i]
                y_bar_np1 = yn + dt*dydx(tn,yn)
                
                y[:,i] = yn + dt/2*(dydx(tn,yn) + dydx(tnp1,y_bar_np1))
        else:
            for i in range(1,N):
                yn = y[:,i-1]
                tn = t[i-1]
                tnp1 = t[i]
                y_bar_np1 = yn + dt*dydx(tn,yn,*args)
                
                y[:,i] = yn + dt/2*(dydx(tn,yn,*args) \
                                 + dydx(tnp1,y_bar_np1,*args))
        return y
    
    def f_test(self,t,x):
        '''Test System Stable at X = [0,0]
           negative real eigenvalues'''
        A = np.array([[0.,1.],[-2.,-3.]])
        Y = np.matmul(A,x)
        return Y
