'''Simulation Functions for Cart Pole Simulation And Control Project

Versions Used for Imports:
Python Version 3.7.3, Matplotlib Version 3.0.3, Scipy Version 1.4.1,
Numpy Version 1.16.2

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import colors as mcolors
from itertools import cycle
from matplotlib import animation
from scipy.linalg import solve_continuous_are, inv, eig

def dX_dt(t,X,M,m,L,g,F):
    '''Nonlinear System Model
    Desc: Calculate X dot for Odeint and Simulations
    Inputs: X = state variable vector (x,x dot,theta,theta dot)
            t =  time vector (sec)
            M = Mass of cart (kg)
            m = Mass of pole at center of mass (kg)
            L = Length from pivot to center of mass (meters)
            g = acceleration due to gravity (m/s^2)
            F = Force in the positive x direction on cart (N)
    Output: X_dot = derivative of state vector 
                    (x dot,x dbl dot, dot theta,theta dbl dot)
            
            uses: none'''
    
    # break out for clairity
    x,dx,theta,dtheta = X 

    # compute for speed
    S = np.sin(theta)
    C = np.cos(theta)
    
    # compute X dot
    dx_dt = dx
    ddx_dt = (F + m*S*(L*dtheta*dtheta + g*C))/(M + m*S*S)
    dtheta_dt = dtheta
    ddtheta_dt = ( -F*C - m*L*dtheta*dtheta*S*C - (M + m)*g*S )/(L*(M+m*S*S))
    
    X_dot = [dx_dt,ddx_dt,dtheta_dt,ddtheta_dt]
    return X_dot

def dX_dt_control(t,X,M,m,L,g,K):
    '''Nonlinear System Model with controller
    Desc: Calculate X dot for Odeint and Simulations With Control Law K
    Inputs: X = state variable vector (x,x dot,theta,theta dot)
            t =  time vector (sec)
            M = Mass of cart (kg)
            m = Mass of pole at center of mass (kg)
            L = Length from pivot to center of mass (meters)
            g = acceleration due to gravity (m/s^2)
            K = Control Law Matix
    Output: X_dot = derivative of state vector 
                    (x dot,x dbl dot, dot theta,theta dbl dot)
            
            uses: none'''
    
    # break out for clairity
    x,dx,theta,dtheta = X 
    
    #compute control law
    F = np.matrix(-K)*np.matrix(X).T

    # compute for speed
    S = np.sin(theta)
    C = np.cos(theta)
    
    # compute X dot
    dx_dt = dx
    ddx_dt = (F + m*S*(L*dtheta*dtheta + g*C))/(M + m*S*S)
    dtheta_dt = dtheta
    ddtheta_dt = ( -F*C - m*L*dtheta*dtheta*S*C - (M + m)*g*S )/(L*(M+m*S*S))
    
    X_dot = [dx_dt,ddx_dt,dtheta_dt,ddtheta_dt]
    return X_dot
#=================================================================================

def simulate(X0, t, M = 1, m = 0.1,
             L = 0.5,g = -9.8 ,F = 0):
    '''
    Desc: Simulate system Cart Pole System
    Inputs: X0 = initial conditions (x, x dot, theta, theta dot)
            t =  time vector (sec)
            M = Mass of cart (kg)
            m = Mass of pole at center of mass (kg)
            L = Length from pivot to center of mass (meters)
            g = acceleration due to gravity (m/s^2)
            K = Control Law Matrix for Force on Cart
    Output: x_pend = x coordinate of pendulum (np.array)
            y_pend = y coordinate of pendulum (np.array)
            x_cart = x coordinate of pendulem (np.array)
            X = state variable vector (x,x dot,theta,theta dot)
            uses: none
    '''
    # Constants
    params = (M,m,L,g,F)
    # run simulation and calculate other vars
    X = solve_ivp(dX_dt,(t[0],t[-1]),X0,t_eval = t, args=params,
                      method = 'RK23').y
    x_cart = X[0,:]
    y_pend = L*np.cos(X[2,:])
    x_pend = L*np.sin(X[2,:]) + X[0,:]
    return x_pend,y_pend,x_cart,X

#==============================================================

def simulateControl(X0, t, K, step=1e-4, M = 1,m = 0.1,
             L = 0.5,g = -9.8):
    '''
    Desc: Simulate system Cart Pole With Control System
    Inputs: X0 = initial conditions (x, x dot, theta, theta dot)
            t =  time vector (sec)
            step = min step size for solver
            M = Mass of cart (kg)
            m = Mass of pole at center of mass (kg)
            L = Length from pivot to center of mass (meters)
            g = acceleration due to gravity (m/s^2)
            K = Control Matrix for Force in the positive x direction on cart
    Output: x_pend = x coordinate of pendulum (np.array)
            y_pend = y coordinate of pendulum (np.array)
            x_cart = x coordinate of pendulem (np.array)
            X = state variable vector (x,x dot,theta,theta dot)
            uses: none
    '''
    # Constants
    params = (M,m,L,g,K)
    # run simulation and calculate other vars
    X = solve_ivp(dX_dt_control,(t[0],t[-1]),X0,t_eval = t, args=params,
                      method = 'RK23').y
    x_cart = X[0,:]
    y_pend = L*np.cos(X[2,:])
    x_pend = L*np.sin(X[2,:]) + X[0,:]
    return x_pend,y_pend,x_cart,X


def buildAnimation(t,x_pend,y_pend,x_cart):
    '''
    Desc: Build Animation Object for Cart and Pendulum
    Input: t = time vector,
           x_pend = x coordinate of pendulum
           y_pend = y coordinate of pendulum
           x_cart = x coordinate of cart
    Output: animation object
    Uses: none
    '''
    # Build plot
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 1), ylim=(-1,1))
    ax.grid()

    # Init Objects for Animation
    cart, = ax.plot([],[],linestyle='None',marker='s', markersize=40,
                    markeredgecolor='k', color='r',markeredgewidth=2)
    mass, = ax.plot([],[],linestyle='None',marker='o', markersize=20,
                    markeredgecolor='k', color='b',markeredgewidth=1)
    line, = ax.plot([],[],'o-',color='k',lw=4, markersize=6,
                    markeredgecolor='k', markerfacecolor='k')
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05,0.9,'',transform=ax.transAxes)

    # Animation Functions
    def init():
        cart.set_data([],[])
        mass.set_data([],[])
        line.set_data([],[])
        time_text.set_text('')
        return line, cart, mass, time_text

    def animate(i):
        cart.set_data([x_cart[i],[0]])
        mass.set_data([x_pend[i]],[y_pend[i]])
        line.set_data([x_cart[i],x_pend[i]],[0,y_pend[i]])
        time_text.set_text(time_template % t[i])
        return line, cart, mass, time_text

    # Animate
    anim = animation.FuncAnimation(fig, animate, np.arange(0,len(t)),
                                    interval=42,blit=False,init_func=init)
    return anim


def plotState(t,X):
    '''
    Desc: Plots State Variables vs Time for Cart Pole Simulation
    Inputs: t = time vector
            X = state variable vector (x,x dot,theta,theta dot)
    Outputs: Plots
    Uses: 
    '''
    ROWS = 4
    COL = 1
    COLORS = cycle(mcolors.TABLEAU_COLORS.items())
    f, axes = plt.subplots(ROWS,COL,sharex=True)
    f.set_size_inches(10,5)
    axes = axes.flatten()
    T = ['Cart Position vs Time',
        'Cart Velocity vs Time',
        'Pole Angle vs Time',
        'Pole Angular Velocity vs Time']
    yl = ['Meters','Meters','Radians','Radians']

    for i in range(0,X.shape[0]):
        axes[i].plot(t,X[i,:],color=next(COLORS)[1])
        axes[i].set_title(T[i],fontweight='bold')
        axes[i].set_ylabel(yl[i])
        axes[i].grid()
    axes[-1].set_xlabel('Time (sec)')
    plt.tight_layout()
    
def plotStateForce(t,X,u):
    '''
    Desc: Plots State Variables and Force vs Time for Cart Pole Simulation
    Inputs: t = time vector
            X = state variable vector (x,x dot,theta,theta dot)
            K = control law matrix
    Outputs: Plots
    Uses: 
    '''
    ROWS = 5
    COL = 1
    COLORS = cycle(mcolors.TABLEAU_COLORS.items())
    f, axes = plt.subplots(ROWS,COL,sharex=True)
    f.set_size_inches(10,7)
    axes = axes.flatten()
    T = ['Cart Position vs Time',
        'Cart Velocity vs Time',
        'Pole Angle vs Time',
        'Pole Angular Velocity vs Time',
        'Force on Cart vs Time']
    yl = ['Meters','Meters','Radians','Radians','Newtons']

    for i in range(0,X.shape[0]):
        axes[i].plot(t,X[i,:],color=next(COLORS)[1])
        axes[i].set_title(T[i],fontweight='bold')
        axes[i].set_ylabel(yl[i])
        axes[i].grid()
        
    axes[-1].plot(t,u,color=next(COLORS)[1])
    axes[-1].set_title(T[-1],fontweight='bold')
    axes[-1].set_ylabel(yl[-1])
    axes[-1].grid()
    axes[-1].set_xlabel('Time (sec)')
    plt.tight_layout()
    
    
def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    code from: http://www.mwm.im/lqr-controllers-with-python/
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = np.matrix(solve_continuous_are(A, B, Q, R))

    #compute the LQR gain
    K = np.matrix(inv(R)*(B.T*X))

    eigVals, eigVecs = eig(A-B*K)
    return K, eigVals