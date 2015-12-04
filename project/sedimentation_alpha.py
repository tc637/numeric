# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 08:28:57 2015

@author: Tim
"""

import numpy as np
#mport matplotlib
import matplotlib.pyplot as plt
import yaml
from collections import namedtuple
from scipy import special as sp
#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# make new yaml file for project

plt.close('all')
plt.style.use('ggplot')

initialize = False
yaml_name = 'sedimentation.yaml'
fontsize=28
ticksize=20
# function to load/restart yaml file, and initialize the dictionaries for easy edits
def reset_params():
    with open(yaml_name, "r") as yaml_file:
        yaml_all = yaml.load(yaml_file)
        uservars_init = yaml_all["uservars"]
        initvars_init = yaml_all["initvars"]
    return uservars_init, initvars_init
    
    
def upstream(N_array, M_array, i, j, Vn, Vm):
        N_array[i+1,j] = N_array[i,j] + dt/dz*(Vn[i,j+1]*N_array[i,j+1] - 
                        Vn[i,j]*N_array[i,j])
    
        M_array[i+1,j] = M_array[i,j] + dt/dz*(Vm[i,j+1]*M_array[i,j+1] - 
                        Vm[i,j]*M_array[i,j])

        return N_array, M_array
  
  
def predictor(N_array, M_array, N_tilda, M_tilda, j, Vn, Vm, n_grid):

    if j != 0 or j != n_grid-1:
        N_tilda[0,j] = N_array[0,j] + dt/(2*dz)*(Vn[0,j-1]*N_array[0,j-1] - 
            Vn[i,j+1]*N_array[i,j+1])
        M_tilda[0,j] = M_array[0,j] + dt/(2*dz)*(Vm[0,j-1]*M_array[0,j-1] - 
            Vm[0,j+1]*M_array[0,j+1])
    
        if N_tilda[0,j] < 0:
            N_tilda[0,j] = 0
        elif N_tilda[0,j] > np.max(N_array[0,:]):
            N_tilda[0,j] = np.max(N_array[0,:])
        
        if M_tilda[0,j] < 0:
            M_tilda[0,j] = 0
        elif M_tilda[0,j] > np.max(M_array[0,:]):
            M_tilda[0,j] = np.max(M_array[0,:])
            
    return N_tilda,M_tilda
        
        
def mid_solution(N_array, M_array, N_tilda, M_tilda):
    N_half = np.mean([N_tilda[0,:],N_array[0,:]],axis=0)
    M_half = np.mean([M_tilda[0,:],M_array[0,:]],axis=0)
    
    return N_half,M_half
    
    
def corrector(N_array, M_array, N_half, M_half, j, Vn, Vm, n_grid):
    
    if j != 0 or j != n_grid-1:
        N_array[1,j] = N_array[0,j] + dt/(2*dz)*(Vn[0,j-1]*N_half[j-1] - 
            Vn[0,j+1]*N_half[j+1])
        M_array[1,j] = M_array[0,j] + dt/(2*dz)*(Vm[0,j-1]*M_half[j-1] - 
            Vm[0,j+1]*M_half[j+1])
            
        if N_array[1,j] < 0:
            N_array[1,j] = 0
        elif N_array[1,j] > np.max(N_array[0,:]):
            N_array[1,j] = np.max(N_array[0,:])
        
        if M_array[1,j] < 0:
            M_array[1,j] = 0
        elif M_array[1,j] > np.max(M_array[0,:]):
            M_array[1,j] = np.max(M_array[0,:])
    
    return N_array,M_array

    
def leapfrog(N_array, M_array, i, j, Vn, Vm, n_grid):
    
    if j != 0 or j != n_grid-1:
        N_array[i+1,j] = N_array[i-1,j] + dt/dz*(Vn[i,j-1]*N_array[i,j-1] - 
            Vn[i,j+1]*N_array[i,j+1])
        M_array[i+1,j] = M_array[i-1,j] + dt/dz*(Vm[i,j-1]*M_array[i,j-1] - 
            Vm[i,j+1]*M_array[i,j+1])
            
        if N_array[i,j] < 0:
            N_array[i,j] = 0
        elif N_array[i,j] > np.max(N_array[i,:]):
            N_array[i,j] = np.max(N_array[i,:])
        
        if M_array[i,j] < 0:
            M_array[i,j] = 0
        elif M_array[i,j] > np.max(M_array[i,:]):
            M_array[i,j] = np.max(M_array[i,:])

    return N_array,M_array
    
def set_labels(axis, titlestr, xlabel, ylabel, fontsize, ticksize, legend=True):
    
    axis.set_title(titlestr, fontsize=fontsize, fontweight="bold")
    axis.set_xlabel(xlabel, fontsize=fontsize, fontweight="bold")
    axis.set_ylabel(ylabel, fontsize=fontsize, fontweight="bold")
    axis.tick_params(axis='x',labelsize=ticksize)
    axis.tick_params(axis='y',labelsize=ticksize)
    
    if legend:
        axis.legend(loc="lower right", fontsize=fontsize)

    return None
    

if initialize:
    timevars={'timevars':{'dt':3.,'tstart':0.0,'tend':240.0}}
    # constants
    uservars={'uservars':{'rho_w':1000, 'rho_a':1.225, 'a':842, 'b':0.8, 'alpha':0}} 
    
    # initial conditions and spatial parameters
    initvars={'initvars':{'M':1e-3, 'N':1e4*1.225, 'Nbot':4300, 'Ntop':5050, 'zmin':0,
                          'zmax':6000,'dz':100.}} 
                          
    # adaptive timestep
    adaptvars={'adaptvars':{'dtpassmin':0.1,'dtfailmax':0.5,'dtfailmin':0.1,'s':0.9,'rtol':1.0e-05,
               'atol':1.0e-05,'maxsteps':2000.0,'maxfail':60.0,'dtpassmax':5.0}}
    
    with open(yaml_name,'w') as f:
        f.write(yaml.dump(timevars,default_flow_style=False))
        f.write(yaml.dump(uservars,default_flow_style=False))
        f.write(yaml.dump(initvars,default_flow_style=False))
        f.write(yaml.dump(adaptvars,default_flow_style=False))   


with open(yaml_name, 'rb') as f:
    config = yaml.load(f)
    
    timevars = namedtuple('timevars',config['timevars'].keys())
    timevars = timevars(**config['timevars'])
    # read in dtpassmin dtpassmax dtfailmin dtfailmax s rtol atol maxsteps maxfail
    adaptvars = namedtuple('adaptvars', config['adaptvars'].keys())
    adaptvars = adaptvars(**config['adaptvars'])
    
    
    uservars = namedtuple('uservars', config['uservars'].keys())
    uservars = uservars(**config['uservars'])
   
    initvars = namedtuple('initvars', config['initvars'].keys())
    initvars = initvars(**config['initvars'])
            
    yinit = np.array([initvars.M, initvars.N])
    nvars = len(yinit)
    
    n_grid = round((initvars.zmax - initvars.zmin)/initvars.dz)
    n_time = round((timevars.tend - timevars.tstart)/timevars.dt)
    rainbot = round((initvars.Nbot - initvars.zmin)/initvars.dz)
    raintop = round((initvars.Ntop - initvars.zmin)/initvars.dz)

# ====================================================================
# bulk upstream

M_array = np.zeros([n_time, n_grid])
N_array = np.zeros([n_time, n_grid])

M_array[0,rainbot:raintop+1] = initvars.M
M_array[:, 0] = 0
M_array[:, -1] = 0
N_array[0,rainbot:raintop+1] = initvars.N
N_array[:, 0] = 0
N_array[:, -1] = 0

a = uservars.a
b = uservars.b
alpha = uservars.alpha
rho_w = uservars.rho_w
rho_a = uservars.rho_a
gamma = np.pi/6.*rho_w*sp.gamma(alpha+4)
beta = sp.gamma(alpha+1)
dt = timevars.dt
dz = initvars.dz
    
Vm = np.zeros([n_time, n_grid])
Vn = np.zeros([n_time, n_grid])

for i in range(0,n_time-1):

    for j in range(0,n_grid):
        if M_array[i,j] <= 0. or N_array[i,j] <= 0.:
            lamb = 0.
            N0 = 0.
            Vn[i,j] = 0.
            Vm[i,j] = 0.
        else:
            lamb = ((gamma*N_array[i,j])/(beta*M_array[i,j]))**(1./3)
            N0 = N_array[i,j]/beta*lamb**(alpha+1)
            Vn[i,j] = (a/lamb**b)*sp.gamma(alpha+b+1)/sp.gamma(alpha+1)
            Vm[i,j] = (a/lamb**b)*sp.gamma(alpha+b+4)/sp.gamma(alpha+4)

        
        if Vm[i,j] > 10:
            Vm[i,j] = 10
        
        if Vn[i,j] > 10:
            Vn[i,j] = 10
        
    for j in range(0,n_grid-1):
        N_array,M_array = upstream(N_array, M_array, i, j, Vn, Vm)

fig1,ax1 = plt.subplots(1,1,figsize=(6,6))
fig2,ax2 = plt.subplots(1,1,figsize=(6,6))

for i in np.arange(0,n_time+dt,5*dt):
    if i == 0:
        ax1.plot(M_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz), '--g',label="t = {} s".format(i*dt))
        ax2.plot(N_array[i,:]/rho_a,np.arange(initvars.zmin,initvars.zmax,dz), '--g',label="t = {} s".format(i*dt))
    else:
        ax1.plot(M_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz),label="t = {} s".format(i*dt))
        ax2.plot(N_array[i,:]/rho_a,np.arange(initvars.zmin,initvars.zmax,dz),label="t = {} s".format(i*dt))
ax1.set_xlim([0, 1.5])
ax2.set_xlim([0, 15000])
ax1.set_title(r"Advection of M with the Bulk Scheme, $\alpha$ = {}".format(alpha))
ax2.set_title(r"Advection of N with the Bulk Scheme, $\alpha$ = {}".format(alpha))
ax1.set(xlabel=r"$M\ (g\ kg^{-1})$", ylabel=r"$z\ (m)$")
ax2.set(xlabel=r"$N\ (kg^{-1})$", ylabel=r"$z\ (m)$")
ax1.legend(loc="lower right")
ax2.legend(loc="lower right")


#################################
# alpha = 1
M_array = np.zeros([n_time, n_grid])
N_array = np.zeros([n_time, n_grid])

M_array[0,rainbot:raintop+1] = initvars.M
M_array[:, 0] = 0
M_array[:, -1] = 0
N_array[0,rainbot:raintop+1] = initvars.N
N_array[:, 0] = 0
N_array[:, -1] = 0

a = uservars.a
b = uservars.b
alpha = 10
rho_w = uservars.rho_w
rho_a = uservars.rho_a
gamma = np.pi/6.*rho_w*sp.gamma(alpha+4)
beta = sp.gamma(alpha+1)
dt = timevars.dt
dz = initvars.dz
    
Vm = np.zeros([n_time, n_grid])
Vn = np.zeros([n_time, n_grid])

for i in range(0,n_time-1):

    for j in range(0,n_grid):
        if M_array[i,j] <= 0. or N_array[i,j] <= 0.:
            lamb = 0.
            N0 = 0.
            Vn[i,j] = 0.
            Vm[i,j] = 0.
        else:
            lamb = ((gamma*N_array[i,j])/(beta*M_array[i,j]))**(1./3)
            N0 = N_array[i,j]/beta*lamb**(alpha+1)
            Vn[i,j] = (a/lamb**b)*sp.gamma(alpha+b+1)/sp.gamma(alpha+1)
            Vm[i,j] = (a/lamb**b)*sp.gamma(alpha+b+4)/sp.gamma(alpha+4)

        
        if Vm[i,j] > 10:
            Vm[i,j] = 10
        
        if Vn[i,j] > 10:
            Vn[i,j] = 10
        
    for j in range(0,n_grid-1):
        N_array,M_array = upstream(N_array, M_array, i, j, Vn, Vm)

fig3,ax3 = plt.subplots(1,1,figsize=(6,6))
fig4,ax4 = plt.subplots(1,1,figsize=(6,6))

for i in np.arange(0,n_time+dt,5*dt):
    if i == 0:
        ax3.plot(M_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz), '--g',label="t = {} s".format(i*dt))
        ax4.plot(N_array[i,:]/rho_a,np.arange(initvars.zmin,initvars.zmax,dz), '--g',label="t = {} s".format(i*dt))
    else:
        ax3.plot(M_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz),label="t = {} s".format(i*dt))
        ax4.plot(N_array[i,:]/rho_a,np.arange(initvars.zmin,initvars.zmax,dz),label="t = {} s".format(i*dt))
ax3.set_xlim([0, 1.5])
ax4.set_xlim([0, 15000])
ax3.set_title(r"Advection of M with the Bulk Scheme, $\alpha$ = {}".format(alpha))
ax4.set_title(r"Advection of N with the Bulk Scheme, $\alpha$ = {}".format(alpha))
ax3.set(xlabel=r"$M\ (g\ kg^{-1})$", ylabel=r"$z\ (m)$")
ax4.set(xlabel=r"$N\ (kg^{-1})$", ylabel=r"$z\ (m)$")
ax3.legend(loc="lower right")
ax4.legend(loc="lower right")

########################################


a = uservars.a
b = uservars.b
alphas = np.arange(0,5.1,0.5)
rho_w = uservars.rho_w
rho_a = uservars.rho_a
gamma = np.pi/6.*rho_w*sp.gamma(alpha+4)
beta = sp.gamma(alpha+1)
dt = timevars.dt
dz = initvars.dz
    
end_gameN = np.zeros([alphas.shape[0],n_grid])
end_gameM = np.zeros([alphas.shape[0],n_grid])

for alpha_ind in np.arange(alphas.shape[0]):
    alpha = alphas[alpha_ind]
    M_array = np.zeros([n_time, n_grid])
    N_array = np.zeros([n_time, n_grid])
    
    M_array[0,rainbot:raintop+1] = initvars.M
    M_array[:, 0] = 0
    M_array[:, -1] = 0
    N_array[0,rainbot:raintop+1] = initvars.N
    N_array[:, 0] = 0
    N_array[:, -1] = 0
    Vm = np.zeros([n_time, n_grid])
    Vn = np.zeros([n_time, n_grid])

    for i in range(0,n_time-1):
    
        for j in range(0,n_grid):
            if M_array[i,j] <= 0. or N_array[i,j] <= 0.:
                lamb = 0.
                N0 = 0.
                Vn[i,j] = 0.
                Vm[i,j] = 0.
            else:
                lamb = ((gamma*N_array[i,j])/(beta*M_array[i,j]))**(1./3)
                N0 = N_array[i,j]/beta*lamb**(alpha+1)
                Vn[i,j] = (a/lamb**b)*sp.gamma(alpha+b+1)/sp.gamma(alpha+1)
                Vm[i,j] = (a/lamb**b)*sp.gamma(alpha+b+4)/sp.gamma(alpha+4)
    
            
            if Vm[i,j] > 10:
                Vm[i,j] = 10
            
            if Vn[i,j] > 10:
                Vn[i,j] = 10
            
        for j in range(0,n_grid-1):
            N_array,M_array = upstream(N_array, M_array, i, j, Vn, Vm)

    end_gameN[alpha_ind, :] = N_array[-1,:]
    end_gameM[alpha_ind, :] = M_array[-1,:]
    
fig5,ax5 = plt.subplots(1,1,figsize=(6,6))
fig6,ax6 = plt.subplots(1,1,figsize=(6,6))

ax5.plot(M_array[0,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz), '--g',label="t = {} s".format(0))
ax6.plot(N_array[0,:]/rho_a,np.arange(initvars.zmin,initvars.zmax,dz), '--g',label="t = {} s".format(0))
        
for alpha_ind in np.arange(alphas.shape[0]):
    ax5.plot(end_gameM[alpha_ind,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz),label=r"$\alpha\ =\ {}$".format(alphas[alpha_ind]))
    ax6.plot(end_gameN[alpha_ind,:]/rho_a,np.arange(initvars.zmin,initvars.zmax,dz),label=r"$\alpha\ =\ {}$".format(alphas[alpha_ind]))
ax5.set_xlim([0, 1.5])
ax6.set_xlim([0, 15000])

titlestr = r"Advection of M with the Upstream-Bulk Scheme, t = {} s".format(n_time*dt)
xlabel = r"$M\ (g\ kg^{-1})$"
ylabel = r"$z\ (m)$"
set_labels(ax5, titlestr, xlabel, ylabel, fontsize, ticksize, legend=False)

titlestr = r"Advection of N with the Upstream-Bulk Scheme, t = {} s".format(n_time*dt)
xlabel = r"$N\ (kg^{-1})$"
ylabel = r"$z\ (m)$"
set_labels(ax6, titlestr, xlabel, ylabel, fontsize, ticksize, legend=False)

#####################
# courant
a = uservars.a
b = uservars.b
alphas1 = np.arange(0,0.5,0.05)
alphas2 = np.arange(0.5,10.1,0.5)
alphas = np.append(alphas1, alphas2)
rho_w = uservars.rho_w
rho_a = uservars.rho_a
gamma = np.pi/6.*rho_w*sp.gamma(alpha+4)
beta = sp.gamma(alpha+1)
dt = timevars.dt
dz = initvars.dz
    


max_courm = []
max_courn = []

for alpha_ind in np.arange(alphas.shape[0]):
    alpha = alphas[alpha_ind]
    M_array = np.zeros([n_time, n_grid])
    N_array = np.zeros([n_time, n_grid])
    
    M_array[0,rainbot:raintop+1] = initvars.M
    M_array[:, 0] = 0
    M_array[:, -1] = 0
    N_array[0,rainbot:raintop+1] = initvars.N
    N_array[:, 0] = 0
    N_array[:, -1] = 0
    Vm = np.zeros([n_time, n_grid])
    Vn = np.zeros([n_time, n_grid])

    courm = np.zeros_like(Vm)
    courn = np.zeros_like(Vn)
    
    for i in range(0,n_time-1):
    
        for j in range(0,n_grid):
            if M_array[i,j] <= 0. or N_array[i,j] <= 0.:
                lamb = 0.
                N0 = 0.
                Vn[i,j] = 0.
                Vm[i,j] = 0.
            else:
                lamb = ((gamma*N_array[i,j])/(beta*M_array[i,j]))**(1./3)
                N0 = N_array[i,j]/beta*lamb**(alpha+1)
                Vn[i,j] = (a/lamb**b)*sp.gamma(alpha+b+1)/sp.gamma(alpha+1)
                Vm[i,j] = (a/lamb**b)*sp.gamma(alpha+b+4)/sp.gamma(alpha+4)
    
            
            courn[i,j] = Vn[i,j]*dt/dz
            courm[i,j] = Vm[i,j]*dt/dz     
       
        for j in range(0,n_grid-1):
            N_array,M_array = upstream(N_array, M_array, i, j, Vn, Vm)


    max_courm.append(np.max(courm))
    max_courn.append(np.max(courn))
    
fig7,ax7 = plt.subplots(1,1,figsize=(6,6))
fig8,ax8 = plt.subplots(1,1,figsize=(6,6))

ax7.plot(alphas,max_courm,'or',label=r"$\alpha\ =\ {}$".format(alphas[alpha_ind]))
ax8.plot(alphas,max_courn,'ob',label=r"$\alpha\ =\ {}$".format(alphas[alpha_ind]))
ax7.set_ylim([0,5])
ax8.set_ylim([0,5])
#titlestr = r"Peak Courant Numbers of M with the Upstream-Bulk Scheme, Changing $\alpha$"
titlestr = ""
ylabel = r"$C$"
xlabel = r"$\alpha$"
set_labels(ax7, titlestr, xlabel, ylabel, fontsize, ticksize, legend=False)

#titlestr = r"Peak Courant Numbers of N with the Upstream-Bulk Scheme, Changing $\alpha$"
titlestr = ""
ylabel = r"$C$"
xlabel = r"$\alpha$"
set_labels(ax8, titlestr, xlabel, ylabel, fontsize, ticksize, legend=False)





fig9,ax9 = plt.subplots(1,1,figsize=(6,6))
fig10,ax10 = plt.subplots(1,1,figsize=(6,6))
for i in np.arange(0,n_time+dt,5*dt):
    if i == 0:
        ax9.plot(M_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz), '--g',label="t = {} s".format(i*dt))
        ax10.plot(N_array[i,:]/rho_a,np.arange(initvars.zmin,initvars.zmax,dz), '--g',label="t = {} s".format(i*dt))
    else:
        ax9.plot(M_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz),label="t = {} s".format(i*dt))
        ax10.plot(N_array[i,:]/rho_a,np.arange(initvars.zmin,initvars.zmax,dz),label="t = {} s".format(i*dt))

ax9.set_xlim([0, 1.5])
ax10.set_xlim([0, 15000])
titlestr = r"Advection of M with the Upstream-Bulk Scheme, $\alpha$ = 10, $V_m$ Unbounded"
xlabel = r"$M\ (g\ kg^{-1})$"
ylabel = r"$z\ (m)$"
set_labels(ax9, titlestr, xlabel, ylabel, fontsize, ticksize, legend=True)

titlestr = r"Advection of N with the Upstream-Bulk Scheme, $\alpha$ = 10, $V_N$ Unbounded"
xlabel = r"$N\ (kg^{-1})$"
ylabel = r"$z\ (m)$"
set_labels(ax10, titlestr, xlabel, ylabel, fontsize, ticksize, legend=True)


maxn = []
maxm = []
for eachn in courn:
    maxn.append((np.max(eachn)))
for eachm in courm:
    maxm.append((np.max(eachm)))
fig11,ax11 = plt.subplots(1,1,figsize=(6,6))
fig12,ax12 = plt.subplots(1,1,figsize=(6,6))
ax11.plot(dt*np.arange(0,n_time),maxm)
ax12.plot(dt*np.arange(0,n_time),maxn,'-b')