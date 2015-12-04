# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:26:24 2015

@author: Tim
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from collections import namedtuple
from scipy import special as sp
# make new yaml file for project

plt.close('all')
plt.style.use('ggplot')

initialize = False
yaml_name = 'sedimentation.yaml'
fontsize = 28
ticksize = 20

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
        N_tilda[0,j] = N_array[0,j] - dt/(2*dz)*(Vn[0,j-1]*N_array[0,j-1] - 
            Vn[i,j+1]*N_array[i,j+1])
        M_tilda[0,j] = M_array[0,j] - dt/(2*dz)*(Vm[0,j-1]*M_array[0,j-1] - 
            Vm[0,j+1]*M_array[0,j+1])
    
        if N_tilda[0,j] < 0:
            N_tilda[0,j] = -N_tilda[0,j]
        elif N_tilda[0,j] > np.max(N_array[0,:]):
            N_tilda[0,j] = np.max(N_array[0,:])
        
        if M_tilda[0,j] < 0:
            M_tilda[0,j] = -M_tilda[0,j]
        elif M_tilda[0,j] > np.max(M_array[0,:]):
            M_tilda[0,j] = np.max(M_array[0,:])
            
    return N_tilda,M_tilda
        
        
def mid_solution(N_array, M_array, N_tilda, M_tilda):
    N_half = np.mean([N_tilda[0,:],N_array[0,:]],axis=0)
    M_half = np.mean([M_tilda[0,:],M_array[0,:]],axis=0)
    
    return N_half,M_half
    
    
def corrector(N_array, M_array, N_half, M_half, j, Vn, Vm, n_grid):
    
    if j != 0 or j != n_grid-1:
        N_array[1,j] = N_array[0,j] - dt/(2*dz)*(Vn[0,j-1]*N_half[j-1] - 
            Vn[0,j+1]*N_half[j+1])
        M_array[1,j] = M_array[0,j] - dt/(2*dz)*(Vm[0,j-1]*M_half[j-1] - 
            Vm[0,j+1]*M_half[j+1])
            
        if N_array[1,j] < 0:
            N_array[1,j] = -N_array[1,j]
        elif N_array[1,j] > np.max(N_array[0,:]):
            N_array[1,j] = np.max(N_array[0,:])
        
        if M_array[1,j] < 0:
            M_array[1,j] = -M_array[1,j]
        elif M_array[1,j] > np.max(M_array[0,:]):
            M_array[1,j] = np.max(M_array[0,:])
    
    return N_array,M_array

    
def leapfrog(N_array, M_array, i, j, Vn, Vm, n_grid):
    
    if j != 0 or j != n_grid-1:
        N_array[i+1,j] = N_array[i-1,j] - dt/dz*(Vn[i,j-1]*N_array[i,j-1] - 
            Vn[i,j+1]*N_array[i,j+1])
        M_array[i+1,j] = M_array[i-1,j] - dt/dz*(Vm[i,j-1]*M_array[i,j-1] - 
            Vm[i,j+1]*M_array[i,j+1])
            
        if N_array[i,j] < 0:
            N_array[i,j] = -N_array[i,j]
        elif N_array[i,j] > np.max(N_array[0,:]):
            N_array[i,j] = np.max(N_array[0,:])
        
        if M_array[i,j] < 0:
            M_array[i,j] = -M_array[i,j]
        elif M_array[i,j] > np.max(M_array[0,:]):
            M_array[i,j] = np.max(M_array[0,:])

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
    
###########################################################
def calc_nk(D, N0, alpha, lamb, dD):
    # D {mm}
    # N0 {m}
    # alpha {}
    # lamb {m}
    # dD {mm}


    D = D*1e-3 # convert to m
    dD = dD*1e-3 # convert to m
    
    nk = np.sum([N0*((each_D)**alpha)*np.exp(-lamb*each_D) for each_D in D])*dD
    
    return nk
    
def calc_mk(D, N0, alpha, lamb, rho_w, dD):
    # D {mm}
    # N0 {m}
    # alpha {}
    # lamb {m}
    # dD {mm}

    D = D*1e-3 # convert to m
    dD = dD*1e-3 # convert to m
    
    mk = np.sum([np.pi/6*rho_w*((each_D)**3)*N0*(each_D)**alpha*np.exp(-lamb*each_D) for each_D in D])*dD
    
    return mk
    
def calc_Vk(D, a, b):
    D = D*1e-3 # convert to m
    Vk = a*(np.mean(D)**b)
    
    return Vk
    
def upstream_bins(nk, mk, Vk, delta_D, Y, N_array, M_array, i, j):
    
    #N_temp = np.zeros([Y])
    #M_temp = np.zeros([Y])
    
    N_sum = 0
    M_sum = 0
    # convert delta_D to m
    delta_D = delta_D
    
    N_lim = np.max(N_array[i,:])
    M_lim = np.max(M_array[i,:])
    
    add_n = True
    add_m = True
    
    for k in range(0,Y):

        #N_temp[k] = (nk[i,j,k] + dt/dz*(Vk[k]*nk[i,j+1,k]-Vk[k]*nk[i,j,k]))
        #M_temp[k] = (mk[i,j,k] + dt/dz*(Vk[k]*mk[i,j+1,k]-Vk[k]*mk[i,j,k]))

        if add_n:
            n_current_bin =(nk[i,j,k] + dt/dz*(Vk[k]*nk[i,j+1,k]-Vk[k]*nk[i,j,k]))
            if (N_sum + n_current_bin) < N_lim: 
                N_sum = N_sum + n_current_bin
            #else:
                #add_n = False
            
        if add_m:
            m_current_bin = (mk[i,j,k] + dt/dz*(Vk[k]*mk[i,j+1,k]-Vk[k]*mk[i,j,k]))
            if (M_sum + m_current_bin) < M_lim:
                M_sum = M_sum + m_current_bin
            #else:
                #add_m = False
                
        if (add_n == False) and (add_m == False):
            print("Hello")
            print("i={} N={} M={}".format(i,N_sum,M_sum))
            break

            
    #N_array[i+1,j] = np.sum(N_temp)*delta_D
    #M_array[i+1,j] = np.sum(M_temp)*delta_D
            
    N_array[i+1,j] = N_sum        
    M_array[i+1,j] = M_sum
    
    return N_array, M_array
    
    
a = uservars.a
b = uservars.b
alpha = 10
rho_w = uservars.rho_w
rho_a = uservars.rho_a
gamma = np.pi/6.*rho_w*sp.gamma(alpha+4)
beta = sp.gamma(alpha+1)
dt = timevars.dt
dz = initvars.dz

dD = 0.01 # mm
D_array = np.arange(0,5,dD) # mm
Y = 10

delta_D = D_array.shape[0]/Y

bins = np.zeros([Y, delta_D])

for k in range(0,Y):
    bins[k,:] = D_array[k*delta_D:k*delta_D + delta_D] # mm


M_array = np.zeros([n_time, n_grid])
N_array = np.zeros([n_time, n_grid])

M_array[0,rainbot:raintop+1] = initvars.M
M_array[:, 0] = 0
M_array[:, -1] = 0
N_array[0,rainbot:raintop+1] = initvars.N
N_array[:, 0] = 0
N_array[:, -1] = 0


# calculate Vk first
Vk = np.zeros([Y])

for k in range(0,Y):
    Vk[k] = calc_Vk(bins[k,:], a, b)
    if Vk[k] > 10:
        Vk[k] = 10
 
 
fig,ax=plt.subplots(1,2,figsize=(6,6))
nk = np.zeros([n_time, n_grid, Y])
mk = np.zeros([n_time, n_grid, Y])

for i in range(0,n_time-1):
    
    for j in range(0,n_grid):
                    
        if M_array[i,j] <= 0. or N_array[i,j] <= 0.:
            lamb = 0.
            N0 = 0.
        else:
            lamb = ((gamma*N_array[i,j])/(beta*M_array[i,j]))**(1./3)
            N0 = N_array[i,j]/beta*lamb**(alpha+1)
            
        for k in range(0,Y):
            nk[i,j,k] = calc_nk(bins[k,:], N0, alpha, lamb, dD)
            mk[i,j,k] = calc_mk(bins[k,:], N0, alpha, lamb, rho_w, dD)
            
    for j in range(0,n_grid-1):
        N_array,M_array = upstream_bins(nk, mk, Vk, delta_D, Y, N_array, M_array, i, j)


for i in range(0,n_time):
    ax[0].plot(i,np.sum(N_array[i,:]),'.b')
    ax[1].plot(i,np.sum(M_array[i,:]),'.r')

    

fig1,ax1 = plt.subplots(1,1,figsize=(6,6))
fig2,ax2 = plt.subplots(1,1,figsize=(6,6))

#for i in np.arange(0,1,3*dt):
for i in np.arange(0,n_time+dt,5*dt):
    if i == 0:
        ax1.plot(M_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz), '--g',label="t = {} s".format(i*dt))
        ax2.plot(N_array[i,:]/rho_a,np.arange(initvars.zmin,initvars.zmax,dz), '--g',label="t = {} s".format(i*dt))
    else:
        ax1.plot(M_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz),label="t = {} s".format(i*dt))
        ax2.plot(N_array[i,:]/rho_a,np.arange(initvars.zmin,initvars.zmax,dz),label="t = {} s".format(i*dt))
ax1.set_xlim([0, 1.5])
ax2.set_xlim([0, 15000])

titlestr = r"Advection of M with the Upwind-Hybrid Scheme, $\alpha$ = {}".format(alpha)
xlabel = r"$M\ (g\ kg^{-1})$"
ylabel = r"$z\ (m)$"
set_labels(ax1, titlestr, xlabel, ylabel, fontsize, ticksize, legend=True)

titlestr = r"Advection of N with the Upwind-Hybrid Scheme, $\alpha$ = {}".format(alpha)
xlabel = r"$N\ (kg^{-1})$"
ylabel = r"$z\ (m)$"
set_labels(ax2, titlestr, xlabel, ylabel, fontsize, ticksize, legend=True)


 
    
    