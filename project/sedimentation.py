# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:27:14 2015

@author: Tim
sedimentation.py
main script
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
    N_tilda = np.zeros([1,n_grid])    
    M_tilda = np.zeros([1,n_grid])
    
    if j != 0 or j != n_grid-1:
        N_tilda[0,j] = N_array[0,j] + dt/(2*dz)*(Vn[0,j-1]*N_array[0,j-1] - 
            Vn[i,j+1]*N_array[i,j+1])
        M_tilda[0,j] = M_array[0,j] + dt/(2*dz)*(Vm[0,j-1]*M_array[0,j-1] - 
            Vm[0,j+1]*M_array[0,j+1])
    
    return N_tilda,M_tilda
        
        
def mid_solution(N_array, M_array, N_tilda, M_tilda):
    N_half = np.mean([N_tilda[0,:],N_array[0,:]],axis=0)
    M_half = np.mean([M_tilda[0,:],M_array[0,:]],axis=0)
    
    return N_half,M_half
    
    
def corrector(N_array, M_array, N_half, M_half, j, Vn, Vm, n_grid):
    
    if j != 0 or j != n_grid-1:
        N_array[1,j] = N_array[0,j] + dt/(2*dz)*(Vn[0,j-1]*N_half[0,j-1] - 
            Vn[0,j+1]*N_half[0,j+1])
        M_array[1,j] = M_array[0,j] + dt/(2*dz)*(Vm[0,j-1]*M_half[0,j-1] - 
            Vm[0,j+1]*M_half[0,j+1])
    
    return N_array,M_array

    
def leapfrog(N_array, M_array, i, j, Vn, Vm, n_grid):
    
    if j != 0 or j != n_grid-1:
        N_array[i+1,j] = N_array[i,j] + dt/(2*dz)*(Vn[i,j-1]*N_array[i,j-1] - 
            Vn[i,j+1]*N_array[i,j+1])
        M_array[i+1,j] = M_array[i,j] + dt/(2*dz)*(Vm[i,j-1]*M_array[i,j-1] - 
            Vm[i,j+1]*M_array[i,j+1])

    return N_array,M_array

if initialize:
    timevars={'timevars':{'dt':3.,'tstart':0.0,'tend':240.0}}
    # constants
    uservars={'uservars':{'rho_w':1000, 'rho_a':1.225, 'a':842, 'b':0.8, 'alpha':0}} 
    
    # initial conditions and spatial parameters
    initvars={'initvars':{'M':1, 'N':1e4, 'Nbot':4300, 'Ntop':5050, 'zmin':0,
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
gamma = np.pi/6.*rho_w*sp.gamma(alpha+4)
beta = sp.gamma(alpha+1)
dt = timevars.dt
dz = initvars.dz
    
Vm = np.zeros([n_time, n_grid])
Vn = np.zeros([n_time, n_grid])

for i in range(0,n_time-1):

    for j in range(0,n_grid):
        if M_array[i,j] == 0. or N_array[i,j] == 0.:
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

for i in np.arange(0,n_time,2*dt):
    if i == 0:
        ax1.plot(M_array[i,:],np.arange(initvars.zmin,initvars.zmax,dz), '--m',label="t = {}".format(i*dt))
        ax2.plot(N_array[i,:],np.arange(initvars.zmin,initvars.zmax,dz), '--m',label="t = {}".format(i*dt))
    else:
        ax1.plot(M_array[i,:],np.arange(initvars.zmin,initvars.zmax,dz),label="t = {}".format(i*dt))
        ax2.plot(N_array[i,:],np.arange(initvars.zmin,initvars.zmax,dz),label="t = {}".format(i*dt))
ax1.set_xlim([0, 1.5])
ax2.set_xlim([0, 15000])
ax1.set_title("M")
ax2.set_title("N")
ax1.legend()
ax2.legend()
    

    

