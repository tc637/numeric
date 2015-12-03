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

###############
# Courant

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
courm = np.zeros_like(Vm)
courn = np.zeros_like(Vn)
for i in range(0,n_time-1):

    for j in range(0,n_grid):
        if M_array[i,j] <= 0 or N_array[i,j] <= 0:
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

maxn = []
maxm = []
for eachn in courn:
    maxn.append((np.max(eachn)))
for eachm in courm:
    maxm.append((np.max(eachm)))
fig4,ax4 = plt.subplots(1,1,figsize=(6,6))
fig5,ax5 = plt.subplots(1,1,figsize=(6,6))

ax4.plot(dt*np.arange(0,n_time),maxm)
ax5.plot(dt*np.arange(0,n_time),maxn,'-b')
ax4.set_title(r"Max Courant Number for M, $\Delta t$ = {} s, $\Delta z$ = {} m".format(dt,dz))
ax5.set_title(r"Max Courant Number for N, $\Delta t$ = {} s, $\Delta z$ = {} m".format(dt,dz))
ax4.set(xlabel=r"$t\ (s)$", ylabel=r"$C$")
ax5.set(xlabel=r"$t\ (s)$", ylabel=r"$C$")

breakn = 11
breakm = 8

###############################################
# constant V
    
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
Vm[:,:] = 1
Vn = np.zeros([n_time, n_grid])
Vn[:,:] = 1

for i in range(0,n_time-1):

    for j in range(0,n_grid-1):
        N_array[i+1,j] = N_array[i,j+1]
    
        M_array[i+1,j] = M_array[i,j+1]

fig3,ax3 = plt.subplots(1,1,figsize=(6,6))


for i in np.arange(0,n_time,4.5*dt):
    if i == 0:
        ax3.plot(N_array[i,:]/rho_a,np.arange(initvars.zmin,initvars.zmax,dz), '--g',label="t = {} s".format(i*dt))
    else:
        ax3.plot(N_array[i,:]/rho_a,np.arange(initvars.zmin,initvars.zmax,dz),label="t = {}".format(i*dt))

ax3.set_xlim([0, 15000])
ax3.set_title("Advection of N with Uniform Fall Speed".format(alpha))
ax3.set(xlabel=r"$N\ (kg^{-1})$", ylabel=r"$z\ (m)$")


################################################


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

    for j in range(0,n_grid): #n_grid
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
        
    if i == 0:
        N_tilda = np.zeros([1,n_grid])    
        M_tilda = np.zeros([1,n_grid])
        for j in range(0,n_grid-1):
            N_tilda,M_tilda = predictor(N_array, M_array, N_tilda, M_tilda, j, Vn, Vm, n_grid)
            
        N_half,M_half = mid_solution(N_array, M_array, N_tilda, M_tilda)
        
        for j in range(0,n_grid-1):
            N_array,M_array = corrector(N_array, M_array, N_half, M_half, j ,Vn, Vm, n_grid)
            
    else:
        for j in range(0,n_grid-1):
            N_array,M_array = leapfrog(N_array, M_array, i, j, Vn, Vm, n_grid)
        

fig11,ax11 = plt.subplots(1,1,figsize=(6,6))
fig12,ax12 = plt.subplots(1,1,figsize=(6,6))

for i in np.arange(0,n_time,5*dt):
    if i == 0:
        ax11.plot(M_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz), '--m',label="t = {}".format(i*dt))
        ax12.plot(N_array[i,:]/1.225,np.arange(initvars.zmin,initvars.zmax,dz), '--m',label="t = {}".format(i*dt))
    else:
        ax11.plot(M_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz),label="t = {}".format(i*dt))
        ax12.plot(N_array[i,:]/1.225,np.arange(initvars.zmin,initvars.zmax,dz),label="t = {}".format(i*dt))
ax11.set_xlim([-0.5, 1.5])
ax12.set_xlim([-1000, 15000])
ax11.set_title("M Leapfrog")
ax12.set_title("N Leapfrog")
ax11.legend(loc="lower right")
ax12.legend(loc="lower right")





######################################################
# hybrid

"""
def calc_nk(D, N0, alpha, lamb, dD):
    lamb = lamb*1e3
    D = D*1e3
    nk = np.sum([N0*((each_D)**alpha)*np.exp(-lamb*each_D) for each_D in D])*dD
    
    return nk
    
def calc_mk(D, N0, alpha, lamb, rho_w, dD):
    lamb = lamb*1e3
    D = D*1e3
    mk = np.sum([np.pi/6*rho_w*((each_D)**3)*N0*(each_D)**alpha*np.exp(-lamb*each_D) for each_D in D])*dD
    
    return mk
    
def calc_Vk(D, a, b):
    Vk = a*(np.mean(D)**b)
    
    return Vk
    
def upstream_bins(nk, mk, Vk, delta_D, Y, N_array, M_array, i, j):
    
    N_temp = np.zeros([Y])
    M_temp = np.zeros([Y])
    
    for k in range(0,Y):
        N_temp[k] = (nk[k,j] + dt/dz*(Vk[k]*nk[k,j+1]-Vk[k]*nk[k,j]))*delta_D
        M_temp[k] = (mk[k,j] + dt/dz*(Vk[k]*mk[k,j+1]-Vk[k]*mk[k,j]))*delta_D
    
    N_array[i+1,j] = np.sum(N_temp)
    M_array[i+1,j] = np.sum(M_temp)
    
    return N_array, M_array
    
    
dD = 0.0001
D_array = np.arange(0,4+dD,dD) # mm
Y = 10

bin_width_index = 40
delta_D = bin_width_index * Y

bins = np.zeros([Y, bin_width_index])

for k in range(0,Y):
    bins[k,:] = D_array[k*bin_width_index:40+k*bin_width_index]


M_array = np.zeros([n_time, n_grid])
N_array = np.zeros([n_time, n_grid])

M_array[0,rainbot:raintop+1] = initvars.M
M_array[:, 0] = 0
M_array[:, -1] = 0
N_array[0,rainbot:raintop+1] = initvars.N
N_array[:, 0] = 0
N_array[:, -1] = 0
gamma = np.pi/6.*rho_w*sp.gamma(alpha+4)
beta = sp.gamma(alpha+1)
a = uservars.a
b = uservars.b
alpha = uservars.alpha
rho_w = uservars.rho_w
dt = timevars.dt
dz = initvars.dz

# calculate Vk first
Vk = np.zeros([Y])

for k in range(0,Y):
    Vk[k] = calc_Vk(bins[k,:], a, b)
    if Vk[k] > 10:
        Vk[k] = 10
    
# then integrate everything else
for i in range(0,n_time-1):
    
    nk = np.zeros([Y, n_grid])
    mk = np.zeros([Y, n_grid])

    for j in range(0,n_grid):
                    
        if M_array[i,j] <= 0. or N_array[i,j] <= 0.:
            lamb = 0.
            N0 = 0.
        else:
            lamb = ((gamma*N_array[i,j])/(beta*M_array[i,j]))**(1./3)
            N0 = N_array[i,j]/beta*lamb**(alpha+1)
            if i == 0 or i == 1:
                print(lamb)
                print(N0)
            
        for k in range(0,Y):
            nk[k,j] = calc_nk(bins[k,:], N0, alpha, lamb, dD)
            mk[k,j] = calc_mk(bins[k,:], N0, alpha, lamb, rho_w, dD)
            
       
            
    for j in range(0,n_grid-1):
        N_array,M_array = upstream_bins(nk, mk, Vk, delta_D, Y, N_array, M_array, i, j)
        


fig6,ax6 = plt.subplots(1,1,figsize=(6,6))
fig7,ax7 = plt.subplots(1,1,figsize=(6,6))

#for i in np.arange(0,1,3*dt):
for i in np.arange(0,2):
    if i == 0:
        ax6.plot(M_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz), '--g',label="t = {} s".format(i*dt))
        ax7.plot(N_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz), '--g',label="t = {} s".format(i*dt))
    else:
        ax6.plot(M_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz),label="t = {} s".format(i*dt))
        ax7.plot(N_array[i,:]*1e3,np.arange(initvars.zmin,initvars.zmax,dz),label="t = {} s".format(i*dt))
ax6.set_xlim([0, 1.5])
ax7.set_xlim([0, 15000])
ax6.set_title(r"Advection of M with the Hybrid Scheme, $\alpha$ = {}".format(alpha))
ax7.set_title(r"Advection of N with the Hybrid Scheme, $\alpha$ = {}".format(alpha))
ax6.set(xlabel=r"$M\ (g\ kg^{-1})$", ylabel=r"$z\ (m)$")
ax7.set(xlabel=r"$N\ (kg^{-1})$", ylabel=r"$z\ (m)$")
ax6.legend(loc="best")
ax7.legend(loc="best")
"""

"""
#####################################################
# gamma distribution
fig10,ax10 = plt.subplots(1,1,figsize=(6,6))
D_array = np.arange(0,4,0.01)
params = [[1,1,1],[1,2,1],[2,1,1],[1,1,2],[1,0,1,]]
for param in params:
    N_prime = param[0]*D_array**(param[1])*np.exp(-param[2]*D_array)
    label_str = r"$N_0\ =\ {},\ \alpha\ =\ {},\ \lambda\ =\ {}$".format(param[0],param[1],param[2])
    ax10.plot(D_array, N_prime, label = label_str)

ax10.legend(loc="best")
ax10.set(title="Raindrop PSDs with Different Parameters", xlabel="Diameter (mm)", 
        ylabel="Number")
"""