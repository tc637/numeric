# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 22:13:26 2015

@author: Tim
precipitate.py
Precipitate class, derived from Integrator class by P. Austin
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy import special as sp
import copy
import yaml

class Precipitate(object):

    def set_yinit(self, uservars_init, initvars_init):
        
        
        # set new initial conditions
        for init_key in initvars_init:
            self.config["initvars"][init_key] = initvars_init[init_key]
        
        # set new parameters
        for user_key in uservars_init:
            self.config["uservars"][user_key] = uservars_init[user_key]
        
        uservars = namedtuple('uservars', self.config['uservars'].keys())
        self.uservars = uservars(**self.config['uservars'])
   
        initvars = namedtuple('initvars', self.config['initvars'].keys())
        self.initvars = initvars(**self.config['initvars'])
                
        self.yinit = np.array([self.initvars.M, self.initvars.N])
        self.nvars = len(self.yinit)
        
        self.n_grid = round((self.initvars.zmax - self.initvars.zmin)/self.initvars.dz)
        self.n_time = round((self.timevars.tend - self.timevars.tstart)/self.timevars.dt)
        self.rainbot = round((self.initvars.Nbot - self.initvars.zmin)/self.initvars.dz)
        self.raintop = round((self.initvars.Ntop - self.initvars.zmin)/self.initvars.dz)
        # Storage for values at previous, current, and next time step
        self.M_prev = np.zeros(self.n_grid)
        self.M_now = np.zeros(self.n_grid)
        self.M_next = np.zeros(self.n_grid)
        
        self.N_prev = np.zeros(self.n_grid)
        self.N_now = np.zeros(self.n_grid)
        self.N_next = np.zeros(self.n_grid)
        # Storage for results at each time step.  In a bigger model
        # the time step results would be written to disk and read back
        # later for post-processing (such as plotting).
        self.store = np.empty((self.n_grid, self.n_time))
        
        return None
        

    def __init__(self, coeff_file_name, uservars_init, initvars_init):
        with open(coeff_file_name, 'rb') as f:
            config = yaml.load(f)
        self.config = config
        timevars = namedtuple('timevars',config['timevars'].keys())
        self.timevars = timevars(**config['timevars'])
        # read in dtpassmin dtpassmax dtfailmin dtfailmax s rtol atol maxsteps maxfail
        adaptvars = namedtuple('adaptvars', config['adaptvars'].keys())
        self.adaptvars = adaptvars(**config['adaptvars'])
        
        # change parameters and initial values
        self.set_yinit(uservars_init, initvars_init)
        
        
    def store_timestep(self, timestep):
        """Copy the values for the specified time step to the storage
        array.

        The `attr` argument is the name of the attribute array (prev,
        now, or next) that we are going to store.  Assigning the value
        'next' to it in the function def statement makes that the
        default, chosen because that is the most common use (in the
        time step loop).
        """
        # The __getattribute__ method let us access the attribute
        # using its name in string form;
        # i.e. x.__getattribute__('foo') is the same as x.foo, but the
        # former lets us change the name of the attribute to operate
        # on at runtime.
        self.M.store[:, timestep] = self.M_next
        self.N.store[:, timestep] = self.N_next

    def shift(self):
        """Copy the .now values to .prev, and the .next values to .new.

        This reduces the storage requirements of the model to 3 n_grid
        long arrays for each quantity, which becomes important as the
        domain size and model complexity increase.  It is possible to
        reduce the storage required to 2 arrays per quantity.
        """
        # Note the use of the copy() method from the copy module in
        # the standard library here to get a copy of the array, not a
        # copy of the reference to it.  This is an important and
        # subtle aspect of the Python data model.
        self.M_prev = copy.copy(self.M_now)
        self.M_now = copy.copy(self.M_next)
        self.N_prev = copy.copy(self.N_now)
        self.N_now = copy.copy(self.N_next)
        
    def calc_params(self):
        M = self.M_now
        N = self.N_now
        alpha = self.uservars.alpha
        
        
        beta = sp.gamma(alpha+1)
        gamma = np.pi/6.*self.uservars.rho_w*sp.gamma(alpha+4)
        
        lamb = (gamma*N/(beta*M[48]))**(1/3)
        print(lamb.shape)        
        N0 = N*lamb**(alpha+1)/beta**2
        print(N0.shape)
        
        return(lamb,N0)
        
    def calc_speeds(self):
        lamb,N0 = np.calc_params()
        alpha = self.initvars.alpha
        a = self.initvars.a
        b = self.initvars.b
        Vn = (a/lamb**(b))*sp.gamma(alpha+b+1)/sp.gamma(alpha+1)
        Vm = (a/lamb**(b))*sp.gamma(alpha+b+4)/sp.gamma(alpha+4)
        
        return(Vn, Vm)
        
    def upstream(self):
        """y[0]=mass mixing ratio
           y[1]=number mixing ratio
        """
        
        f = np.empty_like(y)
        f[0] = 0
        return
    

    def initial_conditions(self):
        """Set the initial condition values.
        """
        self.M_prev[self.rainbot:self.raintop] = self.initvars.M
        self.N_prev[self.rainbot:self.raintop] = self.initvars.N
        
    
    def boundary_conditions(self, M_array, N_array):
        """Set the boundary condition values.
        """
        M_array[0] = 0 # u1
        M_array[n_grid-1] = 0 # un
        N_array[0] = 0 # u1
        N_array[n_grid-1] = 0 # un
    
        
             
    def rain(args):
        """Run the model.
    
        args is a 3-tuple; (number-of-time-steps, number-of-grid-points, time step)
        """
        n_time = int(args[0])
        n_grid = int(args[1])
    #     Alternate implementation:
    #     n_time, n_grid = map(int, args)
    
        # Constants and parameters of the model
        g = 980                     # acceleration due to gravity [cm/s^2]
        H = 1                       # water depth [cm]
        dt = float(args[2])         # user-defined time step [s]
        #dt = 0.001                  # time step [s]
        dx = 1                      # grid spacing [cm]
        ho = 0.01                   # initial perturbation of surface [cm]
        gu = g * dt / dx            # first handy constant
        gh = H * dt / dx            # second handy constant
        # Create velocity and surface height objects
        u = Quantity(n_grid, n_time)
        h = Quantity(n_grid, n_time)
        # Set up initial conditions and store them in the time step
        # results arrays
        initial_conditions(u, h, ho)
        u.store_timestep(0, 'prev')
        h.store_timestep(0, 'prev')
        # Calculate the first time step values from the
        # predictor-corrector, apply the boundary conditions, and store
        # the values in the time step results arrays
        first_time_step(u, h, g, H, dt, dx, ho, gu, gh, n_grid)
        boundary_conditions(u.now, h.now, n_grid)
        u.store_timestep(1, 'now')
        h.store_timestep(1, 'now')
        # Time step loop using leap-frog scheme
        for t in np.arange(2, n_time):
            # Advance the solution and apply the boundary conditions
            leap_frog(u, h, gu, gh, n_grid)
            boundary_conditions(u.next, h.next, n_grid)
            # Store the values in the time step results arrays, and shift
            # .now to .prev, and .next to .now in preparation for the next
            # time step
            u.store_timestep(t)
            h.store_timestep(t)
            u.shift()
            h.shift()
    
        # Plot the results as colored graphs
        make_graph(u, h, dt, n_time)
        return
          
    
if __name__ == "__main__":
    print("hello")