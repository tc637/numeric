# -*- coding: utf-8 -*-
"""
# ATSC 409
# ./miniproject1_tchui.py

Timothy Chui
37695129


Module for Miniproject 1 of ATSC 409 2015
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

### define functions for calculating dI/dd, A_h and dA_h/dd; and plotting ###

def get_I_change(depth, alpha, beta, gamma):
    """Returns dI/dd at depth d
    
    Inputs: depth >= 0 in m, float
            alpha, light attenuation rate in 1/m, float
            beta, ice fraction, float
            gamma, water albedo, float
            
    Outputs:
            i_change, W/m^3
    """
    
    I_change = (-100*alpha*(1-beta)*(1-gamma)*np.exp(-alpha*depth))
    return(I_change)


def get_A_h(depth, A_max, A_depth, A_dip, h_depth):
    """Returns A_h and dA_h/dd at depth d (eddy-viscosity coefficient)
    
    Inputs: depth >= 0 in m, float
            A_max, parameter in m^2 s^-1, float
            A_depth, parameter in m^2 s^-1, float
            A_dip, parameter in m^2 s^-1, float
            h_depth >= 0 in m, thickness of surface layer, float
            
    Outputs:
            a_hi, m^2 s^-1
            a_hi_diff, m^2 s^-2
    """
    if depth <= h_depth:
        A_hi = A_max
        A_hi_diff = 0
        
    else:
        A_hi = (A_depth + (A_max - A_depth - A_dip*(depth-h_depth))*
            np.exp(-0.5*(depth-h_depth)))
        A_hi_diff = ((-0.5*A_max + 0.5*A_depth - A_dip*(1-0.5*(depth-h_depth)))*
            np.exp(-0.5*(depth-h_depth)))
            
    return(A_hi, A_hi_diff)
   
   
def plot_profile(temps, depths, fig=None, ax=None, params=None):
    """ Makes temperature profile with depth
    
    Inputs: temps, list of arrays of temperatures in C
            depths, array of depths in m
            fig, current figure
            ax, current axis
            params, tuple of parameters for the plot
    
            ([linestyle_strs], [legend_labels])
            
    Outputs:
            fig, current figure
            axis, current axis
    """
    
    
    if (type(fig) == type(None)) or (type(ax) == type(None)):
        fig,ax = plt.subplots(1,1)
        
    fig.set_size_inches(16., 8.)
    
    if len(temps.shape) < 2:
        for k in range(0, temps.shape[0]):
            ax.plot(temps, depths, params[0][0], label=params[1][0])
            
    else:
        for i in range(0, temps.shape[1]):
            ax.plot(temps[:,i], depths, params[0][i], label=params[1][i])
                             
    return fig,ax
    
    
### main script ###  
    
if __name__ == "__main__":
    
    # close all existing figures
    plt.close('all')
    
    # define constants and coefficients
    delta = 1.0 # m, step-size
    cp = 4.0e6 # specific heat, J m^-3 C^-1
    alpha = 1./10 # light attenuation rates, 1/m
    beta = 0.5 # ice fraction
    gamma = 0.1 # water albedo
    h_depth = 10. # surface layer depth, m
    A_max = 1.0e-2 # m^2 s^-1 
    A_depth = 1.0e-4 # m^2 s^-1
    A_dip = 1.5e-3 # m^2 s^-1
    
    
    # shove parameters in dictionary
    keys = ["alpha", "beta", "h_depth", "A_max", "A_depth", "A_dip"]
    multipliers = [1,1.1,1.2,1.3,1/1.1,1/1.2,1/1.3] # for perturbations
    #multipliers = [1,2,5,10,1/2.,1/5.,1/10.]
    init_dict = {keys[0]:alpha, keys[1]:beta, keys[2]:h_depth, 
                  keys[3]:A_max, keys[4]:A_depth, keys[5]:A_dip}   

    # for plots
    fontsize = 20
    xlabel = r'$Temperature\ (^{o}C)$'
    ylabel = r'$Depth\ (m)$'
    
    # define arrays
    depths = np.arange(0, 200+delta, delta) # depths, increasing downwards
    N = len(depths) - 1 # number of points, 0-based index
    T_array = np.zeros([N+1, N+1, len(keys), len(multipliers)]) # initialize T_array 
    
    # expected appearance of T_array, for a single set of parameters
    
    """        "T_array"                      "augmented"
    [[1                                      | boundary0 ]
     [theta1 phi1 kappa1                     | forcing1  ]
     [       theta2 phi2 kappa2              | forcing2  ]
     [              ...                      | ...       ]
     [              thetaN-1 phiN-1 kappaN-1 | forcingN-1]
     [                              1        | boundaryN ]]
    
    """
    
    # initialize condition number and solution arrays
    cond_array = np.zeros([N+1, len(keys), len(multipliers)])
    solution_array = np.zeros([N+1, len(keys), len(multipliers)])
    
    # initialize NSTM arrays
    NSTM = np.zeros([len(keys), len(multipliers)])
    NSTM_depths = np.zeros([len(keys), len(multipliers)])
    
    # construct T_array
    for key_index in range(0, len(keys)): # iterate over all keys
        for mult_index in range(0, len(multipliers)): # iterate over all mults
            
            # construct augmented column
            augmented = np.zeros(N+1)
            # apply boundary conditions
            augmented[0] = -1
            augmented[-1] = -2
            
            # parameters in current iteration with appropriate multiplier
            current_key = keys[key_index]
            
            parameters = [init_dict[key]*multipliers[mult_index] \
                            if key == current_key else init_dict[key] \
                                for key in keys]
                        
            # define current parameter dictionary for clarity
            parameter_dict = dict(zip(keys, parameters))

            # apply forcing from I at each depth
            augmented[1:N] = [1/cp*get_I_change(depths[i], parameter_dict['alpha'], 
                                  parameter_dict['beta'], gamma) 
                                      for i in range(1,len(depths)-1)]
                               
            # fill in T_array with coefficients
            for i in range(0, N+1):
                if i == 0: # first boundary
                    T_array[i, i, key_index, mult_index] = 1
                elif i == N: # second boundary
                    T_array[i, i, key_index, mult_index] = 1
                else: # for all other rows
                    # get A_h and A_h_diff for current depth
                    A_hi, A_hi_diff = get_A_h(depths[i], parameter_dict['A_max'], 
                                              parameter_dict['A_depth'], 
                                                parameter_dict['A_dip'], 
                                                    parameter_dict['h_depth'])
                                                    
                    # deal with negative A_hi; set to 0 at current depth          
                    if A_hi < 0:
                        print("depth = {}, key_index = {}, mult_index = {}".format(depths[i],
                              key_index,mult_index))
                        A_hi = 0
                    
                    # coefficient for T_i-1
                    theta_i = -A_hi_diff*(1/(2*delta)) + A_hi/delta**2
                    
                    # coefficient for T_i
                    phi_i = -2*A_hi/delta**2
                    
                    # coefficient for T_i+1
                    kappa_i = A_hi_diff*(1/(2*delta)) + A_hi/delta**2
                     
                    # place into T_array
                    T_array[i, i-1, key_index, mult_index] = theta_i
                    T_array[i, i, key_index, mult_index] = phi_i
                    T_array[i, i+1, key_index, mult_index] = kappa_i
                    
               
            # solve the system
            solution_array[:,key_index,mult_index]=np.linalg.solve(T_array[:,:,key_index,mult_index],
                                                        augmented)        
            
            # solve the system
            cond_array[:,key_index,mult_index]=np.linalg.cond(T_array[:,:,key_index,mult_index])
            
            # grab NSTM magnitudes and depths
            NSTM[key_index, mult_index] = max(solution_array[:,key_index,mult_index])
            NSTM_depths[key_index, mult_index] = \
                    depths[np.where(solution_array[:,key_index,mult_index] == \
                            NSTM[key_index, mult_index])[0]]
            
               
    # plot base case
    fig,ax = plot_profile(np.array(solution_array[:,0,0]), depths, fig=None, 
                                  ax=None, params=(['-r'],['']))
                                               
    original_title = "Temperature Profile for Base-Case Parameters"
    plt.gca().invert_yaxis()
    plt.title(original_title, fontsize=fontsize, fontweight='bold')
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    
    # now iterate over all cases
    linestyles = ['-r','xb','xg','xc','.y','.m','.k']
    
    for i in range(0, solution_array.shape[1]):
        
        # define list of strings for labelling
        var_strs = [r"$\alpha$", r"$\beta$", r"$h_{depth}$", r"$A_{max}$", 
                    r"$A_{depth}$", r"$A_{dip}$"]
        
        units = [r"$\ m^{-1}$", "", r"$\ m$", r"$\ m^{2}\ s^{-1}$", 
                 r"$\ m^{2}\ s^{-1}$", r"$\ m^{2}\ s^{-1}$"]
        label_str = "{} = {:.1e}{}"
        label_array = [label_str.format(var_strs[i], 
                                        init_dict[keys[i]]*mult,units[i]) \
                                        for mult in multipliers]
        
        # call plot_profile for each varying parameter
        fig = None
        ax = None
        
        if type(fig) == type(None) or type(ax) == type(None):
            fig,ax = plot_profile(np.array(solution_array[:,i,:]), depths, fig=None, 
                                      ax=None, params=(linestyles,label_array))
        else:
            fig,ax = plot_profile(np.array(solution_array[:,i,:]), depths, fig=fig, 
                                      ax=ax, params=(linestyles,label_array))
        
        # increase depth going downwards                              
        plt.gca().invert_yaxis()
        
        # labels
        plot_title = "Temperature Profiles for Varying {}".format(var_strs[i])
        plt.title(plot_title, fontsize=fontsize, fontweight='bold')
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)   
        
        plt.legend(loc='lower right')
        plt.gca().set_xlim(left=-2.0)
        
    
    # plot NSTM comparisons
    fig1, ax1 = plt.subplots(1,1)
    plt.xlabel(r'$Multiplier$', fontsize=fontsize)
    plt.ylabel(r'$NSTM\ (^oC)$', fontsize=fontsize)
    plt.title('Comparsion of NSTM for Each Parameter', fontsize=fontsize, fontweight='bold')
    
    fig2, ax2 = plt.subplots(1,1)
    plt.xlabel(r'$Multiplier$', fontsize=fontsize)
    plt.ylabel(r'$NSTM\ Depth\ (m)$', fontsize=fontsize)
    plt.title('Comparsion of NSTM Depths for Each Parameter', fontsize=fontsize, fontweight='bold')
    
    linestyles = ['.r','xb','og','.c','xy','om']
    for i in range(0,NSTM.shape[0]):
        ax1.plot(multipliers, NSTM[i,:], linestyles[i], label=var_strs[i])
        ax2.plot(multipliers, NSTM_depths[i,:], linestyles[i], label=var_strs[i])

    
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    ax1.set_xlim(right=1.4)   
    ax2.set_xlim(right=1.4)                   
    
    
    
    