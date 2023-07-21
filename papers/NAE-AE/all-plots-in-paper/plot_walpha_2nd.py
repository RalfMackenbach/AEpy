###########################################################
# Figure 2: plot of second order dependence of precession #
###########################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import  special
from matplotlib import rc

# Plotting features
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 18})
rc('text', usetex=True)
rc('figure', **{'figsize': (12.5, 5.5)})  

# Save figure
save_fig = True

# Define k grid
N_k = int(1e4)
k_grid = np.linspace(0,1,N_k,endpoint=True)

# Second order expressions: Eqs. (3.4a)-(3.4c)
E_o_K = special.ellipe(k_grid*k_grid)/special.ellipk(k_grid*k_grid)
G = -4.0*E_o_K*E_o_K + 2*(3-2*k_grid*k_grid)*E_o_K + (-1+2*k_grid*k_grid)
G_20 = -2+0.0*G
G_2c = -4*(E_o_K*E_o_K - 2*k_grid*k_grid*E_o_K + (k_grid*k_grid-0.5))

########
# Plot #
########

######
# Make figure
fig, axes = plt.subplot_mosaic([['left', 'top right'],
                                ['left', 'bottom right']])

# Left plot: G, G20 and G2c functions, Eqs. (3.4a)-(3.4c)
axes['left'].plot(k_grid, G,label=r'$\mathcal{G}$',linewidth=1.5)
axes['left'].plot(k_grid, G_20,label=r'$\mathcal{G}_{20}$',linewidth=1.5)
axes['left'].plot(k_grid, G_2c,label=r'$\mathcal{G}_{2c}$',linewidth=1.5)
axes['left'].plot(k_grid, 0.0*k_grid,'k',linestyle='dashed') # Zero line

axes['left'].set_xlabel(r'$ k$')
axes['left'].set_xlim([0,1.004])
axes['left'].set_ylim([-2.5,2])
axes['left'].legend(loc = 'lower right')
for axis in ['top','bottom','left','right']:
     axes['left'].spines[axis].set_linewidth(2)

######
# Top right plot: G_delta^axi for different values of elongation, alpha

# Alpha values
N_alpha = 5
alpha_array = np.logspace(-1,1,N_alpha,endpoint=True)
colors = plt.cm.Purples(np.linspace(0,1,N_alpha+1)) # Coors for different alpha
# Plot each alpha
for jalpha, alpha in enumerate(alpha_array):
    G_delta = (3/(3+alpha))*(G_20-G_2c) - alpha/(3+alpha)*(3*G_20-G_2c) # Eq. (D5.b)
    if jalpha == 0:
        name = r'$\alpha = {:10.1f}$'.format(alpha) 
    else:
        name = r'${:10.1f}$'.format(alpha) 
    axes['top right'].plot(k_grid, G_delta, label=name, color = colors[jalpha+1],linewidth=1.5)
axes['top right'].plot(k_grid, 0.0*k_grid,'k',linestyle='dashed') # Zero line
axes['top right'].set_xlabel(r'$ k$')
axes['top right'].set_ylabel(r'$\mathcal{G}^{\mathrm{axi}}_\delta$')
axes['top right'].set_xlim([0,1.004])
axes['top right'].xaxis.tick_top()
axes['top right'].xaxis.set_label_position('top')
axes['top right'].yaxis.tick_right()
axes['top right'].yaxis.set_label_position('right')
axes['top right'].set_ylim([-4.0,6.0])
name = r'$\alpha = {:10.1f}-{:10.1f}$'.format(alpha_array[0],alpha_array[-1])
axes['top right'].legend([name])
for axis in ['top','bottom','left','right']:
     axes['top right'].spines[axis].set_linewidth(2)

#######
# Bottom right: G_p2^axi for different f's

# f values
N_fac = 5
fac_array = np.logspace(-1,1,N_alpha,endpoint=True)
colors = plt.cm.Purples(np.linspace(0,1,N_alpha+1))
# Plot for each f
for jfac, fac in enumerate(fac_array):
    G_p2 = G_20 + fac*(G_20-0.5*G_2c) # Eq. (D5.a)
    if jfac == 0:
        name = r'$f = {:10.1f}$'.format(alpha) 
    else:
        name = r'${:10.1f}$'.format(alpha) 
    axes['bottom right'].plot(k_grid, G_p2, label=name, color = colors[jfac+1],linewidth=1.5)
axes['bottom right'].plot(k_grid, 0.0*k_grid,'k',linestyle='dashed')
axes['bottom right'].set_xlabel(r'$ k$')
axes['bottom right'].set_ylabel(r'$\mathcal{G}^{\mathrm{axi}}_{p_2}$')
axes['bottom right'].set_xlim([0,1.004])
axes['bottom right'].set_ylim([-20,0])
axes['bottom right'].yaxis.tick_right()
axes['bottom right'].yaxis.set_label_position('right')
name = r'$f = {:10.1f}-{:10.1f}$'.format(fac_array[0],fac_array[-1])
axes['bottom right'].legend([name])
for axis in ['top','bottom','left','right']:
     axes['bottom right'].spines[axis].set_linewidth(2)

if save_fig:
    plt.savefig('walpha_G_2.png', dpi=300)
plt.show()