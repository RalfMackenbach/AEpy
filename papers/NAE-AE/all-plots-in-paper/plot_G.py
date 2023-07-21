########################################################
# Plot in Figure 1b: the leading order precession G(k) #
########################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import  special
from matplotlib import rc

# Specify plotting details
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 19})
rc('text', usetex=True)
rc('figure', **{'figsize': (6.5, 5.5)})  

# Create k grid
N_k = int(1e4)
k_grid = np.linspace(0,1,N_k,endpoint=True)

# Define the function G(k), Eq. (3.2)
E_o_K = special.ellipe(k_grid*k_grid)/special.ellipk(k_grid*k_grid)
G = 2*E_o_K - 1

# Plot
plt.plot(k_grid, G, 'k', label=r'$\mathcal{G}$',linewidth=1.5)
plt.plot(k_grid, 0.0*k_grid,'k',linestyle='dashed')
plt.xlabel(r'$ k$')
plt.ylabel(r'$G$')
plt.xlim([0,1.004])
ax = plt.gca()
for axis in ['top','bottom','left','right']:
     ax.spines[axis].set_linewidth(1.5)
plt.tight_layout()

plt.savefig('walpha_G.png', dpi=300)
plt.show()