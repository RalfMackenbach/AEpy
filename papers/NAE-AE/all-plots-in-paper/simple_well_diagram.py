##########################################
# Plot well diagrams for Figures 6 and 7 #
##########################################
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

# Definition of F(c1)
def F_func(c1):
    num = 2*np.sqrt(c1)*(15 + 4*c1)*np.exp(-c1) + 3*np.sqrt(np.pi)*(2*c1-5)*special.erf(np.sqrt(c1))
    den = 8*c1*c1
    return num/den

# Definition of G(k)
def G1(k):
    val = 2*special.ellipe(k*k)/special.ellipk(k*k)-1
    return val

# Domain of the magnetic well (phi)
N_phi = 4000
phi = np.linspace(0,1,N_phi)*2*np.pi
y_well = np.cos(phi)

# B well shape
delta_B = 0.1
B = 1+delta_B*y_well
lam = 1/B # Definition of lambda
k = np.sqrt(0.5*(1+y_well)) # Definition of k

# Evaluate G
G1_val = G1(k)
mask_precess = G1_val < 0.0 # Negative precession

# Choose plot: depending on weight given to different k populations
plot_colour_well = True # False, only the function F(c1) is plotted
full_energy = False # True, the full energy is plotted, i.e. the weighting 
                    # by bounce-time is included and an additional factor k from dlam/dk

if plot_colour_well:
    # Plot lines of different colour according to AE within the well
    def plot_well_ae(fac_c1, ax):
        c = F_func(fac_c1/G1_val)
        if full_energy == True:
            fac_energy = k * special.ellipk(k*k)
            c = fac_energy * c
        c[mask_precess] = 0.0
        c_max = c.max()
        c = c/c_max
        # Plot lines
        for jc_val, c_val in enumerate(c):
            color = pl.cm.Oranges(c_val)
            ax.plot([phi[jc_val],2*np.pi-phi[jc_val]], [y_well[jc_val],y_well[jc_val]], color = color, linewidth = 0.3)
        ax.plot(phi, y_well,'k',linewidth=1.5)
        # Eliminate axes
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])


    fig, axs = plt.subplots(1,2, figsize = (9,3.5))

    # Weak and strong regimes
    plot_well_ae(fac_c1 = 1e-1, ax = axs[0])
    plot_well_ae(fac_c1 = 1e4, ax = axs[1])
    if full_energy:
        plt.savefig('diagram_well_AE_energy.png', dpi=300)
    else: 
        plt.savefig('diagram_well_AE.png', dpi=300)
    plt.show()
else:
    # Plot function F(c1(k))
    fac_c1 = 1e4
    c = F_func(fac_c1/G1_val)
    if full_energy == True:
        fac_energy = k * special.ellipk(k*k)
        c = fac_c1 * c
    plt.plot(y_well, c, 'k')
    plt.xlim([y_well.min(), y_well.max()])
    ax = plt.gca()
    ax.yaxis.set_visible(False)
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    plt.savefig('plot_F.png', dpi = 300)
    plt.show()


    
