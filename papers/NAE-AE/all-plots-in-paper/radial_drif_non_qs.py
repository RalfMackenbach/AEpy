########################################################
# Figure 10 in Appendix B: radial bounce-average drift #
########################################################

import sys
from BAD import bounce_int
import numpy as np
import matplotlib.pyplot as plt
import matplotlib        as mpl
from scipy.interpolate import CubicSpline as spline
from scipy import interpolate as interp
from qsc import Qsc
from matplotlib import rc
from scipy import special
import matplotlib.colors as colors

# Plotting details
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 14})
rc('text', usetex=True)
save_fig = False

# Interpolation function (within NAE, pyQSC)
def convert_to_spline_varphi(self,array):
    sp=spline(np.append(self.varphi,2*np.pi/self.nfp+self.varphi[0]), np.append(array,array[0]), bc_type='periodic')
    return sp

############################
# Numerical parameters
#########
N_alpha = 100
N_k = 100
N_chi = 1000

alpha_array = np.linspace(0,1,N_alpha,endpoint = False)*2*np.pi
k_array = np.linspace(0.01,1,N_k,endpoint = False)
chi_array = np.linspace(-1,1,N_chi,endpoint = False)*np.pi/2

############################
# Construct near-axis field
#########
stel = Qsc.from_paper("precise QH", nphi = 301)
stel.etabar = -np.abs(stel.etabar) # Make etabar>0 (to align with paper)
stel.calculate()

# Define B20 and its derivative
x = np.matmul(stel.d_d_varphi,stel.B20)
dB20_spline = stel.convert_to_spline(x)
x = stel.varphi-stel.phi
nu_spline = convert_to_spline_varphi(stel, x)

# Adjust alpha array to irreducible unit
alpha_array = alpha_array*stel.iotaN/stel.nfp

# Function I1(k), for bounc time
I1 = 2*special.ellipk(k_array*k_array)

############################
# Compute integrals along fieldline
#########
rad_drift_array = np.zeros((N_alpha, N_k))
roots_list = np.zeros((N_alpha, N_k))
# Loop over field lines
for jalpha, alpha in enumerate(alpha_array):
    # Evaluate B20'(varphi(chi)), which is defined in the phi_cyl grid inside pyqsc
    phi_arg = (2*chi_array-alpha)/stel.iotaN
    dB20 = dB20_spline(phi_arg - nu_spline(phi_arg))
    # Loop over trapped particle classes, k
    for jk, k in enumerate(k_array):
        # Compute bounce integral using BAD
        f = k*k - np.sin(chi_array)*np.sin(chi_array)
        rad_drift, roots  = bounce_int.bounce_integral_wrapper(f,dB20,chi_array,return_roots=True)
        # make into list of lists
        roots_list[jalpha, jk] = roots[1]
        rad_drift_array[jalpha, jk] = rad_drift[0]
    # Normalise to bounce time
    rad_drift_array[jalpha, :] = rad_drift_array[jalpha, :]/I1

# Final scaling following Eq. (B5)
rad_drift_array = rad_drift_array/stel.iotaN
print(r'Max $\omega_\psi$: ', rad_drift_array.max())

############################
# Plotting
#########
fig, axes = plt.subplots(1,2, figsize=(10,5))

# Left axis: B20 as a function of varphi
axes[0].plot(stel.varphi,stel.B20,'k')
axes[0].set_xlabel(r'$\varphi$')
axes[0].set_ylabel(r'$B_{20}$')

# Right axis: omega_psi as a function of alpha, k
[alpha_grid, k_grid] = np.meshgrid(alpha_array, k_array) 
pcm = axes[1].pcolormesh(alpha_grid, k_grid, rad_drift_array.transpose(), cmap='RdBu_r', shading='gouraud',
            #    norm=colors.LogNorm(vmin=np.max([rad_drift_array.min(), 1e-8]), vmax=rad_drift_array.max())
               )
axes[1].set_xlabel(r'$\alpha$')
axes[1].set_ylabel(r'$k$')
cbar = fig.colorbar(pcm, ax=axes[1])
cbar.ax.set_ylabel(r'$\omega_\psi$', rotation=90)

if save_fig:
    plt.savefig('B20_radial_drift_precise_QH.png', dpi=300)
plt.show()
