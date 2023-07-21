######################################################
# Figure 9 in Appendix A: check of J_par calculation #
######################################################
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy import optimize, integrate, special

# Plot details 
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 14})
rc('text', usetex=True)
rc('figure', **{'figsize': (5.5, 4.5)})  
plot_save = False

###################################
# Definition of relevant functions
##################

def B_mag(chi, eps, eta, B2c, B20):
    # A QS, near-axis magnetic field magnitude
    return 1 - eps*eta*np.cos(chi) + eps*eps*(B20+B2c*np.cos(2*chi))

def Ji(chi, lam, eps, eta, B2c, B20, B = []): 
    # Integrand of the J_par integral
    if not B:
        B = B_mag(chi, eps, eta, B2c, B20)
    return np.sqrt(1 - lam*B)/B

def rbound(lam, eps, eta, B2c, B20):
    # Right bunce-point, in chi
    def B_zero(chi):
        return 1 - lam*B_mag(chi, eps, eta, B2c, B20)
    val = optimize.root_scalar(B_zero,x0=0,x1=np.pi/2)
    return val.root

def l_val(lam, eps, eta, B2c, B20, nsize = 10000):
    # Domain of integration from the bottom of the well to the bounce point
    return np.linspace(0,rbound(lam,eps,eta, B2c, B20), nsize, endpoint = False)

def J_parallel(k, eps = 0.01, eta = 1, B2c = 1, B20 = 1, nsize = 10000):
    # Numerical calculation of the second adiabatic invariant
    chi_b = 2*np.arcsin(k)
    lam = 1/B_mag(chi_b, eps, eta, B2c, B20)
    domain = l_val(lam,eps,eta, B2c, B20, nsize)
    integrand = Ji(domain, lam, eps, eta, B2c, B20)
    return 2*integrate.trapz(integrand, domain)

def J_approx_0(k, eps = 0.01, eta = 1, B2c = 1, B20 = 1):
    # Leading order asymptotics for J_par
    I1 = 2*(k*k-1)*special.ellipk(k*k) + 2*special.ellipe(k*k)
    return 2*np.sqrt(2*eps*eta)*I1

def J_approx_1(k, eps = 0.01, eta = 1, B2c = 1, B20 = 1):
    # First order asymptotic correction to J_par
    I1 = 2*(k*k-1)*special.ellipk(k*k) + 2*special.ellipe(k*k)
    I2 = -2.0/3.0*(k*k-1)*special.ellipk(k*k) + 2.0/3.0*(2*k*k-1)*special.ellipe(k*k)
    return 2*np.sqrt(2*eps*eta)*eps*eta*(-B2c/eta/eta*(I2-(2*k*k-1)*I1)+(I2-(k*k-0.5)*I1))

###################################
# Definition of parameters
##################
N = 100
eps = np.logspace(-3,-1,N) # r in the near-axis expansion
k = 0.5 # Value of k

# Allocate arrays
J_arr = np.zeros(N)
J_est_0 = np.zeros(N)
J_est_1 = np.zeros(N)
B_test = np.zeros(N)

###################################
# Computation J_par
##################
for i in range(N):
    J_arr[i] = J_parallel(k, eta = 1.1, B2c = 3, B20 = 1.3, eps=eps[i], nsize = 1000000)
    J_est_0[i] = J_approx_0(k, eta = 1.1, B2c = 3, B20 = 1.3, eps=eps[i])
    J_est_1[i] = J_approx_1(k, eta = 1.1, B2c = 3, B20 = 1.3, eps=eps[i])

###################################
# Plotting
##################
plt.loglog(eps,np.abs(J_arr-J_est_0),linewidth=1.5, color='k')
plt.loglog(eps,np.abs(J_arr-J_est_0-J_est_1), linewidth=1.5, color='k', linestyle='dashed')
plt.loglog(eps[int(N/4):int(7*N/8)],eps[int(N/4):int(7*N/8)]**2.5,linestyle='dotted',color='k') # Scaling

ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
plt.legend([r'$\mathcal{J}_\parallel-\mathcal{J}_\parallel^{O(r^{1/2})}$',r'$\mathcal{J}_\parallel-\mathcal{J}_\parallel^{O(r^{3/2})}$',r'$\sim r^{5/2}$'],\
           handletextpad=0., frameon=False)
plt.xlabel('$r$')
plt.ylabel(r'$\Delta\mathcal{J}_\parallel$')
plt.tight_layout()

if plot_save:
    plt.savefig('J_par_num_check_k_hat.png', dpi=300)
plt.show()
