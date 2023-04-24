import scipy
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib        import rc



rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

def f(c1):
    sqrtc1 = np.sqrt(c1)
    erf = scipy.special.erf(sqrtc1)
    exp = np.exp(-c1)
    sqrtpi = np.sqrt(np.pi)
    ans =  15/4 *exp / (sqrtc1 * c1) + exp/sqrtc1 - erf * 15 * sqrtpi / (8 * c1**2) + 3 * sqrtpi * erf / (4 * c1)
    return ans

c = 0.6
fig, ax = plt.subplots(1,1,figsize=(c*6,c*4),tight_layout=True)

c1 = np.logspace(-2,3,100)
plot = ax.loglog(c1,f(c1),color='black',linestyle='solid',label=r'$\mathcal{F}(c_1)$')
ax.set_xlabel(r'$c_1$')
ax.set_ylabel(r'$\mathcal{F}(c_1)$')
ax.axvline(3.90232,color='black',linestyle='dashed')
ax.grid()
ax.tick_params(axis='x',direction='in')
ax.tick_params(axis='y',direction='in')
ax.set_xlim(c1.min(),c1.max())

plt.savefig('Fc1.png', dpi=1000)
plt.show()