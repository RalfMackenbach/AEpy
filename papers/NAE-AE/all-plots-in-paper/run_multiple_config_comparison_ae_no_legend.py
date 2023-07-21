from scipy.special import erf
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from BAD import bounce_int
import numpy as np
from qsc import Qsc

from AEpy import ae_routines as ae
from   matplotlib        import rc
import   matplotlib.pyplot  as plt
import matplotlib.cm as cm
import matplotlib


rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

nphi = 1001


# define names for the comparison
name_array = ["precise QA", "precise QH", "precise QA+well", "precise QH+well", "2022 QA", "2022 QH nfp2", \
       "2022 QH nfp3 vacuum", "2022 QH nfp3 beta", "2022 QH nfp4 long axis", "2022 QH nfp4 well", "2022 QH nfp4 Mercier", \
       "2022 QH nfp7"]


# set plotting parameters
cmap = 'coolwarm'
# set gradient
wstar = 1.0
# set r array
N_r = 6
r_array = np.logspace(-3,-1,N_r)
ae_array = np.zeros(N_r)
ae_nae = np.zeros(N_r)

# set up figure
fig, ax = plt.subplots(1,1,figsize=(0.5*6,0.5*4),constrained_layout=True)

for ind_name, name in enumerate(name_array):
    print(name)
    stel = Qsc.from_paper(name, nphi = nphi)
    stel.r = 1e-1
    for ind, r in enumerate(r_array):
        print(ind, ' out of ', N_r, ' (',100*(ind+1)/N_r,'%)        ', end="\r")
        stel.r = r
        NAE_AE = ae.AE_pyQSC(stel_obj = stel, alpha=-0.0, N_turns=3, nphi=nphi,
                    lam_res=2, get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
        NAE_AE.calc_AE_quad(omn=stel.spsi*stel.r*wstar,omt=stel.spsi*stel.r*0.0,omnigenous=True)
        ae_array[ind] = NAE_AE.ae_tot
        ae_nae[ind] = NAE_AE.nae_ae_asymp_weak(stel.r*wstar,1.0)
    print('\n')
    ax.scatter(ae_nae, ae_array, s=15, label = name, alpha = 0.5, c=r_array, norm=matplotlib.colors.LogNorm(),cmap=cmap)

xlim = ax.get_xlim()
ax.plot([xlim[0],xlim[1]], [xlim[0],xlim[1]],'k', linestyle = 'dashed',zorder=-10)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\widehat{A}_\mathrm{w}$')
ax.set_ylabel(r'$\widehat{A}_\mathrm{numerical}$')
ax.set_xticks([1e-12,1e-8,1e-4])
ax.set_yticks([1e-12,1e-8,1e-4])
ax.grid()
cbar = fig.colorbar(cm.ScalarMappable(norm=matplotlib.colors.LogNorm(vmin=r_array.min(),vmax=r_array.max()), cmap=cmap), ax=ax,ticks=[1e-3,1e-2,1e-1],label=r'$\varrho$')
# save the figure
fig.savefig('NAE-AE_comparison.png', dpi=1000)


plt.show()