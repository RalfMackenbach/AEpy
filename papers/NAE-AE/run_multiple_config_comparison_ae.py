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


rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

nphi = 1001
# stel.spsi = -1
# stel.zs = -stel.zs
# stel.r = 1e-1
# stel.calculate()
# print(stel.B20_variation)


name_array = ["precise QA", "precise QH", "precise QA+well", "precise QH+well", "2022 QA", "2022 QH nfp2", \
       "2022 QH nfp3 vacuum", "2022 QH nfp3 beta", "2022 QH nfp4 long axis", "2022 QH nfp4 well", "2022 QH nfp4 Mercier", \
       "2022 QH nfp7"]
cmap = cm.get_cmap('RdYlBu')

wstar = 1.0
N_r = 10
r_array = np.logspace(-3,-1,N_r)
ae_array = np.zeros(N_r)
ae_nae = np.zeros(N_r)
for ind_name, name in enumerate(name_array):
    print(name)
    stel = Qsc.from_paper(name, nphi = nphi)
    stel.r = 1e-1
    for ind, r in enumerate(r_array):
        print(ind, ' out of ', N_r, ' (',100*(ind+1)/N_r,'%)        ', end="\r")
        stel.r = r
        NAE_AE = ae.AE_pyQSC(stel_obj = stel, alpha=-0.0, N_turns=3, nphi=nphi,
                    lam_res=10001, get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
        NAE_AE.calc_AE(omn=stel.spsi*stel.r*wstar,omt=stel.spsi*stel.r*0.0,omnigenous=True)
        ae_array[ind] = NAE_AE.ae_tot
        ae_nae[ind] = NAE_AE.nae_ae_asymp_weak(stel.r*wstar,1.0)
    print('\n')
    plt.scatter(ae_nae, ae_array, s=40, label = name, alpha = 0.5, c=cmap(ind_name*(0.0*ae_nae+1.0)/len(name_array)),edgecolors='black')
axes = plt.gca()
xlim = axes.get_xlim()
plt.plot([xlim[0],xlim[1]], [xlim[0],xlim[1]],'k', linestyle = 'dashed')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$AE_\mathrm{nae}$')
plt.ylabel(r'$AE_\mathrm{num}$')
plt.legend(ncol = 2)
plt.show()