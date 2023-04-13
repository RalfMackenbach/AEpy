from scipy.special import erf
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from BAD import bounce_int
import numpy as np
from qsc import Qsc
from AEpy import ae_routines as ae
from   matplotlib        import rc
import   matplotlib.pyplot  as plt

rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

nphi = int(1e3+1)
stel = Qsc.from_paper("precise QH", nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs
stel.r = 1e-1
stel.calculate()

NAE_AE = ae.AE_pyQSC(stel_obj = stel, alpha=-0.0, N_turns=3, nphi=nphi,
                 lam_res=1000, get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
# NAE_AE.plot_geom()
NAE_AE.plot_precession(nae = stel)

# # Scaling with w_star, in the weakly and strongly driven limits
# N_wstar = 100
# wstar_array = np.logspace(-2,6,N_wstar)
# ae_array = np.zeros(N_wstar)
# for ind, wstar in enumerate(wstar_array):
#     print(ind, ' out of ', N_wstar, ' (',100*(ind+1)/N_wstar,'%)        ', end="\r")
#     NAE_AE.calc_AE(omn=stel.spsi*stel.r*wstar,omt=stel.spsi*stel.r*0.0,omnigenous=True)
#     ae_array[ind] = NAE_AE.ae_tot
# print('\n')
# ae_min = ae_array.min()
# ae_max = ae_array.max()
# plt.loglog(wstar_array, ae_array, color = 'k')
# plt.loglog(wstar_array, ae_array[0]*(wstar_array/wstar_array[0])**3, color = 'k', linestyle='dashed')
# plt.loglog(wstar_array, ae_array[-1]*(wstar_array/wstar_array[-1]), color = 'k', linestyle='dashed')
# plt.xlabel(r'$\omega_\star$')
# plt.ylabel(r'$AE$')
# plt.ylim([ae_min, ae_max])
# plt.show()


# # Scaling with the radial coordinate r
# wstar = 1e1
# N_r = 20
# r_array = np.logspace(-3,-0.5,N_r)
# ae_array = np.zeros(N_r)
# for ind, r in enumerate(r_array):
#     print(ind, ' out of ', N_r, ' (',100*(ind+1)/N_r,'%)        ', end="\r")
#     stel.r = r
#     NAE_AE = ae.AE_pyQSC(stel_obj = stel, alpha=-0.0, N_turns=3, nphi=nphi,
#                  lam_res=1000, get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
#     NAE_AE.calc_AE(omn=stel.spsi*stel.r*wstar,omt=stel.spsi*stel.r*0.0,omnigenous=True)
#     ae_array[ind] = NAE_AE.ae_tot
# print('\n')
# ae_min = ae_array.min()
# ae_max = ae_array.max()
# ae_nae = r_array**(7/2)/3.0*np.sqrt(2*np.pi/np.abs(stel.etabar))*0.666834*wstar**3/(np.sqrt(np.pi)*np.pi*stel.B0)
# print(NAE_AE.ft_vol)
# plt.loglog(r_array, ae_array, color = 'k')
# plt.loglog(r_array, ae_nae, color = 'k',linestyle='dashed')
# # plt.loglog(r_array, ae_array[0]*(r_array/r_array[0])**(7/2), color = 'k', linestyle='dashed')
# # plt.loglog(r_array, ae_array[0]*(r_array/r_array[0])**(3/2), color = 'k', linestyle='dashed')
# plt.xlabel(r'$r$')
# plt.ylabel(r'$AE$')
# plt.ylim([ae_min, ae_max])
# plt.show()

# # Show AE plots on B profile
# wstar = 1.0
# NAE_AE.calc_AE(omn=stel.spsi*stel.r*wstar,omt=stel.spsi*stel.r*0.0,omnigenous=True)
# NAE_AE.plot_AE_per_lam()