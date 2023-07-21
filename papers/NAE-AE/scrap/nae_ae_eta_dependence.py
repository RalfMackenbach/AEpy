from scipy.special import erf
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from BAD import bounce_int
import numpy as np
from qsc import Qsc
from AEpy import ae_routines as ae
from   matplotlib        import rc
import matplotlib.pyplot as plt

rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)




nphi = int(2e3+1)
res = 10
eta_arr =  -np.logspace(-2.0,2.0,res)
ae_arr  = np.empty_like(eta_arr)


# make base-case stellarator
stel = Qsc.from_paper('precise QA', nphi = nphi)
stel.r = 1e-6


for idx, eta in enumerate(eta_arr):
    stel.etabar = eta 
    stel.calculate()
    omn = 1.0
    omt = 0.0
    NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=0.0, N_turns=3, nphi=nphi,
                lam_res=2000,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
    # NAE_AE.plot_geom()
    NAE_AE.calc_AE(omn=stel.spsi*omn,omt=stel.spsi*omt,omnigenous=True)
    # NAE_AE.plot_AE_per_lam()
    ae_arr[idx] = NAE_AE.ae_tot



fig = plt.figure()
ax = plt.gca()
ax.loglog(np.abs(eta_arr),ae_arr)

ax.loglog(np.abs(eta_arr[0:int(res/3)]),2*ae_arr[0]*(eta_arr[0:int(res/3)]/eta_arr[0])**(3/2),linestyle='dotted',color='black',label=r'$\eta^{3/2}$')
ax.loglog(np.abs(eta_arr[int(2*res/3)::]),1.5*ae_arr[-1]*(eta_arr[int(2*res/3)::]/eta_arr[-1])**(-1/2),linestyle='dashed',color='black',label=r'$\eta^{-1/2}$')
ax.legend()
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\widehat{A}$')
plt.show()