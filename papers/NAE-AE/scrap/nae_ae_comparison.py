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




nphi = int(1e3+1)
names = ["precise QA","precise QA+well","precise QH","precise QH+well"]
r_arr =  np.logspace(-2,-1,10)



def nae_ae_asymp(stel,omn):
    ae_fac  = 0.666834
    r       = stel.r
    eta     = np.abs(stel.etabar)
    prefac  = 1 / (4*np.sqrt(np.pi))
    return prefac * np.sqrt(r/eta) * (r*omn)**3 * ae_fac




ae_arr = []
ae_ana = []

for name  in names:
    print('calculating for', name)
    stel = Qsc.from_paper(name, nphi = nphi)
    stel.spsi = -1
    stel.zs = -stel.zs
    stel.r = 1e-2
    stel.calculate()

    for idx, r in enumerate(r_arr):
        stel.r = r 
        stel.calculate()
        omn = 1.0
        NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=0.0, N_turns=1, nphi=nphi,
                    lam_res=2000,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
        NAE_AE.calc_AE(omn=stel.r*stel.spsi*omn,omt=stel.spsi*0.0,omnigenous=True)
        ae_arr.append(NAE_AE.ae_tot)
        ae_ana.append(nae_ae_asymp(stel,omn))



fig = plt.figure()
ax = plt.gca()
ax.scatter(ae_arr,ae_ana)
line = np.linspace(np.min(ae_arr),np.max(ae_arr),100)
ax.plot(line,line,color='black',linestyle='dotted')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()