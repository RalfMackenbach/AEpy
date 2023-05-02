from scipy import special
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d

from BAD import bounce_int
import numpy as np
from qsc import Qsc
from AEpy import ae_routines as ae
from   matplotlib        import rc
import   matplotlib.pyplot  as plt

rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 14})
rc('text', usetex=True)

nphi = int(1e3+1)
stel = Qsc.from_paper("precise QA", nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs
stel.calculate()
N_r = 3
r_array = np.logspace(-2,-1,N_r)

for jr, r in enumerate(r_array):
    stel.r = r
    NAE_AE = ae.AE_pyQSC(stel_obj = stel, alpha=-0.0, N_turns=3, nphi=nphi,
                    lam_res=1000, get_drifts=True,normalize='ft-vol',AE_lengthscale='None')

    walp_arr = np.nan*np.zeros([len(NAE_AE.walpha),len(max(NAE_AE.walpha,key = lambda x: len(x)))])
    for i,j in enumerate(NAE_AE.walpha):
        walp_arr[i][0:len(j)] = j
    alp_l  = np.shape(walp_arr)[1]
    k_arr = np.nan*np.zeros([len(NAE_AE.roots),len(max(NAE_AE.roots,key = lambda x: len(x)))])
    for i,j in enumerate(NAE_AE.roots):
        k_arr[i][0:len(j)] = np.sin(0.5*np.remainder(j*NAE_AE.stel.iotaN,2*np.pi))
    wpsi_arr = np.nan*np.zeros([len(NAE_AE.wpsi),len(max(NAE_AE.wpsi,key = lambda x: len(x)))])
    for i,j in enumerate(NAE_AE.wpsi):
        wpsi_arr[i][0:len(j)] = j
    # lam_arr = (1-NAE_AE.k2*(np.amax(NAE_AE.modb)-np.amin(NAE_AE.modb))/np.amax(NAE_AE.modb))/np.amin(NAE_AE.modb)
    k2_arr = k_arr * k_arr # np.repeat(NAE_AE.k2,alp_l)
    k2 = np.mean(k2_arr, axis =1)
    k2_arr_anal = np.repeat(k2,alp_l)
    k2_arr = np.repeat(NAE_AE.k2,alp_l)
    

    plt.scatter(k2_arr,walp_arr,s=0.5,marker='.',color='black',facecolors='black')
    plt.scatter(k2_arr_anal,walp_arr,s=0.5,marker='.',color='blue',facecolors='blue')
    E_k_K_k = special.ellipe(NAE_AE.k2)/special.ellipk(NAE_AE.k2)
    wa = -NAE_AE.stel.etabar/NAE_AE.stel.B0*(2*E_k_K_k-1) # Negative sign because derivation for -etabar, no r because y
    # plt.plot(NAE_AE.k2, wa, color = 'orange', linestyle='dotted', label='NAE (1st order)')
    wa_lb = wa-NAE_AE.stel.r*NAE_AE.stel.etabar/NAE_AE.stel.B0*(2/NAE_AE.stel.etabar*NAE_AE.stel.B20.min()+NAE_AE.stel.etabar*(4*E_k_K_k*E_k_K_k - 2*(3-2*NAE_AE.k2)*E_k_K_k + (1-2*NAE_AE.k2)) +\
                                                            2/NAE_AE.stel.etabar*NAE_AE.stel.B2c*(2*E_k_K_k*E_k_K_k - 4*NAE_AE.k2*E_k_K_k + (2*NAE_AE.k2-1)))
    wa_ub = wa-NAE_AE.stel.r*NAE_AE.stel.etabar/NAE_AE.stel.B0*(2/NAE_AE.stel.etabar*NAE_AE.stel.B20.max()+NAE_AE.stel.etabar*(4*E_k_K_k*E_k_K_k - 2*(3-2*NAE_AE.k2)*E_k_K_k + (1-2*NAE_AE.k2)) +\
                                                            2/NAE_AE.stel.etabar*NAE_AE.stel.B2c*(2*E_k_K_k*E_k_K_k - 4*NAE_AE.k2*E_k_K_k + (2*NAE_AE.k2-1)))
    wa = wa-NAE_AE.stel.r*NAE_AE.stel.etabar/NAE_AE.stel.B0*(2/NAE_AE.stel.etabar*NAE_AE.stel.B20_mean+NAE_AE.stel.etabar*(4*E_k_K_k*E_k_K_k - 2*(3-2*NAE_AE.k2)*E_k_K_k + (1-2*NAE_AE.k2)) +\
                                                            2/NAE_AE.stel.etabar*NAE_AE.stel.B2c*(2*E_k_K_k*E_k_K_k - 4*NAE_AE.k2*E_k_K_k + (2*NAE_AE.k2-1)))
    plt.plot(NAE_AE.k2, wa, color = 'orange', linewidth=1.5, linestyle='dashed', label='NAE (2nd order)')
    plt.fill_between(NAE_AE.k2, wa_lb, wa_ub, alpha=0.5, facecolor='orange')
# plt.legend()
plt.plot(NAE_AE.k2,0.0*NAE_AE.k2,color='black',linestyle='dotted')

plt.xlim([0,1])
plt.xlabel(r'$k^2$')
plt.ylabel(r'$r\omega_\alpha$',color='black')
plt.show()

# NAE_AE.plot_geom()
# NAE_AE.plot_precession(nae = stel)