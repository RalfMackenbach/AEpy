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

def drift_asymptotic(stel,a_minor,k2):
    from scipy import  special
    E_k_K_k =  special.ellipe(k2)/special.ellipk(k2)
    wa0 = a_minor * -stel.etabar/stel.B0*(2*E_k_K_k-1)
    wa1 = a_minor * -stel.r*stel.etabar/stel.B0*(2/stel.etabar*stel.B20_mean+stel.etabar*(4*E_k_K_k*E_k_K_k - 2*(3-2*k2)*E_k_K_k + (1-2*k2)) +\
                                                               2/stel.etabar*stel.B2c*(2*E_k_K_k*E_k_K_k - 4*k2*E_k_K_k + (2*k2-1)))
    return wa0, wa1

def plot_precession_func(AE_obj,save=False,filename='AE_precession.eps',nae=False,stel=None,alpha=0.0):
    r"""
    Plots the precession as a function of the bounce-points and k2.
    """
    import matplotlib.pyplot as plt
    import matplotlib        as mpl

    plt.close('all')

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 10}

    mpl.rc('font', **font)

    # reshape for plotting
    walp_arr = np.nan*np.zeros([len(AE_obj.walpha),len(max(AE_obj.walpha,key = lambda x: len(x)))])
    for i,j in enumerate(AE_obj.walpha):
        walp_arr[i][0:len(j)] = j
    wpsi_arr = np.nan*np.zeros([len(AE_obj.wpsi),len(max(AE_obj.wpsi,key = lambda x: len(x)))])
    for i,j in enumerate(AE_obj.wpsi):
        wpsi_arr[i][0:len(j)] = j
    alp_l  = np.shape(walp_arr)[1]
    k2_arr = np.repeat(AE_obj.k2,alp_l)
    fig, ax = plt.subplots(3, 2, tight_layout=True, figsize=(2*3.5, 3/2*5.0))
    ax[1,0].scatter(k2_arr,walp_arr,s=0.2,marker='.',color='black',facecolors='black')
    ax[1,0].plot(AE_obj.k2,0.0*AE_obj.k2,color='red',linestyle='dashed')
    ax[1,1].scatter(k2_arr,wpsi_arr,s=0.2,marker='.',color='black',facecolors='black')
    ax[1,0].set_xlim(0,1)
    ax[1,1].set_xlim(0,1)
    ax[1,0].set_xlabel(r'$k^2$')
    ax[1,1].set_xlabel(r'$k^2$')
    ax[1,0].set_ylabel(r'$\langle \mathbf{v}_D \cdot \nabla y \rangle$',color='black')
    ax[1,1].set_ylabel(r'$\langle \mathbf{v}_D \cdot \nabla x \rangle$',color='black')


    # now do plot as a function of bounce-angle
    walpha_bounceplot = []
    roots_bounceplot  = []
    wpsi_bounceplot   = []
    for lam_idx, lam_val in enumerate(AE_obj.lam):
        root_at_lam = AE_obj.roots[lam_idx]
        wpsi_at_lam = AE_obj.wpsi[lam_idx]
        walpha_at_lam= AE_obj.walpha[lam_idx]
        roots_bounceplot.extend(root_at_lam)
        for idx in range(len(wpsi_at_lam)):
            wpsi_bounceplot.extend([wpsi_at_lam[idx]])
            wpsi_bounceplot.extend([wpsi_at_lam[idx]])
            walpha_bounceplot.extend([walpha_at_lam[idx]])
            walpha_bounceplot.extend([walpha_at_lam[idx]])

    roots_ordered, wpsi_bounceplot = (list(t) for t in zip(*sorted(zip(roots_bounceplot, wpsi_bounceplot))))
    roots_ordered, walpha_bounceplot = (list(t) for t in zip(*sorted(zip(roots_bounceplot, walpha_bounceplot))))
    ax[0,0].plot(AE_obj.z,AE_obj.modb,color='black')
    ax001= ax[0,0].twinx()
    ax001.plot(roots_ordered,walpha_bounceplot,color='tab:blue')
    ax001.plot(np.asarray(roots_ordered),0.0*np.asarray(walpha_bounceplot),color='tab:red',linestyle='dashed')
    ax[0,1].plot(AE_obj.z,AE_obj.modb,color='black')
    ax011= ax[0,1].twinx()
    ax011.plot(roots_ordered,wpsi_bounceplot,color='tab:blue')
    ax[0,0].set_xlim(AE_obj.z.min(),AE_obj.z.max())
    ax[0,1].set_xlim(AE_obj.z.min(),AE_obj.z.max())
    ax[0,0].set_xlabel(r'$z$')
    ax[0,1].set_xlabel(r'$z$')
    ax[0,0].set_ylabel(r'$B$')
    ax[0,1].set_ylabel(r'$B$')
    ax001.set_ylabel(r'$\langle \mathbf{v}_D \cdot \nabla y \rangle$',color='tab:blue')
    ax011.set_ylabel(r'$\langle \mathbf{v}_D \cdot \nabla x \rangle$',color='tab:blue')
    if nae:
        wa0, wa1 = drift_asymptotic(stel,AE_obj.a_minor,AE_obj.k2)
        ax[1,0].plot(AE_obj.k2, wa0, color = 'orange', linestyle='dotted', label='NAE (1st order)')
        ax[1,0].plot(AE_obj.k2, wa0+wa1, color = 'green', linestyle='dashed', label='NAE (2nd order)')
        ax[1,0].legend()
        ax[2,0].plot(AE_obj.k2, wa0, color = 'orange', linestyle='dotted', label='NAE (1st order)')
        ax[2,0].plot(AE_obj.k2, wa0+wa1, color = 'green', linestyle='dashed', label='NAE (2nd order)')
        ax[2,0].legend()

    # transform to k hat
    roots_ordered_chi = stel.iotaN*np.asarray(roots_ordered) - alpha
    khat = np.sin(np.mod(roots_ordered_chi/2.0,2*np.pi))
    
    # plot as function of khat^2
    ax[2,0].scatter(khat**2,walpha_bounceplot,s=0.2,marker='.',color='black',facecolors='black')
    ax[2,1].scatter(khat**2,wpsi_bounceplot,s=0.2,marker='.',color='black',facecolors='black')
    ax[2,0].set_xlabel(r'$\hat{k}^2$')
    ax[2,1].set_xlabel(r'$\hat{k}^2$')
    ax[2,0].set_ylabel(r'$\langle \mathbf{v}_D \cdot \nabla y \rangle$',color='black')
    ax[2,1].set_ylabel(r'$\langle \mathbf{v}_D \cdot \nabla x \rangle$',color='black')
    ax[2,0].set_xlim(0,1)
    ax[2,1].set_xlim(0,1)

    if save==True:
        plt.savefig(filename,dpi=1000)
    plt.show()

nphi = 1001
stel = Qsc.from_paper("precise QA+well", nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs
stel.r = 1e-1
stel.calculate()
print(stel.B20_variation)

alpha = 1.0

NAE_AE = ae.AE_pyQSC(stel_obj = stel, alpha=-alpha, N_turns=3, nphi=nphi,
                 lam_res=1001, get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
# NAE_AE.plot_geom()
plot_precession_func(NAE_AE,nae=True,stel=stel,alpha=alpha)


