import scipy
import numpy as np
from    qsc     import  Qsc
from    AEpy    import  ae_routines as      ae
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from   matplotlib        import rc




def plot_AE_per_lam_func(AE_obj,save=False,filename='AE_per_lam.eps',scale=1.0):
    r"""
    Plots AE per bouncewell
    """
    import  matplotlib.pyplot   as      plt
    import  matplotlib          as      mpl
    from    matplotlib          import  cm
    import  matplotlib.colors   as      mplc
    plt.close('all')

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 10}

    mpl.rc('font', **font)
    fig ,ax = plt.subplots(1, 1, figsize=(scale*6, scale*4.0))
    ax.set_xlim(min(AE_obj.z)/np.pi,max(AE_obj.z)/np.pi)

    lam_arr   = np.asarray(AE_obj.lam).flatten()
    ae_per_lam = AE_obj.ae_per_lam
    list_flat = []
    for val in ae_per_lam:
        list_flat.extend(val)
    max_ae_per_lam = max(list_flat)

    roots=AE_obj.roots

    cm_scale = lambda x: x
    colors_plot = [cm.plasma(cm_scale(np.asarray(x) * 1.0/max_ae_per_lam)) for x in ae_per_lam]

    # iterate over all values of lambda
    for idx_lam, lam in enumerate(lam_arr):
        b_val = 1/lam

        # iterate over all bounce wells
        for idx_bw, _ in enumerate(ae_per_lam[idx_lam]):
            bws = roots[idx_lam]
            # check if well crosses boundary
            if(bws[2*idx_bw] > bws[2*idx_bw+1]):
                ax.plot([bws[2*idx_bw]/np.pi, max(AE_obj.z)/np.pi], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])
                ax.plot([min(AE_obj.z)/np.pi, bws[2*idx_bw+1]/np.pi], [b_val, b_val],color=colors_plot[idx_lam][idx_bw])
            # if not normal plot
            else:
                ax.plot([bws[2*idx_bw]/np.pi, bws[2*idx_bw+1]/np.pi], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])

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


    roots_ordered = [root/np.pi for root in roots_ordered]

    ax.plot(AE_obj.z/np.pi,AE_obj.modb,color='black',linewidth=2)
    ax2 = ax.twinx()
    

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        print(idx,array[idx])
        return idx


    # ax2.plot(roots_ordered, wpsi_bounceplot, 'cornflowerblue',linestyle='dashdot',label=r'$\omega_\psi$')
    # ax2.plot(roots_ordered, walpha_bounceplot, 'tab:green',linestyle='dashed',label=r'$\omega_\alpha$')
    # ax2.plot(AE_obj.z/np.pi,AE_obj.z*0.0,linestyle='dotted',color='red')
    idx0=find_nearest(AE_obj.walpha, 0.0)
    lam0= AE_obj.lam[idx0][0]
    ax.axhline(1/lam0,color='green',linestyle='dashed')
    ax.set_ylabel(r'$B$')
    # ax2.set_ylabel(r'$\omega_\alpha$',color='tab:green')
    ax2.tick_params(axis='y', colors='black',direction='in')
    ax2.set_yticks([])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(r'$\chi$')
    # ax.tick_params(axis='both',direction='in')
    # ax2.legend(loc='lower right')
    max_norm = 1.0
    cbar = plt.colorbar(cm.ScalarMappable(norm=mplc.Normalize(vmin=0.0, vmax=max_norm, clip=False), cmap=cm.plasma), ticks=[0, max_norm], ax=ax,location='right',label=r'$\widehat{A}_\lambda/\widehat{A}_{\lambda,\mathrm{max}}$') #'%.3f'
    # cbar.ax.set_ticklabels([0, round(max_norm)])
    if save==True:
        plt.savefig(filename, format='png',
            #This is recommendation for publication plots
            dpi=1000)
    plt.show()





rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

# inputs
filename= 'precise QA'
omn_weak= 0.1
omn_strong= 1e4
omt     = 0.0
r       = 1e-3
omnigenous = True


nphi = int(1e3+1)
stel = Qsc.from_paper(filename, nphi = nphi)
stel.spsi = -1
stel.r = r
stel.zs = -stel.zs

NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=0.0, N_turns=1, nphi=nphi,
            lam_res=10001,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
NAE_AE.calc_AE(omn=omn_weak,omt=omt,omnigenous=omnigenous)
plot_AE_per_lam_func(NAE_AE,scale=0.5,save=True,filename='AE_per_lam.png')


NAE_AE.calc_AE(omn=omn_strong,omt=omt,omnigenous=omnigenous)
plot_AE_per_lam_func(NAE_AE,scale=0.5,save=True,filename='AE_per_lam_strongly_driven.png')