##############################################################################
# Figure 4: precession of global QS equilibria - precise QA, precise QH & HSX
##############################################################################
from scipy.special import erf
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from BAD import bounce_int
import numpy as np
from qsc import Qsc
from AEpy import ae_routines as ae
from   matplotlib        import rc
import   matplotlib.pyplot  as plt
from    simsopt.mhd.vmec                import  Vmec
from    simsopt.mhd.vmec_diagnostics    import  vmec_fieldlines
from    simsopt.util.mpi                import  MpiPartition
from simsopt.mhd.boozer import Boozer
from tqdm import tqdm
from scipy import  special

# Plotting details
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)



def drift_asymptotic(stel,AE_obj, k2 = None):
    # Analytic NAE drifts, leading order and first order correction
    # normalized as true drift * q * B_ref * L_ref**2 * varrho / H
    # where varrho / r = sqrt(B_0/2 psi_edge)
    if not isinstance(k2, list) or not isinstance(k2, np.ndarray):  
        k2 = AE_obj.k2
    scale = AE_obj.Bref * AE_obj.Lref**2 * np.sqrt( stel.B0 / ( 2 * AE_obj.psi_edge_over_two_pi ))
    E_k_K_k =  special.ellipe(k2)/special.ellipk(k2)
    etabar = np.abs(stel.etabar)
    # Leading order
    wa0 = scale * etabar / stel.B0 *(2*E_k_K_k-1)
    # First order
    G = -(4*E_k_K_k*E_k_K_k - 2*(3-2*k2)*E_k_K_k + (1-2*k2))
    G_20 = -2
    G_2c = -2*(2*E_k_K_k*E_k_K_k - 4*k2*E_k_K_k + (2*k2-1))
    # Total precession
    wa1 = scale * stel.r*etabar / stel.B0 *(etabar*G + G_20/etabar*stel.B20_mean + G_2c/etabar*stel.B2c)
    return wa0, wa1

def lam_2_k(lam, stel):
    # Find k for the analytics given lambda, using the 2nd order approximation
    etabar = np.abs(stel.etabar)
    # Polynomial on k2: definition of lambda to second orderEq. (C1b)
    p0 = stel.r*stel.r*(stel.B20_mean + stel.B2c)/stel.B0 - stel.r*etabar + 1
    p2 = -8*stel.r*stel.r*stel.B2c/stel.B0 + 2*stel.r*etabar
    p4 = 8*stel.r*stel.r*stel.B2c/stel.B0

    k_out = np.empty_like(lam)
    for jlam, lam_val in enumerate(lam):
        if p2*p2 < 4*p4*(p0 - 1/lam_val/stel.B0):
            k_out[jlam] = np.nan
        else:
            # Compute solutions and choose largest 0 <= k2 <= 1
            k_roots = np.roots([p4, p2, p0 - stel.B0/lam_val])    
            if k_roots.max() >= 0 and k_roots.max() < 1:
                k_out[jlam] = np.sqrt(k_roots.max())
            elif k_roots.min() >= 0 and k_roots.min() < 1:
                k_out[jlam] = np.sqrt(k_roots.min())
            else:
                k_out[jlam] = np.nan
    return k_out

def plot_precession_func(ax,AE_obj,save=False,filename='AE_precession.eps',nae=False,stel=None,alpha=0.0,color='blue',flip_sign=True, k_chi_b = False):
    r"""
    Plots the precession as a function of the bounce-points and k2.
    """
    # reshape for plotting
    sgn = 1.0
    if flip_sign:
        sgn = -1.0
    # Collect walpha and root data
    walpha_bounceplot = []
    roots_bounceplot  = []
    lam_bounceplot = []
    k2_bounceplot = []
    # wpsi_bounceplot   = []
    for lam_idx, lam_val in enumerate(AE_obj.lam):
        root_at_lam = AE_obj.roots[lam_idx]
        # wpsi_at_lam = AE_obj.wpsi[lam_idx]
        walpha_at_lam= AE_obj.walpha[lam_idx]
        roots_bounceplot.extend(root_at_lam)
        for idx in range(len(walpha_at_lam)):
            # wpsi_bounceplot.extend([wpsi_at_lam[idx]])
            # wpsi_bounceplot.extend([wpsi_at_lam[idx]])
            # Doubled because two roots per lam
            walpha_bounceplot.extend([sgn*walpha_at_lam[idx]])
            walpha_bounceplot.extend([sgn*walpha_at_lam[idx]])
            lam_bounceplot.extend([lam_val])
            lam_bounceplot.extend([lam_val])
            k2_bounceplot.extend([AE_obj.k2[lam_idx]])
            k2_bounceplot.extend([AE_obj.k2[lam_idx]])


    # Decide which k to use: the bounce point definition or usual Roach definition (default)
    if k_chi_b:
        # Location of the minimum of B
        z_min = AE_obj.z[np.argmin(AE_obj.modb)]
        pos = int(len(AE_obj.modb)*(1-0.5/1.0))
        # Location of the tops
        z_max = AE_obj.z[pos+np.argmax(AE_obj.modb[pos:])]
        # Order walpha according to roots
        # roots_ordered, wpsi_bounceplot = (list(t) for t in zip(*sorted(zip(roots_bounceplot, wpsi_bounceplot))))
        roots_ordered, walpha_bounceplot = (list(t) for t in zip(*sorted(zip(roots_bounceplot, walpha_bounceplot))))
        walpha_bounceplot = np.array(walpha_bounceplot)
        # Define an effective chi that is a rescaled and recentred z so that 0 at the bottom, pi at the top
        roots_ordered     = np.asarray(z_min-roots_ordered)/(z_min-z_max)*np.pi
        # There is some arbitrariness in how k is defined 
        khat = np.sin(roots_ordered/2.0)
        mask = roots_ordered >= 0

        # Plot nae
        if stel:
            wa0, wa1 = drift_asymptotic(stel,AE_obj)

        # plot as function of khat^2
        ax.scatter(khat[mask]**2,np.asarray(walpha_bounceplot[mask]),s=0.2,marker='.',color=color,facecolors=color)

        if stel:
            ax.plot(AE_obj.k2, wa0, color = 'red',     linestyle='dotted')
            ax.plot(AE_obj.k2, wa0+wa1, color = 'green', linestyle='dashed')
    else:
        walpha_bounceplot = np.array(walpha_bounceplot)
        lam_bounceplot = np.squeeze(np.array(lam_bounceplot))
        k2_bounceplot = np.array(k2_bounceplot)

        # Plot nae
        if stel:
            k = lam_2_k(lam_bounceplot, stel)
            wa0, wa1 = drift_asymptotic(stel,AE_obj, k*k)
        
        # plot as function of khat^2
        ax.scatter(k2_bounceplot,walpha_bounceplot,s=0.2,marker='.',color=color,facecolors=color)
        if stel:
            ax.plot(k2_bounceplot, wa0, color = 'red',     linestyle='dotted')
            ax.plot(k2_bounceplot, wa0+wa1, color = 'green', linestyle='dashed')


# construct HSX stellarator
stel_HSX = Qsc( nfp=4,
                spsi=1.,
                rc=[ 1.22015235e+00, 2.06770090e-01, 1.82712358e-02, 2.01793457e-04,
                -5.40158003e-06],
                zs=[-0.00000000e+00,-1.66673918e-01,-1.65358508e-02,-4.95694105e-04,
                9.65041347e-05],
                etabar=-1.25991504,
                B2c=-0.36522788,
                order='r2')
stel_HSX.zs = -stel_HSX.zs


stel_QA     = Qsc.from_paper('precise QA')
stel_QA.zs  = -stel_QA.zs
stel_QA.spsi= -1

stel_QH  = Qsc.from_paper('precise QH')
stel_QH.zs  = -stel_QH.zs
stel_QH.spsi = -1

k_chi_b = True

# make stellarator in VMEC
file_HSX = "../configs/wout_HSX.nc"
file_QA = "../configs/wout_precise_QA.nc"
file_QH = "../configs/wout_precise_QH.nc"
mpi = MpiPartition()
mpi.write()
vmec_HSX = Vmec(file_HSX,mpi=mpi,verbose=True)
vmec_QA = Vmec(file_QA,mpi=mpi,verbose=True)
vmec_QH = Vmec(file_QH,mpi=mpi,verbose=True)
vmec_HSX.run()
vmec_QA.run()
vmec_QH.run()
wout_HSX = vmec_HSX.wout
wout_QA = vmec_QA.wout
wout_QH = vmec_QH.wout
a_minor_HSX = wout_HSX.Aminor_p 
a_minor_QA = wout_QA.Aminor_p
a_minor_QH = wout_QH.Aminor_p



bs = {}
names = ["HSX", "QA", "QH"]
for name in names:
    print(name)
    bs[name] = Boozer(locals()["vmec_"+name])
    bs[name].register(locals()["vmec_"+name].s_full_grid)
    bs[name].verbose = False
    bs[name].bx.verbose = False
    bs[name].run()


lam_res = 1001

rho_arr = np.asarray([0.1,1.0])
omnigenous = True

num_plots = 15
alpha_arr = np.linspace(0.0,2.0,num_plots)

color = 'black'

fig, ax = plt.subplots(2,3,figsize=(6,4),sharex=True,sharey=True,tight_layout=True)


booz_bool = True
n_chi_turns = 1.0


# now, HSX
booz_bool = bs["HSX"]
for idx, rho in enumerate(rho_arr):
    # find r
    print(rho)
    helicity_HSX = +4
    with tqdm(total=num_plots, desc='Computing...') as pbar:
        for idy, alpha in enumerate(alpha_arr):
            VMEC_AE = ae.AE_vmec(vmec_HSX,rho**2,n_turns=n_chi_turns,
                        booz=booz_bool,lam_res=lam_res,gridpoints=10001,plot=False,
                        mod_norm=None,helicity=helicity_HSX, QS_mapping="QS", adjust = True, alpha = alpha)
            VMEC_AE.rho = rho
            plot_precession_func(ax[idx,2],VMEC_AE,alpha=alpha,color='lightgray', flip_sign = True, k_chi_b = k_chi_b)
            pbar.update(1)
    edge_toroidal_flux_over_2pi = np.abs(vmec_HSX.wout.phi[-1] / (2 * np.pi))
    stel_HSX.r = rho * np.sqrt(2 * edge_toroidal_flux_over_2pi/stel_HSX.B0) 
    # omn and omt not needed, but let's set anyhow
    stel_HSX.calculate()
    fl =vmec_fieldlines(vmec_HSX,rho**2,0.0,theta1d=np.linspace(-1,1,3))
    VMEC_AE_HSX = ae.AE_vmec(vmec_HSX,rho**2,n_turns=n_chi_turns,
                        booz=booz_bool,lam_res=lam_res,gridpoints=10001,plot=False,
                        mod_norm=None,helicity=helicity_HSX,QS_mapping=True, adjust = True)
    plot_precession_func(ax[idx,2],VMEC_AE_HSX,stel=stel_HSX,alpha=0.0,color=color, k_chi_b = k_chi_b)

# do precise QA first
booz_bool = bs["QA"]
for idx, rho in enumerate(rho_arr):
    with tqdm(total=num_plots, desc='Computing...') as pbar:
        for idy, alpha in enumerate(alpha_arr):
            VMEC_AE = ae.AE_vmec(vmec_QA,rho**2,n_turns=n_chi_turns,
                        booz=booz_bool,lam_res=lam_res,gridpoints=10001,plot=False,
                        mod_norm=None,helicity=0, QS_mapping="QS", adjust = True, alpha = alpha)
            VMEC_AE.rho = rho
            # cmap = plt.get_cmap('Greens')  # Choose any colormap you prefer
            # color = cmap(ind/num_plots) 
            plot_precession_func(ax[idx,0],VMEC_AE,alpha=alpha,color='lightgray', flip_sign = False, k_chi_b = k_chi_b)
            pbar.update(1)
    # find r
    edge_toroidal_flux_over_2pi = np.abs(vmec_QA.wout.phi[-1] / (2 * np.pi))
    stel_QA.r = rho * np.sqrt(2 * edge_toroidal_flux_over_2pi/stel_QA.B0) 
    stel_QA.calculate()
    VMEC_AE_QA = ae.AE_vmec(vmec_QA,rho**2,n_turns=n_chi_turns,
                        booz=booz_bool,lam_res=lam_res,gridpoints=10001,plot=False,
                        mod_norm=None,helicity=0,QS_mapping=True)
    plot_precession_func(ax[idx,0],VMEC_AE_QA,stel=stel_QA,alpha=0.0,color=color,flip_sign=False, k_chi_b = k_chi_b)

# now, precise QH
booz_bool = bs["QH"]
for idx, rho in enumerate(rho_arr):
    with tqdm(total=num_plots, desc='Computing...') as pbar:
        for idy, alpha in enumerate(alpha_arr):
            VMEC_AE = ae.AE_vmec(vmec_QH,rho**2,n_turns=n_chi_turns,
                        booz=booz_bool,lam_res=lam_res,gridpoints=10001,plot=False,
                        mod_norm=None,helicity=-4, QS_mapping="QS", adjust = True, alpha = alpha)
            VMEC_AE.rho = rho
            # cmap = plt.get_cmap('Greens')  # Choose any colormap you prefer
            # color = cmap(ind/num_plots) 
            plot_precession_func(ax[idx,1],VMEC_AE,alpha=alpha,color='lightgray', flip_sign = False, k_chi_b = k_chi_b)
            pbar.update(1)
    # find r
    edge_toroidal_flux_over_2pi = np.abs(vmec_QH.wout.phi[-1] / (2 * np.pi))
    stel_QH.r = rho * np.sqrt(2 * edge_toroidal_flux_over_2pi/stel_QH.B0) #a_minor_QH*rho
    # omn and omt not needed, but let's set anyhow
    stel_QH.calculate()
    VMEC_AE_QH = ae.AE_vmec(vmec_QH,rho**2,n_turns=n_chi_turns,
                        booz=booz_bool,lam_res=lam_res,gridpoints=10001,plot=False,
                        mod_norm=None,helicity=-4,QS_mapping=True)
    plot_precession_func(ax[idx,1],VMEC_AE_QH,stel=stel_QH,alpha=0.0,color=color,flip_sign=False, k_chi_b = k_chi_b)


ax[0,0].set_xlim([0.0,1.0])
ax[0,1].set_xlim([0.0,1.0])
ax[0,2].set_xlim([0.0,1.0])
ax[1,0].set_xlim([0.0,1.0])
ax[1,1].set_xlim([0.0,1.0])
ax[1,2].set_xlim([0.0,1.0])

ax[0,0].set_ylim([-0.1, 0.27])

ax[0,0].set_ylabel(r'$\hat{\omega}_\alpha$')
ax[1,0].set_ylabel(r'$\hat{\omega}_\alpha$')
ax[1,0].set_ylabel(r'$\hat{\omega}_\alpha$')
ax[1,0].set_xlabel(r'$k^2$')
ax[1,1].set_xlabel(r'$k^2$')
ax[1,1].set_xlabel(r'$k^2$')
ax[1,2].set_xlabel(r'$k^2$')

ax[0,0].grid()
ax[0,1].grid()
ax[1,0].grid()
ax[1,1].grid()
ax[0,2].grid()
ax[1,2].grid()

ax[0,0].set_title(r"precise QA"
                    "\n"
                    r"$\varrho = 0.1$")
ax[0,1].set_title(r"precise QH"
                    "\n"
                    r"$\varrho = 0.1$")
ax[0,2].set_title(r"HSX"
                    "\n"
                    r"$\varrho = 0.1$")
ax[1,0].set_title(r"$\varrho = 1.0$")
ax[1,1].set_title(r"$\varrho = 1.0$")
ax[1,2].set_title(r"$\varrho = 1.0$")


plt.savefig('./precession_comparison_shaded_chi_b.png',dpi=1000)

plt.show()