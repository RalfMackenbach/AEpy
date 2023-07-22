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

rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)





def drift_asymptotic(stel,a_minor,k2):
    from scipy import  special
    # NAE drifts, leading order and first order correction
    # normalized as true drift * q * a * B0 * r / H
    # 1/B0 dependence due to choice of Bref
    E_k_K_k =  special.ellipe(k2)/special.ellipk(k2)
    wa0 = a_minor * -stel.etabar / stel.B0 *(2*E_k_K_k-1)
    wa1 = a_minor * -stel.r*stel.etabar / stel.B0 *(2/stel.etabar*stel.B20_mean+stel.etabar*(4*E_k_K_k*E_k_K_k - 2*(3-2*k2)*E_k_K_k + (1-2*k2)) +\
                                                               2/stel.etabar*stel.B2c*(2*E_k_K_k*E_k_K_k - 4*k2*E_k_K_k + (2*k2-1)))
    return wa0, wa1





def plot_precession_func(ax,AE_obj,save=False,filename='AE_precession.eps',nae=False,stel=None,alpha=0.0,iota=1.0,color='blue',flip_sign=True):
    r"""
    Plots the precession as a function of the bounce-points and k2.
    """
    # reshape for plotting
    sgn = 1.0

    if flip_sign:
        sgn = -1.0

    # calculate bounce angles
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
            walpha_bounceplot.extend([sgn*walpha_at_lam[idx]])
            walpha_bounceplot.extend([sgn*walpha_at_lam[idx]])

    roots_ordered, wpsi_bounceplot = (list(t) for t in zip(*sorted(zip(roots_bounceplot, wpsi_bounceplot))))
    roots_ordered, walpha_bounceplot = (list(t) for t in zip(*sorted(zip(roots_bounceplot, walpha_bounceplot))))
    wa0, wa1 = drift_asymptotic(stel,AE_obj.a_minor,AE_obj.k2)
    ax.plot(AE_obj.k2, wa0+wa1, color = 'red', linestyle='dashed')
    roots_ordered     = np.asarray(roots_ordered)/iota
    roots_ordered_chi = stel.iotaN*roots_ordered - alpha
    khat = np.sin(np.mod(roots_ordered_chi/2.0,2*np.pi))

    # plot as function of khat^2
    ax.scatter(khat**2,np.asarray(walpha_bounceplot),s=0.2,marker='.',color=color,facecolors=color)



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

stel_QA  = Qsc.from_paper('precise QH')

stel_HSX.zs = -stel_HSX.zs
stel_QA.zs  = -stel_QA.zs
stel_QA.spsi = -1



# make stellarator in VMEC
file_HSX = "./configs/wout_HSX.nc"
file_QA = "./configs/wout_precise_QH.nc"
mpi = MpiPartition()
mpi.write()
vmec_HSX = Vmec(file_HSX,mpi=mpi,verbose=True)
vmec_QA = Vmec(file_QA,mpi=mpi,verbose=True)
vmec_HSX.run()
vmec_QA.run()
wout_HSX = vmec_HSX.wout
wout_QA = vmec_QA.wout
a_minor_HSX = wout_HSX.Aminor_p
a_minor_QA = wout_QA.Aminor_p


lam_res = 1001

rho = 0.1
omn = 1.0
omt = 1.0
omnigenous = True


color = 'black'

fig, ax = plt.subplots(2,2,figsize=(6,4),sharex=True,sharey=True,tight_layout=True)


# do HSX first


# find r
stel_HSX.r = a_minor_HSX*rho
# omn and omt not needed, but let's set anyhow
omn_input = omn
omt_input = omt
stel_HSX.calculate()
NAE_AE_HSX = ae.AE_pyQSC(stel_obj = stel_HSX, r=stel_HSX.r, alpha=0.0, N_turns=1.0, nphi=10001,
                    lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None',
                    a_minor=a_minor_HSX)
fl =vmec_fieldlines(vmec_HSX,rho**2,0.0,theta1d=np.linspace(-1,1,3))
iota = fl.iota[0]
iotaN = iota - 4
VMEC_AE_HSX = ae.AE_vmec(vmec_HSX,rho**2,n_turns=1.0*np.abs(iota/iotaN),
                    booz=True,lam_res=lam_res,gridpoints=10001,plot=False,
                    mod_norm='T')

plot_precession_func(ax[0,1],NAE_AE_HSX,stel=stel_HSX,alpha=0.0,iota=1.0,color=color)
ax[0,1].set_title(r'HSX (NAE)')
plot_precession_func(ax[1,1],VMEC_AE_HSX,stel=stel_HSX,alpha=0.0,iota=iota,color=color)
ax[1,1].set_title(r'HSX (VMEC)')


# do precise QA next

# find r
stel_QA.r = a_minor_QA*rho
# omn and omt not needed, but let's set anyhow
omn_input = omn
omt_input = omt
stel_QA.calculate()
NAE_AE_QA = ae.AE_pyQSC(stel_obj = stel_QA, r=stel_QA.r, alpha=0.0, N_turns=1.0, nphi=10001,
                    lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None',
                    a_minor=a_minor_QA)
fl =vmec_fieldlines(vmec_QA,rho**2,0.0,theta1d=np.linspace(-1,1,3))
iota = fl.iota[0]
iotaN = iota - 1
VMEC_AE_QA = ae.AE_vmec(vmec_QA,rho**2,n_turns=1.0*np.abs(stel_QA.iota/stel_QA.iotaN),
                    booz=True,lam_res=lam_res,gridpoints=10001,plot=False,
                    mod_norm='T')

plot_precession_func(ax[0,0],NAE_AE_QA,stel=stel_QA,alpha=0.0,iota=1.0,color=color,flip_sign=False)
ax[0,0].set_title(r'precise QA (NAE)')
plot_precession_func(ax[1,0],VMEC_AE_QA,stel=stel_QA,alpha=0.0,iota=iota,color=color,flip_sign=False)
ax[1,0].set_title(r'precise QA (VMEC)')



ax[0,0].set_xlim([0.0,1.0])
ax[0,1].set_xlim([0.0,1.0])
ax[1,0].set_xlim([0.0,1.0])
ax[1,1].set_xlim([0.0,1.0])

# ax[0,0].set_ylim([-0.2,0.2])
# ax[0,1].set_ylim([-0.2,0.2])
# ax[1,0].set_ylim([-0.2,0.2])
# ax[1,1].set_ylim([-0.2,0.2])

ax[0,0].set_ylabel(r'$\hat{\omega}_\alpha$')
ax[1,0].set_ylabel(r'$\hat{\omega}_\alpha$')
ax[1,0].set_xlabel(r'$\hat{k}$')
ax[1,1].set_xlabel(r'$\hat{k}$')

ax[0,0].grid()
ax[0,1].grid()
ax[1,0].grid()
ax[1,1].grid()

plt.savefig('./figures/precession_comparison.png',dpi=1000)

plt.show()