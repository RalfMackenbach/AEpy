import  numpy   as      np
from    qsc     import  Qsc
from    AEpy    import  ae_routines as      ae
from    simsopt.util.mpi            import  MpiPartition
from    simsopt.mhd.vmec            import  Vmec
from    simsopt.mhd.boozer            import  Boozer
from    simsopt.mhd.vmec_diagnostics import  vmec_fieldlines
import  matplotlib.pyplot           as      plt
from    matplotlib                  import  rc



def nae_ae_asymp(stel,omn,a_minor):
    ae_fac  = 0.666834
    r       = stel.r
    eta     = np.abs(stel.etabar)
    prefac  = np.sqrt(2) / (3 * np.pi)
    return prefac * np.sqrt(r/ a_minor**2 / eta) * (omn)**3 * ae_fac
# set up matplotlib
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)


# force omnigenous
omnigenous = True

# set lam res
lam_res = 1001


# omn/rho and omt/rho
omn = 0.01
omt = 0.0


# Set up arrays
res = 4
rho_arr = np.logspace(-np.log10(1001)/2,0.0,10) #np.logspace(np.log10(np.sqrt(1/201)),0,res) # rho = r / a_minor
nae_ae  = np.empty_like(rho_arr)
vmec_ae = np.empty_like(rho_arr)
booz_ae = np.empty_like(rho_arr)
asym_ae_weak = np.empty_like(rho_arr)
asym_ae_strong = np.empty_like(rho_arr)




# make stellarator in VMEC
file = "./configs/wout_precise_QA.nc"
# file = 'wout_precise_QA_000_000000.nc'
mpi = MpiPartition()
mpi.write()
vmec = Vmec(file,mpi=mpi,verbose=True)
vmec.run()
wout = vmec.wout
a_minor = wout.Aminor_p
R_major = wout.Rmajor_p
B_ref = 2 * np.abs(wout.phi[-1] / (2 * np.pi)) / (wout.Aminor_p)**2
print(B_ref,R_major,a_minor,a_minor/R_major)

# make Booz
bs = Boozer(vmec)
bs.register(vmec.s_full_grid)
bs.run()

# make stellarator in QSC
nphi = int(1e3+1)
stel = Qsc.from_paper('precise_QA', nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs
stel.B0 = B_ref
stel.R0 = R_major




# loop over r
for idx, rho in enumerate(rho_arr):
    stel.r = a_minor*rho
    omn_input = rho*omn
    omt_input = rho*omt
    stel.calculate()
    NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=0.0, N_turns=1, nphi=nphi,
                lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None',a_minor=a_minor)
    # NAE_AE.plot_geom()
    NAE_AE.calc_AE(omn=stel.spsi*omn_input,omt=stel.spsi*omt_input,omnigenous=omnigenous)
    # NAE_AE.plot_AE_per_lam()
    nae_ae[idx] = NAE_AE.ae_tot

    # loop over rho for vmec
    VMEC_AE = ae.AE_vmec(vmec,rho**2,n_turns=1,lam_res=lam_res)
    VMEC_AE.calc_AE(omn=-omn_input,omt=-omt_input,omnigenous=omnigenous)
    # VMEC_AE.plot_precession()
    # VMEC_AE.plot_AE_per_lam()
    vmec_ae[idx] = VMEC_AE.ae_tot

    BOOZ_AE = ae.AE_vmec(vmec,rho**2,booz=bs,n_turns=1,lam_res=lam_res)
    BOOZ_AE.calc_AE(omn=-omn_input,omt=-omt_input,omnigenous=omnigenous)
    # VMEC_AE.plot_precession()
    # VMEC_AE.plot_AE_per_lam()
    booz_ae[idx] = BOOZ_AE.ae_tot

    # Asymptotic AE
    asym_ae_weak[idx] = NAE_AE.nae_ae_asymp_weak(omn_input,a_minor)

    # Strongly driven AE
    asym_ae_strong[idx] = NAE_AE.nae_ae_asymp_strong(omn_input,a_minor)

    






# plot

fig, ax = plt.subplots(1,1,figsize=(10,5),tight_layout=True)

ax.loglog(rho_arr,asym_ae_weak,label=r'$\widehat{A}_\mathrm{asymp,weak}$',color='black',linestyle='dotted')
ax.loglog(rho_arr,asym_ae_strong,label=r'$\widehat{A}_\mathrm{asymp,strong}$',color='black',linestyle='--')
ax.loglog(rho_arr,vmec_ae,label=r'$\widehat{A}_\mathrm{vmec}$')
ax.loglog(rho_arr,booz_ae,label=r'$\widehat{A}_\mathrm{booz}$')
print(vmec_ae/asym_ae_weak)
# ax.loglog(rho_arr,nae_ae,label=r'$\widehat{A}_\mathrm{qsc}$')
ax.set_ylabel(r'$\widehat{A}$')
ax.set_xlabel(r'$\varrho$')
ax.set_xlim([rho_arr[0],rho_arr[-1]])
ax.legend()
plt.show()