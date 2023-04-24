import  numpy   as      np
from    qsc     import  Qsc
from    AEpy    import  ae_routines as      ae
from    simsopt.util.mpi            import  MpiPartition
from    simsopt.mhd.vmec            import  Vmec
from    simsopt.mhd.vmec_diagnostics import  vmec_fieldlines
import  matplotlib.pyplot           as      plt



def nae_ae_asymp(stel,omn,a_minor):
    ae_fac  = 0.666834
    r       = stel.r
    eta     = np.abs(stel.etabar)
    prefac  = np.sqrt(2) / (3 * np.pi)
    return prefac * np.sqrt(r/ a_minor**2 / eta) * (omn)**3 * ae_fac


# force omnigenous
omnigenous = True

# make stellarator
file = "../../../Python/Trapped_electron/configs/wout_precise_QA.nc"
mpi = MpiPartition(8)
mpi.write()
vmec = Vmec(file,mpi=mpi,verbose=True)
vmec.run()
wout = vmec.wout
a_minor = wout.Aminor_p
R_major = wout.Rmajor_p
B_ref = 2 * np.abs(wout.phi[-1] / (2 * np.pi)) / (wout.Aminor_p)**2
print(B_ref,R_major,a_minor,a_minor/R_major)

omn = 0.01
omt = 0.0


# Set up arrays
res = 10
rho_arr = np.logspace(np.log10(np.sqrt(1/201)),0,res) # rho = r / a_minor
nae_ae  = np.empty_like(rho_arr)
vmec_ae = np.empty_like(rho_arr)
asym_ae = np.empty_like(rho_arr)

# make stellarator
nphi = int(1e3+1)
stel = Qsc.from_paper('precise QA', nphi = nphi)
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
                lam_res=10001,get_drifts=True,normalize='ft-vol',AE_lengthscale='None',a_minor=a_minor)
    # NAE_AE.plot_geom()
    NAE_AE.calc_AE(omn=stel.spsi*omn_input,omt=stel.spsi*omt_input,omnigenous=omnigenous)
    # NAE_AE.plot_AE_per_lam()
    nae_ae[idx] = NAE_AE.ae_tot

    # loop over rho for vmec
    VMEC_AE = ae.AE_vmec(vmec,rho**2,n_turns=1,lam_res=10001)
    VMEC_AE.calc_AE(omn=-omn_input,omt=-omt_input,omnigenous=omnigenous)
    # VMEC_AE.plot_precession()
    # VMEC_AE.plot_AE_per_lam()
    vmec_ae[idx] = VMEC_AE.ae_tot

    # Asymptotic AE
    asym_ae[idx] = nae_ae_asymp(stel,omn_input,a_minor)

    







# plt.loglog(rho_arr,,label=r'$\widehat{A}_\mathrm{num,qsc}$')
plt.loglog(rho_arr,asym_ae,label=r'$\widehat{A}_\mathrm{asymp}$')
plt.loglog(rho_arr,vmec_ae,label=r'$\widehat{A}_\mathrm{vmec}$')
plt.loglog(rho_arr,nae_ae,label=r'$\widehat{A}_\mathrm{qsc}$')
# plt.loglog(rho_arr,vmec_ae)
# plt.loglog(rho_arr,nae_ae)
plt.ylabel(r'$\widehat{A}$')
plt.xlabel(r'$\rho$')
plt.legend()
plt.tight_layout()
# plt.loglog(rho_arr,ana_arr,label=r'$\widehat{A}_\mathrm{ana,qsc}$')
plt.show()