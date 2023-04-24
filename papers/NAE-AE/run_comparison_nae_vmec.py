import  numpy                           as      np
from    qsc                             import  Qsc
from    AEpy                            import  ae_routines     as      ae
from    simsopt.util.mpi                import  MpiPartition
from    simsopt.mhd.vmec                import  Vmec
from    simsopt.mhd.vmec_diagnostics    import  vmec_fieldlines
import  matplotlib.pyplot               as      plt
from    matplotlib                      import  rc





# set up matplotlib
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

# do we plot during iteration?
plot = True

# force omnigenous
omnigenous = True

# set lam res
lam_res = 10001

# set rho res
rho_res = 10



# omn/rho and omt/rho
omn = 0.01
omt = 0.0


# Set up arrays
rho_arr = np.logspace(-3,0,rho_res) # rho = r / a_minor
ae_num_qsc      = np.empty_like(rho_arr)*np.nan
ae_num_vmec     = np.empty_like(rho_arr)*np.nan
asym_ae_weak    = np.empty_like(rho_arr)*np.nan
asym_ae_strong  = np.empty_like(rho_arr)*np.nan




# make stellarator in VMEC
# file = "./configs/wout_precise_QA.nc"
file = 'wout_precise_QA.nc'
mpi = MpiPartition()
mpi.write()
vmec = Vmec(file,mpi=mpi,verbose=True)
vmec.run()
wout = vmec.wout
a_minor = wout.Aminor_p
R_major = wout.Rmajor_p
B_ref = 2 * np.abs(wout.phi[-1] / (2 * np.pi)) / (wout.Aminor_p)**2
print(B_ref,R_major,a_minor,a_minor/R_major)


# construct break point
# set rho break point (< break: pyQSC, > break: VMEC)
N_s=int(1/vmec.ds+1)
rho_break = np.sqrt(1/N_s)

# make stellarator in QSC
nphi = int(1e3+1)
stel = Qsc.from_paper('precise QA', nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs
stel.B0 = B_ref
stel.R0 = R_major




# loop over r
for idx, rho in enumerate(rho_arr):
    stel.r = a_minor*rho
    omn_input = omn
    omt_input = omt
    stel.calculate()
    if rho < rho_break:
        NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=0.0, N_turns=1, nphi=nphi,
                    lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None',a_minor=a_minor)
        NAE_AE.calc_AE(omn=stel.spsi*omn_input,omt=stel.spsi*omt_input,omnigenous=omnigenous)
        ae_num_qsc[idx] = NAE_AE.ae_tot
        if plot:
            # NAE_AE.plot_AE_per_lam()
            NAE_AE.plot_precession(nae=True,stel=stel)
    if rho > rho_break:
        VMEC_AE = ae.AE_vmec(vmec,rho**2,n_turns=1,lam_res=lam_res)
        VMEC_AE.calc_AE(omn=-omn_input,omt=-omt_input,omnigenous=omnigenous)
        ae_num_vmec[idx] = VMEC_AE.ae_tot
        if plot:
            # VMEC_AE.plot_AE_per_lam()
            VMEC_AE.plot_precession(nae=True,stel=stel)

    # Asymptotic AE
    asym_ae_weak[idx] = NAE_AE.nae_ae_asymp_weak(omn_input,a_minor)

    # Strongly driven AE
    asym_ae_strong[idx] = NAE_AE.nae_ae_asymp_strong(omn_input,a_minor)

    






# plot

fig, ax = plt.subplots(1,1,figsize=(10,5),tight_layout=True)

ax.loglog(rho_arr,asym_ae_weak,label=r'$\widehat{A}_\mathrm{asymp,weak}$',color='black',linestyle='dotted')
# ax.loglog(rho_arr,asym_ae_strong,label=r'$\widehat{A}_\mathrm{asymp,strong}$',color='black',linestyle='--')
ax.scatter(rho_arr,ae_num_qsc,label=r'$\widehat{A}_\mathrm{qsc}$',color='black',marker='x')
ax.scatter(rho_arr,ae_num_vmec,label=r'$\widehat{A}_\mathrm{vmec}$',color='black',marker='o')
# ax.loglog(rho_arr,nae_ae,label=r'$\widehat{A}_\mathrm{qsc}$')
ax.set_ylabel(r'$\widehat{A}$')
ax.set_xlabel(r'$\varrho$')
ax.grid()
ax.tick_params(direction='in')
ax.set_xlim([rho_arr[0],rho_arr[-1]])
ax.set_ylim([np.nanmin(ae_num_qsc)/2,np.nanmax(ae_num_vmec)*2])
ax.legend()
plt.show()