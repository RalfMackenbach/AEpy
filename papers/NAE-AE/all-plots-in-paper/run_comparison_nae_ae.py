#############################################
# Figure 8a: weak and strong regimes in nae
#############################################

import  numpy   as      np
from    qsc     import  Qsc
from    AEpy    import  ae_routines as      ae
from    simsopt.util.mpi            import  MpiPartition
from    simsopt.mhd.vmec            import  Vmec
from    simsopt.mhd.boozer            import  Boozer
from    simsopt.mhd.vmec_diagnostics import  vmec_fieldlines, vmec_splines
import  matplotlib.pyplot           as      plt
from    matplotlib                  import  rc



# set up matplotlib
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

# do we plot during iteration?
plot = False

# save figure?
save_fig = False

# force omnigenous
omnigenous = True

# set lam res
lam_res = 2

# set rho res
rho_res = 15



# omn/rho and omt/rho
omn = 1000.0
omt = 0.0


# Set up arrays
rho_arr = np.logspace(-5,-1,rho_res,endpoint=True)
ae_num_qsc      = np.empty_like(rho_arr)*np.nan
asym_ae_weak    = np.empty_like(rho_arr)*np.nan
asym_ae_strong  = np.empty_like(rho_arr)*np.nan


# make stellarator in QSC
nphi = int(1e3+1)
stel = Qsc.from_paper('precise QA', nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs


# loop over r
for idx, rho in enumerate(rho_arr):
    print('rho is', round(rho,6))
    stel.r = rho
    omn_input = omn * rho
    omt_input = omt * rho
    stel.calculate()
    NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=0.0, N_turns=1, nphi=nphi,
                lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
    NAE_AE.calc_AE_quad(omn=stel.spsi*omn_input,omt=stel.spsi*omt_input,omnigenous=omnigenous)
    ae_num_qsc[idx] = NAE_AE.ae_tot
    if plot:
        NAE_AE.plot_AE_per_lam()
        NAE_AE.plot_precession(nae=True,stel=stel)

    # Asymptotic AE
    asym_ae_weak[idx] = NAE_AE.nae_ae_asymp_weak(omn_input)

    # Strongly driven AE
    asym_ae_strong[idx] = NAE_AE.nae_ae_asymp_strong(omn_input)


# plot


fig, ax = plt.subplots(1,1,figsize=(3,2),constrained_layout=True)


# fig.suptitle(r'precise QA')
ax.axvline(1.61591*np.abs(stel.etabar)/omn,color='blue',linestyle='dashdot')#label=r'$\varrho_\mathrm{crit}$')
ax.loglog(rho_arr,asym_ae_weak,color='red',linestyle='dotted')#,label=r'$\widehat{A}_\mathrm{weak}$')
ax.loglog(rho_arr,asym_ae_strong,color='red',linestyle='--')#,label=r'$\widehat{A}_\mathrm{strong}$')
ax.scatter(rho_arr,ae_num_qsc,color='black',marker='x')#,label=r'$\widehat{A}_\mathrm{num}$')
# ax.scatter(rho_arr,ae_num_booz,label=r'$\widehat{A}_\mathrm{booz}$',color='blue',marker='o')
ax.set_ylabel(r'$\widehat{A}$')
ax.set_xlabel(r'$\varrho$')
ax.grid()
ax.tick_params(direction='in')
ax.set_xlim([rho_arr[0],rho_arr[-1]])
ax.set_ylim([np.nanmin(ae_num_qsc)/2,np.nanmax(ae_num_qsc)*2])
# ax.legend()
if save_fig:
    plt.savefig('comparison_nae_AE.png',dpi=1000)
plt.show()
