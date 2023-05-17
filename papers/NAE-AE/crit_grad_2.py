import  numpy                           as      np
from    qsc                             import  Qsc
from    simsopt.util.mpi                import  MpiPartition
from    simsopt.mhd.vmec                import  Vmec
from    simsopt.mhd.boozer              import  Boozer
from    simsopt.mhd.vmec_diagnostics    import  vmec_fieldlines, vmec_splines
import  matplotlib.pyplot               as      plt
from    matplotlib                      import  rc
import  matplotlib.ticker               as mtick
import  matplotlib                      as mpl
import  matplotlib.colors as colors
from    AEpy    import  ae_routines     as      ae
from matplotlib.lines import Line2D



nphi = int(1e3+1)
name='precise QH'
stel = Qsc.from_paper(name, nphi = nphi)
B20_base = stel.B20


omn_weak = 1e-3
omn_strong= 1e3


c_arr = np.linspace(0.9,1.1,10)
crit_arr = np.empty_like(c_arr)
crit_lead = np.empty_like(c_arr)
B20_arr = np.empty_like(c_arr)


fig, ax = plt.subplots(1,2,figsize=(8,4),tight_layout=True)

for c_idx,c_val in enumerate(c_arr):
    # adjust B20 
    stel.B20 = B20_base*c_val
    stel.r = 1e-2
    crit_lead_val = np.abs(1.61591 * stel.etabar)
    NAE_AE = ae.AE_pyQSC(stel_obj = stel, alpha=-0.0, N_turns=3, nphi=10*nphi,
                lam_res=1, get_drifts=True,normalize='ft-vol',AE_lengthscale='None',epsrel=1e-5)
    NAE_AE.calc_AE_quad(omn=stel.spsi*crit_lead_val*omn_weak,omt=stel.spsi*0.0,omnigenous=True)
    ae_weak = NAE_AE.ae_tot
    NAE_AE.calc_AE_quad(omn=stel.spsi*crit_lead_val*omn_strong,omt=stel.spsi*0.0,omnigenous=True)
    ae_strong = NAE_AE.ae_tot
    # A_weak * (omn/omn_weak)**3 = A_strong * (omn/omn_strong)
    crit_arr[c_idx] =  np.sqrt(ae_strong/ae_weak)*np.sqrt(omn_weak/omn_strong)*omn_weak
    B20_arr[c_idx] = stel.B20_mean*c_val
    crit_lead[c_idx] = crit_lead_val
    print('\n')


ax[0].scatter(crit_lead,crit_arr,alpha=0.5)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel(r'$1.61 \dots \cdot \eta$')
ax[0].set_ylabel(r'$\omega_\mathrm{crit}$')

ax[1].scatter(B20_arr,crit_arr/crit_lead-1,alpha=0.5)
ax[1].set_ylabel(r'$\omega_{\mathrm{crit,num}}/\omega_{\mathrm{crit,0}}-1$')
ax[1].set_xlabel(r'$\langle B_{20} \rangle$')


fig.suptitle(r'$r = 10^{-2}$, precise QA+well')



plt.show()