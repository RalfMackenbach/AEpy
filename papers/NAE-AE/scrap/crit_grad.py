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



name_array = np.asarray(["precise QA", "precise QH", "precise QA+well", "precise QH+well", "2022 QA", "2022 QH nfp2", \
       "2022 QH nfp3 vacuum", "2022 QH nfp3 beta", "2022 QH nfp4 long axis", "2022 QH nfp4 well", "2022 QH nfp4 Mercier", \
       "2022 QH nfp7"])

crit_arr = np.empty(len(name_array))
crit_lead = np.empty(len(name_array))
B20_arr = np.empty(len(name_array))

nphi = 1001
omn_weak_base = 1e-3
omn_strong_base = 1e3

c_arr = np.asarray([1,10**(1/3),10**(2/3),10])

color_arr=['red','orange','green','blue']


fig, ax = plt.subplots(1,2,figsize=(8,4),tight_layout=True)

for c_idx,c_val in enumerate(c_arr):
    omn_weak = omn_weak_base/c_val
    omn_strong = c_val * omn_strong_base
    color=color_arr[c_idx]
    for ind_name, name in enumerate(name_array):
        print(name)
        stel = Qsc.from_paper(name, nphi = nphi)
        stel.r = 1e-2
        NAE_AE = ae.AE_pyQSC(stel_obj = stel, alpha=-0.0, N_turns=3, nphi=10*nphi,
                    lam_res=1, get_drifts=True,normalize='ft-vol',AE_lengthscale='None',epsrel=1e-4)
        NAE_AE.calc_AE_quad(omn=stel.spsi*omn_weak,omt=stel.spsi*0.0,omnigenous=True)
        ae_weak = NAE_AE.ae_tot
        NAE_AE.calc_AE_quad(omn=stel.spsi*omn_strong,omt=stel.spsi*0.0,omnigenous=True)
        ae_strong = NAE_AE.ae_tot
        # A_weak * (omn/omn_weak)**3 = A_strong * (omn/omn_strong)
        crit_arr[ind_name] =  np.sqrt(ae_strong/ae_weak)*np.sqrt(omn_weak/omn_strong)*omn_weak
        B20_arr[ind_name] = stel.B20_mean
        print(stel.B20_mean)
        crit_lead[ind_name] = np.abs(1.61591 * stel.etabar)
        print('\n')


    print('error in percent:', 100*np.abs(crit_arr/crit_lead-1))


    ax[0].scatter(crit_lead,crit_arr,color=color,alpha=0.5)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$1.61 \dots \cdot \eta$')
    ax[0].set_ylabel(r'$\omega_\mathrm{crit}$')

    ax[1].scatter(B20_arr,crit_arr/crit_lead,color=color,alpha=0.5)
    ax[1].set_yscale('log')
    ax[1].set_ylabel(r'$\omega_{\mathrm{crit,num}}/\omega_{\mathrm{crit,0}}$')
    ax[1].set_xlabel(r'$\langle B_{20} \rangle$')


fig.suptitle(r'$r = 10^{-2}$')



plt.show()