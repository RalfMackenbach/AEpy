import  numpy   as      np
from    qsc     import  Qsc
from    AEpy    import  ae_routines as      ae
from    simsopt.util.mpi            import  MpiPartition
from    simsopt.mhd.vmec            import  Vmec
from    simsopt.mhd.boozer            import  Boozer
from    simsopt.mhd.vmec_diagnostics import  vmec_fieldlines, vmec_splines
import  matplotlib.pyplot           as      plt
from    matplotlib                  import  rc
import matplotlib.ticker as mtick



# set up matplotlib
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

# numerical scaling parameters
c = int(1)

# set parameters
plot = False
omnigenous = True
lam_res = 1001
rho_res = 50
omn = 10000.0
omt = 0.0
file_qsc = "precise QA"


# Set up arrays
rho_arr = np.logspace(-8,-1,rho_res) # rho = r / a_minor
ae_num_qsc      = np.empty_like(rho_arr)*np.nan
asym_ae_weak    = np.empty_like(rho_arr)*np.nan
asym_ae_strong  = np.empty_like(rho_arr)*np.nan


# make stellarator in QSC
nphi = int(1e3+1)
stel = Qsc.from_paper(file_qsc, nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs





# loop over r
for idx, rho in enumerate(rho_arr):
    stel.r = rho
    omn_input = omn * rho
    omt_input = omt * rho
    stel.calculate()
    NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=0.0, N_turns=1, nphi=10*nphi,
                lam_res=lam_res,get_drifts=False,normalize='ft-vol',AE_lengthscale='None')
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

fig, ax = plt.subplots(1,1,figsize=(10,5),tight_layout=True)

print(np.abs(1-ae_num_qsc/asym_ae_weak))
print(np.abs(1-ae_num_qsc/asym_ae_strong))

ax.loglog(rho_arr,np.abs(1-ae_num_qsc/asym_ae_weak)*100,color='black',linestyle='--',label=r'weakly driven')
ax.loglog(rho_arr,np.abs(1-ae_num_qsc/asym_ae_strong)*100,color='black',linestyle='dashdot',label=r'strongly driven')
ax.set_ylabel(r'Relative error')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.set_xlabel(r'$\varrho$')
ax.grid()
ax.legend()

plt.show()