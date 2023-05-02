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



# set up matplotlib
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

# numerical scaling parameters
c = int(1)

# set parameters
plot = False
omnigenous = True
alpha       = 0.0
lam_res = 2
rho_res = 5
omn_arr = np.asarray([1,100,10000])
file_qsc = "precise QA"


# make stellarator in QSC
nphi = int(1e3+1)
stel = Qsc.from_paper(file_qsc, nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs
r_sing = stel.r_singularity




# Set up arrays
rho_arr = np.logspace(-3,np.log10(r_sing),rho_res,endpoint=False) # rho = r / a_minor
# mesh the arrays
# first strong
ae_num_qsc_strong   = np.empty([rho_res,len(omn_arr)])*np.nan
asym_ae_strong      = np.empty([rho_res,len(omn_arr)])*np.nan
# now weak
ae_num_qsc_weak = np.empty([rho_res,len(omn_arr)])*np.nan
asym_ae_weak    = np.empty([rho_res,len(omn_arr)])*np.nan






# loop over r
for rho_idx, rho in enumerate(rho_arr):
    stel.r = rho
    stel.calculate()
    for omn_idx, omn in enumerate(omn_arr):
        omn = omn_arr[omn_idx] * rho
        NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=alpha, N_turns=1, nphi=10*nphi+1,
                    lam_res=lam_res,get_drifts=False,normalize='ft-vol',AE_lengthscale='None')
        NAE_AE.calc_AE_quad(omn=stel.spsi*omn,omt=0.0,omnigenous=omnigenous)
        ae_num_qsc_weak[rho_idx,omn_idx] = NAE_AE.ae_tot
        ae_num_qsc_strong[rho_idx,omn_idx] = NAE_AE.ae_tot
        if plot:
            NAE_AE.plot_AE_per_lam()
            NAE_AE.plot_precession(nae=True,stel=stel)
    

        # Asymptotic AE
        asym_ae_weak[rho_idx,omn_idx] = NAE_AE.nae_ae_asymp_weak(omn)

        # Strongly driven AE
        asym_ae_strong[rho_idx,omn_idx] = NAE_AE.nae_ae_asymp_strong(omn)

    






# plot

fig, ax = plt.subplots(1,1,figsize=(6, 4.0),tight_layout=True)



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = truncate_colormap(plt.cm.gray, 0.0, 0.8, 100)

col = cmap(np.linspace(0,1.0,len(omn_arr)))  
norm = mpl.colors.LogNorm(vmin=omn_arr.min(), vmax=omn_arr.max())

for idx,omn in enumerate(omn_arr):
    ax.loglog(rho_arr,np.abs(1-ae_num_qsc_weak[:,idx]/asym_ae_weak[:,idx])*100,color=col[idx],linestyle='-')
for idx,omn in enumerate(omn_arr):
    ax.loglog(rho_arr,np.abs(1-ae_num_qsc_strong[:,idx]/asym_ae_strong[:,idx])*100,color=col[idx],linestyle='dashed')
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax,label=r'$\hat{\omega}_n/\varrho$')


ax.axvline(r_sing,color='red',linestyle='dotted')
ax.set_ylabel(r'Relative error')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.set_xlabel(r'$\varrho$')
ax.grid()


custom_lines = [Line2D([0], [0], color='black', lw=1, linestyle='-'),
                Line2D([0], [0], color='black', lw=1, linestyle='dashed'),
                Line2D([0], [0], color='red', lw=1, linestyle='dotted')]

ax.legend(custom_lines, [r'weakly driven', 'strongly driven',r'$\varrho_{\mathrm{singularity}}$'],loc='lower right')

ax.set_xlim([rho_arr.min(),1.0])
ax.set_ylim(bottom=0.1)
plt.title(r'precise QA')

plt.savefig('figures/ae_asymptotics.png',dpi=1000,bbox_inches='tight')

plt.show()

print(ae_num_qsc_weak)