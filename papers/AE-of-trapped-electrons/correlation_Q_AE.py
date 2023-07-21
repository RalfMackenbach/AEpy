import  itertools
import  matplotlib.lines    as      mlines
import  numpy               as      np
import  numpy               as      np
from    AEpy                import  mag_reader
import  matplotlib.pyplot   as      plt
import  matplotlib          as      mpl
from    matplotlib          import  cm
from    AEpy                import  ae_routines     as ae
from    matplotlib.colors   import  ListedColormap, LinearSegmentedColormap
plt.close('all')

from matplotlib import rc

# mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size': 13})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)


def boundary_and_refinement(gist_file,refine=True,refined_grid=10001,boundary='None',plot=False):
    gist_file.include_endpoint()
    if boundary=='truncate':
        gist_file.truncate_domain()
    if boundary=='extend':
        gist_file.extend_domain()
    if refine==True:
        gist_file.refine_profiles(refined_grid)
    if plot==True:
        gist_file.plot_geometry()
    return gist_file
    



############## some variables to changes ##############
lam_res      = 1000
make_plots   = False
force_omnigenous = False
normalize   = 'ft-vol'
ae_length   = 'None'










Q_list=np.asarray([7.645847154751877,37.11184344194595,66.28563923274653,88.89918001896879,0.2214852174780842,0.8211983903213222,1.848364581960216,8.272401253718314,0.42191750695843805,0.31121588405080763,0.9969773188039313,5.042615883328997,0.9204760920288282,79.72394526587243,2272.082920266666])


AE_list = []
omn_list = np.asarray([1.0,2.0,3.0,4.0])





path = "../../tests/gist_files/"



print('calcuting AE for D3D')
# datapoints exist for D3D omn=[1.0,2.0,3.0,4.0]
file = 'gist_D3D_vac_05'
gist_file = path+file
gist_file = mag_reader.mag_data(gist_file)
gist_file = boundary_and_refinement(gist_file)
ae_dat = ae.AE_gist(gist_file,normalize=normalize,AE_lengthscale=ae_length,lam_res=lam_res)
for omn in omn_list:
    ae_dat.calc_AE(omn,omt=0.0,omnigenous=True)
    if make_plots==True:
        ae_dat.plot_AE_per_lam()
    AE_list = np.append(AE_list,ae_dat.ae_tot)
# Plot last four to see where we are in terms of scaling law
if make_plots==True:
    plt.plot(omn_list,AE_list[-4::])
    plt.show()



print('calcuting AE for HSX')
# datapoints exist for HSX omn=[1.0,2.0,3.0,4.0]
file = 'gist_HSX.txt'
gist_file = path+file
gist_file = mag_reader.mag_data(gist_file)
gist_file = boundary_and_refinement(gist_file)
ae_dat = ae.AE_gist(gist_file,normalize=normalize,AE_lengthscale=ae_length,lam_res=lam_res)
for omn in omn_list:
    if make_plots==False:
        ae_dat.calc_AE_fast(omn,omt=0.0,omnigenous=force_omnigenous)
    if make_plots==True:
        ae_dat.calc_AE(omn,omt=0.0,omnigenous=force_omnigenous)
        ae_dat.plot_AE_per_lam()
    AE_list = np.append(AE_list,ae_dat.ae_tot)
# Plot last four to see where we are in terms of scaling law
if make_plots==True:
    plt.plot(omn_list,AE_list[-4::])
    plt.show()


print('calcuting AE for W7-X (HM)')
# datapoints exist for W7X (HM) omn=[1.0,2.0,3.0,4.0]
file = 'gist_W7XHM.txt'
gist_file = path+file
gist_file = mag_reader.mag_data(gist_file)
gist_file = boundary_and_refinement(gist_file)
ae_dat = ae.AE_gist(gist_file,normalize=normalize,AE_lengthscale=ae_length,lam_res=lam_res)
for omn in omn_list:
    if make_plots==False:
        ae_dat.calc_AE_fast(omn,omt=0.0,omnigenous=force_omnigenous)
    if make_plots==True:
        ae_dat.calc_AE(omn,omt=0.0,omnigenous=force_omnigenous)
        ae_dat.plot_AE_per_lam()
    AE_list = np.append(AE_list,ae_dat.ae_tot)
# Plot last four to see where we are in terms of scaling law
if make_plots==True:
    plt.plot(omn_list,AE_list[-4::])
    plt.show()




print('calcuting AE for W7-X (SC)')
# datapoints exist for W7X (SC) [omn,omt]=[3.0,0.0] and [omn,omt]=[0.0,3.0]
file = 'gist_W7XSC.txt'
gist_file = path+file
gist_file = mag_reader.mag_data(gist_file)
gist_file.include_endpoint()
gist_file = boundary_and_refinement(gist_file)
ae_dat = ae.AE_gist(gist_file,normalize=normalize,AE_lengthscale=ae_length,lam_res=lam_res)
omn_list = [3.0,0.0]
omt_list = [0.0,3.0]
for idx, _ in enumerate(omn_list):
    if make_plots==False:
        ae_dat.calc_AE_fast(omn_list[idx],omt=omt_list[idx],omnigenous=force_omnigenous)
    if make_plots==True:
        ae_dat.calc_AE(omn_list[idx],omt_list[idx],omnigenous=force_omnigenous)
        ae_dat.plot_AE_per_lam()
    AE_list = np.append(AE_list,ae_dat.ae_tot)


print('calcuting AE for D3D (omt)')
# one more datapoint exists for DIII-D omt=3.0 omn=0.0
file = 'gist_D3D_vac_05'
gist_file = path+file
gist_file = mag_reader.mag_data(gist_file)
gist_file = boundary_and_refinement(gist_file)
ae_dat = ae.AE_gist(gist_file,normalize=normalize,AE_lengthscale=ae_length,lam_res=lam_res)
if make_plots==False:
    ae_dat.calc_AE_fast(0.0,3.0,omnigenous=True)
if make_plots==True:
    ae_dat.calc_AE(0.0,3.0,omnigenous=True)
    ae_dat.plot_AE_per_lam()
AE_list = np.append(AE_list,ae_dat.ae_tot)
plt.close('all')


############### All data is in AE list now. Let us first normalize the fluxes Te
tau_list    = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7])
Q_norm       = np.asarray(Q_list)/(tau_list**(5/2))

# Supposing that the AE of the trapped ions is the same as that of the trapped electrons
# we thus have the total AE is AE_electrons+ AE_ions/tau) = AE_electrons(1 + 1/tau)
AE_total =  np.asarray(AE_list)



c = 1.0

fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(c*5/6*6, c*5/6*4.5))


marker = itertools.cycle(('P', 'P', 'P', 'P',
                          'o', 'o', 'o', 'o',
                          '^', '^', '^', '^',
                          'v','^','P'))
cmap = mpl.colormaps['plasma_r'].resampled(5)#cm.get_cmap('plasma_r', 5) # deprecated, but the LUT argument is not available in alternatives... >:(
all_vals = cmap([0,1,2,3,4])
colormap = ListedColormap(all_vals[1:5])

color  = itertools.cycle((1, 2, 3, 4,
                          1, 2, 3, 4,
                          1, 2, 3, 4,
                          3))

for idx, AE_val in enumerate(AE_total):
    if (tau_list[idx]==1.0):
        color_val = ((next(color)-1)/3 - 1.5)*1.0 + 1.5
        color_plt = colormap(color_val)
    else:
        color_plt='lightgrey'
    ax.scatter(AE_val, Q_norm[idx], marker=next(marker), color=color_plt, zorder=100, s=20)

print(AE_total)

norm = mpl.colors.Normalize(vmin=.5,vmax=4.5)
cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=colormap),ax=ax,ticks=[1, 2, 3, 4],label=r"$L_\mathrm{ref}/L_n$")
ax.xaxis.set_tick_params(which='major', direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
ax.yaxis.set_tick_params(which='major', direction='in', top='on')
ax.yaxis.set_tick_params(which='minor', direction='in', top='on')
ax.set_xlabel(r"$\widehat{A}$")
ax.set_ylabel(r"$\widehat{Q}_\mathrm{sat}$")
ax.grid(which='major', color='0.90', zorder=0)

# now do least squares
lnA = np.log(AE_total)
lnQ = np.log(Q_norm)

fit, cov = np.polyfit(lnA[0:-2], lnQ[0:-2], 1, cov=True)

err      = np.sqrt(np.diag(cov))

print("power law is:", fit[0], '±', err[0])



ms=5.5

DIIID       = mlines.Line2D([], [], color='black', marker='P', linestyle='None',
                          markersize=ms, label='DIII-D')
HSX         = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=ms, label='HSX')
W7XHM       = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                          markersize=ms, label='W7-X (HM)')
W7XSC       = mlines.Line2D([], [], color='black', marker='v', linestyle='None',
                          markersize=ms, label='W7-X (SC)')
NCSX        = mlines.Line2D([], [], color='black', marker='s', linestyle='None',
                          markersize=ms, label='NCSX')
ax.legend(handles=[DIIID, HSX, W7XHM, W7XSC])#,borderpad=0.2,labelspacing=0.3)


A_logspace = np.logspace(np.log10(AE_total.min()),np.log10(AE_total.max()),1000)
Q_fit = np.exp(fit[0]*np.log(A_logspace)+fit[1])
ax.loglog(A_logspace,Q_fit,color='black',zorder=10)

ax.set_ylim(Q_fit.min()/1.5,Q_fit.max()*1.5)

ax.set_yscale('log')
ax.set_xscale('log')
plt.savefig('plot_AE_Q_corr.png', format='png',
            #This is recommendation for publication plots
            dpi=1000)
plt.show()