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
force_omnigenous = False
normalize   = 'ft-vol'
ae_length   = 'None'





path = "../../tests/gist_files/"



#
omn_arr = np.logspace(-3,3,100)
D3D_arr = np.zeros_like(omn_arr)
W7X_arr = np.zeros_like(omn_arr)
HSX_arr = np.zeros_like(omn_arr)


print('calcuting AE for D3D')
file = 'gist_D3D.txt'
gist_file = path+file
gist_file = mag_reader.mag_data(gist_file)
gist_file = boundary_and_refinement(gist_file)
ae_dat = ae.AE_gist(gist_file,normalize=normalize,AE_lengthscale=ae_length,lam_res=lam_res)
for idx, val in enumerate(omn_arr):
    print(idx)
    ae_dat.calc_AE_fast(omn=val,omt=0.0,omnigenous=True)
    D3D_arr[idx] = ae_dat.ae_tot


print('calcuting AE for W7XLM')
file = 'gist_W7XLM.txt'
gist_file = path+file
gist_file = mag_reader.mag_data(gist_file)
gist_file = boundary_and_refinement(gist_file)
ae_dat = ae.AE_gist(gist_file,normalize=normalize,AE_lengthscale=ae_length,lam_res=lam_res)
for idx, val in enumerate(omn_arr):
    print(idx)
    ae_dat.calc_AE_fast(omn=val,omt=0.0,omnigenous=False)
    W7X_arr[idx] = ae_dat.ae_tot


print('calcuting AE for HSX')
file = 'gist_HSX.txt'
gist_file = path+file
gist_file = mag_reader.mag_data(gist_file)
gist_file = boundary_and_refinement(gist_file)
ae_dat = ae.AE_gist(gist_file,normalize=normalize,AE_lengthscale=ae_length,lam_res=lam_res)
for idx, val in enumerate(omn_arr):
    print(idx)
    ae_dat.calc_AE_fast(omn=val,omt=0.0,omnigenous=False)
    HSX_arr[idx] = ae_dat.ae_tot



fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(5/6*6, 5/6*4.5))

ax.loglog(omn_arr,D3D_arr,label=r'tokamak')
ax.loglog(omn_arr,W7X_arr,label=r'stellarator')
ax.loglog(omn_arr,HSX_arr,label=r'HSX')

filter_first    = np.ones_like(omn_arr)
filter_first[0:2*int(len(filter_first)/3)]    = np.nan*filter_first[0:2*int(len(filter_first)/3)]
filter_last    = np.ones_like(omn_arr)
filter_last[int(len(filter_first)/3)::]    = np.nan*filter_last[int(len(filter_first)/3)::]

ax.loglog(omn_arr,filter_first*0.1*W7X_arr[-1]*(omn_arr/omn_arr[-1]),linestyle='dashed',color='black')
ax.loglog(omn_arr,filter_last*10*W7X_arr[0]*(omn_arr/omn_arr[0])**2,linestyle='dashdot',color='black')
ax.loglog(omn_arr,filter_last*0.1*D3D_arr[0]*(omn_arr/omn_arr[0])**3,linestyle='dotted',color='black')
ax.set_xlabel(r'$L_\mathrm{ref}/L_n$')
ax.set_ylabel(r'$\widehat{A}$')
ax.set_xlim([omn_arr.min(),omn_arr.max()])
ax.legend()
plt.savefig('gradient-scaling.png', format='png',
            #This is recommendation for publication plots
            dpi=1000)
plt.show()