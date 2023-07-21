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



def boundary_and_refinement(gist_file,refine=True,refined_grid=10001,boundary='None',plot=False,include_end=True):
    if include_end==True:
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
make_plots   = True
force_omnigenous = False
normalize   = 'ft-vol'
ae_length   = 'None'





path = "../../tests/gist_files/"





print('calcuting AE for D3D')
file = 'gist_D3D_vac_05'
gist_file = path+file
gist_file = mag_reader.mag_data(gist_file)
gist_file = boundary_and_refinement(gist_file,include_end=False)
ae_dat = ae.AE_gist(gist_file,normalize=normalize,AE_lengthscale=ae_length,lam_res=10*lam_res)
ae_dat.calc_AE(omn=3.0,omt=0.0,omnigenous=True)
ae_dat.plot_AE_per_lam(save=True,filename='AE_per_lam_D3D.png')



print('calcuting AE for HSX')
file = 'gist_HSX.txt'
gist_file = path+file
gist_file = mag_reader.mag_data(gist_file)
gist_file = boundary_and_refinement(gist_file)
ae_dat = ae.AE_gist(gist_file,normalize=normalize,AE_lengthscale=ae_length,lam_res=lam_res)
ae_dat.calc_AE(omn=3.0,omt=0.0,omnigenous=force_omnigenous)
ae_dat.plot_AE_per_lam(save=True,filename='AE_per_lam_HSX.png')


print('calcuting AE for W7X-SC')
file = 'gist_W7XSC.txt'
gist_file = path+file
gist_file = mag_reader.mag_data(gist_file)
gist_file = boundary_and_refinement(gist_file)
ae_dat = ae.AE_gist(gist_file,normalize=normalize,AE_lengthscale=ae_length,lam_res=lam_res)
ae_dat.calc_AE(omn=3.0,omt=0.0,omnigenous=force_omnigenous)
ae_dat.plot_AE_per_lam(save=True,filename='AE_per_lam_W7XSC.png')




