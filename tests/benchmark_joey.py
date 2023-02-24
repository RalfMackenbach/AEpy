from AEpy import ae_routines as ae
import numpy as np
from AEpy import mag_reader
import matplotlib.pyplot as plt
import matplotlib        as mpl
plt.close('all')

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10}

mpl.rc('font', **font)



lam_res = 10000
omn_res = 100
plot=False


# Quick tests on D3D, checking scaling laws, difference between omnigenous and non-omnigenous,
# and make some plots

# First import gist data
file_name='gist_s7alpha5.txt'
data = mag_reader.mag_data("gist_files/"+file_name)
data.truncate_domain()
data.refine_profiles(10001)
data.plot_geometry()



# now do scaling law:
omn_range = np.logspace(-4,4,omn_res)
ae_vals   = np.zeros_like(omn_range)
AE_dat    = ae.AE_gist(data,quad=False,lam_res=lam_res)
for omn_idx, omn_val in np.ndenumerate(omn_range):
    AE_dat.calc_AE(omn=omn_range[omn_idx],omt=0.0,omnigenous=True,Delta_x=AE_dat.q0,Delta_y=AE_dat.q0)
    ae_vals[omn_idx]   = AE_dat.ae_tot * 4 * np.sqrt(np.pi)
    if plot==True:
        if omn_idx[0]%int(omn_res/5)==0:
            AE_dat.plot_AE_per_lam()


AE_dat.calc_AE(omn=1.0,omt=0.0,omnigenous=True,Delta_x=AE_dat.q0,Delta_y=AE_dat.q0)

print('AE @ omn=1.0 is',AE_dat.ae_tot  * 4 * np.sqrt(np.pi))

plt.loglog(omn_range,ae_vals)
plt.loglog(omn_range,(omn_range/omn_range[0])**3.0*ae_vals[0]*10,color='black',linestyle="dashed")
plt.loglog(omn_range,(omn_range/omn_range[-1])*ae_vals[-1]*10,color='black',linestyle="dashed")
plt.xlabel(r'omn')
plt.ylabel(r'$\widehat{A}$')
plt.savefig('density-dependence.png',dpi=1000)
plt.show()