from AEpy import ae_routines as ae
import numpy as np
from AEpy import mag_reader


lam_res = 1000
omn_res = 50

# Quick tests on D3D, checking scaling laws, difference between omnigenous and non-omnigenous,
# and make some plots

# First import gist data
file_name='gist_D3D.txt'
data = mag_reader.mag_data("gist_files/"+file_name)
data.include_endpoint()
data.refine_profiles(10001)
data.plot_geometry()

# Calculate AE and plot, find difference between omnigenous and non omnigenous
AE_dat = ae.AE_gist(data,quad=False,lam_res=lam_res)
AE_dat.plot_precession()
AE_dat.calc_AE(omn=1.0,omt=1.0,omnigenous=True,Delta_x=AE_dat.q0,Delta_y=AE_dat.q0)
AE_dat.plot_AE_per_lam()
omnigenous      = AE_dat.ae_tot
AE_dat.calc_AE(omn=1.0,omt=1.0,omnigenous=False ,Delta_x=AE_dat.q0,Delta_y=AE_dat.q0)
not_omnigenous  = AE_dat.ae_tot
# find difference
print('DIII-D: 1 - omnigenous/nonomnigenous = ', 1 - omnigenous/not_omnigenous)


# Calculate AE and find difference for slow and fast method
AE_dat = ae.AE_gist(data,quad=False,lam_res=lam_res)
AE_dat.plot_precession()
AE_dat.calc_AE(omn=1.0,omt=1.0,omnigenous=False,Delta_x=AE_dat.q0,Delta_y=AE_dat.q0)
slow      = AE_dat.ae_tot
AE_dat.calc_AE_fast(omn=1.0,omt=1.0,omnigenous=False ,Delta_x=AE_dat.q0,Delta_y=AE_dat.q0)
fast  = AE_dat.ae_tot
# find difference between slow and fast method
print('(slow-fast)/fast = ', (slow-fast)/fast)



# now do scaling law:
omn_range = np.logspace(-3,3,omn_res)
ae_vals   = np.zeros_like(omn_range)
for omn_idx, omn_val in np.ndenumerate(omn_range):
    AE_dat.calc_AE(omn=omn_range[omn_idx],omt=0.0,omnigenous=True,Delta_x=AE_dat.q0,Delta_y=AE_dat.q0)
    ae_vals[omn_idx]   = AE_dat.ae_tot


# now do scaling for stellarator
# First import gist data
file_name='gist_W7XHM.txt'
data = mag_reader.mag_data("gist_files/"+file_name)
data.include_endpoint()
data.refine_profiles(10001)
data.plot_geometry()
AE_dat = ae.AE_gist(data,quad=False,lam_res=lam_res)
ae_vals_stell   = np.zeros_like(omn_range)
for omn_idx, omn_val in np.ndenumerate(omn_range):
    print('currently at',int(omn_idx[0]/len(omn_range)*100),'% for stellarator')
    AE_dat.calc_AE_fast(omn=omn_range[omn_idx],omt=0.0,omnigenous=False,Delta_x=AE_dat.q0,Delta_y=AE_dat.q0)
    ae_vals_stell[omn_idx]   = AE_dat.ae_tot
# make plot for stellarator
AE_dat.calc_AE(omn=3.0,omt=0.0,omnigenous=False,Delta_x=AE_dat.q0,Delta_y=AE_dat.q0)
AE_dat.plot_AE_per_lam()

import matplotlib.pyplot as plt
plt.loglog(omn_range,ae_vals,label=r'$\widehat{A}_\mathrm{DIII-D}$')
plt.loglog(omn_range,ae_vals_stell,label=r'$\widehat{A}_\mathrm{W7XHM}$')
first_nan = np.ones_like(omn_range)
second_nan= np.ones_like(omn_range)
second_nan[0:int(omn_res/2)] = np.nan
first_nan[int(omn_res/2)-1::]= np.nan
plt.loglog(omn_range,(omn_range/omn_range[0])**3*ae_vals[0]/10*first_nan,color='black',linestyle='dashed',label=r'$\mathrm{omn}^3$')
plt.loglog(omn_range,(omn_range/omn_range[0])**2*ae_vals_stell[0]*10*first_nan,color='black',linestyle='dashdot',label=r'$\mathrm{omn}^2$')
plt.loglog(omn_range,(omn_range/omn_range[-1])**1*ae_vals[-1]*10*second_nan,color='black',linestyle='dotted',label=r'$\mathrm{omn}$')
plt.xlim(omn_range.min(),omn_range.max())
plt.legend()
plt.show()