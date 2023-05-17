from scipy.special import erf
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from BAD import bounce_int
import numpy as np
from qsc import Qsc
from AEpy import ae_routines as ae
from   matplotlib        import rc
import   matplotlib.pyplot  as plt

rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)



# make stellarator in QSC
nphi = int(1e3+1)
stel = Qsc.from_paper('precise QH', nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs


# set input variables
lam_res = 1001
omn = 1.0
omt = 1.0
omnigenous = False

# loop over r
stel.r = 0.001
omn_input = omn
omt_input = omt
stel.calculate()

alpha = 0.0
NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=alpha, N_turns=3, nphi=nphi,
                     lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
NAE_AE.calc_AE(omn=stel.spsi*omn_input,omt=stel.spsi*omt_input,omnigenous=omnigenous)
print("total AE is", NAE_AE.ae_tot)


NAE_AE.plot_precession(nae=True,stel=stel,alpha=-alpha)
NAE_AE.plot_AE_per_lam()