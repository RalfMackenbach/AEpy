from scipy.special import erf
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from BAD import bounce_int
import numpy as np
from qsc import Qsc
from AEpy import ae_routines as ae
from   matplotlib        import rc
import matplotlib.pyplot as plt

rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

nphi = int(1e3+1)
stel = Qsc.from_paper("precise QH", nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs
stel.r = 1e-2
stel.calculate()

# NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=-0.0, N_turns=3, nphi=nphi,
#                  lam_res=1000, get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
# NAE_AE.plot_precession(nae = stel)
# NAE_AE.calc_AE(omn=stel.r*stel.spsi*1.0,omt=stel.r*stel.spsi*0.0,omnigenous=True)
# NAE_AE.plot_AE_per_lam()