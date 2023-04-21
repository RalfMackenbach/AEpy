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

nphi = 1001
stel = Qsc.from_paper("precise QH", nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs
stel.r = 1e-1
stel.calculate()
print(stel.B20_variation)

NAE_AE = ae.AE_pyQSC(stel_obj = stel, alpha=-0.0, N_turns=3, nphi=nphi,
                 lam_res=1000, get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
# NAE_AE.plot_geom()
NAE_AE.plot_precession(nae = stel)