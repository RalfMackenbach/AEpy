from scipy.special import erf
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from BAD import bounce_int
import numpy as np
from qsc import Qsc
from AEpy import ae_routines as ae
import numpy as np

stel = Qsc.from_paper("precise QH+well")
NAE_AE = ae.AE_pyQSC(stel_obj = stel, nphi=int(1e3+1),N_turns=3,r=1e-4,lam_res=int(1e3))
# NAE_AE.plot_geom()
# NAE_AE.plot_precession()
NAE_AE.calc_AE(omn=10.0,omt=0.0,omnigenous=True)
print(NAE_AE.ae_tot)
NAE_AE.plot_AE_per_lam()