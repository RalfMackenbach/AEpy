from scipy.special import erf
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from BAD import bounce_int
import numpy as np
from qsc import Qsc
from AEpy import ae_routines as ae
import numpy as np

stel = Qsc.from_paper("precise QA")
NAE_AE = ae.AE_pyQSC(stel_obj = stel, nphi=int(1e3+1),N_turns=3,r=1e-8,lam_res=int(1e3))
NAE_AE.plot_geom()
NAE_AE.plot_precession()