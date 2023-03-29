from scipy.special import erf
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from BAD import bounce_int
import numpy as np
from qsc import Qsc
from AEpy import ae_routines as ae
import numpy as np

stel = Qsc.from_paper("precise QA")
stel.spsi = -1
stel.zs = -stel.zs
stel.calculate()

NAE_AE = ae.AE_pyQSC(stel_obj = stel)
NAE_AE.plot_geom()
NAE_AE.plot_precession()