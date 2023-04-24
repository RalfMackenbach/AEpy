import scipy
import numpy as np
from    qsc     import  Qsc
from    AEpy    import  ae_routines as      ae
import matplotlib.pyplot as plt

from   matplotlib        import rc



rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

# inputs
filename= 'precise QA'
omn     = 0.05
omt     = 0.0
r       = 1e-2
omnigenous = True


nphi = int(1e3+1)
stel = Qsc.from_paper(filename, nphi = nphi)
stel.spsi = -1
stel.r = r
stel.zs = -stel.zs

NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=0.0, N_turns=1, nphi=nphi,
            lam_res=10001,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
NAE_AE.calc_AE(omn=omn,omt=omt,omnigenous=omnigenous)
NAE_AE.plot_AE_per_lam(save=True,filename='AE_per_lam.png',scale=0.8)