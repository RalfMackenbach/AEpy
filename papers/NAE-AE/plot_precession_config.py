from scipy.special import erf
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from BAD import bounce_int
import numpy as np
from qsc import Qsc
from AEpy import ae_routines as ae
from   matplotlib        import rc
import   matplotlib.pyplot  as plt
from    simsopt.mhd.vmec                import  Vmec
from    simsopt.mhd.vmec_diagnostics    import  vmec_fieldlines
from    simsopt.util.mpi                import  MpiPartition

rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)



nphi = 1001
stel = Qsc.from_paper("precise QA+well", nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs
stel.r = 1e-1
stel.calculate()
print(stel.B20_variation)

alpha = 0.0


file = 'wout_precise_QA.nc'
mpi = MpiPartition()
mpi.write()
vmec = Vmec(file,mpi=mpi,verbose=True)
vmec.run()


rho = 0.2

VMEC_AE = ae.AE_vmec(vmec,rho**2,n_turns=1,lam_res=1001)

fieldline = vmec_fieldlines(vmec,rho**2,alpha,theta1d=np.linspace(-2,2,100),phi_center=0.0)
q = 1.0/fieldline.iota[0]



VMEC_AE = ae.AE_vmec(vmec,rho**2,n_turns=1,lam_res=1001)
VMEC_AE.plot_precession(nae=True,stel=stel,alpha=alpha,q=q)