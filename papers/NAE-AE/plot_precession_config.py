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



# make stellarator in VMEC
file = "./configs/wout_precise_QH.nc"
file_qsc = 'precise QH'
mpi = MpiPartition()
mpi.write()
vmec = Vmec(file,mpi=mpi,verbose=True)
vmec.run()
wout = vmec.wout
a_minor = wout.Aminor_p
R_major = wout.Rmajor_p
B_ref = 2 * np.abs(wout.phi[-1] / (2 * np.pi)) / (wout.Aminor_p)**2
print(B_ref,R_major,a_minor,a_minor/R_major)


# construct break point
# set rho break point (< break: pyQSC, > break: VMEC)
N_s=int(1/vmec.ds+1)
rho_break = np.sqrt(1/N_s)

# make stellarator in QSC
nphi = int(1e3+1)
stel = Qsc.from_paper(file_qsc, nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs
stel.B0 = B_ref
stel.R0 = R_major


lam_res = 1001

rho = 0.1
omn = 1.0
omt = 1.0
omnigenous = True

# loop over r
stel.r = a_minor*rho
omn_input = omn
omt_input = omt
stel.calculate()
NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=0.0, N_turns=3.0, nphi=nphi,
                     lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None',a_minor=a_minor)
NAE_AE.calc_AE(omn=stel.spsi*omn_input,omt=stel.spsi*omt_input,omnigenous=omnigenous)

NAE_AE.plot_precession(nae=True,stel=stel)
VMEC_AE = ae.AE_vmec(vmec,rho**2,n_turns=np.abs(stel.iota/stel.iotaN),booz=True,lam_res=lam_res,gridpoints=1001,plot=False)
VMEC_AE.calc_AE(omn=-omn_input,omt=-omt_input,omnigenous=omnigenous)
VMEC_AE.plot_AE_per_lam()

fl =vmec_fieldlines(vmec,rho**2,0.0,theta1d=np.linspace(-1,1,1000))

q = 1/fl.iota[0]

VMEC_AE.plot_precession(nae=True,stel=stel,q=q)
