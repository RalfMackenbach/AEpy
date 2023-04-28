import  numpy   as      np
from    qsc     import  Qsc
from    AEpy    import  ae_routines     as      ae
from    simsopt.util.mpi                import  MpiPartition
from    simsopt.mhd.vmec                import  Vmec
from    simsopt.mhd.boozer              import  Boozer
from    simsopt.mhd.vmec_diagnostics    import  vmec_fieldlines, vmec_splines
import  matplotlib.pyplot               as      plt
from    matplotlib                      import  rc



# set up matplotlib
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

# set parameters
plot = True
omnigenous = True
lam_res = 10001
rho = 0.5
file_vmec = "./../../../balloon_ITG/wout_precise_QH.nc" # "./configs/wout_precise_QH.nc"
file_qsc  = 'precise QH'


# make stellarator in VMEC
mpi = MpiPartition()
mpi.write()
vmec = Vmec(file_vmec,mpi=mpi,verbose=True)
vmec.run()
wout = vmec.wout
a_minor = wout.Aminor_p
R_major = wout.Rmajor_p
B_ref = 2 * np.abs(wout.phi[-1] / (2 * np.pi)) / (wout.Aminor_p)**2
splines = vmec_splines(vmec)
iota_s = splines.iota(rho**2)

# make Booz
bs = Boozer(vmec)
bs.verbose = False
bs.bx.verbose = False
bs.register(vmec.s_full_grid)
bs.run()

# construct break point
# set rho break point (< break: pyQSC, > break: VMEC)
N_s=int(1/vmec.ds+1)
rho_break = 0.0#np.sqrt(1/N_s)

# make stellarator in QSC
nphi = int(1e3+1)
stel = Qsc.from_paper(file_qsc, nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs
stel.B0 = B_ref
stel.R0 = R_major



splines = vmec_splines(vmec)
iota_s = splines.iota(rho**2)

# do NAE first
stel.r = a_minor*rho

stel.calculate()
helicity = stel.iota - stel.iotaN

# NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=0.0, N_turns=1, nphi=nphi,
#             lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None',a_minor=a_minor)
# if plot:
#     NAE_AE.plot_precession(nae=True,stel=stel,filename='NAE_booz_phi.png',save=True)

# next try vmec using vmec fieldlines
# VMEC_AE = ae.AE_vmec(vmec,rho**2,n_turns=1,lam_res=lam_res)
# iota_s = splines.iota(rho**2)
# if plot:
#     VMEC_AE.plot_AE_per_lam()
#     VMEC_AE.plot_precession(nae=True,stel=stel,q=1/iota_s)


print(stel.iota,stel.iotaN)
print(iota_s,iota_s+4)

# next try vmec using booz fieldlines
BOOZ_AE = ae.AE_vmec(vmec,rho**2,booz=bs,n_turns=1,helicity=helicity,lam_res=lam_res)
if plot:
    BOOZ_AE.plot_precession(nae=True,stel=stel,filename='vmec_booz_phi.png',save=True)