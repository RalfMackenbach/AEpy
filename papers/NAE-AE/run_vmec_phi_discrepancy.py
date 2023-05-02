import  numpy   as      np
from    qsc     import  Qsc
from    AEpy    import  ae_routines     as      ae
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
lam_res = 1001
rho = 0.1
file_vmec = "./../../../Python/Trapped_electron/configs/wout_precise_QH.nc"  # "./../../../balloon_ITG/wout_precise_QH.nc" # "./configs/wout_precise_QH.nc"
file_qsc  = 'precise QH'
helicity = -4


# make stellarator in VMEC
vmec = Vmec(file_vmec,verbose=True)
vmec.run()
wout = vmec.wout

vs = vmec_splines(vmec)

iota_s = vs.iota(rho*rho)
iotaN_s = iota_s - helicity

N_theta = 100
theta = np.linspace(-1,1,N_theta)*np.pi*iota_s/iotaN_s
print(theta)

# # make Booz
# bs = Boozer(vmec)
# bs.verbose = False
# bs.bx.verbose = False
# bs.register(vmec.s_full_grid)
# bs.run()

# construct break point
# set rho break point (< break: pyQSC, > break: VMEC)
# N_s=int(1/vmec.ds+1)
# rho_break = 0.0#np.sqrt(1/N_s)

# # make stellarator in QSC
# nphi = int(1e3+1)
# stel = Qsc.from_paper(file_qsc, nphi = nphi)
# stel.spsi = -1
# stel.zs = -stel.zs
# stel.B0 = B_ref
# stel.R0 = R_major


# do NAE first
# stel.r = a_minor*rho

# stel.calculate()
# helicity = stel.iota - stel.iotaN

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

# print(helicity)
# print(stel.iota,stel.iotaN)
# print(iota_s,iota_s+4)

# next try vmec using booz fieldlines
BOOZ_AE = ae.AE_vmec(vmec,rho**2,booz=[],n_turns=1,helicity=helicity,lam_res=lam_res)
if plot:
    BOOZ_AE.plot_precession(nae=False,stel=[],filename='vmec_booz_phi.png',save=True)