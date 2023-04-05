import  numpy       as      np
from    AEpy        import  ae_routines as ae
from    simsopt.mhd.vmec    import Vmec
from    qsc         import  Qsc
from    matplotlib  import  rc
from    AEpy        import  mag_reader
import  os

rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)



### NAE plots ###
# make base-case stellarator
nphi = int(2e3+1)
stel = Qsc.from_paper('precise QA', nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs
stel.r = 1e-1
stel.calculate()
omn = 1.0
omt = 0.0
NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=0.0, N_turns=3, nphi=nphi,
            lam_res=2000,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
NAE_AE.plot_geom()
NAE_AE.calc_AE(omn=stel.spsi*omn,omt=stel.spsi*omt,omnigenous=True)
NAE_AE.plot_precession()
NAE_AE.plot_AE_per_lam()



### GIST plots ###
# Do plots for HSX
file = '../tests/gist_files/gist_HSX.txt'
gist_file = mag_reader.mag_data(file)
gist_file.include_endpoint()
gist_file.refine_profiles(1001)
gist_file.plot_geometry()
ae_dat = ae.AE_gist(gist_file,normalize='ft_vol',AE_lengthscale='None',lam_res=1001)
ae_dat.calc_AE(omn=3.0,omt=0.0,omnigenous=False)
ae_dat.plot_precession()
ae_dat.plot_AE_per_lam(save=False)




### VMEC plots ###
# Do plots for NCSX
file = '../tests/vmec_files/input.ncsx'
vmec = Vmec(file)
ae_dat = ae.AE_vmec(vmec,s_val=1/4,n_turns=1)
ae_dat.calc_AE(omn=-3.0,omt=0.0,omnigenous=False)
ae_dat.plot_precession()
ae_dat.plot_AE_per_lam()

os.remove("input.ncsx_000_000000")
os.remove("parvmecinfo.txt")
os.remove("threed1.ncsx")
os.remove("wout_ncsx_000_000000.nc")