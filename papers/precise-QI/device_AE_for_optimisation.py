import  numpy                           as      np
from    AEpy    import  ae_routines     as      ae 
from    scipy.interpolate               import  InterpolatedUnivariateSpline
from    scipy.integrate                 import  quad
from    simsopt.mhd                     import  vmec_splines
    

# this one is meant for optimisation purposes


def dAE_ds(vmec,n=1.0,T=1.0,omn=2.0,omt=2.0,s=0.5,omnigenous=False, plot=False,gridpoints=1001,n_turns=4,symmetry='QI'):
    epsrel_AE = 1e-2
    vmec.run()
    splines = vmec_splines(vmec)
    iota = splines.iota(s)
    surf = vmec.boundary
    nfp = surf.nfp
    if symmetry=='QI':
        turns_fac = np.abs(iota/nfp)
    elif symmetry=='QH':
        turns_fac = 1
    elif symmetry=='QA':
        turns_fac = 3/n_turns
    # set up AE object
    VMEC_AE = ae.AE_vmec(vmec,s,n_turns=n_turns*turns_fac,lam_res=101,gridpoints=gridpoints,plot=False,epsrel=epsrel_AE)
    # calculate AE per thermal energy
    VMEC_AE.calc_AE(omn=omn,omt=omt,omnigenous=omnigenous,fast=True)
    if plot:
        VMEC_AE = ae.AE_vmec(vmec,s,n_turns=n_turns*turns_fac,lam_res=1000,gridpoints=gridpoints,plot=False,epsrel=epsrel_AE)
        VMEC_AE.calc_AE(omn=omn,omt=omt,omnigenous=omnigenous)
        VMEC_AE.plot_AE_per_lam()
    # store AE
    ae_tot = VMEC_AE.ae_tot
    # calculate dV/ds
    s_half_grid_arr = vmec.s_half_grid
    dVds_arr = 4 * np.pi * np.pi * np.abs(vmec.wout.gmnc[0, 1:])
    dVds_f = InterpolatedUnivariateSpline(s_half_grid_arr, dVds_arr, ext='extrapolate')
    # calculate dAE/ds
    # importantly, we have rho*^2 scaling, impying an extra factor of T
    dAE_ds = dVds_f(s) * ae_tot * n * T**2
    return dAE_ds


def device_AE(vmec,n_f,T_f,omn_f,omt_f,s_res=10,omnigenous=False,plot=False,symmetry='QI'):
    # construct list of s values
    s_arr = np.linspace(0,1,s_res+2)
    s_arr = s_arr[1:-1]
    AE_arr = np.zeros_like(s_arr)
    # loop over s values
    for i in range(len(s_arr)):
        s = s_arr[i]
        AE_arr[i] = dAE_ds(vmec,n_f(s),T_f(s),omn_f(s),omt_f(s),s,omnigenous=omnigenous,plot=plot,symmetry=symmetry)
    # also find the total thermal energy
    def integrand_2(s):
        s_half_grid_arr = vmec.s_half_grid
        dVds_arr = 4 * np.pi * np.pi * np.abs(vmec.wout.gmnc[0, 1:])
        dVds_f = InterpolatedUnivariateSpline(s_half_grid_arr, dVds_arr, ext='extrapolate')
        return n_f(s)*T_f(s) * dVds_f(s)
    thermal_energy_arr = quad(integrand_2,0,1)
    thermal_energy = thermal_energy_arr[0]
    return AE_arr/thermal_energy
    
