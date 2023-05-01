import  numpy                           as      np
from    AEpy    import  ae_routines     as      ae 

# construct AE functions and the logarithmic derivative
# using sympy

def AE_per_thermal_energy(vmec,omn,omt,rho,omnigenous=False):
    vmec.run()
    VMEC_AE = ae.AE_vmec(vmec,rho**2,n_turns=3)
    VMEC_AE.calc_AE(omn=-omn,omt=-omt,omnigenous=omnigenous)
    ae_tot = VMEC_AE.ae_tot
    


def device_AE(vmec,n,omn,T,omt):