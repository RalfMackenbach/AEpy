from    device_AE_for_optimisation import *
import  numpy as np
import  sympy as sp
from    simsopt.mhd.vmec import Vmec
import time



# construct input parameters
omnigenous = False
eta = 0.0

# construct density and temperature profiles
s_sym = sp.Symbol('s_sym')
dsdrho = 2 * s_sym**(1/2)
n_sym = (1 - s_sym)
T_sym = (1 - s_sym)**eta
dnds = n_sym.diff(s_sym)
dTds = T_sym.diff(s_sym)
omn_sym = dnds/n_sym * dsdrho
omt_sym = dTds/T_sym * dsdrho
# lambdify
n_f = sp.lambdify(s_sym,n_sym)
T_f = sp.lambdify(s_sym,T_sym)
omn_f = sp.lambdify(s_sym,omn_sym)
omt_f = sp.lambdify(s_sym,omt_sym)

# read in vmec file
vmec = Vmec('./configs/wout_nfp2_beta_0.00.nc',verbose=True)


start_time = time.time()
ans = device_AE(vmec,n_f,T_f,omn_f,omt_f,s_res=10,omnigenous=False,plot=False,symmetry='QI')
print("data generated in       --- %s seconds ---" % (time.time() - start_time))

_ = device_AE(vmec,n_f,T_f,omn_f,omt_f,s_res=1,omnigenous=False,plot=True,symmetry='QI')

print(ans)