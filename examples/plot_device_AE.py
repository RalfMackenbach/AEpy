import  os
import  numpy               as      np
import  matplotlib.pyplot   as      plt
from    simsopt.mhd.vmec    import  Vmec
from    matplotlib          import  rc
from    sympy               import  *
from    AEpy                import  ae_routines as  ae


# Define the profile functions
s = Symbol('s')
dens    =   (1.0-1.0*s)**1.0
temp    =   (1.0-1.0*s)**1.0
omn     =    dens.diff(s)/dens * 2 * s**(1/2)
omt     =    temp.diff(s)/temp * 2 * s**(1/2)
# convert to numpy functions
n_f     = lambdify(s,   dens,   'numpy')
T_f     = lambdify(s,   temp,   'numpy')
omn_f   = lambdify(s,   omn,    'numpy')
omt_f   = lambdify(s,   omt,    'numpy')



# Do plots for NCSX
file = "wout_precise_QA_000_000000.nc" #'../tests/vmec_files/input.precise_QA'
vmec = Vmec(file)



# Function to return AE at a given s
def AE_at_s(vmec,s_val,omn,omt,omnigenous=False,n_turns=1,plot=False):
    ae_dat = ae.AE_vmec(vmec,s_val=s_val,n_turns=n_turns,plot=plot,lam_res=1001,gridpoints=1001)
    ae_dat.calc_AE(omn,omt,omnigenous)
    return ae_dat.ae_tot



# Make arrays holding values at s
s_res  = 21
s_vals = np.linspace(0,1,s_res+2)
s_vals = s_vals[1:-1]
n_vals = n_f(s_vals)
T_vals = T_f(s_vals)
omn_vals = omn_f(s_vals)
omt_vals = omt_f(s_vals)
AE_vals = np.zeros(s_res)

# Calculate AE at each s
for i in range(s_res):
    print('Calculating AE at s = ',s_vals[i])
    AE_vals[i] = n_vals[i] * T_vals[i] * AE_at_s(vmec,s_vals[i],omn_vals[i],omt_vals[i],omnigenous=True,n_turns=5,plot=False)

# Plot AE vs s
plt.figure()
plt.plot(s_vals,AE_vals)
plt.xlabel('s')
plt.ylabel('AE')
plt.show()


os.remove("input.ncsx_000_000000")
os.remove("parvmecinfo.txt")
os.remove("threed1.ncsx")
os.remove("wout_ncsx_000_000000.nc")