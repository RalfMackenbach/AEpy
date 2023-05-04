from    device_AE import *
import  numpy as np
import  sympy as sp
from    simsopt.mhd.vmec import Vmec
from    os import listdir
from    os.path import isfile, join
import matplotlib.pyplot as plt
import multiprocessing as mp
import time


# I ran with fairly loose tolerances to increase speed
# For optimisation schemes, one would want to use tighter tolerances
# set parameters here
omnigenous = False
eta = (2/3)

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


# now construct list of vmec objects
path = './configs'

# retrieve all config files
all_files = sorted([f for f in listdir(path) if isfile(join(path, f))])

# sort files into device categories
nfp2_files = sorted([i for i in all_files if 'nfp2' in i])
nfp3_files = sorted([i for i in all_files if 'nfp3' in i])
PQA_well_files = sorted([i for i in all_files if 'PQA_well' in i])
QH_nowell_files = sorted([i for i in all_files if 'QH_nowell' in i])
W7X_files = sorted([i for i in all_files if 'W7-X' in i])

# set up arrays
nfp2_AE     = np.zeros(len(nfp2_files))
nfp2_beta   = np.zeros(len(nfp2_files))
nfp3_AE     = np.zeros(len(nfp3_files))
nfp3_beta   = np.zeros(len(nfp3_files))
PQA_well_AE = np.zeros(len(PQA_well_files))
PQA_well_beta = np.zeros(len(PQA_well_files))
QH_nowell_AE = np.zeros(len(QH_nowell_files))
QH_nowell_beta = np.zeros(len(QH_nowell_files))
W7X_AE      = np.zeros(len(W7X_files))
W7X_beta    = np.zeros(len(W7X_files))

# set up function to calculate AE and beta for a given device
def device_wrapper(path,name,symmetry,n_f=n_f,T_f=T_f,omn_f=omn_f,omt_f=omt_f,omnigenous=False):
    path_to_file = path + '/' + name
    vmec = Vmec(path_to_file)
    vmec.run()
    print(path_to_file)
    # if file contains 0.00 set plot to True
    if 'no_plotting_today' in name:
        plot = True 
    else:
        plot = False
    AE = device_AE(vmec,n_f,T_f,omn_f,omt_f,omnigenous=omnigenous,plot=plot,symmetry=symmetry)
    beta = vmec.wout.betatotal
    return [AE, beta]




# now loop over all devices and calculate AE 
# we do so parallelised over devices




if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the calculation


    # now, PQA_well
    print('Calculating AE for PQA_well')
    start_time = time.time()
    output_list = pool.starmap(device_wrapper, [(path,val,'QA') for val in PQA_well_files])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))
    # transfer data to arrays
    for i in range(len(PQA_well_files)):
        PQA_well_AE[i] = output_list[i][0]
        PQA_well_beta[i] = output_list[i][1]

    # now, QH_nowell
    print('Calculating AE for QH_nowell')
    start_time = time.time()
    output_list = pool.starmap(device_wrapper, [(path,val,'QH') for val in QH_nowell_files])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))
    # transfer data to arrays
    for i in range(len(QH_nowell_files)):
        QH_nowell_AE[i] = output_list[i][0]
        QH_nowell_beta[i] = output_list[i][1]

    # start with nfp=2
    print('Calculating AE for nfp2')
    start_time = time.time()
    output_list = pool.starmap(device_wrapper, [(path,val,'QI') for val in nfp2_files])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))
    # transfer data to arrays
    for i in range(len(nfp2_files)):
        nfp2_AE[i] = output_list[i][0]
        nfp2_beta[i] = output_list[i][1]

    
    # now, nfp=3
    print('Calculating AE for nfp3')
    start_time = time.time()
    output_list = pool.starmap(device_wrapper, [(path,val,'QI') for val in nfp3_files])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))
    # transfer data to arrays
    for i in range(len(nfp3_files)):
        nfp3_AE[i] = output_list[i][0]
        nfp3_beta[i] = output_list[i][1]

    # now, W7X
    print('Calculating AE for W7X')
    start_time = time.time()
    output_list = pool.starmap(device_wrapper, [(path,val,'QI') for val in W7X_files])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))
    # transfer data to arrays
    for i in range(len(W7X_files)):
        W7X_AE[i] = output_list[i][0]
        W7X_beta[i] = output_list[i][1]

    # close the pool
    pool.close()




    # now, plot AE vs beta
    fig, ax = plt.subplots()
    ax.semilogy(nfp2_beta,nfp2_AE,label='nfp=2')
    ax.semilogy(nfp3_beta,nfp3_AE,label='nfp=3')
    ax.semilogy(PQA_well_beta,PQA_well_AE,label='PQA_well')
    ax.semilogy(QH_nowell_beta,QH_nowell_AE,label='QH_nowell')
    ax.semilogy(W7X_beta,W7X_AE,label='W7X')
    ax.set_xlabel(r'$\beta_{total}$')
    ax.set_ylabel(r'$\widehat{A}$')
    ax.legend()
    fig.suptitle(r'$\eta = {{{}}}$'.format(round(eta,2)))
    plt.show()


    # save data
    np.savez('AE_vs_beta_eta{}.npz'.format(round(eta,2)),nfp2_beta=nfp2_beta,nfp2_AE=nfp2_AE,nfp3_beta=nfp3_beta,nfp3_AE=nfp3_AE,PQA_well_beta=PQA_well_beta,PQA_well_AE=PQA_well_AE,QH_nowell_beta=QH_nowell_beta,QH_nowell_AE=QH_nowell_AE,W7X_beta=W7X_beta,W7X_AE=W7X_AE)