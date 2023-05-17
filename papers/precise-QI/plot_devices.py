import numpy as np
import matplotlib.pyplot as plt

filenames = ['AE_vs_beta_eta0.0.npz','AE_vs_beta_eta0.67.npz','AE_vs_beta_eta1000.npz']


# make figure
fig, ax = plt.subplots(1,3,figsize=(15,5),sharey=True,sharex=True,tight_layout=True)

# loop over filenames
for idx,filename in enumerate(filenames):

    print(idx)

    # read data
    data = np.load(filename)

    if filename=='AE_vs_beta_eta0.0.npz':
        suptitle = r'$n = (1 -s)$ and $T = 1$'
    if filename=='AE_vs_beta_eta0.67.npz':
        suptitle = r'$n = (1 -s)$ and $T = (1 - s)^{2/3}$'
    if filename=='AE_vs_beta_eta1000.npz':
        suptitle = r'$n = 1$ and $T = (1 - s)$'

    # load nfp2
    AE_nfp2 = data["nfp2_AE.npy"]
    beta_nfp2 = data["nfp2_beta.npy"]

    # load nfp3
    AE_nfp3 = data["nfp3_AE.npy"]
    beta_nfp3 = data["nfp3_beta.npy"]

    # load W7X
    AE_W7X = data["W7X_AE.npy"]
    beta_W7X = data["W7X_beta.npy"]

    # load QH_nowell
    AE_QH_nowell = data["QH_nowell_AE.npy"]
    beta_QH_nowell = data["QH_nowell_beta.npy"]

    # load PQA_well
    AE_PQA_well = data["PQA_well_AE.npy"]
    beta_PQA_well = data["PQA_well_beta.npy"]

    # make plot
    ax[idx].semilogy(100*beta_QH_nowell, AE_QH_nowell, label='QH_nowell',linestyle=':')
    ax[idx].semilogy(100*beta_nfp3, AE_nfp3, label='nfp=3',linestyle='-')
    ax[idx].semilogy(100*beta_nfp2, AE_nfp2, label='nfp=2',linestyle='--')
    ax[idx].semilogy(100*beta_W7X, AE_W7X, label='W7X',linestyle='-',linewidth=3)
    ax[idx].semilogy(100*beta_PQA_well, AE_PQA_well, label='PQA_well',linestyle='-.')
    ax[idx].set_xlim([0,4])
    ax[idx].grid()
    ax[idx].set_xlabel(r'$\beta$ [%]')
    if idx==0:
        ax[idx].set_ylabel(r'AE/$E_t$')
    if idx==len(filenames)-1:
        ax[idx].legend(loc='lower right')
    ax[idx].set_title(suptitle)


plt.savefig('AE_vs_beta.png',dpi=1000)
plt.show()