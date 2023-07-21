###############################################################################
# Figure 3: precession dependence on pressure and triangularity in QS examples
###############################################################################
from qsc import Qsc
import numpy as np
from scipy import integrate as integ
from    scipy           import  special
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib import rc

# Plotting feature
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 16})
rc('text', usetex=True)
rc('figure', **{'figsize': (9,8)})  

def add_default_args(kwargs_old, **kwargs_new):
    """
    Take any key-value arguments in ``kwargs_new`` and treat them as
    defaults, adding them to the dict ``kwargs_old`` only if
    they are not specified there.
    """
    for key in kwargs_new:
        if key not in kwargs_old:
            kwargs_old[key] = kwargs_new[key]

def load_local_stel(name, **kwargs):
    """
    Load the near-axis fields for designs not in the default pyQSC database
    """
    if name == "ariescs":
        add_default_args(
            kwargs,
            nfp=3,
            spsi=-1.,
            rc=[ 8.02919827e+00, 6.64951023e-01, 4.29853382e-02, 2.68991828e-03,
                -1.91516635e-04,-1.09052599e-04, 9.09665890e-06],
            zs=[-0.00000000e+00,-3.44851001e-01,-3.87201846e-02,-2.79011081e-03,
                -4.21521144e-04, 1.37678950e-04,-1.17894627e-06],
            etabar=-0.07408501,
            B2c=-0.06909306,
            order='r2',
    )
    elif name == "estell_24_scaled":
        add_default_args(
            kwargs,
            nfp=2,
            spsi=-1.,
            rc=[ 1.39432552e+00,-1.51070998e-01, 9.41773230e-03,-6.95382868e-04,
                7.16066286e-05,-6.26212560e-06],
            zs=[-0.00000000e+00, 1.40157846e-01,-8.93005420e-03, 5.42162265e-04,
                -1.33534345e-04, 1.79296985e-05],
            etabar=-0.56400278,
            B2c=0.10675459,
            order='r2',
    )
    elif name == "GAR":
        add_default_args(
            kwargs,
            nfp=2,
            spsi=-1.,
            rc=[2.09518646e+00,3.23649241e-01,3.56724544e-02,4.24595439e-03,
                3.95133235e-04,3.06945016e-05,4.31493571e-06],
            zs=[-0.00000000e+00,-2.76102418e-01,-3.31974768e-02,-4.30920148e-03,
                -4.27205590e-04,-3.30445272e-05,-3.88603376e-06],
            etabar=-0.34943169,
            B2c=-0.00904748,
            order='r2',
    )
    elif name == "HSX":
        add_default_args(
            kwargs,
            nfp=4,
            spsi=1.,
            rc=[ 1.22015235e+00, 2.06770090e-01, 1.82712358e-02, 2.01793457e-04,
                -5.40158003e-06],
            zs=[-0.00000000e+00,-1.66673918e-01,-1.65358508e-02,-4.95694105e-04,
                9.65041347e-05],
            etabar=-1.25991504,
            B2c=-0.36522788,
            order='r2',
    )
    elif name == "NCSX":
        add_default_args(
            kwargs,
            nfp=3,
            spsi=-1.,
            rc=[ 1.44865815e+00, 1.07575273e-01, 7.42455165e-03, 3.21897533e-04,
                -3.94687029e-05,-1.20561105e-05, 7.82032340e-06],
            zs=[-0.00000000e+00,-6.89115445e-02,-5.73158561e-03,-3.94028708e-04,
                -5.10622628e-05, 1.48769319e-05,-3.26281178e-06],
            etabar=-0.38289141,
            B2c=-0.49296162,
            order='r2',
    )
    elif name == "qhs48":
        add_default_args(
            kwargs,
            nfp=4,
            spsi=-1.,
            rc=[7.95415764e+00,1.12501469e+00,1.18931388e-01,1.21107462e-02,
                1.73588389e-03,4.39174548e-04,6.76696398e-05],
            zs=[0.00000000e+00,9.82797564e-01,1.25944557e-01,1.45159810e-02,
                1.82922388e-03,3.66124252e-04,5.88447058e-05],
            etabar=-0.14491763,
            B2c=0.00323481,
            order='r2',
    )
    else:
        raise ValueError('Unrecognized configuration name')

    return Qsc(**kwargs)

# Position of labels
def step_annotation(name):
    dx = 0.0
    dy = 0.0
    if name == "precise QH":
        dy = +1.5
        dx = -1.1
    elif name == "precise QH+well":
        dy = +1.0
        dx = -0.75
    elif name == "2022 QH nfp4 well":
        dy = +1.2
        dx = -0.0
    elif name == "HSX":
        dy = +0.8
        dx = -0.05
    elif name == "qhs48":
        dy = +1.3
        dx = -0.9
    elif name == "2022 QA":
        dy = +0.5
        dx = 1.2
    elif name == "2022 QH nfp7":
        dy = -1
        dx = 0.3
    elif name == "2022 QH nfp3 beta":
        dy = 0.9
        dx = 0.5
    elif name == "2022 QH nfp4 Mercier":
        dy = -1
        dx = 1.0
    elif name == "ariescs":
        dy = -0.0
        dx = 2
    elif name == "precise QA+well":
        dy = -0.3
        dx = 1.6
    elif name == "precise QA":
        dy = -0.3
        dx = 1.6
    elif name == "GAR":
        dy = -0.3
        dx = 1.6
    elif name == "2022 QH nfp4 long axis":
        dy = -0.4
        dx = -1.1
    elif name == "2022 QH nfp2":
        dy = -1.8
        dx = -1.38
    elif name == "2022 QH nfp3 vacuum":
        dy = -1.3
        dx = -1.2
    return dx, dy


save_fig = False
name_list =  ["precise QA",\
       "precise QA+well",\
       "precise QH",\
       "precise QH+well",\
       "2022 QA",\
       "2022 QH nfp2",\
       "2022 QH nfp3 vacuum",\
       "2022 QH nfp3 beta",\
       "2022 QH nfp4 long axis",\
       "2022 QH nfp4 well",\
       "2022 QH nfp4 Mercier",\
       "2022 QH nfp7",\
       "ariescs",\
       "GAR",\
       "HSX",\
       "qhs48"]

# Initialise arrays
sz_names = np.size(name_list)
true_features = np.zeros((sz_names,4))
est_features = np.zeros((sz_names,4))
ind = 0

# Define k grid
N_k = 1000
k_array = np.linspace(0,1.0,N_k)

# Initialise more arrays
G_delta_array  = np.zeros((sz_names,N_k))
G_p2_array  = np.zeros((sz_names,N_k))
triang_array  = np.zeros(sz_names)
texts = []

for jname, name in enumerate(name_list):
    try:
        stel = Qsc.from_paper(name, B0=1)
        stel = Qsc.from_paper(name, etabar=np.abs(stel.etabar), B0=1, nphi=1001)
    except:
        # Deal with configurations not included in pyQSC
        stel = load_local_stel(name)
        stel = load_local_stel(name, etabar=np.abs(stel.etabar), nphi=1001)
    # Compute elongation
    elongation_vert = (stel.curvature/stel.etabar)**2
    mid = int((stel.nphi+1)/2) # Point near the middle of phi
    print(name, elongation_vert[0], elongation_vert[mid])
    # Check which cross-section is most vertically elongated and choose it to represent the stellarator
    if elongation_vert[0] < elongation_vert[mid]:
        # If cross-section not at zero, rotate the configuration
        rc = stel.rc
        rc = [rc[i]*(-1)**i for i in range(len(rc))]
        stel.rc = rc
        zs = stel.zs
        zs = [zs[i]*(-1)**i for i in range(len(zs))]
        stel.zs = zs
        stel.calculate()
        print(name)

    orig = 0

    # Define scaled quantities
    dldp = stel.abs_G0_over_B0
    etabar = stel.etabar*dldp
    curvature = stel.curvature*dldp
    d_curvature = np.matmul(stel.d_d_varphi, curvature)
    d2_curvature = np.matmul(stel.d_d_varphi, d_curvature)
    torsion = -stel.torsion*dldp
    d_torsion = np.matmul(stel.d_d_varphi, torsion)
    d2_torsion = np.matmul(stel.d_d_varphi, d_torsion)
    I2 = stel.I2*dldp
    iota0 = stel.iotaN

    # Compute F_bar
    f = (I2+torsion)/curvature**2
    f_average = sum(stel.d_varphi_d_phi*f)
    g = 1+stel.sigma**2+etabar**4/curvature**4
    g_average = sum(stel.d_varphi_d_phi*g)
    F = 2*(f*g_average/f_average/g-1)
    a = etabar**4/curvature**4
    a0 = a[orig] # Evaluate at phi=0
    F0 = F[orig]

    # G2s calculation for k_hats
    G_20 = -2
    E_o_K = special.ellipe(k_array*k_array)/special.ellipk(k_array*k_array)
    G_2c = -4*(E_o_K*E_o_K - 2*k_array*k_array*E_o_K + (k_array*k_array-0.5))
    G_p2 = G_20 + (stel.etabar*stel.G0/stel.B0/stel.iotaN)**2*2*np.sqrt(a0)/((a0+3)-(a0+1)*F0)*(2*G_20 - G_2c)
    G_delta = (3*(1-a0)*G_20-(3-a0)*G_2c)/((3+a0)-(a0+1)*F0) + ((a0+1)*(3*G_20-G_2c)*F0)/((3+a0)-(a0+1)*F0)

    # Compute triangularity
    triang = -2*np.sign(stel.etabar)*(stel.X2c/stel.X1c-stel.Y2s/stel.Y1s)

    G_delta_array[jname,:] = G_delta
    G_p2_array[jname,:] = G_p2
    triang_array[jname] = triang[orig]

    ########
    # Plot
    ########
    plt.plot(G_delta, G_p2, 'k', linewidth=1.5, label = name)
    plt.scatter(G_delta[0], G_p2[0], facecolor='k')
    dx, dy = step_annotation(name) 
    # Attach label
    plt.annotate(name,(G_delta[0], G_p2[0]),fontsize=11,xytext = (G_delta[0]+dx, G_p2[0]+dy), horizontalalignment='left',arrowprops=dict(arrowstyle='-', lw=0.5))
    
# Plot details
plt.xlabel(r'$\mathcal{G}_\delta$')
plt.xscale('log')
plt.ylabel(r'$\mathcal{G}_{p_2}$')
plt.ylim([-6.5, 0.0])
plt.xlim([0.21, 10])
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)

if save_fig:
    plt.savefig('qs_examples_G2s.png', dpi=300)
plt.show()