from scipy.special import erf
from scipy.integrate import quad, quad_vec, simpson, dblquad
from scipy.signal   import argrelextrema
from scipy.interpolate import interp1d
from BAD import bounce_int
import numpy as np
from qsc import Qsc
from termcolor import colored

######################################################
################## functions for AE ##################
######################################################


# diamagnetic drift
def w_diamag(dlnndx,dlnTdx,z):
    return ( dlnndx/z + dlnTdx * ( 1.0 - 3.0 / (2.0 * z) ) )

# Integrand of AE
def AE_per_lam_per_z(walpha,wpsi,wdia,tau_b,z):
    r"""
    The available energy per lambda per z.
    """
    geometry = ( wdia - walpha ) * walpha - wpsi**2 + np.sqrt( ( wdia - walpha )**2 + wpsi**2  ) * np.sqrt( walpha**2 + wpsi**2 )
    envelope = np.exp(-z) * np.power(z,5/2)
    jacobian = tau_b
    val      = geometry * envelope * jacobian
    return val/(4*np.sqrt(np.pi))

# Integrand of AE with z-dependence integrated out 
# only possible for omnigenous devices.
def AE_per_lam(c0,c1,tau_b,walpha):
    r"""
    function containing the integral over z for exactly omnigenous systems.
    This is the available energy per lambda.
    """
    condition1 = np.logical_and((c0>=0),(c1<=0))
    condition2 = np.logical_and((c0>=0),(c1>0))
    condition3 = np.logical_and((c0<0),(c1<0))
    ans = np.zeros(len(c1))
    ans[condition1]  = (2 * c0[condition1] - 5 * c1[condition1])
    ans[condition2]  = (2 * c0[condition2] - 5 * c1[condition2]) * erf(np.sqrt(c0[condition2]/c1[condition2])) + 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0[condition2] + 15 * c1[condition2] ) * np.sqrt(c0[condition2]/c1[condition2]) * np.exp( - c0[condition2]/c1[condition2] )
    ans[condition3]  = (2 * c0[condition3] - 5 * c1[condition3]) * (1 - erf(np.sqrt(c0[condition3]/c1[condition3]))) - 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0[condition3] + 15 * c1[condition3] ) * np.sqrt(c0[condition3]/c1[condition3]) * np.exp( - c0[condition3]/c1[condition3] )
    return 3/16*ans*tau_b*walpha**2


######################################################
######################################################
######################################################


















######################################################
################# functions for BAD ##################
######################################################


def all_drifts(f,h,h1,h2,x,is_func=False,sinhtanh=False):
    r"""
    ``all_drifts`` does the bounce integral
    and wraps the root finding routine into one function.
    Does the bounce int for three functions
    I0 = ∫h(x)/sqrt(f(x))  dx
    I1 = ∫h1(x)/sqrt(f(x)) dx
    I2 = ∫h2(x)/sqrt(f(x)) dx,
    and returns all of these values.
    Can be done by either quad if is_func=True, or
    gtrapz if is_func=False. When is_func=True
    both f and h need to be functions. Otherwise
    they should be arrays. Also returns roots
     Args:
        f: function or array containing f
        h: function or array containing h
        hy:function or array containing hy
        hx:function of array containing hx
        is_func: are h, hy, hx, and f functions or not.
        return_roots: returns roots or not
        sinhtanh: use sinhtanh quadrature
    """
    # if f is not a function use gtrapz
    if is_func==False:
        # if false use array for root finding
        index,root = bounce_int._find_zeros(f,x,is_func=False)
        # check if first well is edge, if so roll
        first_well = bounce_int._check_first_well(f,x,index,is_func=False)
        if first_well==False:
            index = np.roll(index,1)
            root = np.roll(root,1)
        # do bounce integral
        I0 = bounce_int._bounce_integral(f,h, x,index,root,is_func=False,sinhtanh=False)
        I1 = bounce_int._bounce_integral(f,h1,x,index,root,is_func=False,sinhtanh=False)
        I2 = bounce_int._bounce_integral(f,h2,x,index,root,is_func=False,sinhtanh=False)
    # if is_func is true, use function for both root finding and integration
    if is_func==True:
        index,root = bounce_int._find_zeros(f,x,is_func=True)
        first_well = bounce_int._check_first_well(f,x,index,is_func=True)
        if first_well==False:
            index = np.roll(index,1)
            root = np.roll(root,1)
        I0 = bounce_int._bounce_integral(f,h, x,index,root,is_func=True,sinhtanh=sinhtanh)
        I1 = bounce_int._bounce_integral(f,h1,x,index,root,is_func=True,sinhtanh=sinhtanh)
        I2 = bounce_int._bounce_integral(f,h2,x,index,root,is_func=True,sinhtanh=sinhtanh)
    return [I0,I1,I2], root




def drift_from_gist(theta,modb,sqrtg,L1,L2,my_dpdx,lam_res,quad=False,interp_kind='cubic'):
    r"""
    Calculate the drift given GIST input arrays.

    """
    # make dldzeta
    dldz    = sqrtg*modb
    # make lam arr
    lam_arr = np.linspace(1/modb.max(),1/modb.min(),lam_res+1,endpoint=False)
    lam_arr = np.delete(lam_arr,  0)

    # routine if quad is true:
    # make interpolated functions
    if quad==True:
        L1_f    = interp1d(theta,L1,kind=interp_kind)
        L2_f    = interp1d(theta,L2,kind=interp_kind)
        dldz_f  = interp1d(theta,dldz,kind=interp_kind)
        modb_f  = interp1d(theta,modb,kind=interp_kind)
        # we're going to make a much finer resolved
        # theta_arr to evaluate the function on
        theta   = np.linspace(theta.min(),theta.max(),1000)

        # loop over lambda
        # and save results in list of lists
        wpsi_list   = []
        walpha_list = []
        tau_b_list  = []
        roots_list  = []
        lam_list    = []
        # start the loop
        for lam_idx, lam_val in enumerate(lam_arr):
            # construct interpolated drift for lambda vals
            f  = lambda x: 1.0 - lam_val * modb_f(x)
            h  = lambda x: dldz_f(x)
            hx = lambda x: ( lam_val + 2 * (1/modb_f(x) - lam_val) ) * L1_f(x) * dldz_f(x)
            hy = lambda x: ( ( lam_val + 2 * (1/modb_f(x) - lam_val) ) * L2_f(x) - my_dpdx * (1 - lam_val * modb_f(x))/modb_f(x)**2 ) * dldz_f(x)
            list, roots = all_drifts(f,h,hx,hy,theta,is_func=True,sinhtanh=False)
            roots       = np.asarray(roots)
            tau_b       = np.asarray(list[0])
            delta_psi   = np.asarray(list[1])
            delta_alpha = np.asarray(list[2])
            walpha      = delta_alpha/tau_b
            wpsi        = delta_psi/tau_b
            wpsi_list.append(wpsi)
            walpha_list.append(walpha)
            tau_b_list.append(tau_b)
            roots_list.append(roots)
            lam_list.append([lam_val])
            print(lam_idx)

    # routine if quad is False:
    if quad==False:
        # loop over lambda
        # and save results in list of lists
        wpsi_list   = []
        walpha_list = []
        tau_b_list  = []
        roots_list  = []
        lam_list    = []
        # start the loop
        for lam_idx, lam_val in enumerate(lam_arr):
            # construct interpolated drift for lambda vals
            f  = 1 - lam_val * modb
            h  = dldz
            hx =( lam_val + 2 * (1/modb - lam_val) ) * L1 * dldz
            hy =( ( lam_val + 2 * (1/modb - lam_val) ) * L2 - my_dpdx * (1 - lam_val * modb)/modb**2 ) * dldz
            list, roots = all_drifts(f,h,hx,hy,theta,is_func=False,sinhtanh=False)
            roots       = np.asarray(roots)
            tau_b       = np.asarray(list[0])
            delta_psi   = np.asarray(list[1])
            delta_alpha = np.asarray(list[2])
            walpha      = delta_alpha/tau_b
            wpsi        = delta_psi/tau_b
            wpsi_list.append(wpsi)
            walpha_list.append(walpha)
            tau_b_list.append(tau_b)
            roots_list.append(roots)
            lam_list.append([lam_val])
        k2         = (1 - lam_arr*np.amin(modb))*np.amax(modb)/(np.amax(modb)-np.amin(modb))

    return roots_list,wpsi_list,walpha_list,tau_b_list,lam_list,k2




def drift_from_pyQSC(theta,modb,dldz,L1,L2,my_dpdx,lam_res,quad=False,interp_kind='cubic',direct=False):
    r"""
    Calculate the drift given pyQSC input arrays.

    """
    if direct==False:
        # make lam arr
        lam_arr = np.linspace(1/modb.max(),1/modb.min(),lam_res+1,endpoint=False)
        lam_arr = np.delete(lam_arr,  0)
    if direct==True:
        lam_arr = np.asarray([lam_res])

    # routine if quad is true:
    # make interpolated functions
    if quad==True:
        L1_f    = interp1d(theta,L1,kind=interp_kind)
        L2_f    = interp1d(theta,L2,kind=interp_kind)
        dldz_f  = interp1d(theta,dldz,kind=interp_kind)
        modb_f  = interp1d(theta,modb,kind=interp_kind)
        # we're going to make a much finer resolved
        # theta_arr to evaluate the function on
        theta   = np.linspace(theta.min(),theta.max(),1000)

        # loop over lambda
        # and save results in list of lists
        wpsi_list   = []
        walpha_list = []
        tau_b_list  = []
        roots_list  = []
        lam_list    = []
        # start the loop
        for lam_idx, lam_val in enumerate(lam_arr):
            # construct interpolated drift for lambda vals
            f  = lambda x: 1.0 - lam_val * modb_f(x)
            h  = lambda x: dldz_f(x)
            hx = lambda x: ( lam_val + 2 * (1/modb_f(x) - lam_val) ) * L1_f(x) * dldz_f(x)
            hy = lambda x: ( ( lam_val + 2 * (1/modb_f(x) - lam_val) ) * L2_f(x) - my_dpdx * (1 - lam_val * modb_f(x))/modb_f(x)**2 ) * dldz_f(x)
            list, roots = all_drifts(f,h,hx,hy,theta,is_func=True,sinhtanh=False)
            roots       = np.asarray(roots)
            tau_b       = np.asarray(list[0])
            delta_psi   = np.asarray(list[1])
            delta_alpha = np.asarray(list[2])
            walpha      = delta_alpha/tau_b
            wpsi        = delta_psi/tau_b
            wpsi_list.append(wpsi)
            walpha_list.append(walpha)
            tau_b_list.append(tau_b)
            roots_list.append(roots)
            lam_list.append([lam_val])
            print(lam_idx)

    # routine if quad is False:
    if quad==False:
        # loop over lambda
        # and save results in list of lists
        wpsi_list   = []
        walpha_list = []
        tau_b_list  = []
        roots_list  = []
        lam_list    = []
        # start the loop
        for lam_idx, lam_val in enumerate(lam_arr):
            # construct interpolated drift for lambda vals
            f  = 1 - lam_val * modb
            h  = dldz
            hx =( lam_val + 2 * (1/modb - lam_val) ) * L1 * dldz
            hy =( ( lam_val + 2 * (1/modb - lam_val) ) * L2 - my_dpdx * (1 - lam_val * modb)/modb**2 ) * dldz
            list, roots = all_drifts(f,h,hx,hy,theta,is_func=False,sinhtanh=False)
            roots       = np.asarray(roots)
            tau_b       = np.asarray(list[0])
            delta_psi   = np.asarray(list[1])
            delta_alpha = np.asarray(list[2])
            walpha      = delta_alpha/tau_b
            wpsi        = delta_psi/tau_b
            wpsi_list.append(wpsi)
            walpha_list.append(walpha)
            tau_b_list.append(tau_b)
            roots_list.append(roots)
            lam_list.append([lam_val])
        k2         = (1 - lam_arr*np.amin(modb))*np.amax(modb)/(np.amax(modb)-np.amin(modb))

    return roots_list,wpsi_list,walpha_list,tau_b_list,lam_list,k2





def drift_from_vmec(theta,modb,dldz,L1,L2,K1,K2,lam_res,quad=False,interp_kind='cubic',direct=False):
    r"""
    Calculate the drift given vmec input arrays.

    """
    if direct==False:
        # make lam arr
        lam_arr = np.linspace(1/modb.max(),1/modb.min(),lam_res+2,endpoint=True)[1:-1]

    if direct==True:
        lam_arr = np.asarray([lam_res])
    # routine if quad is true:
    # make interpolated functions
    if quad==True:
        L1_f    = interp1d(theta,L1,kind=interp_kind)
        L2_f    = interp1d(theta,L2,kind=interp_kind)
        K1_f    = interp1d(theta,K1,kind=interp_kind)
        K2_f    = interp1d(theta,K2,kind=interp_kind)
        dldz_f  = interp1d(theta,dldz,kind=interp_kind)
        modb_f  = interp1d(theta,modb,kind=interp_kind)
        # we're going to make a much finer resolved
        # theta_arr to evaluate the function on
        theta   = np.linspace(theta.min(),theta.max(),1000)

        # loop over lambda
        # and save results in list of lists
        wpsi_list   = []
        walpha_list = []
        tau_b_list  = []
        roots_list  = []
        lam_list    = []
        # start the loop
        for lam_idx, lam_val in enumerate(lam_arr):
            # construct interpolated drift for lambda vals
            f  = lambda x: 1.0 - lam_val * modb_f(x)
            h  = lambda x: dldz_f(x)
            hx = lambda x: ( lam_val * L1_f(x) + 2 * (1/modb_f(x) - lam_val) * K1_f(x) ) * dldz_f(x)
            hy = lambda x: ( lam_val * L2_f(x) + 2 * (1/modb_f(x) - lam_val) * K2_f(x) ) * dldz_f(x)
            list, roots = all_drifts(f,h,hx,hy,theta,is_func=True,sinhtanh=False)
            roots       = np.asarray(roots)
            tau_b       = np.asarray(list[0])
            delta_psi   = np.asarray(list[1])
            delta_alpha = np.asarray(list[2])
            walpha      = delta_alpha/tau_b
            wpsi        = delta_psi/tau_b
            wpsi_list.append(wpsi)
            walpha_list.append(walpha)
            tau_b_list.append(tau_b)
            roots_list.append(roots)
            lam_list.append([lam_val])
            print(lam_idx)

    # routine if quad is False:
    if quad==False:
        # loop over lambda
        # and save results in list of lists
        wpsi_list   = []
        walpha_list = []
        tau_b_list  = []
        roots_list  = []
        lam_list    = []
        # start the loop
        for lam_idx, lam_val in enumerate(lam_arr):
            # construct drift arrays
            f  = 1 - lam_val * modb
            h  = dldz
            hx =( lam_val * L1 + 2 * (1/modb - lam_val) * K1 ) * dldz
            hy =( lam_val * L2 + 2 * (1/modb - lam_val) * K2 ) * dldz
            list, roots = all_drifts(f,h,hx,hy,theta,is_func=False,sinhtanh=False)
            roots       = np.asarray(roots)
            tau_b       = np.asarray(list[0])
            delta_psi   = np.asarray(list[1])
            delta_alpha = np.asarray(list[2])
            walpha      = delta_alpha/tau_b
            wpsi        = delta_psi/tau_b
            wpsi_list.append(wpsi)
            walpha_list.append(walpha)
            tau_b_list.append(tau_b)
            roots_list.append(roots)
            lam_list.append([lam_val])
        k2         = (1 - lam_arr*np.amin(modb))*np.amax(modb)/(np.amax(modb)-np.amin(modb))

    return roots_list,wpsi_list,walpha_list,tau_b_list,lam_list,k2





def drift_asymptotic(stel,a_minor,k2):
    from scipy import  special
    # NAE drifts, leading order and first order correction
    # normalized as true drift * q * a * B0 * r / H
    # 1/B0 dependence due to choice of Bref
    E_k_K_k =  special.ellipe(k2)/special.ellipk(k2)
    wa0 = a_minor * -stel.etabar / stel.B0 *(2*E_k_K_k-1)
    wa1 = a_minor * -stel.r*stel.etabar / stel.B0 *(2/stel.etabar*stel.B20_mean+stel.etabar*(4*E_k_K_k*E_k_K_k - 2*(3-2*k2)*E_k_K_k + (1-2*k2)) +\
                                                               2/stel.etabar*stel.B2c*(2*E_k_K_k*E_k_K_k - 4*k2*E_k_K_k + (2*k2-1)))
    return wa0, wa1


######################################################
######################################################
######################################################





























######################################################
################# functions for geo ##################
######################################################

def vmec_geo(vmec,s_val,alpha=0.0,phi_center=0.0,gridpoints=1001,n_turns=1,helicity=0,plot=False,QS_mapping=False):
    import numpy as np
    from simsopt.mhd.vmec_diagnostics import vmec_fieldlines, vmec_splines

    vmec_s = vmec_splines(vmec)
    iota_s = vmec_s.iota(s_val)
    iotaN_s = iota_s-helicity

    if QS_mapping==True:
        theta_arr = (np.linspace(-n_turns,n_turns,gridpoints)*np.pi-helicity*alpha/iota_s)*np.abs(iota_s/iotaN_s)
    if QS_mapping==False:
        theta_arr = np.linspace(-n_turns,n_turns,gridpoints)*np.pi
        
    fieldline = vmec_fieldlines(vmec_s,s_val,alpha=alpha,theta1d=theta_arr,phi_center=phi_center)

    if plot==True:
        plot_surface_and_fl(vmec,fieldline,s_val,transparant=False,trans_val=0.9,title='')

    modB            = (fieldline.modB).flatten()
    Bref            = (fieldline.B_reference)
    Lref            = (fieldline.L_reference)
    grad_d_psi      = (fieldline.B_cross_grad_B_dot_grad_psi).flatten()
    grad_d_alpha    = (fieldline.B_cross_grad_B_dot_grad_alpha).flatten()
    curv_d_psi      = (fieldline.B_cross_kappa_dot_grad_psi).flatten()
    curv_d_alpha    = (fieldline.B_cross_kappa_dot_grad_alpha).flatten()
    jac_inv         = (fieldline.B_sup_theta_pest).flatten()
    Bhat            = modB/Bref
    
    dpsidr          = Bref * Lref * np.sqrt(s_val)
    dalphady        = 1/(Lref*np.sqrt(s_val))

    L1              = Lref * grad_d_psi/modB**2/dpsidr
    K1              = Lref * curv_d_psi/modB/dpsidr
    L2              = Lref * grad_d_alpha/modB**2/dalphady
    K2              = Lref * curv_d_alpha/modB/dalphady
    dldtheta        = modB/jac_inv

    return L1,K1,L2,K2,dldtheta,Bhat,theta_arr,Lref,Bref, iota_s, iotaN_s


def booz_geo(vmec,s_val,bs = [], alpha=0.0,phi_center=0.0,gridpoints=1001,n_turns=1, helicity=0,plot=False,QS_mapping=True):
    import numpy as np
    from simsopt.mhd.boozer import Boozer
    from AEpy.mag_reader import boozxform_fieldlines, boozxform_splines
    from simsopt.mhd.vmec_diagnostics import vmec_fieldlines, vmec_splines

    # If Boozer object not provided, then run boozxform for given input Vmec
    if not bs:
        bs = Boozer(vmec)
        bs.register(vmec.s_full_grid)
        bs.verbose = False
        bs.bx.verbose = False
        bs.run()

    bs = boozxform_splines(bs, vmec)
    iota_s = bs.iota(s_val)
    iotaN_s = iota_s-helicity

    if QS_mapping==False:
        theta_arr = np.linspace(-n_turns,n_turns,gridpoints)*np.pi
    else:
        theta_arr = (np.linspace(-n_turns,n_turns,gridpoints)*np.pi-helicity*alpha/iota_s)*np.abs(iota_s/iotaN_s)
    

    fieldline = boozxform_fieldlines(vmec,bs,s_val,alpha,theta1d=theta_arr,phi_center=phi_center)

    if plot==True:
        plot_surface_and_fl(vmec,fieldline,s_val,transparant=False,trans_val=0.9,title='')

    modB            = (fieldline.modB).flatten()
    Bref            = (fieldline.B_reference)
    Lref            = (fieldline.L_reference)
    grad_d_psi      = (fieldline.B_cross_grad_B_dot_grad_psi).flatten()
    grad_d_alpha    = (fieldline.B_cross_grad_B_dot_grad_alpha).flatten()
    curv_d_psi      = (fieldline.B_cross_kappa_dot_grad_psi).flatten()
    curv_d_alpha    = (fieldline.B_cross_kappa_dot_grad_alpha).flatten()
    Bhat            = modB/Bref
    dldtheta        = 1/Bhat # only proportionality matters
    
    dpsidr          = Bref * Lref * np.sqrt(s_val)
    dalphady        = 1/(Lref*np.sqrt(s_val))


    L1              = Lref * grad_d_psi/modB**2/dpsidr
    K1              = Lref * curv_d_psi/modB/dpsidr
    L2              = Lref * grad_d_alpha/modB**2/dalphady
    K2              = Lref * curv_d_alpha/modB/dalphady

    return L1,K1,L2,K2,dldtheta,Bhat,theta_arr,Lref,Bref, iota_s, iotaN_s


def nae_geo(stel, r, alpha,N_turns=1,gridpoints=1001,a_minor=1.0):
    # phi input is in cylindrical coordinates
    # B x grad(B) . grad(psi)
    # alpha = 0
    phi_start   = (-N_turns*np.pi - alpha)/stel.iotaN
    phi_end     = (+N_turns*np.pi - alpha)/stel.iotaN
    phi         = np.linspace(phi_start, phi_end, gridpoints)
    # Extract basic properties from pyQSC
    B0 = stel.B0
    B1c = stel.etabar*stel.B0
    B1s = 0

    vals = {}
    vals["nu"] = stel.varphi - stel.phi

    var_names = ["curvature","X1c","X1s","Y1c","Y1s","B20","X20","X2c", \
                "X2s", "Y20", "Y2c", "Y2s", "Z20", "Z2c", "Z2s"]
    for name in var_names:
        vals[name] = getattr(stel,name)

    # Compute derivatives
    var_names = ["X1c","X1s","Y1c","Y1s","B20","X20","X2c", \
                "X2s", "Y20", "Y2c", "Y2s", "Z20", "Z2c", "Z2s"]
    dvar_names = ["dX1c_dvarphi","dX1s_dvarphi","dY1c_dvarphi","dY1s_dvarphi","dB20_dvarphi","dX20_dvarphi","dX2c_dvarphi", \
                "dX2s_dvarphi", "dY20_dvarphi", "dY2c_dvarphi", "dY2s_dvarphi", "dZ20_dvarphi", "dZ2c_dvarphi", "dZ2s_dvarphi"]

    for name, dname in zip(var_names,dvar_names):
        vals[dname] = np.matmul(stel.d_d_varphi, vals[name])

    # Evaluate in the input grid specified
    var_splines = {}
    var_names = ["nu", "curvature", "X1c", "X1s", "Y1c", "Y1s","X20", "X2c","X2s","Y20","Y2c","Y2s","Z20","Z2c","Z2s", \
                "B20","dX1c_dvarphi","dX1s_dvarphi","dY1c_dvarphi","dY1s_dvarphi","dB20_dvarphi","dX20_dvarphi","dX2c_dvarphi", \
                "dX2s_dvarphi", "dY20_dvarphi", "dY2c_dvarphi", "dY2s_dvarphi", "dZ20_dvarphi", "dZ2c_dvarphi", "dZ2s_dvarphi"]
    for name in var_names:
        x = vals[name]
        var_splines[name] = stel.convert_to_spline(x)
        vals[name] = var_splines[name](phi)

    varphi = phi + vals["nu"]
    chi = alpha + stel.iotaN * varphi

    B1 = B1c * np.cos(chi) + B1s * np.sin(chi)
    B2 = vals["B20"] + stel.B2c * np.cos(2*chi) + stel.B2s * np.sin(2*chi)
    dB1_dvarphi = 0 # (stel.iota-stel.iotaN) * B1c * np.sin(chi) - (stel.iota-stel.iotaN) * B1s * np.cos(chi)
    dB1_dtheta = -B1c * np.sin(chi) + B1s * np.cos(chi)
    dB2_dvarphi = vals["dB20_dvarphi"] + 2*(stel.iota-stel.iotaN)*stel.B2c * np.sin(2*chi) - 2*(stel.iota-stel.iotaN)*stel.B2s * np.cos(2*chi)
    dB2_dtheta = -2*stel.B2c * np.sin(2*chi) + 2*stel.B2s * np.cos(2*chi)

    Y1 = vals["Y1c"] * np.cos(chi) + vals["Y1s"] * np.sin(chi)
    X1 = vals["X1c"] * np.cos(chi) + vals["X1s"] * np.sin(chi)
    Y2 = vals["Y20"] + vals["Y2c"] * np.cos(2*chi) + vals["Y2s"] * np.sin(2*chi)
    X2 = vals["X20"] + vals["X2c"] * np.cos(2*chi) + vals["X2s"] * np.sin(2*chi)
    # Z2 = vals["Z20"] + vals["Z2c"] * np.cos(2*chi) + vals["Z2s"] * np.sin(2*chi)
    # dX1_dvarphi = -stel.iotaN * vals["X1c"] * np.sin(chi) +stel.iotaN * vals["X1s"] * np.cos(chi) + vals["dX1c_dvarphi"] * np.cos(chi) + vals["dX1s_dvarphi"] * np.sin(chi)
    dX1_dtheta = -vals["X1c"] * np.sin(chi) + vals["X1s"] * np.cos(chi)
    # dY1_dvarphi = -stel.iotaN * vals["Y1c"] * np.sin(chi) + stel.iotaN * vals["Y1s"] * np.cos(chi) + vals["dY1c_dvarphi"] * np.cos(chi) + vals["dY1s_dvarphi"] * np.sin(chi)
    dY1_dtheta = -vals["Y1c"] * np.sin(chi) + vals["Y1s"] * np.cos(chi)
    # dX2_dvarphi = -2*stel.iotaN*X2c * np.sin(2*chi) + 2*stel.iotaN*vals["X2s"] * np.cos(2*chi) + vals["dX20_dvarphi"] + vals["dX2c_dvarphi"] * np.cos(2*chi) + vals["dX2s_dvarphi"] * np.sin(2*chi)
    dX2_dtheta = -2*vals["X2c"] * np.sin(2*chi) + 2*vals["X2s"] * np.cos(2*chi)
    # dY2_dvarphi = -2*stel.iotaN*vals["Y2c"] * np.sin(2*chi) + 2*stel.iotaN*vals["Y2s"] * np.cos(2*chi) + vals["dY20_dvarphi"] + vals["dY2c_dvarphi"] * np.cos(2*chi) + vals["dY2s_dvarphi"] * np.sin(2*chi)
    dY2_dtheta = -2*vals["Y2c"] * np.sin(2*chi) + 2*vals["Y2s"] * np.cos(2*chi)
    # dZ2_dvarphi = -2*stel.iotaN*vals["Z2c"] * np.sin(2*chi) + 2*stel.iotaN*vals["Z2s"] * np.cos(2*chi) + vals["dZ20_dvarphi"] + vals["dZ2c_dvarphi"] * np.cos(2*chi) + vals["dZ2s_dvarphi"] * np.sin(2*chi)
    # dZ2_dtheta = -2*vals["Z2c"] * np.sin(2*chi) + 2*vals["Z2s"] * np.cos(2*chi)

    # Evaluate the quantities required for AE to the right order
    BxdBdotdpsi_1 = stel.spsi*B0*B0*dB1_dtheta*(Y1*dX1_dtheta - X1*dY1_dtheta)

    BxdBdotdpsi_2 = stel.spsi*B0*(6*B1*dB1_dtheta*(Y1*dX1_dtheta-X1*dY1_dtheta) + B0*(2*Y2*dB1_dtheta*dX1_dtheta + Y1*dB2_dtheta*dX1_dtheta+\
                        Y1*dB1_dtheta*dX2_dtheta-2*X2*dB1_dtheta*dY1_dtheta + 3*X1*X1*vals["curvature"]*dB1_dtheta*dY1_dtheta-\
                        X1*(3*Y1*vals["curvature"]*dB1_dtheta*dX1_dtheta + dB2_dtheta*dY1_dtheta+dB1_dtheta*dY2_dtheta)))

    BdotdB_1 = B0*(dB1_dvarphi + stel.iota * dB1_dtheta)/stel.G0
    BdotdB_2 = (B1*(dB1_dvarphi + stel.iota * dB1_dtheta) + B0*(dB2_dvarphi + stel.iota * dB2_dtheta))/stel.G0

    BxdBdotdalpha_m1 = B0*B1*(X1*dY1_dtheta-Y1*dX1_dtheta)
    BxdBdotdalpha_0 = 2*B0*B2*(X1*dY1_dtheta-Y1*dX1_dtheta) + B1*B1*(-6*Y1*dX1_dtheta+6*X1*dY1_dtheta) + \
                        B0*B1*(-2*Y2*dX1_dtheta - Y1*dX2_dtheta+2*X2*dY1_dtheta - 3*X1*X1*vals["curvature"]*dY1_dtheta+ \
                        X1*(3*Y1*vals["curvature"]*dX1_dtheta+dY2_dtheta))

    BxdBdotdpsi = r*BxdBdotdpsi_1 + r*r*BxdBdotdpsi_2
    BxdBdotdalpha = BxdBdotdalpha_m1/r + BxdBdotdalpha_0
    BdotdB = r*BdotdB_1 + r*r*BdotdB_2

    # mod B
    B   = B0 + r * B1 + r*r * B2

    # Jacobian
    jac_cheeky = (stel.G0+r*r*stel.G2+stel.iota*stel.I2)/B/B

    # Transform to Boozer coordinates (and now use phi as phi_boozer)
    from scipy.interpolate import splev, splrep

    BxdBdotdalpha_spline = splrep(varphi, BxdBdotdalpha)
    BxdBdotdpsi_spline = splrep(varphi, BxdBdotdpsi)
    B_spline = splrep(varphi, B)
    B2_spline = splrep(varphi, B2)

    BxdBdotdalpha = splev(phi, BxdBdotdalpha_spline)
    BxdBdotdpsi = splev(phi, BxdBdotdpsi_spline)
    B = splev(phi, B_spline)
    B2 = splev(phi, B2_spline)

    # assign to self, same units as GIST uses
    dpsidr      = stel.B0*r # psi = B0 * r^2 / 2
    dalphady    = 1/r # y = r * alpha
    z      = phi
    L1     = a_minor*BxdBdotdpsi/B/B/dpsidr
    L2     = a_minor*BxdBdotdalpha/B/B/dalphady
    a_minor= a_minor
    modb   = B
    dldphi = 1/B #jac_cheeky * B # 1/B when boozer phi is field-line following coordinate
    B2     = B2
    

    # we return B instead of B/stel.B0,
    # so that one can choose the units of B on the fly
    # example, if one wishes to express B in units of B_ref, then
    # simply set stel.B0 = 1/B_ref
    return z, L2, L1, B, dldphi


######################################################
######################################################
######################################################











######################################################
################ classes for AE data #################
######################################################


class AE_gist:
    r"""
    Class which calculates data related to AE. Contains several plotting
    routines, useful for assessing drifts and spatial structure of AE.
    """
    def __init__(self, gist_data, lam_res=1000, quad=False, interp_kind='cubic',
                 get_drifts=True,normalize='ft-vol',AE_lengthscale='None'):

        # import relevant data
        self.L1         = gist_data.L1
        self.L2         = gist_data.L2
        self.sqrtg      = gist_data.sqrtg
        self.modb       = gist_data.modb
        self.z          = gist_data.theta
        self.q0         = np.abs(gist_data.q0)
        self.interp_kind= interp_kind
        self.lam_res    = lam_res
        self.quad       = quad
        try:
            self.my_dpdx = gist_data.my_dpdx
        except:
            print('my_dpdx is unavailable - defaulting to zero.')
            self.my_dpdx = 0.0
        try:
            self.s0 = gist_data.s0
        except:
            print('s0 is unavailable - defaulting to 0.5.')
            self.s0 = 0.5
        self.ft_vol = simpson(self.sqrtg,self.z)/simpson(self.sqrtg*self.modb,self.z)
        self.normalize = normalize

        self.Delta_x = 1.0
        self.Delta_y = 1.0/np.sqrt(self.s0)
        if AE_lengthscale=='q':
            self.Delta_x = self.q0 * self.Delta_x
            self.Delta_y = self.q0 * self.Delta_y



        if get_drifts==True:
            self.calculate_drifts()

    def calculate_drifts(self):

        # calculate drifts
        roots_list,wpsi_list,walpha_list,tau_b_list,lam_list,k2 = drift_from_gist(self.z,
        self.modb,self.sqrtg,self.L1,self.L2,self.my_dpdx,self.lam_res,quad=self.quad,
        interp_kind=self.interp_kind)
        # assign to self
        self.roots  = roots_list
        self.wpsi   = wpsi_list
        self.walpha = walpha_list
        self.taub   = tau_b_list
        self.lam    = lam_list
        self.k2     = k2




    def calc_AE(self,omn,omt,omnigenous):
        # loop over all lambda
        Delta_x = self.Delta_x
        Delta_y = self.Delta_y
        L_tot  = simpson(self.sqrtg*self.modb,self.z)
        ae_at_lam_list = []
        if omnigenous==False:
            for lam_idx, lam_val in enumerate(self.lam):
                wpsi_at_lam     = Delta_y*self.wpsi[lam_idx]
                walpha_at_lam   = Delta_x*self.walpha[lam_idx]
                taub_at_lam     = self.taub[lam_idx]
                integrand       = lambda x: AE_per_lam_per_z(walpha_at_lam,wpsi_at_lam,Delta_x*w_diamag(-omn,-omt,x),taub_at_lam,x)
                ae_at_lam, _    = quad_vec(integrand,0.0,np.inf, epsrel=1e-6,epsabs=1e-20, limit=1000)
                ae_at_lam_list.append(ae_at_lam/L_tot)
        if omnigenous==True:
            for lam_idx, lam_val in enumerate(self.lam):
                walpha_at_lam   = Delta_x*self.walpha[lam_idx]
                taub_at_lam     = self.taub[lam_idx]
                dlnndx          = -omn
                dlnTdx          = -omt
                c0 = Delta_x * (dlnndx - 3/2 * dlnTdx) / walpha_at_lam
                c1 = 1.0 - Delta_x * dlnTdx / walpha_at_lam
                ae_at_lam       = AE_per_lam(c0,c1,taub_at_lam,walpha_at_lam)
                ae_at_lam_list.append(ae_at_lam/L_tot)

        self.ae_per_lam     = ae_at_lam_list

        # now do integral over lam to find total AE
        lam_arr   = np.asarray(self.lam).flatten()
        ae_per_lam_summed = np.zeros_like(lam_arr)
        for lam_idx, lam_val in enumerate(lam_arr):
            ae_per_lam_summed[lam_idx] = np.sum(self.ae_per_lam[lam_idx])
        ae_tot = simpson(ae_per_lam_summed,lam_arr)
        if self.normalize=='ft-vol':
            ae_tot = ae_tot/self.ft_vol
        self.ae_tot     = ae_tot


    def calc_AE_fast(self,omn,omt,omnigenous):
        # loop over all lambda
        L_tot  = simpson(self.sqrtg*self.modb,self.z)
        Delta_x = self.Delta_x
        Delta_y = self.Delta_y
        ae_at_lam_list = []
        if omnigenous==False:
            for lam_idx, lam_val in enumerate(self.lam):
                wpsi_at_lam     = Delta_y*self.wpsi[lam_idx]
                walpha_at_lam   = Delta_x*self.walpha[lam_idx]
                taub_at_lam     = self.taub[lam_idx]
                integrand       = lambda x: np.sum(AE_per_lam_per_z(walpha_at_lam,wpsi_at_lam,Delta_x*w_diamag(-omn,-omt,x),taub_at_lam,x))
                ae_at_lam, _    = quad(integrand,0.0,np.inf, epsrel=1e-6,epsabs=1e-20, limit=1000)
                ae_at_lam_list.append(ae_at_lam/L_tot)
        if omnigenous==True:
            for lam_idx, lam_val in enumerate(self.lam):
                walpha_at_lam   = Delta_x*self.walpha[lam_idx]
                taub_at_lam     = self.taub[lam_idx]
                dlnndx          = -omn
                dlnTdx          = -omt
                c0 = Delta_x * (dlnndx - 3/2 * dlnTdx) / walpha_at_lam
                c1 = 1.0 - Delta_x * dlnTdx / walpha_at_lam
                ae_at_lam       = np.sum(AE_per_lam(c0,c1,taub_at_lam,walpha_at_lam))
                ae_at_lam_list.append(ae_at_lam/L_tot)

        # now do integral over lam to find total AE
        lam_arr   = np.asarray(self.lam).flatten()
        ae_per_lam_summed = np.zeros_like(lam_arr)
        for lam_idx, lam_val in enumerate(lam_arr):
            ae_per_lam_summed[lam_idx] = np.sum(ae_at_lam_list[lam_idx])
        ae_tot = simpson(ae_per_lam_summed,lam_arr)
        if self.normalize=='ft-vol':
            ae_tot = ae_tot/self.ft_vol
        self.ae_tot     = ae_tot


    def plot_precession(self,save=False,filename='AE_precession.eps'):
        plot_precession_func(self,save=save,filename=filename)


    def plot_AE_per_lam(self,save=False,filename='AE_per_lam.eps',scale=1.0):
        plot_AE_per_lam_func(self,save=save,filename=filename,scale=scale)


class AE_pyQSC:
    def __init__(self, stel_obj=None, name='precise QH', r=[], alpha=0.0, N_turns=3, nphi=1001,
                 lam_res=1000, get_drifts=True,normalize='ft-vol',AE_lengthscale='None',a_minor=1.0,
                 epsrel=1e-4):
        import matplotlib.pyplot as plt
        # Construct stellarator
        # if no stellarator given, use from paper
        if stel_obj==None:
            stel = Qsc.from_paper(name,nphi=nphi,B0=1)
            stel.etabar = -np.abs(stel.etabar)
            stel.spsi = -1
            stel.zs = -stel.zs
            stel.calculate()
        else:
            stel = stel_obj

        self.stel = stel_obj

        self.normalize = normalize

        # set lengthscales
        self.Delta_r = 1.0
        self.Delta_y = 1.0

        if not hasattr(stel, 'r'):
            if r:
                stel.r = r
                print('Set r in the near-axis construction to value specified explicitly to the constructor.')
            else:
                stel.r = r
                print('Set r in the near-axis construction to default value 1e-6.')
        else:
            r = stel.r
            print('Using r in the near-axis object given.')
        z, L2, L1, modb, dldz = nae_geo(stel, r, alpha,N_turns=N_turns,gridpoints=nphi,a_minor=a_minor)

        self.z = z
        self.L2= L2
        self.L1= L1
        self.modb = modb
        self.B  = modb
        self.dldz = dldz
        self.dldphi = dldz
        self.phi    = z
        self.a_minor = a_minor
        self.epsrel = epsrel

        

        # calculate normalized flux-tube volume
        self.ft_vol = simpson(self.dldphi/self.modb,self.phi)/simpson(self.dldphi,self.phi)

        self.my_dpdx = 0.0 #stel.r**2 * stel.p2


        roots_list,wpsi_list,walpha_list,tau_b_list,lam_list,k2 = drift_from_pyQSC(self.phi,self.modb,self.dldphi,self.L1,self.L2,self.my_dpdx,lam_res,quad=False,interp_kind='cubic')
        # assign to self
        self.stel   = stel
        self.roots  = roots_list
        self.wpsi   = wpsi_list
        self.walpha = walpha_list
        self.taub   = tau_b_list
        self.lam    = lam_list
        self.k2     = k2


    def calc_AE(self,omn,omt,omnigenous):
        # loop over all lambda
        Delta_r = self.Delta_r
        Delta_y = self.Delta_y
        L_tot  = simpson(self.dldphi,self.phi)
        ae_at_lam_list = []
        if omnigenous==False:
            for lam_idx, lam_val in enumerate(self.lam):
                wpsi_at_lam     = Delta_y*self.wpsi[lam_idx]
                walpha_at_lam   = Delta_r*self.walpha[lam_idx]
                taub_at_lam     = self.taub[lam_idx]
                integrand       = lambda x: AE_per_lam_per_z(walpha_at_lam,wpsi_at_lam,Delta_r*w_diamag(-omn,-omt,x),taub_at_lam,x)
                ae_at_lam, _    = quad_vec(integrand,0.0,np.inf, epsrel=self.epsrel,epsabs=1e-20, limit=1000)
                ae_at_lam_list.append(ae_at_lam/L_tot)
        if omnigenous==True:
            for lam_idx, lam_val in enumerate(self.lam):
                walpha_at_lam   = Delta_r*self.walpha[lam_idx]
                taub_at_lam     = self.taub[lam_idx]
                dlnndx          = -omn
                dlnTdx          = -omt
                c0 = Delta_r * (dlnndx - 3/2 * dlnTdx) / walpha_at_lam
                c1 = 1.0 - Delta_r * dlnTdx / walpha_at_lam
                ae_at_lam       = AE_per_lam(c0,c1,taub_at_lam,walpha_at_lam)
                ae_at_lam_list.append(ae_at_lam/L_tot)

        self.ae_per_lam     = ae_at_lam_list

        # now do integral over lam to find total AE
        lam_arr   = np.asarray(self.lam).flatten()
        ae_per_lam_summed = np.zeros_like(lam_arr)
        for lam_idx, lam_val in enumerate(lam_arr):
            ae_per_lam_summed[lam_idx] = np.sum(self.ae_per_lam[lam_idx])
        ae_tot = simpson(ae_per_lam_summed,lam_arr)
        if self.normalize=='ft-vol':
            ae_tot = ae_tot/self.ft_vol
        self.ae_tot     = ae_tot


    def calc_AE_quad(self,omn,omt,omnigenous):
        # import matplotlib.pyplot as plt
        Delta_r = self.Delta_r
        Delta_y = self.Delta_y
        modb = self.modb
        bmax = (self.modb).max()
        bmin = (self.modb).min()
        # special_modb_max = modb[argrelextrema(modb, np.greater)[0]]
        # special_modb_min = modb[argrelextrema(modb, np.less)[0]]
        # special_points  = np.concatenate((special_modb_max,special_modb_min))
        # special_lam     = 1/special_points
        # special_k2      = (bmax - special_lam * bmax * bmin) / (bmax - bmin)
        # special_k2      = np.sort(np.unique(special_k2.round())) # remove duplicates
        # special_k2      = np.delete(special_k2, 0)
        # special_k2      = tuple(np.delete(special_k2, -1))
        # print(special_k2)
        L_tot  = simpson(self.dldphi,self.phi)
        if omnigenous==False:
            # construct integrand
            def integrand(k2):
                lam = - (k2 * (bmax - bmin) - bmax) / (bmax*bmin)
                _, wpsi_list, walpha_list,tau_b_list, _, _ = drift_from_pyQSC(self.phi,self.modb,self.dldphi,self.L1,self.L2,self.my_dpdx,lam,direct=True)
                wpsi_arr        = Delta_y*np.asarray(wpsi_list).flatten()
                walpha_arr      = Delta_r*np.asarray(walpha_list).flatten()
                taub_arr        = np.asarray(tau_b_list).flatten()
                integrand_2     = lambda z: np.sum(AE_per_lam_per_z(walpha_arr,wpsi_arr,Delta_r*w_diamag(-omn,-omt,z),taub_arr,z))
                ans, _ = quad(integrand_2,0.0,np.inf,epsrel=self.epsrel)
                # plt.scatter(k2,ans,marker='o',s=0.1)
                return ans
            # do integral
            ae_arr = quad(integrand,0.0,1.0,epsrel=self.epsrel,epsabs=1e-16,limit=10000)#,points=special_k2)
        if omnigenous==True:
            # construct integrand
            def integrand(k2):
                lam = - (k2 * (bmax - bmin) - bmax) / (bmax*bmin)
                dlnndx          = -omn
                dlnTdx          = -omt
                _, _, walpha_list,tau_b_list, _, _ = drift_from_pyQSC(self.phi,self.modb,self.dldphi,self.L1,self.L2,self.my_dpdx,lam,direct=True)
                walpha_arr = Delta_r*np.asarray(walpha_list).flatten()
                taub_arr = np.asarray(tau_b_list).flatten()
                c0 = Delta_r * (dlnndx - 3/2 * dlnTdx) / walpha_arr
                c1 = 1.0 - Delta_r * dlnTdx / walpha_arr
                ans = np.sum(AE_per_lam(c0,c1,taub_arr,walpha_arr))
                # plt.scatter(k2,ans,color='black',marker='o',s=0.1)
                # print(k2,ans)
                return ans
            # do integral
            ae_arr =  quad(integrand,0.0,1.0,epsrel=self.epsrel,epsabs=1e-16,limit=10000)#,points=special_k2)
            
        # plt.show()
        if self.normalize=='ft-vol':
            ae_tot = ae_arr[0]/self.ft_vol
        self.ae_tot = ae_tot/L_tot * (bmax - bmin)/(bmax * bmin)


    def nae_ae_asymp_weak(self,omn,a_minor=1.0):
        stel    = self.stel
        ae_fac  = 0.666834
        varrho  = stel.r/a_minor
        aspect  = 1/np.abs(stel.etabar*a_minor)
        prefac  = np.sqrt(2) / (3 * np.pi)
        return prefac * np.sqrt(varrho * aspect) * (omn)**3 * ae_fac
    

    def nae_ae_asymp_strong(self,omn,a_minor=1.0):
        stel    = self.stel
        varrho  = stel.r/a_minor
        aspect  = 1/np.abs(stel.etabar*a_minor)
        prefac  = 1/(np.sqrt(2) * np.pi) * 1.1605
        return prefac * np.sqrt(varrho * aspect) * (omn) /(aspect)**2


    def plot_geom(self):
        plot_geom_nae(self)


    def plot_precession(self,save=False,filename='AE_precession.eps', nae=False,stel=None,alpha=0.0):
        plot_precession_func(self,save=save,filename=filename,nae=nae,stel=stel,alpha=alpha)



    def plot_AE_per_lam(self,save=False,filename='AE_per_lam.eps',scale=1.0):
        plot_AE_per_lam_func(self,save=save,filename=filename,scale=scale)


class AE_vmec:
    def __init__(self, vmec,s_val,booz = False, alpha=0.0,phi_center=0.0,gridpoints=1001,lam_res=1001,n_turns=3, helicity=0,plot=False,mod_norm='None',QS_mapping=False,epsrel=1e-4):
        import matplotlib.pyplot    as      plt
        from simsopt.mhd.vmec       import  Vmec
        from simsopt.mhd.boozer     import  Boozer

        if booz:
            if isinstance(booz, Boozer):
                L1,K1,L2,K2,dldz,modb,theta, Lref,Bref, iota, iotaN = booz_geo(vmec,s_val,bs = booz, alpha=alpha,phi_center=phi_center,
                                                             gridpoints=gridpoints,n_turns=n_turns,helicity=helicity,plot=plot,QS_mapping=QS_mapping)
            else:
                L1,K1,L2,K2,dldz,modb,theta, Lref,Bref, iota, iotaN = booz_geo(vmec,s_val, alpha=alpha,phi_center=phi_center,gridpoints=gridpoints,
                                                             n_turns=n_turns,helicity=helicity,plot=plot,QS_mapping=QS_mapping)
        else:
            L1,K1,L2,K2,dldz,modb,theta, Lref,Bref, iota, iotaN = vmec_geo(vmec,s_val,alpha=alpha,phi_center=phi_center,gridpoints=gridpoints,
                                                         n_turns=n_turns,helicity=helicity,plot=plot,QS_mapping=QS_mapping)
        
        if mod_norm=='fl-ave':
            print('using fl-ave normalization')
            fl_ave_modb = simpson(modb*dldz,theta)/simpson(dldz,theta)
            modb=modb/fl_ave_modb
        if mod_norm=='T':
            print('using [T] normalization')
            modb=modb*Bref


        # get drifts
        roots_list,wpsi_list,walpha_list,tau_b_list,lam_list,k2 = drift_from_vmec(theta,modb,dldz,L1,L2,K1,K2,lam_res)
        self.modb  = modb
        self.z      = theta 
        self.dldz   = dldz
        self.normalize = 'ft-vol'
        self.Lref      = Lref
        self.a_minor   = vmec.wout.Aminor_p
        self.epsrel    = epsrel
        
        # assign to self
        self.roots  = roots_list
        self.wpsi   = wpsi_list
        self.walpha = walpha_list
        self.taub   = tau_b_list
        self.lam    = lam_list
        self.k2     = k2
        # set AE length-scale
        self.Delta_r = 1.0
        self.Delta_y = 1.0
        # set geometry 
        self.L1 = L1
        self.L2 = L2
        self.K1 = K1
        self.K2 = K2
        self.iota = iota
        self.iotaN = iotaN

        # set other scalars
        self.psi_edge_over_two_pi = np.abs(vmec.wout.phi[-1]/(2*np.pi))
        self.Bref = Bref

        
        # set ft_vol
        self.ft_vol = simpson(self.dldz/self.modb,self.z)/simpson(self.dldz,self.z)

        # set vmec variable
        self.vmec   = True
        
        

    def calc_AE(self,omn,omt,omnigenous,fast=False):
        # loop over all lambda
        Delta_r = self.Delta_r
        Delta_y = self.Delta_y
        L_tot  = simpson(self.dldz,self.z)
        ae_at_lam_list = []
        if omnigenous==False:
            for lam_idx, lam_val in enumerate(self.lam):
                wpsi_at_lam     = Delta_y*self.wpsi[lam_idx]
                walpha_at_lam   = Delta_r*self.walpha[lam_idx]
                taub_at_lam     = self.taub[lam_idx]
                if fast==False:
                    integrand       = lambda x: AE_per_lam_per_z(walpha_at_lam,wpsi_at_lam,Delta_r*w_diamag(-omn,-omt,x),taub_at_lam,x)
                    ae_at_lam, _    = quad_vec(integrand,0.0,np.inf, epsrel=self.epsrel,epsabs=1e-20, limit=10000)
                if fast==True:
                    integrand       = lambda x: np.sum(AE_per_lam_per_z(walpha_at_lam,wpsi_at_lam,Delta_r*w_diamag(-omn,-omt,x),taub_at_lam,x))
                    ae_at_lam, _    = quad(integrand,0.0,np.inf, epsrel=self.epsrel,epsabs=1e-20, limit=50)
                ae_at_lam_list.append(ae_at_lam/L_tot)
        if omnigenous==True:
            for lam_idx, lam_val in enumerate(self.lam):
                walpha_at_lam   = Delta_r*self.walpha[lam_idx]
                taub_at_lam     = self.taub[lam_idx]
                dlnndx          = -omn
                dlnTdx          = -omt
                c0 = Delta_r * (dlnndx - 3/2 * dlnTdx) / walpha_at_lam
                c1 = 1.0 - Delta_r * dlnTdx / walpha_at_lam
                ae_at_lam       = AE_per_lam(c0,c1,taub_at_lam,walpha_at_lam)
                ae_at_lam_list.append(ae_at_lam/L_tot)

        self.ae_per_lam = ae_at_lam_list
        # now do integral over lam to find total AE
        lam_arr   = np.asarray(self.lam).flatten()
        ae_per_lam_summed = np.zeros_like(lam_arr)
        for lam_idx, lam_val in enumerate(lam_arr):
            ae_per_lam_summed[lam_idx] = np.sum(self.ae_per_lam[lam_idx])
        ae_tot = simpson(ae_per_lam_summed,lam_arr)
        if self.normalize=='ft-vol':
            ae_tot = ae_tot/self.ft_vol
        self.ae_tot     = ae_tot

        
    def calc_AE_quad(self,omn,omt,omnigenous):
        # loop over all lambda
        # import matplotlib.pyplot as plt
        Delta_r = self.Delta_r
        Delta_y = self.Delta_y
        L_tot  = simpson(self.dldz,self.z)
        bmax = (self.modb).max()
        bmin = (self.modb).min()
        if omnigenous==False:
            def integrand(k2):
                lam = - (k2 * (bmax - bmin) - bmax) / (bmax*bmin)
                _,wpsi_list,walpha_list,tau_b_list,_,_ = drift_from_vmec(self.z,self.modb,self.dldz,self.L1,self.L2,self.K1,self.K2,lam,direct=True)
                wpsi_arr        = Delta_y*np.asarray(wpsi_list).flatten()
                walpha_arr      = Delta_r*np.asarray(walpha_list).flatten()
                taub_arr        = np.asarray(tau_b_list).flatten()
                integrand_2     = lambda z: np.sum(AE_per_lam_per_z(walpha_arr,wpsi_arr,Delta_r*w_diamag(-omn,-omt,z),taub_arr,z))
                ans, _ = quad(integrand_2,0.0,np.inf,epsrel=self.epsrel)
                return ans
            ae_arr = quad(integrand,0.0,1.0,epsrel=self.epsrel)#,points=special_k2)
        if omnigenous==True:
            def integrand(k2):
                lam = - (k2 * (bmax - bmin) - bmax) / (bmax*bmin)
                _,wpsi_list,walpha_list,tau_b_list,_,_ = drift_from_vmec(self.z,self.modb,self.dldz,self.L1,self.L2,self.K1,self.K2,lam,direct=True)
                walpha_arr      = Delta_r*np.asarray(walpha_list).flatten()
                taub_arr        = np.asarray(tau_b_list).flatten()
                dlnndx          = -omn
                dlnTdx          = -omt
                c0 = Delta_r * (dlnndx - 3/2 * dlnTdx) / walpha_arr
                c1 = 1.0 - Delta_r * dlnTdx / walpha_arr
                ans     = np.sum(AE_per_lam(c0,c1,taub_arr,walpha_arr))
                # plt.scatter(k2,ans,color='black',marker='o',s=0.2)
                return ans
            ae_arr = quad(integrand,0.0,1.0,epsrel=self.epsrel)#,points=special_k2)
        # plt.show()
        if self.normalize=='ft-vol':
            ae_tot = ae_arr[0]/self.ft_vol
        self.ae_tot = ae_tot/L_tot * (bmax - bmin)/(bmax * bmin)

    


    def plot_precession(self,save=False,filename='AE_precession.eps', nae=False,stel=None,alpha=0.0,iota=1.0):
        plot_precession_func(self,save=save,filename=filename,nae=nae,stel=stel,alpha=alpha,iota=iota)



    def plot_AE_per_lam(self,save=False,filename='AE_per_lam.eps',scale=1.0):
        plot_AE_per_lam_func(self,save=save,filename=filename,scale=scale)


######################################################
######################################################
######################################################



























######################################################
################# plotting routines ##################
######################################################

def plot_surface_and_fl(vmec,fl,s_val,transparant=False,trans_val=0.9,title=''):
    import math
    from    matplotlib          import  cm
    from mayavi import mlab
    phi = vmec.wout.phi
    iotaf = vmec.wout.iotaf
    presf = vmec.wout.presf
    iotas = vmec.wout.iotas
    pres = vmec.wout.pres
    ns = vmec.wout.ns
    nfp = vmec.wout.nfp
    xn = vmec.wout.xn
    xm = vmec.wout.xm
    xn_nyq = vmec.wout.xn_nyq
    xm_nyq = vmec.wout.xm_nyq
    rmnc = vmec.wout.rmnc.T
    zmns = vmec.wout.zmns.T
    bmnc = vmec.wout.bmnc.T
    raxis_cc = vmec.wout.raxis_cc
    zaxis_cs = vmec.wout.zaxis_cs
    buco = vmec.wout.buco
    bvco = vmec.wout.bvco
    jcuru = vmec.wout.jcuru
    jcurv = vmec.wout.jcurv
    lasym = vmec.wout.lasym
    iradius = int(np.floor(s_val*(ns-1)))

    ac_aux_s = vmec.wout.ac_aux_s
    ac_aux_f = vmec.wout.ac_aux_f

    mpol = vmec.wout.mpol
    ntor = vmec.wout.ntor
    Aminor_p = vmec.wout.Aminor_p
    Rmajor_p = vmec.wout.Rmajor_p
    aspect = vmec.wout.aspect
    betatotal = vmec.wout.betatotal
    betapol = vmec.wout.betapol
    betator = vmec.wout.betator
    betaxis = vmec.wout.betaxis
    ctor = vmec.wout.ctor
    DMerc = vmec.wout.DMerc
    gmnc = vmec.wout.gmnc

    if lasym == 1:
        rmns = vmec.wout.rmns
        zmnc = vmec.wout.zmnc
        bmns = vmec.wout.bmns
        raxis_cs = vmec.wout.raxis_cs
        zaxis_cc = vmec.wout.zaxis_cc
    else:
        rmns = 0*rmnc
        zmnc = 0*rmnc
        bmns = 0*bmnc
        raxis_cs = 0*raxis_cc
        zaxis_cc = 0*raxis_cc
    try:
        ac = vmec.wout.ac
    except:
        ac = []
    try:
        pcurr_type = vmec.wout.pcurr_type
    except:
        pcurr_type = ""
    nmodes = len(xn)
    s = np.linspace(0,1,ns)
    s_half = [(i-0.5)/(ns-1) for i in range(1,ns)]
    phiedge = phi[-1]
    phi_half = [(i-0.5)*phiedge/(ns-1) for i in range(1,ns)]
    ntheta = 200
    nzeta = 8
    theta = np.linspace(0,2*np.pi,num=ntheta)
    zeta = np.linspace(0,2*np.pi/nfp,num=nzeta,endpoint=False)
    R = np.zeros((ntheta,nzeta))
    Z = np.zeros((ntheta,nzeta))
    for itheta in range(ntheta):
        for izeta in range(nzeta):
            for imode in range(nmodes):
                angle = xm[imode]*theta[itheta] - xn[imode]*zeta[izeta]
                R[itheta,izeta] = R[itheta,izeta] + rmnc[iradius,imode]*math.cos(angle) + rmns[iradius,imode]*math.sin(angle)
                Z[itheta,izeta] = Z[itheta,izeta] + zmns[iradius,imode]*math.sin(angle) + zmnc[iradius,imode]*math.cos(angle)

    Raxis = np.zeros(nzeta)
    Zaxis = np.zeros(nzeta)
    for izeta in range(nzeta):
        for n in range(ntor+1):
            angle = -n*nfp*zeta[izeta]
            Raxis[izeta] += raxis_cc[n]*math.cos(angle) + raxis_cs[n]*math.sin(angle)
            Zaxis[izeta] += zaxis_cs[n]*math.sin(angle) + zaxis_cc[n]*math.cos(angle)
    ntheta = 100
    nzeta = int(200)
    theta1D = np.linspace(0,2*np.pi,num=ntheta)
    zeta1D = np.linspace(0,2*np.pi,num=nzeta)
    zeta2D, theta2D = np.meshgrid(zeta1D,theta1D)
    R = np.zeros((ntheta,nzeta))
    Z = np.zeros((ntheta,nzeta))
    B = np.zeros((ntheta,nzeta))
    for imode in range(nmodes):
        angle = xm[imode]*theta2D - xn[imode]*zeta2D
        R = R + rmnc[iradius,imode]*np.cos(angle) + rmns[iradius,imode]*np.sin(angle)
        Z = Z + zmns[iradius,imode]*np.sin(angle) + zmnc[iradius,imode]*np.cos(angle)

    for imode in range(len(xn_nyq)):
        angle = xm_nyq[imode]*theta2D - xn_nyq[imode]*zeta2D
        B = B + bmnc[iradius,imode]*np.cos(angle) + bmns[iradius,imode]*np.sin(angle)

    X = R * np.cos(zeta2D)
    Y = R * np.sin(zeta2D)
    # Rescale to lie in [0,1]:
    B_rescaled = (B - B.min()) / (B.max() - B.min())


    X_coord = (fl.R).flatten() * (fl.cosphi).flatten()
    Y_coord = (fl.R).flatten() * (fl.sinphi).flatten()
    Z_coord = (fl.Z).flatten()

    from mayavi import mlab
    fig = mlab.figure(size=(1000, 1000))

    colors = np.asarray(cm.binary(B_rescaled))[:,:,1]
    surf3 = mlab.mesh(X, Y, Z, scalars=B,colormap='coolwarm',vmin=np.amin(B),vmax=np.amax(B))
    line = mlab.plot3d(X_coord, Y_coord, Z_coord,color=(0.0,0.0,0.0),tube_radius=0.03) #(0, 0.7, 0.23)
    gridpoints = len(X_coord)
    point = mlab.points3d(X_coord[int(gridpoints/2)], Y_coord[int(gridpoints/2)], Z_coord[int(gridpoints/2)], color=(1.0,1.0,1.0),scale_factor=0.2)

    if transparant == True:
        surf3.actor.property.opacity = trans_val

    if title!='':
       mlab.savefig('3D_plot_'+title+'.png', figure=fig)



    mlab.show()
    mlab.close(all=True)



def plot_precession_func(AE_obj,save=False,filename='AE_precession.eps',nae=False,stel=None,alpha=0.0,iota=1.0):
    r"""
    Plots the precession as a function of the bounce-points and k2.
    """
    import matplotlib.pyplot as plt
    import matplotlib        as mpl

    plt.close('all')

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 10}

    mpl.rc('font', **font)

    # reshape for plotting
    walp_arr = np.nan*np.zeros([len(AE_obj.walpha),len(max(AE_obj.walpha,key = lambda x: len(x)))])
    for i,j in enumerate(AE_obj.walpha):
        walp_arr[i][0:len(j)] = j
    wpsi_arr = np.nan*np.zeros([len(AE_obj.wpsi),len(max(AE_obj.wpsi,key = lambda x: len(x)))])
    for i,j in enumerate(AE_obj.wpsi):
        wpsi_arr[i][0:len(j)] = j
    alp_l  = np.shape(walp_arr)[1]
    k2_arr = np.repeat(AE_obj.k2,alp_l)
    fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(2*3.5, 5.0))
    if nae==False:
        ax[1,0].scatter(k2_arr,walp_arr,s=0.2,marker='.',color='black',facecolors='black')
        ax[1,0].plot(AE_obj.k2,0.0*AE_obj.k2,color='red',linestyle='dashed')
        ax[1,1].scatter(k2_arr,wpsi_arr,s=0.2,marker='.',color='black',facecolors='black')
        ax[1,0].set_xlim(0,1)
        ax[1,1].set_xlim(0,1)
        ax[1,0].set_xlabel(r'$k^2$')
        ax[1,1].set_xlabel(r'$k^2$')
        ax[1,0].set_ylabel(r'$\langle \hat{\mathbf{v}}_D \cdot \nabla y \rangle$',color='black')
        ax[1,1].set_ylabel(r'$\langle \hat{\mathbf{v}}_D \cdot \nabla x \rangle$',color='black')


    # now do plot as a function of bounce-angle
    walpha_bounceplot = []
    roots_bounceplot  = []
    wpsi_bounceplot   = []
    for lam_idx, lam_val in enumerate(AE_obj.lam):
        root_at_lam = AE_obj.roots[lam_idx]
        wpsi_at_lam = AE_obj.wpsi[lam_idx]
        walpha_at_lam= AE_obj.walpha[lam_idx]
        roots_bounceplot.extend(root_at_lam)
        for idx in range(len(wpsi_at_lam)):
            wpsi_bounceplot.extend([wpsi_at_lam[idx]])
            wpsi_bounceplot.extend([wpsi_at_lam[idx]])
            walpha_bounceplot.extend([walpha_at_lam[idx]])
            walpha_bounceplot.extend([walpha_at_lam[idx]])

    roots_ordered, wpsi_bounceplot = (list(t) for t in zip(*sorted(zip(roots_bounceplot, wpsi_bounceplot))))
    roots_ordered, walpha_bounceplot = (list(t) for t in zip(*sorted(zip(roots_bounceplot, walpha_bounceplot))))
    ax[0,0].plot(AE_obj.z,AE_obj.modb,color='black')
    ax001= ax[0,0].twinx()
    ax001.plot(roots_ordered,walpha_bounceplot,color='tab:blue')
    ax001.plot(np.asarray(roots_ordered),0.0*np.asarray(walpha_bounceplot),color='tab:red',linestyle='dashed')
    ax[0,1].plot(AE_obj.z,AE_obj.modb,color='black')
    ax011= ax[0,1].twinx()
    ax011.plot(roots_ordered,wpsi_bounceplot,color='tab:blue')
    ax[0,0].set_xlim(AE_obj.z.min(),AE_obj.z.max())
    ax[0,1].set_xlim(AE_obj.z.min(),AE_obj.z.max())
    ax[0,0].set_xlabel(r'$z$')
    ax[0,1].set_xlabel(r'$z$')
    ax[0,0].set_ylabel(r'$B$')
    ax[0,1].set_ylabel(r'$B$')
    ax001.set_ylabel(r'$\langle \hat{\mathbf{v}}_D \cdot \nabla y \rangle$',color='tab:blue')
    ax011.set_ylabel(r'$\langle \hat{\mathbf{v}}_D \cdot \nabla x \rangle$',color='tab:blue')
    if nae:
        wa0, wa1 = drift_asymptotic(stel,AE_obj.a_minor,AE_obj.k2)
        ax[1,0].plot(AE_obj.k2, wa0, color = 'orange', linestyle='dotted', label='NAE (1st order)')
        ax[1,0].plot(AE_obj.k2, wa0+wa1, color = 'green', linestyle='dashed', label='NAE (2nd order)')
        ax[1,0].legend()
        roots_ordered     = np.asarray(roots_ordered)/iota
        roots_ordered_chi = stel.iotaN*roots_ordered - alpha
        khat = np.sin(np.mod(roots_ordered_chi/2.0,2*np.pi))
    
        # plot as function of khat^2
        ax[1,0].scatter(khat**2,walpha_bounceplot,s=0.2,marker='.',color='black',facecolors='black')
        ax[1,1].scatter(khat**2,wpsi_bounceplot,s=0.2,marker='.',color='black',facecolors='black')
        ax[1,0].set_xlabel(r'$\hat{k}^2$')
        ax[1,1].set_xlabel(r'$\hat{k}^2$')
        ax[1,0].set_ylabel(r'$\langle \hat{\mathbf{v}}_D \cdot \nabla y \rangle$',color='black')
        ax[1,1].set_ylabel(r'$\langle \hat{\mathbf{v}}_D \cdot \nabla x \rangle$',color='black')
        ax[1,0].set_xlim(0,1)
        ax[1,1].set_xlim(0,1)
    
    if save==True:
        plt.savefig(filename,dpi=1000)
    plt.show()



def plot_AE_per_lam_func(AE_obj,save=False,filename='AE_per_lam.eps',scale=1.0):
    r"""
    Plots AE per bouncewell
    """
    import  matplotlib.pyplot   as      plt
    import  matplotlib          as      mpl
    from    matplotlib          import  cm
    import  matplotlib.colors   as      mplc
    plt.close('all')

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 10}

    mpl.rc('font', **font)
    fig ,ax = plt.subplots(1, 1, figsize=(scale*6, scale*3.5))
    ax.set_xlim(min(AE_obj.z)/np.pi,max(AE_obj.z)/np.pi)

    lam_arr   = np.asarray(AE_obj.lam).flatten()
    ae_per_lam = AE_obj.ae_per_lam
    list_flat = []
    for val in ae_per_lam:
        list_flat.extend(val)
    max_ae_per_lam = max(list_flat)

    roots=AE_obj.roots

    cm_scale = lambda x: x
    colors_plot = [cm.plasma(cm_scale(np.asarray(x) * 1.0/max_ae_per_lam)) for x in ae_per_lam]

    # iterate over all values of lambda
    for idx_lam, lam in enumerate(lam_arr):
        b_val = 1/lam

        # iterate over all bounce wells
        for idx_bw, _ in enumerate(ae_per_lam[idx_lam]):
            bws = roots[idx_lam]
            # check if well crosses boundary
            if(bws[2*idx_bw] > bws[2*idx_bw+1]):
                ax.plot([bws[2*idx_bw]/np.pi, max(AE_obj.z)/np.pi], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])
                ax.plot([min(AE_obj.z)/np.pi, bws[2*idx_bw+1]/np.pi], [b_val, b_val],color=colors_plot[idx_lam][idx_bw])
            # if not normal plot
            else:
                ax.plot([bws[2*idx_bw]/np.pi, bws[2*idx_bw+1]/np.pi], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])

    # now do plot as a function of bounce-angle
    walpha_bounceplot = []
    roots_bounceplot  = []
    wpsi_bounceplot   = []
    for lam_idx, lam_val in enumerate(AE_obj.lam):
        root_at_lam = AE_obj.roots[lam_idx]
        wpsi_at_lam = AE_obj.wpsi[lam_idx]
        walpha_at_lam= AE_obj.walpha[lam_idx]
        roots_bounceplot.extend(root_at_lam)
        for idx in range(len(wpsi_at_lam)):
            wpsi_bounceplot.extend([wpsi_at_lam[idx]])
            wpsi_bounceplot.extend([wpsi_at_lam[idx]])
            walpha_bounceplot.extend([walpha_at_lam[idx]])
            walpha_bounceplot.extend([walpha_at_lam[idx]])

    roots_ordered, wpsi_bounceplot = (list(t) for t in zip(*sorted(zip(roots_bounceplot, wpsi_bounceplot))))
    roots_ordered, walpha_bounceplot = (list(t) for t in zip(*sorted(zip(roots_bounceplot, walpha_bounceplot))))


    roots_ordered = [root/np.pi for root in roots_ordered]

    ax.plot(AE_obj.z/np.pi,AE_obj.modb,color='black',linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(roots_ordered, wpsi_bounceplot, 'cornflowerblue',linestyle='dashdot',label=r'$\hat{\omega}_\psi$')
    ax2.plot(roots_ordered, walpha_bounceplot, 'tab:green',linestyle='dashed',label=r'$\hat{\omega}_\alpha$')
    ax2.plot(AE_obj.z/np.pi,AE_obj.z*0.0,linestyle='dotted',color='black')
    ax.set_ylabel(r'$B$')
    ax2.set_ylabel(r'$\hat{\omega}_\alpha, \quad \hat{\omega}_\psi$')
    ax2.tick_params(axis='y', colors='black',direction='in')
    ax.set_xlabel(r'$z/\pi$')
    ax.tick_params(axis='both',direction='in')
    ax2.legend(loc='lower right')
    max_norm = 1.0
    cbar = plt.colorbar(cm.ScalarMappable(norm=mplc.Normalize(vmin=0.0, vmax=max_norm, clip=False), cmap=cm.plasma), ticks=[0, max_norm], ax=ax,location='bottom',label=r'$\widehat{A}_\lambda/\widehat{A}_{\lambda,\mathrm{max}}$') #'%.3f'
    cbar.ax.set_xticklabels([0, round(max_norm)])
    if save==True:
        plt.savefig(filename, format='png',
            #This is recommendation for publication plots
            dpi=1000)
    plt.show()



def plot_geom_nae(AE_obj):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,2,figsize=(3.0*3.0,2.5*3.0),tight_layout=True)
    ax[0,0].plot(AE_obj.phi,AE_obj.modb)
    ax[0,1].plot(AE_obj.phi,AE_obj.dldphi)
    ax[1,0].plot(AE_obj.phi,AE_obj.L2)
    ax[1,1].plot(AE_obj.phi,AE_obj.L1)
    ax[0,0].set_xlabel(r'$\phi$')
    ax[0,1].set_xlabel(r'$\phi$')
    ax[1,0].set_xlabel(r'$\phi$')
    ax[1,1].set_xlabel(r'$\phi$')
    ax[0,0].set_ylabel(r'$|B|$')
    ax[0,1].set_ylabel(r'$\mathrm{d} \ell / \mathrm{d} \phi$')
    ax[1,0].set_ylabel(r'$\frac{B \times \nabla B}{B^2}\cdot \nabla \alpha$')
    ax[1,1].set_ylabel(r'$\frac{ B \times \nabla B }{B^2}\cdot \nabla \psi$')
    plt.show()



######################################################
######################################################
######################################################