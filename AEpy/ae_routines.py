from scipy.special import erf
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from BAD import bounce_int
import numpy as np


def w_diamag(dlnndx,dlnTdx,z):
    return ( dlnndx/z + dlnTdx * ( 1.0 - 3.0 / (2.0 * z) ) )


def AE_per_lam_per_z(walpha,wpsi,wdia,tau_b,z):
    r"""
    The available energy per lambda per z.
    """
    geometry = ( wdia - walpha ) * walpha - wpsi**2 + np.sqrt( ( wdia - walpha )**2 + wpsi**2  ) * np.sqrt( walpha**2 + wpsi**2 )
    envelope = np.exp(-z) * np.power(z,5/2)
    jacobian = tau_b 
    val      = geometry * envelope * jacobian
    return val/(4*np.sqrt(np.pi))


def AE_per_lam(c0,c1,tau_b,walpha):
    r"""
    function containing the integral over z for exactly omnigenous systems.
    This is the available energy per lambda. 
    """
    condition1 = (c0>=0) and (c1<=0)
    condition2 = (c0>=0) and (c1>0)
    condition3 = (c0<0)  and (c1<0)
    ans = np.zeros(len(c1))
    ans[condition1]  = (2 * c0[condition1] - 5 * c1[condition1])
    ans[condition2]  = (2 * c0[condition2] - 5 * c1[condition2]) * erf(np.sqrt(c0[condition2]/c1[condition2])) + 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0[condition2] + 15 * c1[condition2] ) * np.sqrt(c0[condition2]/c1[condition2]) * np.exp( - c0[condition2]/c1[condition2] )
    ans[condition3]  = (2 * c0[condition3] - 5 * c1[condition3]) * (1 - erf(np.sqrt(c0[condition3]/c1[condition3]))) - 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0[condition3] + 15 * c1[condition3] ) * np.sqrt(c0[condition3]/c1[condition3]) * np.exp( - c0[condition3]/c1[condition3] )
    return 3/16*ans*tau_b*walpha**2


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




class AE_gist:
    r"""
    Class which calculates data related to AE. Contains several plotting 
    routines, useful for assessing drifts and spatial structure of AE.
    """
    def __init__(self, gist_data, lam_res=1000, quad=False, interp_kind='cubic',
                 norm='fluxtube_vol'):
        
        # import relevant data
        self.L1      = gist_data.L1 
        self.L2      = gist_data.L2
        self.sqrtg   = gist_data.sqrtg
        self.modb    = gist_data.modb
        self.theta   = gist_data.theta
        self.q0      = np.abs(gist_data.q0)
        try:
            self.my_dpdx = gist_data.my_dpdx
        except:
            print('my_dpdx is unavailable - defaulting to zero.')
            self.my_dpdx = 0.0

        # calculate drifts
        roots_list,wpsi_list,walpha_list,tau_b_list,lam_list,k2 = drift_from_gist(self.theta,
        self.modb,self.sqrtg,self.L1,self.L2,self.my_dpdx,lam_res,quad=quad,interp_kind=interp_kind)
        # assign to self
        self.roots  = roots_list
        self.wpsi   = wpsi_list
        self.walpha = walpha_list
        self.taub  = tau_b_list
        self.lam    = lam_list
        self.k2     = k2
        

    def calc_AE(self,omn,omt,omnigenous,Delta_x,Delta_y):
        # loop over all lambda
        L_tot  = np.trapz(self.sqrtg*self.modb,self.theta)
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
        ae_tot = np.trapz(ae_per_lam_summed,lam_arr)
        self.ae_tot     = Delta_x * Delta_y * ae_tot


    def calc_AE_fast(self,omn,omt,omnigenous,Delta_x,Delta_y):
        # loop over all lambda
        L_tot  = np.trapz(self.sqrtg*self.modb,self.theta)
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
        ae_tot = np.trapz(ae_per_lam_summed,lam_arr)
        self.ae_tot     = Delta_x * Delta_y * ae_tot


    def plot_precession(self,save=False,filename='AE_precession.eps'):
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
        walp_arr = np.nan*np.zeros([len(self.walpha),len(max(self.walpha,key = lambda x: len(x)))])
        for i,j in enumerate(self.walpha):
            walp_arr[i][0:len(j)] = j
        wpsi_arr = np.nan*np.zeros([len(self.wpsi),len(max(self.wpsi,key = lambda x: len(x)))])
        for i,j in enumerate(self.wpsi):
            wpsi_arr[i][0:len(j)] = j
        alp_l  = np.shape(walp_arr)[1]
        k2_arr = np.repeat(self.k2,alp_l)
        fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(2*3.5, 5.0))
        ax[1,0].scatter(k2_arr,walp_arr,s=0.2,marker='.',color='black',facecolors='black')
        ax[1,0].plot(self.k2,0.0*self.k2,color='red',linestyle='dashed')
        ax[1,1].scatter(k2_arr,wpsi_arr,s=0.2,marker='.',color='black',facecolors='black')
        ax[1,0].set_xlim(0,1)
        ax[1,1].set_xlim(0,1)
        ax[1,0].set_xlabel(r'$k^2$')
        ax[1,1].set_xlabel(r'$k^2$')
        ax[1,0].set_ylabel(r'$\langle \mathbf{v}_D \cdot \nabla y \rangle$',color='black')
        ax[1,1].set_ylabel(r'$\langle \mathbf{v}_D \cdot \nabla x \rangle$',color='black')


        # now do plot as a function of bounce-angle
        walpha_bounceplot = []
        roots_bounceplot  = []
        wpsi_bounceplot   = []
        for lam_idx, lam_val in enumerate(self.lam):
            root_at_lam = self.roots[lam_idx]
            wpsi_at_lam = self.wpsi[lam_idx]
            walpha_at_lam= self.walpha[lam_idx]
            roots_bounceplot.extend(root_at_lam)
            for idx in range(len(wpsi_at_lam)):
                wpsi_bounceplot.extend([wpsi_at_lam[idx]])
                wpsi_bounceplot.extend([wpsi_at_lam[idx]])
                walpha_bounceplot.extend([walpha_at_lam[idx]])
                walpha_bounceplot.extend([walpha_at_lam[idx]])

        roots_ordered, wpsi_bounceplot = (list(t) for t in zip(*sorted(zip(roots_bounceplot, wpsi_bounceplot))))
        roots_ordered, walpha_bounceplot = (list(t) for t in zip(*sorted(zip(roots_bounceplot, walpha_bounceplot))))
        ax[0,0].plot(self.theta,self.modb,color='black')
        ax001= ax[0,0].twinx()
        ax001.plot(roots_ordered,walpha_bounceplot,color='tab:blue')
        ax001.plot(np.asarray(roots_ordered),0.0*np.asarray(walpha_bounceplot),color='tab:red',linestyle='dashed')
        ax[0,1].plot(self.theta,self.modb,color='black')
        ax011= ax[0,1].twinx()
        ax011.plot(roots_ordered,wpsi_bounceplot,color='tab:blue')
        ax[0,0].set_xlim(self.theta.min(),self.theta.max())
        ax[0,1].set_xlim(self.theta.min(),self.theta.max())
        ax[0,0].set_xlabel(r'$\theta$')
        ax[0,1].set_xlabel(r'$\theta$')
        ax[0,0].set_ylabel(r'$B$')
        ax[0,1].set_ylabel(r'$B$')
        ax001.set_ylabel(r'$\langle \mathbf{v}_D \cdot \nabla y \rangle$',color='tab:blue')
        ax011.set_ylabel(r'$\langle \mathbf{v}_D \cdot \nabla x \rangle$',color='tab:blue')
        if save==True:
            plt.savefig(filename,dpi=1000)
        plt.show()

    
    def plot_AE_per_lam(self,save=False,filename='AE_per_lam.eps'):
        r"""
        Plots AE per bouncewell
        """
        import matplotlib.pyplot as plt
        import matplotlib        as mpl
        from    matplotlib   import cm
        import  matplotlib.colors   as      mplc
        plt.close('all')

        font = {'family': 'sans-serif',
                'weight': 'normal',
                'size': 10}

        mpl.rc('font', **font)
        c = 0.5
        fig ,ax = plt.subplots()
        fig.set_size_inches(5/6*6, 5/6*3.5)
        ax.set_xlim(min(self.theta)/np.pi,max(self.theta)/np.pi)

        lam_arr   = np.asarray(self.lam).flatten()
        ae_per_lam = self.ae_per_lam
        list_flat = []
        for val in ae_per_lam:
            list_flat.extend(val)
        max_ae_per_lam = max(list_flat)

        roots=self.roots

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
                    ax.plot([bws[2*idx_bw]/np.pi, max(self.theta)/np.pi], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])
                    ax.plot([min(self.theta)/np.pi, bws[2*idx_bw+1]/np.pi], [b_val, b_val],color=colors_plot[idx_lam][idx_bw])
                # if not normal plot
                else:
                    ax.plot([bws[2*idx_bw]/np.pi, bws[2*idx_bw+1]/np.pi], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])

        ax.plot(self.theta/np.pi,self.modb,color='black',linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(self.theta/np.pi, self.L2, 'red')
        ax.set_ylabel(r'$B$')
        ax2.set_ylabel(r'$\mathcal{L}_2$',color='red')
        ax2.plot(self.theta/np.pi,self.theta*0.0,linestyle='dashed',color='red')
        ax2.tick_params(axis='y', colors='black',labelcolor='red',direction='in')
        ax.set_xlabel(r'$\theta/\pi$')
        ax.tick_params(axis='both',direction='in')
        plt.subplots_adjust(left=0.1, right=0.88, top=0.99, bottom=0.08)
        cbar = plt.colorbar(cm.ScalarMappable(norm=mplc.Normalize(vmin=0.0, vmax=max_ae_per_lam, clip=False), cmap=cm.plasma), ticks=[0, max_ae_per_lam], ax=ax,location='bottom',label=r'$\widehat{A}_\lambda$') #'%.3f'
        cbar.ax.set_xticklabels([0, round(max_ae_per_lam, 1)])
        if save==True:
            plt.savefig(filename, format='png',
                #This is recommendation for publication plots
                dpi=1000)
        plt.show()