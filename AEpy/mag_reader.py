#!/usr/bin/env python3


### This reads various file formats containing magnetic field data
import f90nml
import numpy as np
from scipy.interpolate import interp1d


# Open the file for reading
def read_columns(file_name):
    "Entirely written by ChatGPT."
    # Open the file for reading
    with open(file_name) as f:
        # Find the index of the last line that contains the "/" character
        last_slash_index = None
        for i, line in enumerate(f):
            if "/" in line:
                last_slash_index = i

        # If no "/" character was found, raise an error
        if last_slash_index is None:
            raise ValueError("File does not contain '/' character")

        # Read the remaining lines in the file, starting from the last "/"
        f.seek(0)
        lines = f.readlines()[last_slash_index+1:]

        # Remove any empty lines or comments
        lines = [line for line in lines if line.strip() and not line.strip().startswith('!')]

        # Get the row and column count by counting the number of columns in the first line
        col_count = len(lines[0].split())

        # Create an empty numpy array of the required size
        arr = np.zeros((len(lines), col_count))

        # Loop through the lines in the file
        for i, line in enumerate(lines):
            # Split the line into numbers and convert them to floats
            nums = [float(x) for x in line.split()]
            # Store the numbers in the array
            arr[i, :] = nums

    # Return the array
    return arr



def periodic_extender(arr,l_max,r_max):
    arr_app     = arr[1:l_max+1]
    arr_pre     = arr[r_max:-1]
    arr_ext     = np.concatenate((arr_pre,arr,arr_app))
    return  arr_ext



class mag_data:
    """
    Reads magnetic field data. Assumes GIST file.
    """
    def __init__(self, file_name):

        # Let's set all the properties
        file = f90nml.read(file_name)
        params  = file['parameters']
        try:
            self.s0     = params['s0']
        except:
            print('s0 not defined')
        try:
            self.bref   = params['bref']
        except:
            print('Bref is not defined')
        try: 
            self.my_dpdx= params['my_dpdx']
        except:
            print('my_dpdx not defined')
        try:
            self.q0     = params['q0']
        except:
            print('q0 is not defined')
        try:
            self.shat   = params['shat']
        except:
            print('shat is not defined')
        try:
            self.gridpoints = params['gridpoints']
        except:
            print('Error: number of gridpoints needs to be defined')
        try:
            self.n_pol  = params['n_pol']
        except:
            print('Error: number of poloidal turns needs to be defined')
        data_arr    = read_columns(file_name)
        self.g11    = data_arr[:,0]
        self.g12    = data_arr[:,1]
        self.g22    = data_arr[:,2]
        self.modb   = data_arr[:,3]
        self.sqrtg  = data_arr[:,4]
        self.L2     = data_arr[:,5]
        self.L1     = data_arr[:,6]
        self.dBdz   = data_arr[:,7]
        self.theta  = np.linspace(-self.n_pol*np.pi,+self.n_pol*np.pi,self.gridpoints,endpoint=False)
        self._endpoint_included = False
        self._extended          = False

    def plot_geometry(self):
        """
        Plots domain saved in self. One can plot truncated/extended
        domain simply by doing 
        data = mag_data(filename)
        data.truncate_domain()
        data.plot_geometry()
        """
        import matplotlib.pyplot as plt
        import matplotlib        as mpl
        font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10}

        mpl.rc('font', **font)
        fig,axs = plt.subplots(2,4 ,tight_layout=True, figsize=(4*3.5, 2*2.5))
        axs[0,0].plot(self.theta/np.pi,self.g11,label='g11')
        axs[0,0].set_title(r'$g_{11}$')
        axs[0,1].plot(self.theta/np.pi,self.g12,label='g12')
        axs[0,1].set_title(r'$g_{12}$')
        axs[0,2].plot(self.theta/np.pi,self.g22,label='g22')
        axs[0,2].set_title(r'$g_{22}$')
        axs[0,3].plot(self.theta/np.pi,self.modb,label='modB')
        axs[0,3].set_title(r'$|B|$')
        axs[1,0].plot(self.theta/np.pi,self.L1,label='L1')
        axs[1,0].set_title(r'$\mathcal{L}_1$')
        axs[1,1].plot(self.theta/np.pi,self.L2,label='L2')
        axs[1,1].set_title(r'$\mathcal{L}_2$')
        axs[1,2].plot(self.theta/np.pi,self.sqrtg,label='sqrtg')
        axs[1,2].set_title(r'$\sqrt{g}$')
        axs[1,3].plot(self.theta/np.pi,self.dBdz,label='dBdz')
        axs[1,3].set_title(r'$\partial_z B$')
        axs[1,0].set_xlabel(r'$\theta/\pi$')
        axs[1,1].set_xlabel(r'$\theta/\pi$')
        axs[1,2].set_xlabel(r'$\theta/\pi$')
        axs[1,3].set_xlabel(r'$\theta/\pi$')
        plt.show()

    def include_endpoint(self):
        if self._endpoint_included==False:
            # assumes stellarator symmetry
            self.g11    = np.append(self.g11,self.g11[0])
            self.g12    = np.append(self.g12,-1*self.g12[0])
            self.g22    = np.append(self.g22,self.g22[0])
            self.modb   = np.append(self.modb,self.modb[0])
            self.sqrtg  = np.append(self.sqrtg,self.sqrtg[0])
            self.L2     = np.append(self.L2,self.L2[0])
            self.L1     = np.append(self.L1,-1*self.L1[0])
            self.dBdz   = np.append(self.dBdz,-1*self.dBdz[0])
            self.theta  = np.append(self.theta,-1*self.theta[0])
            self.gridpoint = self.gridpoints+1
            self._endpoint_included=True

    def extend_domain(self):
        """
        Extends the domain up to B_max on both sides,
        using the quasi-periodic boundary condition.
        """
        if self._extended==False:
            self.include_endpoint()
            # to enforce the quasiperiodic boundary condition we simply extend the domain
            # we first find all the positions where the magnetic field is maximal
            max_idx = np.asarray(np.argwhere(self.modb == np.amax(self.modb))).flatten()
            l_max   = max_idx[0]
            r_max   = max_idx[-1]

            # make extended theta_arr 
            the_app     = self.theta[1:l_max+1] - self.theta[0] +   self.theta[-1]
            the_pre     = self.theta[r_max:-1]  - self.theta[-1]+   self.theta[0]
            theta_ext   = np.append(self.theta,the_app)
            theta_ext   = np.concatenate((the_pre,theta_ext))

            # make extended g11
            g11_ext     = periodic_extender(self.g11,l_max,r_max)

            # use relations for nonperiodic functions
            secarr      = self.shat*self.theta*self.g11
            g12arr      = self.g12
            # construct periodic part
            g12per      = g12arr-secarr
            # make extended periodic array
            g12per_ext  = periodic_extender(g12per,l_max,r_max)
            # now construct extended g12
            g12_ext     = g12per_ext + self.shat*theta_ext*g11_ext
            
            # now construct extended L2 
            kappag      = self.L1 / np.sqrt(self.g11)
            L2sec       =-kappag * g12arr * self.modb/ np.sqrt(self.g11)
            # subtracting the quasi-periodic part should results in a periodic function
            L2per       = self.L2 - L2sec
            L2per_ext   = periodic_extender(L2per,l_max,r_max)
            # make extended periodic array
            kappag_ext  = periodic_extender(kappag,l_max,r_max)
            modb_ext    = periodic_extender(self.modb,l_max,r_max)
            # now construct L2_ext 
            L2sec_ext   = -kappag_ext*g12_ext*modb_ext/np.sqrt(g11_ext)
            L2_ext      = L2per_ext  + L2sec_ext

            # construct g22
            g22_ext     = (modb_ext**2 + g12_ext**2)/g11_ext**2


            # assign to self 
            self.theta  = theta_ext
            self.g11    = g11_ext
            self.g12    = g12_ext
            self.modb   = modb_ext
            self.sqrtg  = periodic_extender(self.sqrtg,l_max,r_max)
            self.L2     = L2_ext
            self.L1     = periodic_extender(self.L1,l_max,r_max)
            self.dBdz   = periodic_extender(self.dBdz,l_max,r_max)
            self.g22    = g22_ext
            self._extended = True


    def truncate_domain(self):
        """
        Truncates domain between two B_max's.
        Assumes there are at least two B_max of equal value.
        If not, the arrays become of length one.
        """
        self.include_endpoint()
        # to enforce the quasiperiodic boundary condition we simply extend the domain
        # we first find all the positions where the magnetic field is maximal
        max_idx = np.asarray(np.argwhere(self.modb == np.amax(self.modb))).flatten()
        l_max   = max_idx[0]
        r_max   = max_idx[-1]

        
        self.theta  = self.theta[l_max:r_max+1]
        self.g11    = self.g11[l_max:r_max+1]
        self.g12    = self.g12[l_max:r_max+1]
        self.modb   = self.modb[l_max:r_max+1]
        self.sqrtg  = self.sqrtg[l_max:r_max+1]
        self.L2     = self.L2[l_max:r_max+1]
        self.L1     = self.L1[l_max:r_max+1]
        self.dBdz   = self.dBdz[l_max:r_max+1]
        self.g22    = self.g22[l_max:r_max+1]


    def refine_profiles(self,res,interp_kind='cubic'):
        """
        Uses interpolation to refine profiles.
        Resolution of refined profiles is res,
        interpolation kind is interp_kin.
        Highly recommended to keep res an odd number.
        """
        theta_new = np.linspace(self.theta.min(),self.theta.max(),res)
        g11_f   = interp1d(self.theta,self.g11,kind=interp_kind)
        g12_f   = interp1d(self.theta,self.g12,kind=interp_kind)
        g22_f   = interp1d(self.theta,self.g22,kind=interp_kind)
        modb_f  = interp1d(self.theta,self.modb,kind=interp_kind)
        sqrtg_f = interp1d(self.theta,self.sqrtg,kind=interp_kind)
        L2_f    = interp1d(self.theta,self.L2,kind=interp_kind)
        L1_f    = interp1d(self.theta,self.L1,kind=interp_kind)
        dBdz_f  = interp1d(self.theta,self.dBdz,kind=interp_kind)

        self.theta  = theta_new
        self.g11    = g11_f(theta_new)
        self.g12    = g12_f(theta_new)
        self.modb   = modb_f(theta_new)
        self.sqrtg  = sqrtg_f(theta_new)
        self.L2     = L2_f(theta_new)
        self.L1     = L1_f(theta_new)
        self.dBdz   = dBdz_f(theta_new)
        self.g22    = g22_f(theta_new)


