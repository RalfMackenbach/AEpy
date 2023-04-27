#!/usr/bin/env python3


### This reads various file formats containing magnetic field data
import f90nml
import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from simsopt.mhd.boozer import Boozer
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import vmec_splines, vmec_fieldlines
from simsopt._core.util import Struct


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

def boozxform_splines(boozer, vmec):
    """
    Initialize radial splines for a boozxform equilibrium.

    Args:
        boozer: An instance of :obj:`simsopt.mhd.boozer.Boozer`.
        vmec: An instance of :obj:`simsopt.mhd.vmec.Vmec'.

    Returns:
        A structure with the splines as attributes.
    """
    boozer.run()
    results = Struct()

    rmnc = []
    zmns = []
    numns = []
    d_rmnc_d_s = []
    d_zmns_d_s = []
    d_numns_d_s = []
    s_full_grid = boozer.bx.compute_surfs/boozer.bx.ns_in
    for jmn in range(boozer.bx.mnboz):
        rmnc.append(InterpolatedUnivariateSpline(s_full_grid, boozer.bx.rmnc_b[jmn, :]))
        zmns.append(InterpolatedUnivariateSpline(s_full_grid, boozer.bx.zmns_b[jmn, :]))
        numns.append(InterpolatedUnivariateSpline(s_full_grid, boozer.bx.numns_b[jmn, :]))
        d_rmnc_d_s.append(rmnc[-1].derivative())
        d_zmns_d_s.append(zmns[-1].derivative())
        d_numns_d_s.append(numns[-1].derivative())

    gmnc = []
    bmnc = []
    d_bmnc_d_s = []
    for jmn in range(boozer.bx.mnboz):
        gmnc.append(InterpolatedUnivariateSpline(s_full_grid[1:], boozer.bx.gmnc_b[jmn, 1:]))
        bmnc.append(InterpolatedUnivariateSpline(s_full_grid[1:], boozer.bx.bmnc_b[jmn, 1:]))
        d_bmnc_d_s.append(bmnc[-1].derivative())

    # Handle 1d profiles:
    results.pressure = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.pres[1:])
    results.d_pressure_d_s = results.pressure.derivative()
    results.iota = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.iotas[1:])
    results.d_iota_d_s = results.iota.derivative()
    results.I = InterpolatedUnivariateSpline(s_full_grid[1:], boozer.bx.Boozer_I[1:])
    results.d_I_d_s = results.I.derivative()
    results.G = InterpolatedUnivariateSpline(s_full_grid[1:], boozer.bx.Boozer_G[1:])
    results.d_G_d_s = results.G.derivative()

    # Save other useful quantities:
    results.phiedge = vmec.wout.phi[-1]
    variables = ['mnboz', 'xm_b', 'xn_b', 'nfp']
    for v in variables:
        results.__setattr__(v, eval('boozer.bx.' + v))

    variables = ['rmnc', 'zmns', 'numns', 'd_rmnc_d_s', 'd_zmns_d_s', 'd_numns_d_s',
                 'gmnc', 'bmnc', 'd_bmnc_d_s']
    for v in variables:
        results.__setattr__(v, eval(v))

    return results

def boozxform_fieldlines(vs, bs, s, alpha, theta1d=None, phi1d=None, phi_center=0, phi_sec = 0, plot=False, show=True, save = False):
    r"""
    Compute field lines in a vmec configuration, and compute many
    geometric quantities of interest along the field lines. In
    particular, this routine computes the geometric quantities that
    enter the gyrokinetic equation.

    One of the tasks performed by this function is to convert between
    the poloidal angles :math:`\theta_{vmec}` and
    :math:`\theta_{pest}`. The latter is the angle in which the field
    lines are straight when used in combination with the standard
    toroidal angle :math:`\phi`. Note that all angles in this function
    have period :math:`2\pi`, not period 1.

    For the inputs and outputs of this function, a field line label
    coordinate is defined by

    .. math::

        \alpha = \theta_{pest} - \iota (\phi - \phi_{center}).

    Here, :math:`\phi_{center}` is a constant, usually 0, which can be
    set to a nonzero value if desired so the magnetic shear
    contribution to :math:`\nabla\alpha` vanishes at a toroidal angle
    different than 0.  Also, wherever the term ``psi`` appears in
    variable names in this function and the returned arrays, it means
    :math:`\psi =` the toroidal flux divided by :math:`2\pi`, so

    .. math::

        \vec{B} = \nabla\psi\times\nabla\theta_{pest} + \iota\nabla\phi\times\nabla\psi = \nabla\psi\times\nabla\alpha.

    To specify the parallel extent of the field lines, you can provide
    either a grid of :math:`\theta_{pest}` values or a grid of
    :math:`\phi` values. If you specify both or neither, ``ValueError``
    will be raised.

    Most of the arrays that are computed have shape ``(ns, nalpha,
    nl)``, where ``ns`` is the number of flux surfaces, ``nalpha`` is the
    number of field lines on each flux surface, and ``nl`` is the number
    of grid points along each field line. In other words, ``ns`` is the
    size of the input ``s`` array, ``nalpha`` is the size of the input
    ``alpha`` array, and ``nl`` is the size of the input ``theta1d`` or
    ``phi1d`` array. The output arrays are returned as attributes of the
    returned object. Many intermediate quantities are included, such
    as the Cartesian components of the covariant and contravariant
    basis vectors. Some of the most useful of these output arrays are (all with SI units):

    - ``phi``: The standard toroidal angle :math:`\phi`.
    - ``theta_vmec``: VMEC's poloidal angle :math:`\theta_{vmec}`.
    - ``theta_pest``: The straight-field-line angle :math:`\theta_{pest}` associated with :math:`\phi`.
    - ``modB``: The magnetic field magnitude :math:`|B|`.
    - ``B_sup_theta_vmec``: :math:`\vec{B}\cdot\nabla\theta_{vmec}`.
    - ``B_sup_phi``: :math:`\vec{B}\cdot\nabla\phi`.
    - ``B_cross_grad_B_dot_grad_alpha``: :math:`\vec{B}\times\nabla|B|\cdot\nabla\alpha`.
    - ``B_cross_grad_B_dot_grad_psi``: :math:`\vec{B}\times\nabla|B|\cdot\nabla\psi`.
    - ``B_cross_kappa_dot_grad_alpha``: :math:`\vec{B}\times\vec{\kappa}\cdot\nabla\alpha`,
      where :math:`\vec{\kappa}=\vec{b}\cdot\nabla\vec{b}` is the curvature and :math:`\vec{b}=|B|^{-1}\vec{B}`.
    - ``B_cross_kappa_dot_grad_psi``: :math:`\vec{B}\times\vec{\kappa}\cdot\nabla\psi`.
    - ``grad_alpha_dot_grad_alpha``: :math:`|\nabla\alpha|^2 = \nabla\alpha\cdot\nabla\alpha`.
    - ``grad_alpha_dot_grad_psi``: :math:`\nabla\alpha\cdot\nabla\psi`.
    - ``grad_psi_dot_grad_psi``: :math:`|\nabla\psi|^2 = \nabla\psi\cdot\nabla\psi`.
    - ``iota``: The rotational transform :math:`\iota`. This array has shape ``(ns,)``.
    - ``shat``: The magnetic shear :math:`\hat s= (x/q) (d q / d x)` where 
      :math:`x = \mathrm{Aminor_p} \, \sqrt{s}` and :math:`q=1/\iota`. This array has shape ``(ns,)``.

    The following normalized versions of these quantities used in the
    gyrokinetic codes ``stella``, ``gs2``, and ``GX`` are also
    returned: ``bmag``, ``gbdrift``, ``gbdrift0``, ``cvdrift``,
    ``cvdrift0``, ``gds2``, ``gds21``, and ``gds22``, along with
    ``L_reference`` and ``B_reference``.  Instead of ``gradpar``, two
    variants are returned, ``gradpar_theta_pest`` and ``gradpar_phi``,
    corresponding to choosing either :math:`\theta_{pest}` or
    :math:`\phi` as the parallel coordinate.

    The value(s) of ``s`` provided as input need not coincide with the
    full grid or half grid in VMEC, as spline interpolation will be
    used radially.

    The implementation in this routine is similar to the one in the
    gyrokinetic code ``stella``.

    Example usage::

        import numpy as np
        from simsopt.mhd.vmec import Vmec
        from simsopt.mhd.vmec_diagnostics import vmec_fieldlines

        v = Vmec('wout_li383_1.4m.nc')
        theta = np.linspace(-np.pi, np.pi, 50)
        fl = vmec_fieldlines(v, 0.5, 0, theta1d=theta)
        print(fl.B_cross_grad_B_dot_grad_alpha)

    Args:
        vs: Either an instance of :obj:`simsopt.mhd.vmec.Vmec`
          or the structure returned by :func:`vmec_splines`.
        s: Values of normalized toroidal flux on which to construct the field lines.
          You can give a single number, or a list or numpy array.
        alpha: Values of the field line label :math:`\alpha` on which to construct the field lines.
          You can give a single number, or a list or numpy array.
        theta1d: 1D array of :math:`\theta_{pest}` values, setting the grid points
          along the field line and the parallel extent of the field line.
        phi1d: 1D array of :math:`\phi` values, setting the grid points along the
          field line and the parallel extent of the field line.
        phi_center: :math:`\phi_{center}`, an optional shift to the toroidal angle
          in the definition of :math:`\alpha`.
        plot: Whether to create a plot of the main geometric quantities. Only one field line will
          be plotted, corresponding to the leading elements of ``s`` and ``alpha``.
        show: Only matters if ``plot==True``. Whether to call matplotlib's ``show()`` function
          after creating the plot.
    """

    # If given a Boozer object, convert it to vmec_splines:
    if isinstance(bs, Boozer):
        bs = boozxform_splines(bs, vs)

    # Make sure s is an array:
    try:
        ns = len(s)
    except:
        s = [s]
    s = np.array(s)
    ns = len(s)

    # Make sure alpha is an array
    try:
        nalpha = len(alpha)
    except:
        alpha = [alpha]
    alpha = np.array(alpha)
    nalpha = len(alpha)

    if (theta1d is not None) and (phi1d is not None):
        raise ValueError('You cannot specify both theta and phi')
    if (theta1d is None) and (phi1d is None):
        raise ValueError('You must specify either theta or phi')
    if theta1d is None:
        nl = len(phi1d)
    else:
        nl = len(theta1d)

    # Shorthand:
    mnboz = bs.mnboz
    xm = bs.xm_b
    xn = bs.xn_b

    # Note the minus sign. psi in the straight-field-line relation seems to have opposite sign to vmec's phi array.
    # This follows from making the handedness of the Boozer frame match that of the VMEC
    edge_toroidal_flux_over_2pi = -bs.phiedge / (2 * np.pi)

    # Now that we have an s grid, evaluate everything on that grid:
    d_pressure_d_s = bs.d_pressure_d_s(s)
    iota = bs.iota(s)
    d_iota_d_s = bs.d_iota_d_s(s)
    d_iota_d_psi = d_iota_d_s / edge_toroidal_flux_over_2pi
    I = -bs.I(s) # Negative sign for theta_b opposite to theta_vmec
    d_I_d_s = bs.d_I_d_s(s)
    G = bs.G(s)
    d_G_d_s = bs.d_G_d_s(s)
    
    # shat = (r/q)(dq/dr) where r = a sqrt(s)
    #      = - (r/iota) (d iota / d r) = -2 (s/iota) (d iota / d s)
    # shat = (-2 * s / iota) * d_iota_d_s

    rmnc = np.zeros((ns, mnboz))
    zmns = np.zeros((ns, mnboz))
    numns = np.zeros((ns, mnboz))
    d_rmnc_d_s = np.zeros((ns, mnboz))
    d_zmns_d_s = np.zeros((ns, mnboz))
    d_numns_d_s = np.zeros((ns, mnboz))
    for jmn in range(mnboz):
        rmnc[:, jmn] = bs.rmnc[jmn](s)
        zmns[:, jmn] = bs.zmns[jmn](s)
        numns[:, jmn] = bs.numns[jmn](s)
        d_rmnc_d_s[:, jmn] = bs.d_rmnc_d_s[jmn](s)
        d_zmns_d_s[:, jmn] = bs.d_zmns_d_s[jmn](s)
        d_numns_d_s[:, jmn] = bs.d_numns_d_s[jmn](s)

    gmnc = np.zeros((ns, mnboz))
    bmnc = np.zeros((ns, mnboz))
    d_bmnc_d_s = np.zeros((ns, mnboz))
    for jmn in range(mnboz):
        gmnc[:, jmn] = bs.gmnc[jmn](s)
        bmnc[:, jmn] = bs.bmnc[jmn](s)
        d_bmnc_d_s[:, jmn] = bs.d_bmnc_d_s[jmn](s)
        
    theta_booz = np.zeros((ns, nalpha, nl))
    phi_booz = np.zeros((ns, nalpha, nl))

    if theta1d is None:
        # We are given phi_booz. Compute theta_booz:
        for js in range(ns):
            phi_booz[js, :, :] = phi1d[None, :]
            theta_booz[js, :, :] = alpha[:, None] + iota[js] * (phi1d[None, :] - phi_center)
    else:
        # We are given theta_booz. Compute phi_booz:
        for js in range(ns):
            theta_booz[js, :, :] = theta1d[None, :]
            phi_booz[js, :, :] = phi_center + (theta1d[None, :] - alpha[:, None]) / iota[js]

    # Now that we know phi_vmec, compute all the geometric quantities
    angle = xm[:, None, None, None] * theta_booz[None, :, :, :] - xn[:, None, None, None] * phi_booz[None, :, :, :]
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mcosangle = xm[:, None, None, None] * cosangle
    ncosangle = xn[:, None, None, None] * cosangle
    msinangle = xm[:, None, None, None] * sinangle
    nsinangle = xn[:, None, None, None] * sinangle
    # Order of indices in cosangle and sinangle: mn, s, alpha, l
    # Order of indices in rmnc, bmnc, etc: s, mn
    R = np.einsum('ij,jikl->ikl', rmnc, cosangle)
    d_R_d_s = np.einsum('ij,jikl->ikl', d_rmnc_d_s, cosangle)
    d_R_d_theta_b = -np.einsum('ij,jikl->ikl', rmnc, msinangle)
    d_R_d_phi_b = np.einsum('ij,jikl->ikl', rmnc, nsinangle)

    Z = np.einsum('ij,jikl->ikl', zmns, sinangle)
    d_Z_d_s = np.einsum('ij,jikl->ikl', d_zmns_d_s, sinangle)
    d_Z_d_theta_b = np.einsum('ij,jikl->ikl', zmns, mcosangle)
    d_Z_d_phi_b = -np.einsum('ij,jikl->ikl', zmns, ncosangle)

    nu = np.einsum('ij,jikl->ikl', numns, sinangle)
    d_numns_d_s = np.einsum('ij,jikl->ikl', d_numns_d_s, sinangle)
    d_numns_d_theta_b = np.einsum('ij,jikl->ikl', numns, mcosangle)
    d_numns_d_phi_b = -np.einsum('ij,jikl->ikl', numns, ncosangle)

    phi_vmec = phi_booz - nu

    sqrt_g_b = np.einsum('ij,jikl->ikl', gmnc, cosangle)
    modB = np.einsum('ij,jikl->ikl', bmnc, cosangle)
    d_B_d_s = np.einsum('ij,jikl->ikl', d_bmnc_d_s, cosangle)
    d_B_d_theta_b = -np.einsum('ij,jikl->ikl', bmnc, msinangle)
    d_B_d_phi_b = np.einsum('ij,jikl->ikl', bmnc, nsinangle)

    # sqrt_g_vmec_alt = R * (d_Z_d_s * d_R_d_theta_vmec - d_R_d_s * d_Z_d_theta_vmec)

    # *********************************************************************
    # Using R(theta,phi) and Z(theta,phi), compute the Cartesian
    # components of the gradient basis vectors using the dual relations:
    # *********************************************************************
    sinphi = np.sin(phi_vmec)
    cosphi = np.cos(phi_vmec)
    # X = R * cos(phi_vmec):
    # d_phi_vmec_d_theta_b = -d_numns_d_theta_b 
    # d_phi_vmec_d_phi_b = 1-d_numns_d_phi_b  
    d_X_d_theta_b = d_R_d_theta_b * cosphi + R * d_numns_d_theta_b * sinphi
    d_X_d_phi_b = d_R_d_phi_b * cosphi - R * sinphi * (1 - d_numns_d_phi_b)
    d_X_d_s = d_R_d_s * cosphi + R * sinphi * d_numns_d_s
    # Y = R * sin(phi):
    d_Y_d_theta_b = d_R_d_theta_b * sinphi - R * d_numns_d_theta_b * cosphi
    d_Y_d_phi_b = d_R_d_phi_b * sinphi + R * cosphi * (1 - d_numns_d_phi_b)
    d_Y_d_s = d_R_d_s * sinphi - R * cosphi * d_numns_d_s

    jac_alt = 0 \
        + d_X_d_s * d_Y_d_theta_b * d_Z_d_phi_b \
        + d_Y_d_s * d_Z_d_theta_b * d_X_d_phi_b \
        + d_Z_d_s * d_X_d_theta_b * d_Y_d_phi_b \
        - d_Z_d_s * d_Y_d_theta_b * d_X_d_phi_b \
        - d_X_d_s * d_Z_d_theta_b * d_Y_d_phi_b \
        - d_Y_d_s * d_X_d_theta_b * d_Z_d_phi_b
    
    # print('Residual jacobian: ',np.sum(np.squeeze(jac_alt / edge_toroidal_flux_over_2pi - sqrt_g_b)))
    # Now use the dual relations to get the Cartesian components of grad s, grad theta_vmec, and grad phi:
    grad_s_X = (d_Y_d_theta_b * d_Z_d_phi_b - d_Z_d_theta_b * d_Y_d_phi_b) / (sqrt_g_b * edge_toroidal_flux_over_2pi)
    grad_s_Y = (d_Z_d_theta_b * d_X_d_phi_b - d_X_d_theta_b * d_Z_d_phi_b) / (sqrt_g_b * edge_toroidal_flux_over_2pi)
    grad_s_Z = (d_X_d_theta_b * d_Y_d_phi_b - d_Y_d_theta_b * d_X_d_phi_b) / (sqrt_g_b * edge_toroidal_flux_over_2pi)

    grad_theta_b_X = (d_Y_d_phi_b * d_Z_d_s - d_Z_d_phi_b * d_Y_d_s) / (sqrt_g_b * edge_toroidal_flux_over_2pi)
    grad_theta_b_Y = (d_Z_d_phi_b * d_X_d_s - d_X_d_phi_b * d_Z_d_s) / (sqrt_g_b * edge_toroidal_flux_over_2pi)
    grad_theta_b_Z = (d_X_d_phi_b * d_Y_d_s - d_Y_d_phi_b * d_X_d_s) / (sqrt_g_b * edge_toroidal_flux_over_2pi)

    grad_phi_b_X = (d_Y_d_s * d_Z_d_theta_b - d_Z_d_s * d_Y_d_theta_b) / (sqrt_g_b * edge_toroidal_flux_over_2pi)
    grad_phi_b_Y = (d_Z_d_s * d_X_d_theta_b - d_X_d_s * d_Z_d_theta_b) / (sqrt_g_b * edge_toroidal_flux_over_2pi)
    grad_phi_b_Z = (d_X_d_s * d_Y_d_theta_b - d_Y_d_s * d_X_d_theta_b) / (sqrt_g_b * edge_toroidal_flux_over_2pi)
    # End of dual relations.

    # *********************************************************************
    # Compute the Cartesian components of other quantities we need:
    # *********************************************************************

    grad_psi_X = grad_s_X * edge_toroidal_flux_over_2pi
    grad_psi_Y = grad_s_Y * edge_toroidal_flux_over_2pi
    grad_psi_Z = grad_s_Z * edge_toroidal_flux_over_2pi

    grad_B_X = d_B_d_s * grad_s_X + d_B_d_theta_b * grad_theta_b_X + d_B_d_phi_b * grad_phi_b_X
    grad_B_Y = d_B_d_s * grad_s_Y + d_B_d_theta_b * grad_theta_b_Y + d_B_d_phi_b * grad_phi_b_Y
    grad_B_Z = d_B_d_s * grad_s_Z + d_B_d_theta_b * grad_theta_b_Z + d_B_d_phi_b * grad_phi_b_Z

    B_X = (d_X_d_phi_b + iota[:, None, None] * d_X_d_theta_b) / sqrt_g_b
    B_Y = (d_Y_d_phi_b + iota[:, None, None] * d_Y_d_theta_b) / sqrt_g_b
    B_Z = (d_Z_d_phi_b + iota[:, None, None] * d_Z_d_theta_b) / sqrt_g_b

    # *********************************************************************
    # For gbdrift, we need \vect{B} cross grad |B| dot grad alpha.
    # For cvdrift, we also need \vect{B} cross grad s dot grad alpha.
    # Let us compute both of these quantities 2 ways, and make sure the two
    # approaches give the same answer (within some tolerance).
    # *********************************************************************

    B_cross_grad_B_dot_grad_psi = (I * d_B_d_phi_b - G * d_B_d_theta_b) / sqrt_g_b

    B_cross_grad_B_dot_grad_thphi = 0 \
        + B_X * grad_B_Y * (grad_theta_b_Z - iota[:, None, None] * grad_phi_b_Z) \
        + B_Y * grad_B_Z * (grad_theta_b_X - iota[:, None, None] * grad_phi_b_X) \
        + B_Z * grad_B_X * (grad_theta_b_Y - iota[:, None, None] * grad_phi_b_Y) \
        - B_Z * grad_B_Y * (grad_theta_b_X - iota[:, None, None] * grad_phi_b_X) \
        - B_X * grad_B_Z * (grad_theta_b_Y - iota[:, None, None] * grad_phi_b_Y) \
        - B_Y * grad_B_X * (grad_theta_b_Z - iota[:, None, None] * grad_phi_b_Z)
    
    B_cross_grad_B_dot_grad_alpha = B_cross_grad_B_dot_grad_thphi - d_iota_d_psi[:, None, None]*(phi_booz - phi_sec)*B_cross_grad_B_dot_grad_psi

    B_cross_kappa_dot_grad_psi = B_cross_grad_B_dot_grad_psi / modB

    mu_0 = 4 * np.pi * (1.0e-7)
    B_cross_kappa_dot_grad_alpha = B_cross_grad_B_dot_grad_alpha / modB + mu_0 * d_pressure_d_s[:, None, None] / edge_toroidal_flux_over_2pi

    B_dot_grad_B = B_X * grad_B_X + B_Y * grad_B_Y + B_Z * grad_B_Z

    grad_thphi_dot_grad_thphi = (grad_theta_b_X - iota[:, None, None] * grad_phi_b_X) * (grad_theta_b_X - iota[:, None, None] * grad_phi_b_X) +\
        (grad_theta_b_Y - iota[:, None, None] * grad_phi_b_Y) * (grad_theta_b_Y - iota[:, None, None] * grad_phi_b_Y) + \
        (grad_theta_b_Z - iota[:, None, None] * grad_phi_b_Z) * (grad_theta_b_Z - iota[:, None, None] * grad_phi_b_Z)

    grad_thphi_dot_grad_psi = (grad_theta_b_X - iota[:, None, None] * grad_phi_b_X) * grad_psi_X + \
        (grad_theta_b_Y - iota[:, None, None] * grad_phi_b_Y) * grad_psi_Y + \
        (grad_theta_b_Z - iota[:, None, None] * grad_phi_b_Z) * grad_psi_Z

    grad_psi_dot_grad_psi = grad_psi_X * grad_psi_X + grad_psi_Y * grad_psi_Y + grad_psi_Z * grad_psi_Z

    mu_0 = 4 * np.pi * (1.0e-7)
    

    # C components
    B_alpha = G + iota[:, None, None] * I

    L_reference = vs.wout.Aminor_p
    B_reference = 2 * abs(edge_toroidal_flux_over_2pi) / (L_reference * L_reference)
    toroidal_flux_sign = np.sign(edge_toroidal_flux_over_2pi)
    sqrt_s = np.sqrt(s)

    # Package results into a structure to return:
    results = Struct()
    variables = ['ns', 'nalpha', 'nl', 's', 'iota', 'd_iota_d_psi', 'd_pressure_d_s', 'G', 'I',
                 'alpha', 'theta_booz', 'phi_booz', 'phi_vmec',
                 'sqrt_g_b',
                 'L_reference', 'B_reference', 'toroidal_flux_sign', 'sqrt_s',
                 'modB', 'd_B_d_s', 'd_B_d_theta_b', 'd_B_d_phi_b',
                 'edge_toroidal_flux_over_2pi', 'sinphi', 'cosphi',
                 'R', 'd_R_d_s', 'd_R_d_theta_b', 'd_R_d_phi_b', 'Z', 'd_Z_d_s', 'd_Z_d_theta_b', 'd_Z_d_phi_b',
                 'd_X_d_theta_b', 'd_X_d_phi_b', 'd_X_d_s', 'd_Y_d_theta_b', 'd_Y_d_phi_b', 'd_Y_d_s',
                 'grad_s_X', 'grad_s_Y', 'grad_s_Z', 'grad_theta_b_X', 'grad_theta_b_Y', 'grad_theta_b_Z',
                 'grad_phi_b_X', 'grad_phi_b_Y', 'grad_phi_b_Z', 'grad_psi_X', 'grad_psi_Y', 'grad_psi_Z',
                 'grad_B_X', 'grad_B_Y', 'grad_B_Z',
                 'B_X', 'B_Y', 'B_Z', 'B_dot_grad_B',
                 'B_cross_grad_B_dot_grad_psi', 'B_cross_grad_B_dot_grad_thphi', 'B_cross_grad_B_dot_grad_alpha',
                 'B_cross_kappa_dot_grad_psi', 'B_cross_kappa_dot_grad_alpha',
                 'grad_thphi_dot_grad_thphi', 'grad_thphi_dot_grad_psi', 'grad_psi_dot_grad_psi']

    for v in variables:
        results.__setattr__(v, eval(v))

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(13, 7))
        nrows = 3
        ncols = 3
        variables = ['modB', 
                    #  'B_cross_grad_B_dot_grad_thphi', 'B_cross_grad_B_dot_grad_psi',
                     'F_p', 'G_p', 'F_tot', 
                    #  'rho',
                    #  'grad_thphi_dot_grad_thphi', 'grad_thphi_dot_grad_psi', 'grad_psi_dot_grad_psi',
                     'C_p', 'C_psi', 'C_q', 'C_tot', 'crit']
        def name_var(var_name):
            if var_name == "modB":
                return r"$|\mathbf{B}|$"
            elif var_name == "B_cross_grad_B_dot_grad_thphi":
                return r"$\mathbf{B}\times\nabla B\cdot(\nabla\theta-\iota\nabla\varphi)$"
            elif var_name == "B_cross_grad_B_dot_grad_psi":
                return r"$\mathbf{B}\times\nabla B\cdot\nabla\psi$"
            elif var_name == "F_p":
                return r"$\mathcal{F}/|\mu_0 p'|$"
            elif var_name == "G_p":
                return r"$\mathcal{G}/|\mu_0 p'|$"
            elif var_name == "F_tot":
                return r"$F$"
            elif var_name == "rho":
                return r"$\tilde{\rho}$"
            elif var_name == "grad_thphi_dot_grad_thphi":
                return r"$|\nabla\theta-\iota\nabla\varphi|^2$"
            elif var_name == "grad_thphi_dot_grad_psi":
                return r"$\nabla \psi\cdot(\nabla\theta-\iota\nabla\varphi)$"
            elif var_name == "grad_psi_dot_grad_psi":
                return r"$|\nabla \psi|^2$"
            elif var_name == "C_p":
                return r"$C_p$"
            elif var_name == "C_psi":
                return r"$C_\psi$"
            elif var_name == "C_q":
                return r"$C_q$"
            elif var_name == "C_tot":
                return r"$C$"
            elif var_name == "crit":
                return r"$F/C$"
            
        for j, variable in enumerate(variables):
            ax = plt.subplot(nrows, ncols, j + 1)
            plt.plot(phi_booz[0, 0, :], eval(variable + '[0, 0, :]'), linewidth = 1.5, color = 'k')
            plt.xlabel(r'$\varphi$')
            plt.title(name_var(variable))
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(1.5)
        plt.figtext(0.5, 0.995, f'$s={s[0]}, \\alpha={alpha[0]}$', ha='center', va='top')
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi = 300)
        if show:
            plt.show()

    return results

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


