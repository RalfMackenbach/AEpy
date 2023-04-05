import numpy as np
import matplotlib.pyplot as plt
from simsopt.mhd.boozer import Boozer
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import vmec_splines, vmec_fieldlines
from simsopt._core.util import Struct
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.optimize import root_scalar
from matplotlib import rc

import numpy as np
from qsc import Qsc
import matplotlib.pyplot as plt


def nae_geo(name, r, phi, alpha):
    stel = Qsc.from_paper(name)
    stel.spsi = -1
    stel.zs = -stel.zs
    stel.calculate()
    
    # B x grad(B) . grad(psi)
    # alpha = 0

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
    dB1_dvarphi = (stel.iota-stel.iotaN) * B1c * np.sin(chi) - (stel.iota-stel.iotaN) * B1s * np.cos(chi)
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

    return varphi, BxdBdotdalpha, BxdBdotdpsi, BdotdB, stel

def boozxform_splines(boozer, vmec):
    """
    Initialize radial splines for a boozxform equilibrium.

    Args:
        boozxform: An instance of :obj:`simsopt.mhd.boozer.Boozer`.

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
        # bsupumnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsupumnc[jmn, 1:]))  
        # bsupvmnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsupvmnc[jmn, 1:]))  
        # Note that bsubsmns is on the full mesh, unlike the other components:
        # bsubsmns.append(InterpolatedUnivariateSpline(s_full_grid, vmec.wout.bsubsmns[jmn, :]))
        # bsubumnc.append(InterpolatedUnivariateSpline(s_full_grid, vmec.wout.bsubumnc[jmn, 1:]))
        # bsubvmnc.append(InterpolatedUnivariateSpline(s_full_grid, vmec.wout.bsubvmnc[jmn, 1:]))
        d_bmnc_d_s.append(bmnc[-1].derivative())
        # d_bsupumnc_d_s.append(bsupumnc[-1].derivative())
        # d_bsupvmnc_d_s.append(bsupvmnc[-1].derivative())

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

def boozxform_fieldlines(vs, bs, s, alpha, theta1d=None, phi1d=None, phi_center=0, phi_sec = 0, plot=False, show=True, press = False, save = False):
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
    edge_toroidal_flux_over_2pi = -bs.phiedge / (2 * np.pi)

    # Now that we have an s grid, evaluate everything on that grid:
    d_pressure_d_s = bs.d_pressure_d_s(s)
    iota = bs.iota(s)
    d_iota_d_s = bs.d_iota_d_s(s)
    d_iota_d_psi = d_iota_d_s / edge_toroidal_flux_over_2pi
    I = bs.I(s)
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
        # We are given theta_booz. Compute phi:
        for js in range(ns):
            theta_booz[js, :, :] = theta1d[None, :]
            phi_booz[js, :, :] = phi_center + (theta1d[None, :] - alpha[:, None]) / iota[js]

    # def residual(phi_vmec, theta_0, phi_booz_target, jradius):
    #     """
    #     This function is used for computing the value of phi_vmec that
    #     gives a desired phi_booz.
    #     """
    #     return phi_booz_target - (phi_vmec + np.sum(numns[jradius, :] * np.sin(xm * theta_0 - xn * phi_booz_target)))

    # # Solve for theta_vmec corresponding to theta_pest:
    # phi_vmec = np.zeros((ns, nalpha, nl))
    # for js in range(ns):
    #     for jalpha in range(nalpha):
    #         for jl in range(nl):
    #             phi_guess = phi_booz[js, jalpha, jl]
    #             solution = root_scalar(residual,
    #                                    args=(theta_booz[js, jalpha, jl], phi_booz[js, jalpha, jl], js),
    #                                    bracket=(phi_guess - 1.0, phi_guess + 1.0))
    #             phi_vmec[js, jalpha, jl] = solution.root

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

    B_dot_grad_B = B_X * grad_B_X + B_Y * grad_B_Y + B_Z * grad_B_Z

    grad_thphi_dot_grad_thphi = (grad_theta_b_X - iota[:, None, None] * grad_phi_b_X) * (grad_theta_b_X - iota[:, None, None] * grad_phi_b_X) +\
        (grad_theta_b_Y - iota[:, None, None] * grad_phi_b_Y) * (grad_theta_b_Y - iota[:, None, None] * grad_phi_b_Y) + \
        (grad_theta_b_Z - iota[:, None, None] * grad_phi_b_Z) * (grad_theta_b_Z - iota[:, None, None] * grad_phi_b_Z)

    grad_thphi_dot_grad_psi = (grad_theta_b_X - iota[:, None, None] * grad_phi_b_X) * grad_psi_X + \
        (grad_theta_b_Y - iota[:, None, None] * grad_phi_b_Y) * grad_psi_Y + \
        (grad_theta_b_Z - iota[:, None, None] * grad_phi_b_Z) * grad_psi_Z

    grad_psi_dot_grad_psi = grad_psi_X * grad_psi_X + grad_psi_Y * grad_psi_Y + grad_psi_Z * grad_psi_Z

    mu_0 = 4 * np.pi * (1.0e-7)
    
    F_bar = -B_cross_grad_B_dot_grad_thphi / modB - mu_0 * d_pressure_d_s[:, None, None] / edge_toroidal_flux_over_2pi

    G_bar = B_cross_grad_B_dot_grad_psi / modB

    # Divided by |p'|
    sign_p_prime = np.sign(-d_pressure_d_s[:, None, None] / edge_toroidal_flux_over_2pi)
    F_p = 2 * sign_p_prime * F_bar * sqrt_g_b / modB / modB
    G_p = 2 * sign_p_prime * G_bar * sqrt_g_b / modB / modB
    F_tot = F_p + d_iota_d_psi[:, None, None] * (phi_booz - phi_sec) * G_p
    Fp_times_p = []
    Gp_times_p = []
    if press:
        Fp_times_p = np.abs(-mu_0 * d_pressure_d_s[:, None, None] / edge_toroidal_flux_over_2pi) * F_p
        Gp_times_p = np.abs(-mu_0 * d_pressure_d_s[:, None, None] / edge_toroidal_flux_over_2pi) * G_p
        F_tot = np.abs(-mu_0 * d_pressure_d_s[:, None, None] / edge_toroidal_flux_over_2pi) * F_tot

    # C components
    B_alpha = G + iota[:, None, None] * I
    C_p = grad_thphi_dot_grad_thphi / B_alpha
    C_psi = 2*grad_thphi_dot_grad_psi / B_alpha
    C_q = grad_psi_dot_grad_psi / B_alpha
    C_tot = C_p - d_iota_d_psi[:, None, None] * (phi_booz - phi_sec) * C_psi + \
        d_iota_d_psi[:, None, None] * d_iota_d_psi[:, None, None] * (phi_booz - phi_sec) * (phi_booz - phi_sec) * C_q

    # omega^2 normalisation
    rho = sqrt_g_b * (1 + d_iota_d_psi[:, None, None] * d_iota_d_psi[:, None, None] * (phi_booz - phi_sec) * (phi_booz - phi_sec))

    # Ballooning potential
    V_ball = - F_tot/rho 

    # Instability criterion
    crit = F_tot/C_tot

    # Package results into a structure to return:
    results = Struct()
    variables = ['ns', 'nalpha', 'nl', 's', 'iota', 'd_iota_d_psi', 'd_pressure_d_s', 'G', 'I',
                 'alpha', 'theta1d', 'phi1d', 'phi_vmec',
                 'sqrt_g_b',
                 'modB', 'd_B_d_s', 'd_B_d_theta_b', 'd_B_d_phi_b',
                 'edge_toroidal_flux_over_2pi', 'sinphi', 'cosphi',
                 'R', 'd_R_d_s', 'd_R_d_theta_b', 'd_R_d_phi_b', 'Z', 'd_Z_d_s', 'd_Z_d_theta_b', 'd_Z_d_phi_b',
                 'd_X_d_theta_b', 'd_X_d_phi_b', 'd_X_d_s', 'd_Y_d_theta_b', 'd_Y_d_phi_b', 'd_Y_d_s',
                 'grad_s_X', 'grad_s_Y', 'grad_s_Z', 'grad_theta_b_X', 'grad_theta_b_Y', 'grad_theta_b_Z',
                 'grad_phi_b_X', 'grad_phi_b_Y', 'grad_phi_b_Z', 'grad_psi_X', 'grad_psi_Y', 'grad_psi_Z',
                 'grad_B_X', 'grad_B_Y', 'grad_B_Z',
                 'B_X', 'B_Y', 'B_Z', 'B_dot_grad_B',
                 'B_cross_grad_B_dot_grad_psi', 'B_cross_grad_B_dot_grad_thphi',
                 'grad_thphi_dot_grad_thphi', 'grad_thphi_dot_grad_psi', 'grad_psi_dot_grad_psi',
                 'F_p', 'G_p','Fp_times_p', 'Gp_times_p', 'F_tot', 'C_p', 'C_psi', 'C_q', 'C_tot', 'crit', 'rho']

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


rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 14})
rc('text', usetex=True)
rc('figure', **{'figsize': (9, 7)})  

N_phi = 1000
phi = np.linspace(-1,1,N_phi)*10*np.pi
r=0.1
alpha = 0.2
varphi, BxdBdotdalpha, BxdBdotdpsi, BdotdB, stel = nae_geo("precise QA", r, phi, alpha)
v = Vmec('wout_preciseQA.nc')
# print(v.wout.__dict__.keys())
b = Boozer(v)
b.register(v.s_full_grid)
b.run()
print(v.s_full_grid)

if isinstance(b, Boozer) and isinstance(v, Vmec):
    bs = boozxform_splines(b, v)
    # vs = vmec_splines(v)
s = stel.B0*r*r/2/np.abs(bs.phiedge / (2 * np.pi))
fl = boozxform_fieldlines(v, bs, s, alpha, phi1d=phi, phi_sec=0, plot=False, press = True) #save = 'geo_props_w7x_a0_s0_5.png'
# fl_vmec = vmec_fieldlines(vs, 0.5, 0, phi1d=phi, plot=False)

print(stel.iota, bs.iota(0))

plt.figure(figsize=(13, 7))
nrows = 1
ncols = 3
ax = plt.subplot(nrows, ncols, 1)
plt.plot(phi, np.squeeze(fl.B_cross_grad_B_dot_grad_psi), linestyle='dashed', color = 'k')
# plt.plot(phi, np.squeeze(fl_vmec.B_cross_grad_B_dot_grad_psi), linestyle='dashed', color = 'r')
plt.plot(varphi, BxdBdotdpsi, color='k')
plt.xlabel(r'$\varphi$')
plt.title(r'$\mathbf{B}\times\nabla B\cdot\nabla\psi$')
ax = plt.subplot(nrows, ncols, 2)
plt.plot(phi, np.squeeze(fl.B_cross_grad_B_dot_grad_thphi), linestyle='dashed', color = 'k')
# plt.plot(phi, np.squeeze(fl_vmec.B_cross_grad_B_dot_grad_alpha), linestyle='dashed', color = 'r')
plt.plot(varphi, BxdBdotdalpha, color='k')
# inter_fun = interp1d(varphi, BxdBdotdalpha)
# plt.plot(phi, np.squeeze(fl.B_cross_grad_B_dot_grad_thphi)+inter_fun(phi), linestyle='dashed', color = 'k')
plt.xlabel(r'$\varphi$')
plt.title(r'$\mathbf{B}\times\nabla B\cdot\nabla\alpha$')
ax = plt.subplot(nrows, ncols, 3)
plt.plot(phi, np.squeeze(fl.B_dot_grad_B), linestyle='dashed', color = 'k')
plt.plot(varphi, BdotdB, color='k')
# inter_fun = interp1d(varphi, BxdBdotdalpha)
# plt.plot(phi, np.squeeze(fl.B_cross_grad_B_dot_grad_thphi)+inter_fun(phi), linestyle='dashed', color = 'k')
plt.xlabel(r'$\varphi$')
plt.title(r'$\mathbf{B}\cdot\nabla B$')
plt.show()


