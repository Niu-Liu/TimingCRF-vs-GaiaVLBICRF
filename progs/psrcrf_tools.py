#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
# File name: psrcrf_tools.py
"""
Created on Thu Feb 10 10:58:55 2022

@author: Neo(niu.liu@nju.edu.cn)
"""
import sys

import numpy as np
from numpy import sin, cos, pi, sqrt, concatenate
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from astropy.table import Table

import statsmodels.api as sm


# define some function for statistics
# ----------------------------------------------------
def calc_med_pos_err(data_tab, ref="g"):
    """Compute median formal uncertainty
    """

    ra_err_t = np.median(data_tab["ra_err_t"])
    dec_err_t = np.median(data_tab["dec_err_t"])
    pos_err_t = np.median(sqrt(data_tab["ra_err_t"]**2
                               + data_tab["dec_err_t"]**2))

    ra_err_g = np.median(data_tab["ra_err_%s" % ref])
    dec_err_g = np.median(data_tab["dec_err_%s" % ref])
    pos_err_g = np.median(sqrt(data_tab["ra_err_%s" % ref]**2
                               + data_tab["dec_err_%s" % ref]**2))

    return ra_err_t, dec_err_t, pos_err_t, ra_err_g, dec_err_g, pos_err_g


# ----------------------------------------------------
def calc_chi2(x, errx, y, erry):
    """Calculate the Chi-square.


    Parameters
    ----------
    x : array, float
        residuals of x
    errx : array, float
        formal errors of x
    x : array, float
        residuals of x
    errx : array, float
        formal errors of x
    covxy : array, float
        summation of covariance between x and y
    reduced : boolean
        True for calculating the reduced chi-square

    Returns
    ----------
    (reduced) chi-square
    """

    #     chi2_mat = np.zeros(len(x))

    #     for i, (xi, errxi, yi, erryi) in enumerate(zip(x, errx, y, erry)):

    #         wgt_mat = np.linalg.inv(np.array([[errxi**2, 0], [0, erryi**2]]))
    #         res_mat = np.array([xi, yi])

    #         chi2_mat[i] = reduce(np.dot, (res_mat, wgt_mat, res_mat))

    chi2_mat = (x / errx)**2 + (y / erry)**2

    return chi2_mat


def calc_chi2_4_table(data_tab):
    """Calculate chi2 for each pulsars in the sample
    """

    chi2_mat = calc_chi2(data_tab["dra"], data_tab["dra_err"],
                         data_tab["ddec"], data_tab["ddec_err"])

    return chi2_mat


def calc_chi2_4_fit(data_tab, dra, ddec):
    """Calculate prefit and postfit reduced Chi2
    """

    nb_psr = len(data_tab.group_by("psr_name").groups)
    nb_obs = len(data_tab)

    a_chi2 = np.sum(
        calc_chi2(data_tab["dra"], data_tab["dra_err"],
                  data_tab["ddec"], data_tab["ddec_err"]) /
        (nb_obs * 2 - 1))

    p_chi2 = np.sum(
        calc_chi2(dra, data_tab["dra_err"],
                  ddec, data_tab["ddec_err"]) / (nb_obs * 2 - 1))

    return a_chi2, p_chi2

# define some function for fitting
# ----------------------------------------------------


def rot_func(pos, rx, ry, rz):
    """Rotation function

    The transformation function considering onl the rigid rotation
    is given by
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
    d_DE   = +r_x*sin(ra) - r_y*cos(ra)

    Parameters
    ----------
    ra/dec : array of float
        right ascension/declination in radian
    rx/ry/rz : float
        rotational angles around X-, Y-, and Z-axis

    Returns
    -------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences
    """

    N = len(pos) // 2
    ra, dec = pos[:N], pos[N:]

    dra = -rx * cos(ra) * sin(dec) - ry * sin(ra) * sin(dec) + rz * cos(dec)
    ddec = rx * sin(ra) - ry * cos(ra)
    dpos = concatenate((dra, ddec))

    return dpos


def calc_dpos(data_tab, rot_pmt, ref="g"):

    ra_rad = np.deg2rad(data_tab["ra_%s" % ref])
    dec_rad = np.deg2rad(data_tab["dec_%s" % ref])
    pos_rad = concatenate((ra_rad, dec_rad))
    dpos = rot_func(pos_rad, *rot_pmt)

    N = len(data_tab)
    dra, ddec = data_tab["dra"]-dpos[:N], data_tab["ddec"]-dpos[N:]

    return dra, ddec


# ----------------------------------------------------
def calc_cov_mat(dra_err, ddec_err):
    """Generate covariance matrix
    """

    cov_mat = np.diag(concatenate((dra_err**2, ddec_err**2)))

    return cov_mat


# ----------------------------------------------------
def calc_jac_mat(pos, rx, ry, rz):
    """Generate the Jacobian matrix.

    Parameters
    ----------
    ra : array of float
        right ascension in radian
    dec : array of float
        declination in radian


    Returns
    ----------
    JacMat : matrix
        Jacobian matrix and its transpose matrix
    """

    N = len(pos) // 2
    ra, dec = pos[:N], pos[N:]

    # Partial array dRA and dDE, respectively.

    # For RA
    par1_r1 = -cos(ra) * sin(dec)
    par1_r2 = -sin(ra) * sin(dec)
    par1_r3 = cos(dec)

    # For Dec
    par2_r1 = sin(ra)
    par2_r2 = -cos(ra)
    par2_r3 = np.zeros_like(ra)

    # (dRA, dDE).
    par_r1 = concatenate((par1_r1, par2_r1))
    par_r2 = concatenate((par1_r2, par2_r2))
    par_r3 = concatenate((par1_r3, par2_r3))

    # transpose of Jacobian matrix.
    Jac_MatT = concatenate((par_r1.reshape(1, 2 * N), par_r2.reshape(
        1, 2 * N), par_r3.reshape(1, 2 * N)),
        axis=0)

    # Jacobian matrix.
    Jac_Mat = np.transpose(Jac_MatT)

    return Jac_Mat


def calc_jac_mat_new(ra, dec):
    """Generate the Jacobian matrix.

    Parameters
    ----------
    ra : array of float
        right ascension in radian
    dec : array of float
        declination in radian


    Returns
    ----------
    JacMat : matrix
        Jacobian matrix and its transpose matrix
    """

    N = len(ra)

    # Partial array dRA and dDE, respectively.
    # For RA
    par1_r1 = -cos(ra) * sin(dec)
    par1_r2 = -sin(ra) * sin(dec)
    par1_r3 = cos(dec)

    # For Dec
    par2_r1 = sin(ra)
    par2_r2 = -cos(ra)
    par2_r3 = np.zeros_like(ra)

    # (dRA, dDE).
    par_r1 = concatenate((par1_r1, par2_r1))
    par_r2 = concatenate((par1_r2, par2_r2))
    par_r3 = concatenate((par1_r3, par2_r3))

    # transpose of Jacobian matrix.
    Jac_MatT = concatenate((par_r1.reshape(1, 2 * N), par_r2.reshape(
        1, 2 * N), par_r3.reshape(1, 2 * N)),
        axis=0)

    # Jacobian matrix.
    Jac_Mat = np.transpose(Jac_MatT)

    return Jac_Mat, Jac_MatT


# ----------------------------------------------------
def cov_to_cor(cov_mat):
    """Convert covariance matrix to sigma and correlation coefficient matrix
    """

    # Formal uncertainty
    sig = sqrt(cov_mat.diagonal())

    # Correlation coefficient.
    cor_mat = np.array([
        cov_mat[i, j] / sig[i] / sig[j] for j in range(len(sig))
        for i in range(len(sig))
    ])
    cor_mat.resize((len(sig), len(sig)))

    return sig, cor_mat


# ----------------------------------------------------
def calc_nor_mat(ra, dec, dra_err, ddec_err):

    jac_mat, jac_mat_t = calc_jac_mat_new(ra, dec)
    cov_mat = calc_cov_mat(dra_err, ddec_err)
    wgt_mat = np.linalg.inv(cov_mat)
    nor_mat = np.dot(np.dot(jac_mat_t, wgt_mat), jac_mat)

    return nor_mat


def calc_pmt_sig(ra, dec, dra_err, ddec_err):

    nor_mat = calc_nor_mat(ra, dec, dra_err, ddec_err)
    cov_mat = np.linalg.inv(nor_mat)

    sig, cor_mat = cov_to_cor(cov_mat)

    return sig


# ----------------------------------------------------
def rot_fit(ra, dec, dra, ddec, dra_err, ddec_err):
    """A simple rotation fit
    """

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    pos = concatenate((ra_rad, dec_rad))
    dpos = concatenate((dra, ddec))

    cov_mat = calc_cov_mat(dra_err, ddec_err)
    popt, pcov = curve_fit(rot_func,
                           pos,
                           dpos,
                           jac=calc_jac_mat,
                           sigma=cov_mat,
                           absolute_sigma=False)

    # Postfit residuals
    pdpos = dpos - rot_func(pos, *popt)

    return popt, pcov, pdpos


def rot_fit_4_table(timing_table, ref="g"):
    """A simple rotation fit for Astropy.table.Table object
    """

    data_tab = Table(timing_table)
    num_par = 3
    num_iter = len(data_tab) - num_par + 1

    indx_out = []
    pmt_mat = np.zeros((num_iter, num_par))
    err_mat = np.zeros((num_iter, num_par))
    # A priori chi2
    apr_chi2_mat = np.zeros(num_iter)
    # A posterior
    pst_chi2_mat = np.zeros(num_iter)

    print("Iter.  NO.PSR  Apr.Chi2  Pos.Chi2"
          "  R1  R2  R3  rmPSR  rmPSREpoch  rmPSRChi2\n"
          "Unit: mas")

    num_count = 0

    for i in range(num_iter):

        num_obs = len(data_tab)
        num_psr = len(data_tab.group_by("psr_name").groups)

        if num_psr < 3:
            break

        num_count += 1

        num_dof = num_obs * 2 - num_par - 1

        sub_apr_chi2_mat = calc_chi2_4_table(data_tab)
        apr_chi2 = np.sum(sub_apr_chi2_mat) / num_dof
        apr_chi2_mat[i] = apr_chi2

        popt, pcov, pdpos = rot_fit(np.array(data_tab["ra_%s" % ref]),
                                    np.array(data_tab["dec_%s" % ref]),
                                    np.array(data_tab["dra"]),
                                    np.array(data_tab["ddec"]),
                                    np.array(data_tab["dra_err"]),
                                    np.array(data_tab["ddec_err"]))

        pmt_mat[i, :] = popt
        sig, cor_mat = cov_to_cor(pcov)
        err_mat[i, :] = sig

        # Calculate residuals
        data_tab1 = Table(data_tab)
        data_tab1["dra"] = pdpos[:num_obs]
        data_tab1["ddec"] = pdpos[num_obs:]

        sub_pst_chi2_mat = calc_chi2_4_table(data_tab1)
        pst_chi2 = np.sum(sub_pst_chi2_mat) / num_dof
        pst_chi2_mat[i] = pst_chi2

        # Find pulsars having the largest Chi2
        indx = np.argmax(sub_apr_chi2_mat)
        indx_out.append(indx)

        #         data_tab.argsort(chi2_mat)
        #         data_tab = data_tab[:num_psr]

        print("{:2d} {:2d} {:8.1f} {:8.1f}"
              " {:10.2f} {:10.2f} {:10.2f}"
              " {:8s} {:8.0f} {:8.1f}".format(
                  i + 1, num_psr, apr_chi2, pst_chi2, *popt,
                  data_tab1["psr_name"][indx], data_tab1["pos_epoch"][indx],
                  sub_pst_chi2_mat[indx]))

        data_tab.remove_row(indx)

    pmt_mat = pmt_mat[:num_count, :]
    err_mat = err_mat[:num_count, :]
    apr_chi2_mat = apr_chi2_mat[:num_count]
    pst_chi2_mat = pst_chi2_mat[:num_count]

    return pmt_mat, err_mat, indx_out, apr_chi2_mat, pst_chi2_mat


def find_gaia_msp():
    """Find Gaia MSP
    """

    gaia_tab = Table.read("../data/edr3_psrcatv1.67.fits")

    # mask = (gaia_tab["p0"] <= 0.03)
    B_over_G = 3.2e19 * np.sqrt(gaia_tab["p0"] * gaia_tab["p1"])
    mask = (B_over_G <= 1e10)

    gaia_msp_list = gaia_tab["name"][mask]
    gaia_msp_list = gaia_msp_list.tolist()

    return gaia_msp_list


def find_vlbi_msp():
    """Find VLBI MSP
    """

    vlbi_msp_list = ["J1022+1001", "J2010-1323", "J2145-0750", "J2317+1439", "J1012+5307", "J1537+1155"]

    return vlbi_msp_list


def find_msp():
    """Find MSPs in our sample
    """

    gaia_msp_list = find_gaia_msp()
    vlbi_msp_list = find_vlbi_msp()

    msp_list = np.unique(vlbi_msp_list + gaia_msp_list)

    return msp_list


def divide_table(psr_table, msp_list):
    """Divide table into MSP and non-MSP
    """

    # MSP table
    temp_table1 = Table([])

    # Non-MSP table
    temp_table2 = Table(psr_table)

    for msp_name in msp_list:

        mask = ((psr_table["psr_name"] == msp_name))
        temp_table1 = vstack((temp_table1, psr_table[mask]))

        mask = ((temp_table2["psr_name"] != msp_name))
        temp_table2 = temp_table2[mask]

    msp_table = Table(temp_table1, masked=False)
    nmp_table = Table(temp_table2, masked=False)

    return msp_table, nmp_table


# Functions for plot
# ----------------------------------------------------
def simple_rot_fit_4_table(timing_table, ref="g"):
    """A simple rotation fit for Astropy.table.Table object
    """

    # Print basic statistics
    data_tab = Table(timing_table)
    num_obs = len(data_tab)
    num_psr = len(data_tab.group_by("psr_name").groups)
    print("[msg] There are {:d} timing solutions for {:d} pulsars".format(
        num_obs, num_psr))

    if num_psr < 3:
        print("[Warning] The number of pulsars is less than three, "
              "so the least-squares fitting will not be performed.")
        sys.exit(1)

    # Check signigicant position offsets, that is, prefit Chi^2 > 100
    sub_apr_chi2_mat = calc_chi2_4_table(data_tab)
    num_dof = num_obs * 2 - 1
    apr_chi2 = np.sum(sub_apr_chi2_mat) / num_dof
    print(
        "[msg] Before removing outliers, the reduced chi-squared is {:.2f}.".format(apr_chi2))

    mask = sub_apr_chi2_mat < 1e10
    data_tab_cln = Table(data_tab[mask])
    num_obs_cln = len(data_tab_cln)
    num_psr_cln = len(data_tab_cln.group_by("psr_name").groups)
    nb_obs_out = num_obs - num_obs_cln
    nb_psr_out = num_psr - num_psr_cln

    if nb_obs_out == 0:
        print("[msg] There is no outliers.")

        popt, pcov, pdpos = rot_fit(np.array(data_tab["ra_%s" % ref]),
                                    np.array(data_tab["dec_%s" % ref]),
                                    np.array(data_tab["dra"]),
                                    np.array(data_tab["ddec"]),
                                    np.array(data_tab["dra_err"]),
                                    np.array(data_tab["ddec_err"]))
    else:
        print("[msg] There are {:d} outliers for {:d} pulsars.".format(
            nb_obs_out, nb_psr_out))

        if num_psr_cln < 3:
            print("[Warning] The number of pulsars is less than three, "
                  "so the least-squares fitting will not be performed.")
            sys.exit(1)
        else:
            popt, pcov, pdpos = rot_fit(np.array(data_tab_cln["ra_%s" % ref]),
                                        np.array(data_tab_cln["dec_%s" % ref]),
                                        np.array(data_tab_cln["dra"]),
                                        np.array(data_tab_cln["ddec"]),
                                        np.array(data_tab_cln["dra_err"]),
                                        np.array(data_tab_cln["ddec_err"]))

    print("[info] NO.Obs  NO.PSR  Apr.Chi2  Pos.Chi2"
          "  R1  R2  R3  R1_err  R2_err R3_err\n"
          "[info] Unit: mas")

    sig, cor_mat = cov_to_cor(pcov)

    # Extract residuals
    pos = concatenate((np.deg2rad(data_tab["ra_%s" % ref]),
                       np.deg2rad(data_tab["dec_%s" % ref])))
    dpos = concatenate(
        (np.array(data_tab["dra"]), np.array(data_tab["ddec"])))

    # Postfit residuals
    pdpos = dpos - rot_func(pos, *popt)

    data_tab = Table(data_tab)
    data_tab["dra"] = pdpos[:num_obs]
    data_tab["ddec"] = pdpos[num_obs:]

    sub_pst_chi2_mat = calc_chi2_4_table(data_tab)
    num_par = 3
    num_dof = num_obs * 2 - num_par - 1
    pst_chi2 = np.sum(sub_pst_chi2_mat) / num_dof

    print("[info] {:2d} {:2d} {:8.1f} {:8.1f}"
          " {:10.2f} {:10.2f} {:10.2f}"
          " {:10.2f} {:10.2f} {:10.2f}" .format(
              num_obs_cln, num_psr_cln, apr_chi2, pst_chi2, *popt, *sig))

    return popt, sig, data_tab["dra"], data_tab["ddec"], apr_chi2, pst_chi2


def find_pmt_est(pmt):
    """Final parameter estimates
    """

    rot = np.median(pmt, axis=0)
    iqr = np.subtract(*np.percentile(pmt, [75, 25], axis=0)) / 1.35

    return rot, iqr


# Define some functions for plot
# ------------------ FUNCTION --------------------------------
# Hammer Projection
def hammer_projection(lamda, phi):
    """
    Detailed diecription about Hammer Projection can be found in
    https://en.wikipedia.org/wiki/Hammer_projection.

    The input angular coordinate (lamda, phi) are longtitude and latitude in degree respectively.
    For equatorial coordinate system,
        lamda = R.A.
        phi   = Dec.

    Usually Xmax / Y max = 2 : 1
    But it looks not good for this fraction.
    So I change it to Xmax / Ymax = 10 : 6
    """

    # Usually the longtitude ranges from 0 to 360 degree.
    # But for Hammer projection, -180 deg < lamda < +180.
    # So the original longtidude part should minus 180 deg.
    lamda = lamda - 180

    # The input paramter of trigonometric function is given in rad.
    lamda, phi = np.deg2rad(lamda), np.deg2rad(phi)

    # Calculate the corresponding rectangular coordinate (x, y)
    # Common denominator of the projection equation.
    den = sqrt(1 + cos(phi) * cos(lamda / 2))
    x = 2 * sqrt(2) * cos(phi) * sin(lamda / 2) / den
    y = sqrt(2) * sin(phi) / den * 1.2  # A extrax scale foctor 1.2

    # Return the result.
    return x, y


# ------------------------------------------------------
def sou_dist_plot(ra, dec, fig_name=None):

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Uniform sampling in longtitude and latitude
    lon_arr = np.arange(0, 361, 5)
    lat_arr = np.arange(-90, 91, 1)

    # Plot a ellipse border.
    lon_bords = np.array([0, 360])

    # A loop to plot 7 grid.
    for lon_bord in lon_bords:
        lat = lat_arr
        lon = np.ones_like(lat) * lon_bord
        X, Y = hammer_projection(lon, lat)
        ax.plot(X, Y, "k", linewidth=1.5)

    # Grid plot.
    # longtitude_grid_plot()
    # 6 nodes in the longtitude from 0 to 360 deg.
    lon_nodes = np.arange(60, 360, 60)

    # A loop to plot 7 grid.
    for lon_node in lon_nodes:
        lat = lat_arr
        lon = np.ones_like(lat) * lon_node
        X, Y = hammer_projection(lon, lat)
        ax.plot(X, Y, "k", linewidth=0.5)

    # latitude_grid_plot()
    # 5 nodes in the latitude from -60 to +60 deg.
    lat_nodes = np.arange(-60, 61, 30)

    # A loop to plot 7 grid.
    for lat_node in lat_nodes:
        lon = lon_arr
        lat = np.ones_like(lon) * lat_node
        X, Y = hammer_projection(lon, lat)
        ax.plot(X, Y, "k", linewidth=0.5)

    # Grid tickers.
    # For fundamental plane.
    lon0 = np.array([0, 360])
    lat0 = np.array([0, 0])
    X0, Y0 = hammer_projection(lon0, lat0)

    ax.text(X0[0] - 0.2, Y0[0] - 0.025, " 0h")
    ax.text(X0[1] + 0.02, Y0[1] - 0.025, "24h")

    # For latitude grid
    lon = np.array([0, 0])
    lat = np.array([30, 60])
    X, Y = hammer_projection(lon, lat)

    for i in range(lat.size):
        ax.text(X[i] - 0.45, Y[i] - 0.0, '$+%d^\circ$' % lat[i])
        ax.text(X[i] - 0.45, -Y[i] - 0.1, '$-%d^\circ$' % lat[i])

    # Plot the source.
    X, Y = hammer_projection(ra, dec)
    ax.plot(X, Y, "bo", ms=5)

    plt.tight_layout()

    # Save figure.
    if fig_name is not None:
        plt.savefig(fig_name, dpi=100)


#     plt.close()


# ------------------------------------------------------
def simple_plot(data_tab, fig_name=None, axis_lim=None):
    """Simple plot for positional offset
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.errorbar(data_tab["dra"],
                data_tab["ddec"],
                xerr=data_tab["dra_err"],
                yerr=data_tab["ddec_err"],
                fmt="bo",
                ecolor="k",
                elinewidth=1)

    ax.set_xlabel("$\Delta\\alpha\cos\delta$ (mas)", fontsize=15)
    ax.set_ylabel("$\Delta\delta$ (mas)", fontsize=15)

    if axis_lim is not None:
        ax.axis("square")
        ax.axis(axis_lim)

    #     plt.grid(lw=1)
    plt.tight_layout()

    if fig_name is not None:
        plt.savefig("../plots/{:s}".format(fig_name))


# ------------------------------------------------------
def divide_pos_oft(data_tab, dra, ddec, xval):
    """Divide position offsets into MSP and non-MSP

    """

    msp_list = find_msp()

    # Lists for storing index
    msp_indx = []
    nmp_indx = []

    for i, psr_name in enumerate(data_tab["psr_name"]):
        if psr_name in msp_list:
            msp_indx.append(i)
        else:
            nmp_indx.append(i)

    msp_tab = data_tab[msp_indx]
    dra_msp = dra[msp_indx]
    ddec_msp = ddec[msp_indx]
    xval_msp = xval[msp_indx]

    nmp_tab = data_tab[nmp_indx]
    dra_nmp = dra[nmp_indx]
    ddec_nmp = ddec[nmp_indx]
    xval_nmp = xval[nmp_indx]

    return msp_tab, dra_msp, ddec_msp, xval_msp, nmp_tab, dra_nmp, ddec_nmp, xval_nmp

def pos_oft_vs_coord(data_tab, yaxis_lim, dra, ddec, fig_name=None,
                     ref="g", xaxis="ra", add_text=None, divided=False,
                     add_text_msp=None):
    """Plot the dependency of positional offset versus RA/DEC
    """

    fig, (ax0, ax1) = plt.subplots(figsize=(8, 6), nrows=2, sharex=True)

    if xaxis == "ra":
        xval = data_tab["ra_%s" % ref]
        ax1.set_xlim([0, 360])
        ax1.set_xticks(np.arange(0, 361, 30))
        ax1.set_xlabel("$\\alpha$ (${}^\\circ$)", fontsize=15)
    elif xaxis == "dec":
        xval = data_tab["dec_%s" % ref]
        ax1.set_xlim([-90, 90])
        ax1.set_xticks(np.arange(-90, 91, 30))
        ax1.set_xlabel("$\delta$ (${}^\\circ$)", fontsize=15)
    else:
        sys.exit()

    legend_handle0 = []
    legend_handle1 = []

    if divided:
        [msp_tab, dra_msp, ddec_msp, xval_msp, nmp_tab, dra_nmp, ddec_nmp,
            xval_nmp] = divide_pos_oft(data_tab, dra, ddec, xval)

        # For MSPs
        if len(msp_tab):
            # Observables
            line01_msp = ax0.errorbar(xval_msp,
                                      # msp_tab["dra"],
                                      dra_msp,
                                      yerr=msp_tab["dra_err"],
                                      fmt="o",
                                      mfc="none",
                                      mec="b",
                                      mew=0.5,
                                      ms=8,
                                      ecolor="grey",
                                      capsize=3,
                                      elinewidth=0.5,
                                      # label="Prefit (MSP)")
                                      label="MSP")
            legend_handle0.append(line01_msp)

            line11_msp = ax1.errorbar(xval_msp,
                                      #                                       msp_tab["ddec"],
                                      ddec_msp,
                                      yerr=msp_tab["ddec_err"],
                                      fmt="o",
                                      mfc="none",
                                      mec="b",
                                      mew=0.5,
                                      ms=8,
                                      elinewidth=0.5,
                                      capsize=3,
                                      ecolor="grey",
                                      # label="Prefit (MSP)")
                                      label="MSP")
            legend_handle1.append(line11_msp)

            # Predictions
#             line02_msp, = ax0.plot(
#                 xval_msp, dra_msp, "rx", label="Postfit (MSP)")
#             legend_handle0.append(line02_msp)
#
#             line12_msp, = ax1.plot(
#                 xval_msp, ddec_msp, "rx", label="Postfit (MSP)")
#             legend_handle1.append(line12_msp)

        # For Non-MSPs
        if len(nmp_tab):
            # Observables
            line01_nmp = ax0.errorbar(xval_nmp,
                                      # nmp_tab["dra"],
                                      dra_nmp,
                                      yerr=nmp_tab["dra_err"],
                                      fmt="s",
                                      mfc="none",
                                      # mec="b",
                                      mec="r",
                                      mew=0.5,
                                      ms=8,
                                      ecolor="grey",
                                      capsize=3,
                                      elinewidth=0.5,
                                      # label="Prefit (Non-MSP)")
                                      label="Non-MSP")
            legend_handle0.append(line01_nmp)

            line11_nmp = ax1.errorbar(xval_nmp,
                                      # nmp_tab["ddec"],
                                      ddec_nmp,
                                      yerr=nmp_tab["ddec_err"],
                                      fmt="s",
                                      mfc="none",
                                      # mec="b",
                                      mec="r",
                                      mew=0.5,
                                      ms=8,
                                      elinewidth=0.5,
                                      capsize=3,
                                      ecolor="grey",
                                      # label="Prefit (Non-MSP)")
                                      label="Non-MSP")
            legend_handle1.append(line11_nmp)

            # Predictions
#             line02_nmp, = ax0.plot(
#                 xval_nmp, dra_nmp, "r+", label="Postfit (Non-MSP)")
#             legend_handle0.append(line02_nmp)
#
#             line12_nmp, = ax1.plot(
#                 xval_nmp, ddec_nmp, "r+", label="Postfit (Non-MSP)")
#             legend_handle1.append(line12_nmp)

    else:
        # Observables
        line01 = ax0.errorbar(xval,
                              data_tab["dra"],
                              yerr=data_tab["dra_err"],
                              fmt="o", mfc="none", mec="b", mew=0.5, ms=8, capsize=3,
                              ecolor="grey",
                              elinewidth=0.5,
                              label="Prefit")
        legend_handle0.append(line01)

        line11 = ax1.errorbar(xval,
                              data_tab["ddec"],
                              yerr=data_tab["ddec_err"],
                              fmt="o", mfc="none", mec="b", mew=0.5, ms=8, capsize=3,
                              ecolor="grey",
                              elinewidth=0.5,
                              label="Prefit")
        legend_handle1.append(line11)

        # Predictions
        line02, = ax0.plot(xval, dra, "rx", label="Postfit")
        legend_handle0.append(line02)

        line12, = ax1.plot(xval, ddec, "rx", label="Postfit")
        legend_handle1.append(line12)

    # Legends
    ax0.legend(handles=legend_handle0)
    ax1.legend(handles=legend_handle1)

    # Axis limits
    if len(yaxis_lim) == 2:
        ax0.set_ylim(yaxis_lim)
        ax1.set_ylim(yaxis_lim)
    else:
        ax0.set_ylim(yaxis_lim[:2])
        ax1.set_ylim(yaxis_lim[2:])

    # Axis labels
    ax0.set_ylabel("$\Delta\\alpha\cos\delta$ (mas)", fontsize=15)
    ax1.set_ylabel("$\Delta\delta$ (mas)", fontsize=15)

    if add_text is not None:
        ax0.text(add_text[0], add_text[1], add_text[2],
                     fontsize=15, transform=ax0.transAxes)
        ax1.text(add_text[0], add_text[1], add_text[2],
                     fontsize=15, transform=ax1.transAxes)

    if add_text_msp is not None:
        for i in range(len(add_text_msp)):
            if add_text_msp[i][3] == 0:
                ax0.text(add_text_msp[i][0], add_text_msp[i][1], add_text_msp[i][2],
                     fontsize=8, color="b")
            else:
                ax1.text(add_text_msp[i][0], add_text_msp[i][1], add_text_msp[i][2],
                     fontsize=8, color="b")

    plt.tight_layout()

    if fig_name is not None:
        plt.savefig("../plots/{:s}".format(fig_name))


# ------------------------------------------------------
def rot_vs_iter(pmt,
                sig,
                xylim,
                fig_name=None,
                elw=2,
                xaxis_range=None,
                add_text=None,
                apr_chi2=None,
                pst_chi2=None,
                ax_loc=None,
                ax1_loc=None,
                ax1_ylim=None,
                caps=3,
                y_shift=0,):
    """Rotation parameter versus number of iterations
    """

    #     fig, ax = plt.subplots(figsize=(6, 6))
    fig, ax = plt.subplots(figsize=(8, 6))

    xval = np.arange(len(pmt))

    if y_shift:
        ax.errorbar(xval,
                    pmt[:, 0]+y_shift,
                    yerr=sig[:, 0],
                    fmt="bo",
                    ls="dashed",
                    lw=1,
                    ecolor="k",
                    elinewidth=elw,
                    capsize=caps,
                    label="$R_1+${}$\,$mas".format(y_shift))

    else:
        ax.errorbar(xval,
                    pmt[:, 0],
                    yerr=sig[:, 0],
                    fmt="bo",
                    ls="dashed",
                    lw=1,
                    ecolor="k",
                    elinewidth=elw,
                    capsize=caps,
                    label="$R_1$")

    ax.errorbar(xval,
                pmt[:, 1],
                yerr=sig[:, 1],
                fmt="rx",
                ls="dashed",
                lw=1,
                ecolor="k",
                elinewidth=elw,
                capsize=caps,
                label="$R_2$")

    if y_shift:
        ax.errorbar(xval,
                    pmt[:, 2]-y_shift,
                    yerr=sig[:, 2],
                    fmt="g^",
                    ls="dashed",
                    lw=1,
                    ecolor="k",
                    elinewidth=elw,
                    capsize=caps,
                    label="$R_3-${}$\,$mas".format(y_shift))
    else:
        ax.errorbar(xval,
                    pmt[:, 2],
                    yerr=sig[:, 2],
                    fmt="g^",
                    ls="dashed",
                    lw=1,
                    ecolor="k",
                    elinewidth=elw,
                    capsize=caps,
                    label="$R_3$")

#     if y_shift:
#
#         ax.hlines(y_shift, xylim[0], xylim[1], color="b", ls="dashed", lw=0.5)
#         ax.hlines(0, xylim[0], xylim[1], color="r", ls="dashed", lw=0.5)
#         ax.hlines(-y_shift, xylim[0], xylim[1], color="g", ls="dashed", lw=0.5)

    ax.axis(xylim)

    if xaxis_range is not None:
        ax.set_xticks(xaxis_range)

    ax.set_xlabel("Nb. Iteration", fontsize=15)
    ax.set_ylabel("Rotation (mas)", fontsize=15)

    if add_text is not None:
        ax.text(add_text[0], add_text[1], add_text[2],
                fontsize=15, transform=ax.transAxes)

    if ax_loc is None:
        ax.legend()
    else:
        ax.legend(loc=ax_loc, fontsize=12)

    if apr_chi2 is not None or pst_chi2 is not None:
        # Add a new axis
        ax1 = ax.twinx()
        color = "grey"
        # color = "tab:red"
        ax1.set_yscale("log")
        ax1.set_ylabel("Reduced $\chi^2$", color=color, fontsize=15)
        ax1.tick_params(axis="y", labelcolor=color)

        if ax1_ylim is not None:
            ax1.set_ylim(ax1_ylim)

    if apr_chi2 is not None:
        ax1.plot(xval, apr_chi2, color=color, label="$\chi^2_{\\rm pre}$",
                 ls="dotted", lw=2)

    if pst_chi2 is not None:
        ax1.plot(xval, pst_chi2, color=color, label="$\chi^2_{\\rm post}$",
                 ls="dashdot", lw=2)

    if apr_chi2 is not None or pst_chi2 is not None:
        if ax1_loc is None:
            ax1.legend()
        else:
            ax1.legend(loc=ax1_loc, fontsize=12)

    plt.tight_layout()

    if fig_name is not None:
        plt.savefig("../plots/{:s}".format(fig_name))


# ------------------------------------------------------------------------------
def count_sigma(data_arr, n_sigma=1):

    nb = len(data_arr)
    nb_sub = len(data_arr[np.fabs(data_arr) <= n_sigma])

    percent = nb_sub * 100 / nb

    return percent


# ------------------------------------------------------------------------------
def fit_plx_offset(plx1, plx2, plx_err):

    X = np.array(plx1)
    Y = np.array(plx2)
    err = np.array(plx_err)

    X = sm.add_constant(X)
    wls_model = sm.WLS(Y, X, weights=1/err**2)
    results = wls_model.fit()

    return results


# ------------------------------------------------------------------------------
def count_psr_nb(data_tab):

    nb_obs = len(data_tab)

    nb_psr = len(np.unique(data_tab["psr_name"]))

    print("There are {} pulsars with {} measurements.".format(nb_psr, nb_obs))
