#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
# File name: extract-nanogtrav-data.py
"""
Created on Mon Nov 22 12:05:53 2021

@author: Neo(niu.liu@nju.edu.cn)
"""


import sys
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


def create_list():
    """ Create a
    """


def parse_data_line(str_line):
    """Used for most cases
    """

    str_array = str_line.split()

    par1 = float(str_array[1])

    if len(str_array) >= 4:
        par2 = float(str_array[3])
    else:
        par2 = 0  # No formal uncertainty estimated

    return par1, par2


def parse_data_line2(str_line):
    """Used when ra and dec are given in HHMMSS DDMMSS format
    """

    str_array = str_line.split()
    par1 = str_array[1]

    if len(str_array) >= 4:
        par2 = float(str_array[3])
    else:
        par2 = 0  # No formal uncertainty estimated

    return par1, par2


def parse_data_file(data_dir, data_file, f_output):

    with open("{:s}/{:s}".format(data_dir, data_file), "r") as f_input:
        data_lines = f_input.readlines()

        par_names = ["sou_name", "epoch", "l", "l_err", "b", "b_err",
                     "pml", "pml_err", "pmb", "pmb_err", "plx", "p_err"]

        par_values = {}

        nb_par_now = 0
        nb_par_tot = len(par_names)

        for data_line in data_lines:

            if nb_par_now >= nb_par_tot:
                break

            data_array = data_line.strip("\n").split()
            line_head = data_array[0]

            # Pulsar name
            if line_head in ["PSRJ", "PSR"]:
                par_values["sou_name"] = data_array[1]
                nb_par_now += 1
                continue

            # Positional epoch
            if line_head in ["POSEPOCH"]:
                par_values["epoch"] = float(data_array[1])
                nb_par_now += 1
                continue

            # Ecliptic longitude
            if line_head in ["LAMBDA", "ELONG"]:
                lon, lon_err = parse_data_line(data_line)
                par_values["l"] = lon
                par_values["l_err"] = lon_err
                nb_par_now += 2
                continue

            # Ecliptic latitude
            if line_head in ["BETA", "ELAT"]:
                lat, lat_err = parse_data_line(data_line)
                par_values["b"] = lat
                par_values["b_err"] = lat_err
                nb_par_now += 2
                continue

            # Proper motion in Ecliptic longitude
            if line_head in ["PMLAMBDA", "PMELONG"]:
                pmlon, pmlon_err = parse_data_line(data_line)
                par_values["pml"] = pmlon
                par_values["pml_err"] = pmlon_err
                nb_par_now += 2
                continue

            # Proper motion in Ecliptic latitude
            if line_head in ["PMBETA", "PMELAT"]:
                pmlat, pmlat_err = parse_data_line(data_line)
                par_values["pmb"] = pmlat
                par_values["pmb_err"] = pmlat_err
                nb_par_now += 2
                continue

            # Parallax
            if line_head in ["PX"]:
                plx, plx_err = parse_data_line(data_line)
                par_values["plx"] = plx
                par_values["p_err"] = plx_err
                nb_par_now += 2
                continue

        data_values = []
        for par_name in par_names:
            if par_name in par_values.keys():
                data_values.append(par_values[par_name])
            else:
                data_values.append(0)

        line_format = ["{:12s}  " + "{:10.4f}  "
                       + "{:17.13f}  {:15.13f}  " * 2
                       + "{:10.4f}  {:6.4f}" * 3][0]

        print(line_format.format(*data_values), file=f_output)


def parse_data_file2(data_dir, data_file, f_output):

    with open("{:s}/{:s}".format(data_dir, data_file), "r") as f_input:
        data_lines = f_input.readlines()

        par_names = ["sou_name", "epoch", "eph", "ra", "dec", "pmra", "pmdec", "plx",
                     "ra_err", "dec_err", "pmra_err", "pmdec_err", "plx_err"]

        par_values = {}

        nb_par_now = 0
        nb_par_tot = len(par_names)

        for data_line in data_lines:

            if nb_par_now >= nb_par_tot:
                break

            data_array = data_line.strip("\n").split()

            if len(data_array) == 0:
                continue

            line_head = data_array[0]

            # Pulsar name
            if line_head in ["PSRJ", "PSR"]:
                par_values["sou_name"] = data_array[1]
                nb_par_now += 1
                continue

            # Positional epoch
            if line_head in ["POSEPOCH"]:
                par_values["epoch"] = float(data_array[1])
                nb_par_now += 1
                continue

            #  Ephemeris
            if line_head in ["EPHEM"]:
                par_values["eph"] = data_array[1]
                nb_par_now += 1
                continue

            # Equatorial longitude
            if line_head in ["RAJ"]:
                lon, lon_err = parse_data_line2(data_line)
                par_values["ra"] = lon
                par_values["ra_err"] = lon_err
                nb_par_now += 2
                continue

            # Equatorial latitude
            if line_head in ["DECJ"]:
                lat, lat_err = parse_data_line2(data_line)
                par_values["dec"] = lat
                par_values["dec_err"] = lat_err
                nb_par_now += 2
                continue

            # Proper motion in Ecliptic longitude
            if line_head in ["PMRA"]:
                pmlon, pmlon_err = parse_data_line(data_line)
                par_values["pmra"] = pmlon
                par_values["pmra_err"] = pmlon_err
                nb_par_now += 2
                continue

            # Proper motion in Ecliptic latitude
            if line_head in ["PMDEC"]:
                pmlat, pmlat_err = parse_data_line(data_line)
                par_values["pmdec"] = pmlat
                par_values["pmdec_err"] = pmlat_err
                nb_par_now += 2
                continue

            # Parallax
            if line_head in ["PX"]:
                plx, plx_err = parse_data_line(data_line)
                par_values["plx"] = plx
                par_values["plx_err"] = plx_err
                nb_par_now += 2
                continue

        # Convert ra and dec from string to float
        ra_dec_str = par_values["ra"] + " " + par_values["dec"]
        c = SkyCoord(ra_dec_str, unit=(u.hourangle, u.deg))
        par_values["ra"] = c.ra.deg
        par_values["dec"] = c.dec.deg

        # Convert the unit of formal uncertainty
        par_values["ra_err"] = par_values["ra_err"] * \
            15e3 * np.cos(c.dec.rad)
        par_values["dec_err"] = par_values["dec_err"] * 1e3

        data_values = []
        for par_name in par_names:
            if par_name in par_values.keys():
                data_values.append(par_values[par_name])
            else:
                data_values.append(0)
        line_format = ["{:12s}  {:10.4f}  {:5s}"
                       + "  {:17.10f}" * 2
                       + "  {:10.3f}" * 3
                       + "  {:15.8f}" * 2
                       + "  {:12.6f}" * 3
                       ][0]

        print(line_format.format(*data_values), file=f_output)

    return None


def extract_data(data_dir, output_file, mode=1, ls_mode=1):

    if mode == 1:
        exec_func = parse_data_file
    else:
        exec_func = parse_data_file2

    with open("../data/{:}".format(output_file), "w") as f_output:

        # Add head information
        print("# psr_name pos_epoch eph ra dec pmra pmdec plx "
              "ra_err dec_err pmra_err pmdec_err plx_err\n"
              "# None MJD None deg deg mas/yr mas/yr mas mas mas mas/yr mas/yr mas",
              file=f_output)

        # Get the list of data files
        if ls_mode == 1:
            os.system("cd {:s} && ls */*.par > data_list.txt".format(data_dir))
        elif ls_mode == 2:
            os.system("cd {:s} && ls *.par > data_list.txt".format(data_dir))
        elif ls_mode == 3:
            os.system(
                "cd {:s} && ls */*.IPTADR2.par > data_list.txt".format(data_dir))

        file_list = np.genfromtxt(
            "{:s}/data_list.txt".format(data_dir), dtype=str)

        for data_file in file_list:
            exec_func(data_dir, data_file, f_output)


def extract_data2(input_file, output_file, mode=1):

    if mode == 1:
        par_names = ["psr_name", "pos_epoch", "ra_str",
                     "dec_str", "pmra_str", "pmdec_str", "plx_str"]
    elif mode == 2:
        par_names = ["psr_name", "pos_epoch", "ra", "dec", "pmra", "pmdec", "plx",
                     "ra_err", "dec_err", "pmra_err", "pmdec_err", "plx_err"]

    par_values = {}

    with open(output_file, "w") as f_output:

        # Print header
        tmp_str = [par_name + ", " for par_name in par_names]
        print("".join(tmp_str), file=f_output)

        f_ip = open(input_file, "r")
        data_lines = f_ip.readlines()
        f_ip.close()

        for data_line in data_lines:

            if len(data_line) == 0:
                continue

            dat_array = data_line.split()

            if dat_array[0] in par_names:
                par_values[dat_array[0]] = dat_array[1:]

        psr_num = len(par_values["psr_name"])

        for par_name in par_names:
            if par_name not in par_values.keys():
                par_values[par_name] = [""] * psr_num

        line_format = "{:s}, " * len(par_names)

        for i in range(psr_num):
            line_values = [par_values[par_name][i] for par_name in par_names]
            print(line_format.format(*line_values), file=f_output)

        return None


# 1) UTMOST-II
# 1a) PSR
extract_data("../data/TimingDataRelease1-master/pulsars",
             "utmost2-psr.dat", 2, 1)

# 1b) Extended PSR
extract_data("../data/TimingDataRelease1-master/extended_pulsars",
             "utmost2-ext-psr.dat", 2, 1)

# 2) VBC+09
extract_data2("../data/vbc+09_table345.dat", "../data/vbc+09_table.dat")

# 3) TSB+99
extract_data2("../data/tsb+99_table1.dat", "../data/tsb+99_table.dat")

# 4) HBO06
extract_data2("../data/hbo06_table234.dat", "../data/hbo06_table.dat")

# 5) SLR+14
extract_data2("../data/slr+14_table234.dat", "../data/slr+14_table.dat")

# 6) BHL+94
extract_data2("../data/bhl+94_table1.dat", "../data/bhl+94_table.dat")

# 7) CST96
extract_data2("../data/cst96_table4.dat", "../data/cst96_table.dat")

# 8) IPTA DR1
extract_data("../data/ipta-dr1/VersionC", "../data/ipta-dr1.dat", 2, 1)

# 9) IPTA DR2
extract_data("../data/ipta-dr2/VersionB", "../data/ipta-dr2.dat", 2, 3)

# 10) PPTA DR1e
extract_data("../data/ppta_dr1e_ephemerides", "ppta_dr1e.dat", 2, 2)

# 11) PPTA DR2e
extract_data("../data/ppta_dr2e_ephemerides", "ppta_dr2e.dat", 1, 2)
# extract_data2("../data/ppta_dr2e_cache.dat", "../data/ppta_dr2e1.dat")

# 12) PALFA
extract_data("../data/palfa_ephemerides", "palfa.dat", 2, 2)

# 13) DTB09
extract_data2("../data/dtb+09_table3.dat", "../data/dtb+09.dat", 2)

# 14) NANOGrav-5yr
extract_data("../data/NANOGrav_5y/par", "nanograv_5yr.dat", 2, 2)

# 1) NANOGrav
# Narrow band
# extract_data("../data/NANOGrav_12yv4/narrowband/par", "nanograv_nb.dat")

# Wide band
# extract_data("../data/NANOGrav_12yv4/wideband/par", "nanograv_wb.dat")
