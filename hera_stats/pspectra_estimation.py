#!/usr/bin/env python3
"""The purpose of this script is to sort power spectra data based on high delay. At high delays we expect thermal
noise and 21-cm signal to be present. By looking at data sets dominated by thermal noise it can give insight on how
the noise and signal varies as the sky drifts with time.
"""
import argparse
import os
import yaml
from typing import List, Tuple

import hera_pspec as hp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from hera_pspec.data import DATA_PATH
from pyuvdata import UVData


def create_uvp_spec_obj(file_name: str) -> np.ndarray:
    """
    Creates a hera_pspec.conversions.Cosmo_Conversions object from file in order to define the the cosmology being used
    to analyze data set. Will yield a UVPSpec objects that holds all power spectra and its meta-data.

    Parameters
    ----------
    file_name: str
        file name that contains data set to analyze

    Returns
    -------
    uvp: np.ndarray
        UVPSpec object, containing delay spectra for a set of baseline-pairs, times, polarizations, and spectral
        windows.
    """
    uvd = UVData()
    uvd.read(file_name)
    cosmo = hp.conversions.Cosmo_Conversions()
    beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    uvb = hp.pspecbeam.PSpecBeamUV(beamfile, cosmo=cosmo)
    Jy_to_mK = uvb.Jy_to_mK(np.unique(uvd.freq_array), pol='XX')
    uvd.data_array *= Jy_to_mK[None, None, :, None]
    uvd1 = uvd.select(times=np.unique(uvd.time_array)[:-1:2], inplace=False)
    uvd2 = uvd.select(times=np.unique(uvd.time_array)[1::2], inplace=False)
    ds = hp.PSpecData(dsets=[uvd1, uvd2], wgts=[None, None], beam=uvb)
    ds.rephase_to_dset(0)
    ds.dsets[0].vis_units = 'mK'
    ds.dsets[1].vis_units = 'mK'
    baselines = uvd.get_antpairs()

    uvp = ds.pspec(baselines, baselines, (0, 1), [('nn', 'nn')], spw_ranges=[(300, 400)], input_data_weight='identity',
                   norm='I', taper='blackman-harris', verbose=False)
    return uvp


def create_keys(uvp: npt.ArrayLike) -> List[Tuple]:
    """
    Attains a list of all possible tuple keys in the data_array from the uvp object.

    Parameters
    ----------
    uvp: npt.ArrayLike
        UVPSpec object, containing delay spectra for a set of baseline-pairs,
        times, polarizations, and spectral windows.

    Returns
    -------
    key_list: List[Tuple]
        All possible tuple keys from uvp object.
    """
    key_list = uvp.get_all_keys()
    return key_list


def sort_out_bad_keys(key_list: List[Tuple], uvp: npt.ArrayLike) -> List[Tuple]:
    """
    Goes through key list and removes any keys that have the flag sum of 0 or baseline pairs
    where the antennas are the same.

    Parameters
    ----------
    key_list: List[Tuple]
        Key list that contains all possible tuple keys from uvp object.

    uvp: npt.ArrayLike
        UVPSpec object, containing delay spectra for a set of baseline-pairs,
        times, polarizations, and spectral windows.

    Returns
    -------
    key_list: List[Tuple]
        New filtered key list that do not have flag sum 0 or baseline pairs that have same antenna
    """
    list_of_bad_keys = []
    for key in key_list:
        antenna_1, antenna_2 = key[1]
        flg = uvp.get_wgts(key)[:, :, 0]  # dimension is the same as power (time vs delay) {0 means bad, 1 means good}
        sum_of_flag = np.sum(flg)
        if sum_of_flag == 0:
            list_of_bad_keys.append(key)
        elif antenna_1[0] == antenna_1[1] and antenna_1[0] == antenna_2[0] and antenna_1[0] == antenna_2[1]:
            list_of_bad_keys.append(key)

    for bad_key in list_of_bad_keys:
        key_list.remove(bad_key)
    return key_list


def parse_out_metadata(key_list: List[Tuple], uvp: npt.ArrayLike) -> Tuple[List[int], List[Tuple], List[npt.ArrayLike]]:
    """
    Obtains spectral window selection, baseline-pair selection, and polarization pair data from
    each key in key list and puts each data type into their respective list

    Parameters
    ----------
    key_list: List[Tuple]
        Sorted list of key that do not have flag sum 0 or baseline pairs that have same antenna

    uvp: npt.ArrayLike
        uvp object that contains all power-spectra and meta-data

    Returns
    -------
        spw_list: Tuple[List[int]
            Tuple that contains spectral window selection

        baselines_list: List[Tuple]
            List that contains baselines

        powers_list: List[npt.ArrayLike]]
            List that contains power spectra data.
    """
    spw_list = []
    baselines_list = []
    powers_list = []

    for key in key_list:
        # Attains spectral window selection to each corresponding key
        spw_list.append(key[0])
        # Attains base line to each corresponding key
        baselines_list.append(key[1])
        power = np.real(uvp.get_data(key))
        powers_list.append(power)
    return spw_list, baselines_list, powers_list


def get_sidereal_time(key_list: List[Tuple], uvp: npt.ArrayLike) -> Tuple[int, int, npt.ArrayLike]:
    """
    Obtains time indices and sidereal time with each corresponding key

    Parameters
    ----------
    key_list: List[Tuple]
        Sorted list of key that do not have flag sum 0 or baseline pairs that have same antenna

    uvp: npt.ArrayLike
        uvp object that contains all power-spectra and meta-data

    Returns
    -------
    length_of_keys: int
        size of keys list

    length_of_sidereal_time: int
        size of sidereal time list

    sidereal_time: npt.ArrayLike
        corresponding sidereal time to size list
    """
    time_index = uvp.key_to_indices(key_list[0])[1]
    sidereal_time = uvp.lst_avg_array[time_index]
    length_of_keys = len(key_list)
    length_of_sidereal_time = len(sidereal_time)
    return length_of_keys, length_of_sidereal_time, sidereal_time


def get_delays(spw_list: List[int], uvp: npt.ArrayLike) -> List[npt.ArrayLike]:
    """
    Obtains the delay power spectrum output by calculating the values from the beginning and end
    frequency channel of the spectral window

    Parameters
    ----------
    spw_list: List[int]
        list that contains spectral window selection

    uvp: npt.ArrayLike
        uvp object that contains all power-spectra and meta-data

    Returns
    -------
    list_of_delays: List[npt.ArrayLike]
        List that contains delays derived from spw.
    """
    list_of_delays = []
    for spw in spw_list:
        delays = uvp.get_dlys(spw) * 1e9
        list_of_delays.append(delays)
    return list_of_delays


def get_high_delays(list_of_delays: List[npt.ArrayLike], hdelay_val: int) -> Tuple[List[npt.ArrayLike], List[Tuple]]:
    """
    Goes through list of delays and parses out any high delay values (delays >= 2000)

    Parameters
    ----------
    list_of_delays: List[npt.ArrayLike]
        List of unsorted delays
        
    hdelay_val: int
        High delay value to observe at

    Returns
    -------
    high_delays_list: List[npt.ArrayLike]
        List of high delays

    high_delay_indices: List[Tuple]
        List of corresponding indices
    """
    high_delay_indices = []
    high_delays_list = []
    for delays in list_of_delays:
        for delays_index, delays_value in np.ndenumerate(delays):
            if np.abs(delays_value) >= hdelay_val:  
                high_delay_indices.append(delays_index)
        high_delays = np.squeeze(delays[high_delay_indices])
        high_delays_list.append(high_delays)
    return high_delays_list, high_delay_indices


def get_avg_power_array(powers_list: List[npt.ArrayLike], high_delay_indices: List[Tuple], length_of_keys: int,
                        length_of_sidereal_time: int) -> npt.ArrayLike:
    """
    Obtains the average power spectrum for each key

    Parameters
    ----------
    powers_list: List[npt.ArrayLike]
        List that contains power spectra data

    high_delay_indices: List[Tuple]
         List of high delay indices

    length_of_keys: int
        Size of keys list

    length_of_sidereal_time: int
        Size of sidereal time list

    Returns
    -------
    avg_power_array: npt.ArrayLike
        Array of the average power spectra
    """
    avg_power_array = np.zeros((length_of_keys, length_of_sidereal_time), dtype=object)
    for index, power in enumerate(powers_list):
        avg_power = np.mean(power[:, high_delay_indices], axis=1)
        avg_power_array[index] = np.squeeze(avg_power)
    return avg_power_array


def get_std_dev_power_array(powers_list: List[npt.ArrayLike], high_delays_indices: List[Tuple], length_of_keys: int,
                            length_of_sidereal_time: int) -> npt.ArrayLike:
    """
    Obtains the standard deviation of the power spectrum for each key

    Parameters
    ----------
    powers_list: List[npt.ArrayLike]
        List that contains power spectra data

    high_delays_indices: List[Tuple]
        List of high delay indices

    length_of_keys: int
        Size of keys list

    length_of_sidereal_time: int
        Size of sidereal time list

    Returns
    -------
    std_deviation_power_array: npt.ArrayLike
        Array of the the standard deviation of the power spectra
    """
    std_deviation_power_array = np.zeros((length_of_keys, length_of_sidereal_time), dtype=object)
    for index, power in enumerate(powers_list):
        std_dev = np.std(power[:, high_delays_indices], axis=1)
        std_deviation_power_array[index] = np.squeeze(std_dev)
    return std_deviation_power_array


def create_avg_power_array_graph(avg_power_array: npt.ArrayLike, sidereal_time: npt.ArrayLike, key_number: int) -> None:
    """
    Creates line graph of Mean of High Delays vs Local Sidereal Time

    Parameters
    ----------
    avg_power_array: npt.ArrayLike
        Array of the average power spectra

    sidereal_time: npt.ArrayLike
        Array that contains corresponding sidereal time to power array

    key_number: int
        Key that corresponds to the sidereal time and average power away

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        Matplotlib Figure instance
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(sidereal_time, avg_power_array[key_number], color="red", label="key #%s" % (key_number))
    ax.grid()
    ax.set_xlabel("Local Sidereal Time", fontsize=14)
    ax.set_ylabel("Mean of High Delays", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.title("Mean of High Delays vs Local Sidereal Time")
    plt.show()
    return fig


def create_std_dev_power_array_graph(std_dev_power_array: npt.ArrayLike, sidereal_time: npt.ArrayLike,
                                     key_number: int) -> plt.Figure:
    """
    Creates line graph of the standard deviation of High Delays vs Local Sidereal Time

    Parameters
    ----------
    std_dev_power_array: npt.ArrayLike
        Array of the power spectra standard deviation

    sidereal_time: npt.ArrayLike
        Array that contains corresponding sidereal time to power array

    key_number: int
        Key that corresponds to the sidereal time and standard deviation away

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        Matplotlib Figure instance
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(sidereal_time, std_dev_power_array[key_number], color="blue", label="key #%s" % (key_number))
    ax.grid()
    ax.set_xlabel("Local Sidereal Time", fontsize=14)
    ax.set_ylabel("Mean of High Delays", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.title("Standard Deviation vs Local Sidereal Time")
    plt.show()
    return fig


def get_histograms(powers_list: List[npt.ArrayLike], high_delay_indices: List[Tuple], key_list: List[Tuple]) \
        -> plt.Figure:
    """
    Creates histograms of Occurrences of Delay Value vs High Delay Value of each key

    Parameters
    ----------
    powers_list: List[npt.ArrayLike]
        List of power spectra

    high_delay_indices: List[Tuple]
        List of corresponding high delay index

    key_list: List[Tuple]
        List of keys

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        Matplotlib Figure instance
    """
    plt.rcParams.update({'figure.max_open_warning': 0})
    for index, power in enumerate(powers_list):
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.hist(power[:, high_delay_indices[index]].squeeze().flatten())
        ax.grid()
        ax.set_ylabel("Occurance of Delay Value", fontsize=14)
        ax.set_xlabel("High Delay Value", fontsize=14)
        plt.title("Histogram of Key: {}".format(key_list[index]))
        return fig


def show_graph_for_one_baseline(baselines: tuple, key_list: List, avg_power_array: npt.ArrayLike,
                                sidereal_time: npt.ArrayLike, std_dev_power_array: npt.ArrayLike) -> plt.Figure:
    """
    Displays line plots, and statistical tests for one specified baseline

    Parameters
    ----------
    baselines: tuple
        Tuple that contains baselines and its corresponding index

    key_list: List
        list of keys

    avg_power_array: npt.ArrayLike
        Array of average power of power spectra

    sidereal_time: npt.ArrayLike
        Array of corresponding sidereal time

    std_dev_power_array: npt.ArrayLike
        Array of standard deviation of power spectra

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        Matplotlib Figure instance.
    """
    for key in key_list:
        if baselines == key[1]:
            print("Key:", key, "For baseline: ", baselines)
            key_index = key_list.index(key)

            # To make a single line graph
            create_avg_power_array_graph(avg_power_array, sidereal_time, key_index)
            create_std_dev_power_array_graph(std_dev_power_array, sidereal_time, key_index)

            # To make a single histogram 
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.hist(avg_power_array[key_index])
            ax.grid()
            ax.set_xlabel("Mean of high delays", fontsize=14)
            ax.set_ylabel("Frequency", fontsize=14)
            plt.title("Histogram of Key: {}".format(key))
            return fig


def main():
    """
    Driver when running as a script
    """
    file_name = input("Please enter a filename: ")
    hdelay_val = int(input("Please enter the highdelay value you would like to look at. "))

    uvp_object = create_uvp_spec_obj(file_name)
    key_list = create_keys(uvp_object)
    key_list_final = sort_out_bad_keys(key_list, uvp_object)
    spw_list, baselines_list, powers_list = parse_out_metadata(key_list_final, uvp_object)
    length_of_keys, length_of_sidereal_time, sidereal_time = get_sidereal_time(key_list, uvp_object)
    list_of_delays = get_delays(spw_list, uvp_object)
    high_delays_list, high_delays_indicies = get_high_delays(list_of_delays, hdelay_val)

    avg_power_array = get_avg_power_array(powers_list, high_delays_indicies, length_of_keys, length_of_sidereal_time)
    std_dev_power_array = get_std_dev_power_array(powers_list, high_delays_indicies, length_of_keys,
                                                  length_of_sidereal_time)

    baseline_input = input("Please enter the baseline you would like to see line plots, and stastical tests for. ")
    baseline_input_list = baseline_input.replace(" ", "").split(',')

    for index in range(0, len(baseline_input_list)):
        baseline_input_list[index] = int(baseline_input_list[index])

    baseline_it = iter(baseline_input_list)
    baselines = []

    for index in baseline_it:
        baselines.append((index, next(baseline_it)))

    baselines = tuple(baselines)
    show_graph_for_one_baseline(baselines, key_list, avg_power_array, sidereal_time, std_dev_power_array)


def power_spectra_cli():
    """
    CLI option to run script
    """
    power_spectra_parser = argparse.ArgumentParser(
        description="The purpose of this script is to sort power spectra data based on high delay. At high delays we "
                    "expect thermalnoise and 21-cm signal to be present. By looking at data sets dominated by thermal "
                    "noise it can give insight on howthe noise and signal varies as the sky drifts with time.")
    power_spectra_parser.add_argument("-f", "--file", type=str, help="Input file name to read")

    power_spectra_args = power_spectra_parser.parse_args("-f ./example_input.yml".split())

    param_file = power_spectra_args.file

    with open(param_file, "r") as yaml_file:
        param_dict = yaml.safe_load(yaml_file)

    baseline_input_list = []
    for baseline in param_dict["select"].values():
        baseline_input_list = baseline.split(",")

    hdelay_val = param_dict["hdelay_val"]

    for index in range(0, len(baseline_input_list)):
        baseline_input_list[index] = int(baseline_input_list[index])

    baseline_it = iter(baseline_input_list)
    baselines = []

    for index in baseline_it:
        baselines.append((index, next(baseline_it)))

    baselines = tuple(baselines)

    for filename in param_dict["source_data"].values():  
        uvp_object = create_uvp_spec_obj(filename)
        key_list = create_keys(uvp_object)
        key_list_final = sort_out_bad_keys(key_list, uvp_object)
        spw_list, baselines_list, powers_list = parse_out_metadata(key_list_final, uvp_object)
        length_of_keys, length_of_sidereal_time, sidereal_time = get_sidereal_time(key_list, uvp_object)
        list_of_delays = get_delays(spw_list, uvp_object)
        high_delays_list, high_delays_indicies = get_high_delays(list_of_delays, hdelay_val)

        avg_power_array = get_avg_power_array(powers_list, high_delays_indicies, length_of_keys,
                                              length_of_sidereal_time)
        std_dev_power_array = get_std_dev_power_array(powers_list, high_delays_indicies, length_of_keys,
                                                      length_of_sidereal_time)

        show_graph_for_one_baseline(baselines, key_list, avg_power_array, sidereal_time, std_dev_power_array)


if __name__ == "__main__":
    main()