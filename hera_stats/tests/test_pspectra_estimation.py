import copy
import numpy as np

from power_spectra_estimation import create_uvp_spec_obj, create_keys, sort_out_bad_keys, parse_out_metadata, \
    get_sidereal_time, get_delays, get_high_delays, get_avg_power_array, get_std_dev_power_array


def set_up():
    file_name = "/Users/pelmini/hera/highdelay_pspectra/epoch_auto_P_IDR2_field1_x.uvh5"
    uvp_object = create_uvp_spec_obj(file_name)
    return uvp_object


def test_create_keys():
    """Tests if keys are correctly collected make from uvp object, checks instance
    type instead of direct comparison as key object is very long"""
    uvp_object = set_up()
    result = create_keys(uvp_object)
    assert isinstance(result, list)


def test_sort_out_bad_keys():
    """Tests if keys have been properly parsed (created copy of key_list because key list
    was being overwritten)"""
    uvp_object = set_up()
    key_list = create_keys(uvp_object)
    result = sort_out_bad_keys(copy.copy(key_list), uvp_object)
    assert key_list != result


def test_parse_out_metadata():
    """Tests if wanted meta data has been parsed out"""
    uvp_object = set_up()
    key_list = create_keys(uvp_object)
    result_spw, result_baseline, result_powers = parse_out_metadata(key_list, uvp_object)
    assert isinstance(result_spw, list)
    assert isinstance(result_baseline, list)
    assert isinstance(result_powers, list)


def test_get_sidereal_time():
    """Tests if time indices and sidereal time with each corresponding key are obtained"""
    uvp_object = set_up()
    key_list = create_keys(uvp_object)
    length_of_keys, length_of_sidereal_time, sidereal_time = get_sidereal_time(key_list, uvp_object)
    assert length_of_keys == 85
    assert length_of_sidereal_time == 120
    assert isinstance(sidereal_time, np.ndarray)


def test_get_delays():
    """Tests if delays from power spectrum is obtained"""
    uvp_object = set_up()
    key_list = create_keys(uvp_object)
    spw_list = parse_out_metadata(key_list, uvp_object)[0]
    result = get_delays(spw_list, uvp_object)
    assert isinstance(result, list)


def test_get_high_delays():
    """Tests if high delays from delay list are obtained"""
    uvp_object = set_up()
    key_list = create_keys(uvp_object)
    spw_list = parse_out_metadata(key_list, uvp_object)[0]
    delay_list = get_delays(spw_list, uvp_object)
    result_hdelays, result_hindicies = get_high_delays(delay_list)

    assert len(result_hdelays) == 85
    assert len(result_hindicies) == 5185


def test_get_avg_power_array():
    """Test for avg power array function"""
    uvp_object = set_up()
    key_list = create_keys(uvp_object)
    spw_list, powers_list = parse_out_metadata(key_list, uvp_object)[::2]
    length_of_keys, sidereal_time_length = get_sidereal_time(key_list, uvp_object)[:2]
    delay_list = get_delays(spw_list, uvp_object)
    hindicies = get_high_delays(delay_list)[1]
    result = get_avg_power_array(powers_list, hindicies, length_of_keys, sidereal_time_length)
    assert isinstance(result, np.ndarray)


def test_get_std_dev_power_array():
    """Test for std deviation of power array function"""
    uvp_object = set_up()
    key_list = create_keys(uvp_object)
    spw_list, powers_list = parse_out_metadata(key_list, uvp_object)[::2]
    length_of_keys, sidereal_time_length = get_sidereal_time(key_list, uvp_object)[:2]
    delay_list = get_delays(spw_list, uvp_object)
    hindicies = get_high_delays(delay_list)[1]
    result = get_std_dev_power_array(powers_list, hindicies, length_of_keys, sidereal_time_length)
    assert isinstance(result, np.ndarray)