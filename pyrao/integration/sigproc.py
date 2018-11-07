"""An interface to SIGPROC filterbank format."""
import struct

header_params = {
    "HEADER_START": 'flag',
    "telescope_id": 'i',
    "machine_id": 'i',
    "data_type": 'i',
    "rawdatafile": 'str',
    "source_name": 'str',
    "barycentric": 'i',
    "pulsarcentric": 'i',
    "az_start": 'd',
    "za_start": 'd',
    "src_raj": 'd',
    "src_dej": 'd',
    "tstart": 'd',
    "tsamp": 'd',
    "nbits": 'i',
    "nsamples": 'i',
    "nbeams": "i",
    "ibeam": "i",
    "fch1": 'd',
    "foff": 'd',
    "FREQUENCY_START": 'flag',
    "fchannel": 'd',
    "FREQUENCY_END": 'flag',
    "nchans": 'i',
    "nifs": 'i',
    "refdm": 'd',
    "period": 'd',
    "npuls": 'q',
    "nbins": 'i',
    "HEADER_END": 'flag'}


def _append_to_hdr(param, value):
    base = struct.pack('i', len(param)) + param.encode('ascii')
    if header_params[param] == 'd':
        return base + struct.pack('d', float(value))
    elif header_params[param] == 'i':
        return base + struct.pack('i', int(value))
    elif header_params[param] == 'str':
        return base + struct.pack('i', len(value)) + value.encode('ascii')
    elif header_params[param] == 'flag':
        return base
    else:
        raise AttributeError(f"key '{param}' is unknown!")


def write(path_to_output, header, data):
    """
    Write header and data to SIGPROC filterbank format.

    Parameters
    ----------
    path_to_output : str
    header : dict
    data : numpy.ndarray
        2D array containing data with `float` or `uint` type.

    """
    with open(path_to_output, 'wb') as f:
        # Write header
        f.write(_append_to_hdr("HEADER_START", None))
        for paramname in list(header.keys()):
            if paramname in header_params:
                f.write(_append_to_hdr(paramname, header[paramname]))
        f.write(_append_to_hdr("HEADER_END", None))

        # Write observations
        if data is not None:
            dtype = 'float' if header['nbits'] == 32 else 'uint'
            data.flatten().astype(dtype + str(header['nbits'])).tofile(f)
