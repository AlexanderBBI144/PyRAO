"""
Base module for PyRAO.

BSAData is the core class in PyRAO package. It is used to read, store,
calibrate and convert raw data files from BSA telescope.

"""
import warnings
import logging

import numpy as np
import pandas as pd
from angles import AlphaAngle, DeltaAngle, fmt_angle
from astropy.time import Time
from datetime import datetime
from dateutil.parser import parse

from .cinterp1d import cinterp1d
from .coords import dej, local_sid, ra, za
from .sigproc import write as sigproc_write

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG)

logger = logging.getLogger(name="pyrao.integration.bsadata")


class BSAData():
    """
    This is the base class for handling data gathered from BSA telescope.

    The class implements basic interface for interacting with BSA telescope
    observations. It allows to read them from .pnt, .pnthr, .csv files,
    or BSA database, store and calibrate them using .txt, .csv calibration
    files or calibration info from BSA database. Eventually, method
    `write_filterbank` allows to save data in common used SIGPROC
    filterbank (.fil) format.

    Attributes
    ----------
    data : numpy.ndarray
        Array of shape (`nsamples`, `nbeams`, `nbands`)
        with BSA observations.
    paths: array_like
        Array of paths to data and calibration files.
    nbeams : int
        Number of beams (rays) of BSA telescope.
    nbands : int
        Number of frequency bands for each beam.
    nsamples : int
        Number of data points for each frequency channel for each beam.
    stand : 1 or 2
        Index of stand.
    fbands : float
        Width of each frequency channel.
    cfreq : float
        Central frequency (the average value) of all frequency channels.
    resolution : float
        Time resolution in seconds.
    dt : numpy.ndarray
        Vector of datetime of each data point.
    mjd : numpy.ndarray
        Vector of mjd of each data point.
    ra : numpy.ndarray
        2D array of right ascensions for each band for each beam.
    dej : numpy.ndarray
        Vector of declinations for each beam.
    za : numpy.ndarray
        Vector of zenith angles for each beam.
    az : float
        Azimuth of BSA telescope.

    Methods
    -------
    read(PATH_DATA=None, start=0, stop=None)
        Read observations from the specified source.
    calibrate(path_to_calb=None)
        Calibrate observations using calibration info
        from the specified source.
    write_filterbank(path_to_output, beams=None)
        Write header info and observations to .fil file
    convert(paths, limits, beams)
        Read BSA-formatted data, calibrate it and write data
        to widely used .fil file format.

    """
    PATH_DATA = './'
    PATH_CALB = './'
    PATH_OUTPUT = './'
    DATA_EXT = '.pnt'
    CALB_EXT = '.txt'

    def __init__(self, nbeams=48, az=0.0, verbose=False):
        """Initialize variables."""
        self.lat = np.radians(54.8337300)
        self.lon = np.radians(37.620052)  # Must be checked
        self.nbeams = nbeams
        self.az = az
        self.verbose = verbose
        self.is_read = False
        self.is_calibrated = False

    def _correct_confused_beams(self):
        """
        Change confused beams on first stand between 25.07.2014 and 8.02.2016.

        в номерации лучей , как они обозначены в базе данных:
        ВМЕСТО 49, 50, 51,52,53,54,55,56
        БЫЛО: 49, 50, 56, 51,52,53,54,55

        С учетом того, что на первой стойке расположены
        с 33 по 80-й кабеля
        (так эти лучи обозначаются в базе данных)
        ЛУЧИ (если их считать в счете 1-48):
        ВМЕСТО 17, 18, 19, 20, 21, 22, 23, 24
        БЫЛО: 17, 18, 24, 19, 20, 21, 22, 23

        # УТОЧНИТЬ, ВКЛЮЧИТЕЛЬНО ЛИ ДАТЫ

        """
        conf_dates = (np.datetime64('2014-07-25') <= self.dt) * \
                     (self.dt < np.datetime64('2016-02-08'))
        if len(conf_dates) > 0:
            logger.info('Correcting rows with confused beams')
            true_order = [0,  1,  2,  3,  4,  5,
                          6,  7,  8,  9,  10, 11,
                          12, 13, 14, 15, 16, 17,
                          19, 20, 21, 22, 23, 18,
                          24, 25, 26, 27, 28, 29,
                          30, 31, 32, 33, 34, 35,
                          36, 37, 38, 39, 40, 41,
                          42, 43, 44, 45, 46, 47]
            self.data[conf_dates] = self.data[conf_dates][:, true_order]

    def _check_consistensy(self):
        required_fields = ['data', 'stand', 'beams', 'nbeams', 'nbands',
                           'nsamples', 'fbands', 'wbands', 'cfreq',
                           'resolution', 'mjd', 'lat', 'lon', 'dt',
                           'az', 'za', 'alt', 'dej', 'ra']
        try:
            for field in required_fields:
                getattr(self, field)
        except AttributeError:
            raise AttributeError('Some required attributes are missing.')

    def _read_db(self, limits=None):
        raise NotImplementedError

    def _read_pnt(self, path, limits=None):
        """Parse header and read observations from .pnt or .pnthr files."""
        with open(path, 'rb') as f:
            # Read header
            logger.info('Reading header')
            n = int(f.readline().decode('utf-8').split()[1])
            f.seek(0)
            header = {}
            for i in range(n):
                row = f.readline().decode('utf-8').split()
                header[row[0]] = row[1:]

            npoints_tot = int(header['npoints'][0])
            start = 0 if limits is None else limits[0]
            stop = npoints_tot if limits is None else limits[1]
            dt_start = parse(' '.join((header['date_begin'][0],
                                       header['time_begin'][0])))
            dt_stop = parse(' '.join((header['date_end'][0],
                                      header['time_end'][0])))
            self.dt = pd.date_range(dt_start, dt_stop, npoints_tot) \
                        .values[start:stop]
            self.mjd = np.linspace(Time(dt_start).mjd,
                                   Time(dt_stop).mjd,
                                   npoints_tot)[start:stop]
            self.nsamples = stop - start
            self.resolution = float(header['tresolution'][0])/1000
            self.stand = 1 if 'N1' in (path.split('/')[-1].split('_')) else 2
            self.nbands = int(header['nbands'][0]) + 1  # n bands + sum
            self.fbands = np.array(list(map(float, header['fbands'])))
            self.wbands = np.array(list(map(float, header['wbands'])))
            self.cfreq = float(header['fcentral'][0])

            # Read data
            logger.info('Reading data')
            f.seek(4 * start, 1)

            dims = (self.nsamples, self.nbeams, self.nbands)
            self.data = (np.fromfile(f,
                                     dtype='f4',
                                     sep='',
                                     count=np.prod(dims))
                           .reshape(dims))

            logger.info(f'Data is read: {str(self.data.shape)}')

    def _read_csv(self, path, limits=None):
        raise NotImplementedError

        # Load data csv
        df = pd.read_csv(path, index_col='datas_id')
        # Sort and drop NA values
        df = df[df['usability'] > 0].sort_values(['time']).dropna(axis=1)
        # Get df_eqv channels' names and transform
        # the corresponding columns of the df_eqv
        df_cnls = df.filter(like='cnl_').columns
        df[df_cnls] = df[df_cnls].applymap(lambda x: x[1:-1].split(','))

        self.nbands = df[df_cnls].values.shape

    def _read_fil(self, path, limits=None):
        raise NotImplementedError
        """
        # Read 1 byte of 4-bit unsigned integers
        with open(path, 'rb') as f:
            print(f.read(365))
            for i in range(5):
                a = f.read(1)
                print(a[0] & bytes([15])[0])
                print((a[0] & bytes([240])[0]) >> 4)
        """

    def read(self, date, stand, limits=None):
        """
        Read data from .pnt, .pnthr, .csv files or from BSADatabase.

        Parameters
        ----------
        limits : array_like
            Tuple of 2 indices: index to start and index to stop while reading.

        """
        logger.info("Reading data")
        if stand == 1:
            stand_name = 'N1_'
        elif stand == 2:
            stand_name = 'N2_'
        else:
            stand_name = ''
        ext = self.DATA_EXT if self.DATA_EXT is not None else ''
        self.filename = date.strftime(f"%d%m%y_%H_{stand_name}00{ext}")
        path = self.PATH_DATA + self.filename

        if ext == '.pnt' or ext == '.pnthr':
            self._read_pnt(path, limits)
        elif ext == '.csv':
            self._read_csv(path, limits)
        elif ext == '':
            self._read_db(date, limits)

        if self.stand == 1:
            self._correct_confused_beams()
            self.beams = np.arange(33, 81)
        else:
            self.beams = np.arange(81, 129)

        self.za = za(beams=self.beams, freqs=self.fbands)
        self.alt = np.pi / 2 - self.za
        self.dej = dej(lat=self.lat, za=self.za)
        self.ra = ra(az=self.az, alt=self.alt, lat=self.lat,
                     sid=local_sid(mjd=self.mjd, lon=self.lon))

        self._check_consistensy()
        self.is_read = True

    def _load_calb_db(self):
        raise NotImplementedError

    def _load_calb_txt(self, path):
        # ВОЗМОЖНО ТОЖЕ ПОМЕНЯТЬ ЛУЧИ
        logger.info(f"Reading {path}")

        with open(path) as f:
            d = [[y.strip() for y in x.split('|')] for x in f.readlines()]

        self.calb_range_mjd = np.unique([float(x[1]) for x in d])

        if min(self.mjd) < min(self.calb_range_mjd):
            raise ValueError("First date of data preceeds \
                              first date of calibration file.")
        elif max(self.mjd) > max(self.calb_range_mjd):
            raise ValueError("Last date of data exceeds \
                              last date of calibration file.")

        logger.info("Parsing")

        small_signal = np.array([[list(map(float, y[1:-1].split(',')))
                                  for y in x[5:]]
                                 for x in d[::2]])
        big_signal = np.array([[list(map(float, y[1:-1].split(',')))
                                for y in x[5:]]
                               for x in d[1::2]])
        t_eq = np.array([x[3] for x in d[::2]], dtype=float)[:,
                                                             np.newaxis,
                                                             np.newaxis]
        t_gsh = np.array([x[3] for x in d[1::2]], dtype=float)[:,
                                                               np.newaxis,
                                                               np.newaxis]
        self.coef = (t_gsh - t_eq) / (big_signal - small_signal)

    def _load_calb_csv(self, path):
        raise NotImplementedError

    def calibrate(self, filename):
        """
        Calibrate data.

        Perform calibration according to the following formula:

        DATA_calibrated = (DATA_original - ZERO_LEVEL) / ONE_KELVIN
        ZERO_LEVEL = SMALL_SIGNAL - (ONE_KELVIN * T_eq) - (ONE_KELVIN * T)
        T = (T_gsh - BIG_SIGNAL / SMALL_SIGNAL)
        / (BIG_SIGNAL / SMALL_SIGNAL - 1)
        ONE_KELVIN = (BIG_SIGNAL - SMALL_SIGNAL) / (T_gsh - T_eq)

        """
        # Check if data is processed
        if not self.is_read:
            raise ValueError("You have to read data first")

        # Interpolate calibration series and apply them to data
        logger.info("Calibrating data")

        ext = self.CALB_EXT if self.CALB_EXT is not None else ''
        # self.filename_calb = date.strftime("%y%m%d_%H_00") + ext
        self.filename_calb = filename + ext
        path = self.PATH_CALB + self.filename_calb

        if ext == '.txt':
            self._load_calb_txt(path)
        elif ext == '.csv':
            self._load_calb_csv(path)
        elif ext == '':
            self._load_calb_db()

        logger.info("Interpolating and applying to data")

        self.data *= cinterp1d(self.calb_range_mjd, self.coef, self.mjd)

        self.is_calibrated = True

    def _write_fits(self):
        raise NotImplementedError

    def _write_fil(self, ibeam):
        """
        Write calibrated data to .fil files.

        Method produces a separate .fil file for each beam index in `beams`.
        Resulting files contain header and observations data of shape
        (`self.nsamples`, `self.nbands`)

        Parameters
        ----------
        ibeam : int
            Beam index

        """
        header = {
         'az_start': self.az,
         'data_type': 1,
         'fch1': self.fbands[-1],
         'foff': -self.wbands[-1],
         'ibeam': ibeam,
         'machine_id': 10000,
         'nbeams': self.data.shape[1],
         'nbits': 32,
         'nchans': self.data.shape[2] - 1,  # Excluding sum of channels
         'nifs': 1,
         'nsamples': self.data.shape[0],
         'rawdatafile': self.filename,
         # Declination for central frequency channel
         'src_dej': fmt_angle(DeltaAngle(r=np.median(self.dej[ibeam])).d,
                              s1='', s2='', s3='', pre=2),
         # RA at the beginning of the observation, central freq channel
         'src_raj': fmt_angle(AlphaAngle(r=np.median(self.ra[0, ibeam])).h,
                              s1='', s2='', s3='', pre=2),
         'za_start': np.median(self.za[ibeam]),
         'telescope_id': 10000,
         'tsamp': self.resolution,
         'tstart': self.mjd[0]
        }
        path = self.PATH_OUTPUT \
            + f"{self.filename.split('.')[0]}beam{ibeam}.fil"

        beam = self.data[:, ibeam, :-1]
        logger.info(f'Writing {beam.shape[1]} frequency bands '
                    f'of {beam.shape[0]} samples to {path}')
        sigproc_write(path, header, np.flip(beam, 1))

    def write(self, beams=None, output_type='fil'):
        """
        Write calibrated data to .fil or .fits files.

        Resulting files contain header and observations data of shape
        (`self.nsamples`, `self.nbands`)

        Parameters
        ----------
        beams : array_like or None
            Array of beams' indices starting from zero. If set to None,
            the method will write `self.nbeams` files (one file per beam).

        """
        if not self.is_read:
            raise ValueError("You have to process data first")
        if not self.is_calibrated:
            warnings.warn('Data has not been calibrated.')
        for ibeam in beams:
            if ibeam > self.nbeams - 1:
                raise ValueError(f"Beams must be between 0 and {self.nbeams}")

        beams = np.arange(self.nbeams) if beams is None else beams
        beams = np.array([beams]) if isinstance(beams, int) else beams

        if output_type == 'fil':
            for ibeam in beams:
                self._write_fil(ibeam)
        elif output_type == 'fits':
            self._write_fits()
        else:
            raise ValueError("output_type must be one of the following: \
                              ['fil', 'fits']")

    def convert(self, date, limits=None, beams=None, output_type='fil'):
        """
        Convert data from .pnt, .pnthr, .csv files or from BSADatabase to .fil.

        Parameters
        ----------
        paths : array_like or str
            Tuple of 3 strings: paths to data, calibration info and output.
            If str then read from BSA database and write to the specified path.
        limits : array_like or None
            Tuple of 2 indices: index to start and index to stop while reading.
        beams : array_like or None
            Array of beams' indices starting from zero. If set to None,
            the method will write `self.nbeams` files (one file per beam).

        """
        self.read(date=date, limits=limits)
        self.calibrate()
        self.write(beams=beams, output_type=output_type)
