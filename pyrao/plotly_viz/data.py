import os
import json
import pandas as pd
import numpy as np
from pyrao import BSAData
from datetime import datetime
from dateparser import parse

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
with open(os.path.join(__location__, "config.json"), 'r') as config_file:
    config = json.load(config_file)

BSAData.PATH_DATA = config['PATH_DATA']
BSAData.PATH_CALB = config['PATH_CALB']


def _read_file(date, stand):
    data = BSAData()
    data.read(date, stand)
    # data.calibrate()
    return data


def get_data(session_id, date, hour, minute, stand):
    """Read file and return two arrays: data matrix and datetime index."""
    # old_date = date[0]
    # new_date = date[1]
    new_date = datetime.strptime(date, "%Y-%m-%d")
    new_date = new_date.replace(hour=hour, minute=minute)

    # @cache.memoize()
    def query_data(session_id, new_date, stand):
        """
        Query data from source.

        Здесь нужно реализовать взаимодействие с БД
        и доступ к файлу по id времени.
        """
        data = _read_file(new_date, stand)

        return data.data, data.dt

    data, datetimes = query_data(session_id, new_date, stand)

    new_datetime = pd.to_datetime(new_date)
    start_dt = np.datetime64(new_datetime)
    end_dt = start_dt + np.timedelta64(1, 'h')
    start_ix = np.where(datetimes >= start_dt)[0][0]
    end_ix = np.where(datetimes <= end_dt)[0][-1]

    return data[start_ix:end_ix], datetimes[start_ix:end_ix]


def get_available_dates():
    return [datetime(2016, 7, 30, 23, 0)]
