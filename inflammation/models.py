"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname = filename, delimiter = ',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.

    :param data: data from which to compute the mean
    """
    return np.mean(data, axis = 0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.

    :param data: data from which to compute the max
    """
    return np.max(data, axis = 0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.

    :param data: data from which to compute the mean
    """
    return np.min(data, axis = 0)


def patient_normalize(data):
    """Normalize patient data from a 2D inflammation data array.
    NaN values are ignored and normalized to 0.
    Negative values throw an input error.

    :param data: a numpy array of patient data
    """
    if np.any(data < 0):
        raise ValueError('Inflammation values should not be negative')

    patient_max = np.max(data, axis = 1)
    with np.errstate(invalid = 'ignore', divide = 'ignore'):
        normalized = data / patient_max[:, np.newaxis]
    normalized[np.isnan(normalized)] = 0
    normalized[normalized < 0] = 0
    return normalized


def patient_normalise(data):
    """Wrapper in case anyone tries to run the British spelling.

    :param data: a numpy array of patient data
    """
    return patient_normalize(data)
