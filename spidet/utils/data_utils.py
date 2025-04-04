"""Data Utils module.

This module contains functions used to prepare data describing periods of time to numpy arrays, usable for fast computations.
"""

import pandas
import numpy as np


def change_interval(t, a: float, b: float, A: float = 0, B: float = 1):
    """
    Map t from interval [A, B] to interval [a, b]
    """
    t = (t - A) / (B - A)  # Normalize t to [0, 1]
    return b * t + a * (1 - t)


def transform_time_grades(
    time_grades: pandas.DataFrame, duration: float, N: int
) -> pandas.DataFrame:
    """Transforms the time grades into indices corresponding to the indices
    of the `H` matrix.

    Parameters
    ----------
    time_grades : pandas.DataFrame
        Time grades. Needs to contain the rows Description, Onset and Duration.
    duration : float
        The total duration of the corresponding recording.
    N : int
        The maximum index of `H`

    Returns
    -------
    pandas.DataFrame
        A dataframe made up of the indices corresponding to the IED events of the time_grades dataframe. Rows 'Onset' and 'Offset' determine start/end of an event.

    """
    drop = time_grades[time_grades["Description"] == "NOISY"].index
    time_grades = time_grades.drop(drop)
    time_grades = round(
        change_interval(time_grades.loc[:, "Onset":"Duration"], 0, N, 0, duration)
    )
    time_grades["Offset"] = time_grades.loc[:, "Onset"] + time_grades.loc[:, "Duration"]
    return time_grades


def create_from_periods(
    index_df: pandas.DataFrame,
    label_start: string,
    label_end: string,
    max_index: int,
    min_index: int = 0,
) -> np.ndarray:
    """Create a numpy array consisting of ones and zeros. Ones designate periods between start and end values, given by the columns in index_df, whose names should be given in label_start and label_end.
    To ensure that the final array corresponds in size to the size of the original data, max and min indices need to be set.

    Parameters
    ----------
    index_df : pandas.DataFrame
        A dataframe containing at least the columns label_start and label_end.
    label_start : string
        Designates the column name corresponding to start indices contained in index_df.
    label_end : string
        Designates the column name corresponding to end indices contained in index_df.
    max_index : int
        The largest possible index (does not have to be contained in index_df).
    min_index : int
        The smallest possible index (usually zero).

    Returns
    -------
    np.ndarray
        A numpy array mask with ones where events are happening and the rest being zero.

    """
    result = np.zeros(max_index - min_index)
    for _, row in index_df.iterrows():
        if np.isnan(row.loc[label_start]) or np.isnan(row.loc[label_end]):
            continue
        period_start = int(row.loc[label_start])
        period_end = int(row.loc[label_end])

        assert period_end >= period_start

        if period_end < min_index or period_start > max_index:
            continue

        # clamp start and end indices to valid values
        period_start = max(period_start, min_index)
        period_end = min(period_end, max_index)

        result[max(0, period_start) : min(len(result) - 1, period_end)] = 1
    return result


def time_grade_predictions(
    time_grades: pandas.DataFrame, duration: float, N: int
) -> np.ndarray:
    """Create a numpy array with ones on indices where events are indicated by time_grades columns and zeros elsewhere.

    Parameters
    ----------
    time_grades : pandas.DataFrame
        A dataframe containing columns labelled with "Onset" and "Offset", indicating the start and end of events. The columns data is in time format, e. g. if a event was detected after 10 seconds of recording which lasts for 10 seconds, there would be a row in time_grades where the column "Onset" would contain the value 10 and the column "Offset" would contain the value 20.
    duration : float
        The duration of the corresponding data
    N : int
        The size of the corresponding data.

    Returns
    -------
    np.ndarray
        Numpy array made of zeros and ones of shape (N,)

    """
    time_grades = transform_time_grades(time_grades, duration, N)
    return create_from_periods(time_grades, "Onset", "Offset", N, 0)
