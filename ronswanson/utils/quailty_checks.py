import numpy as np

import ronswanson.database as database
from ronswanson.utils.logging import setup_logger

log = setup_logger(__name__)


def first_last_nonzero(arr, axis, invalid_val=-1):
    mask = arr > 0.0

    first = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

    mask = np.flip(mask, axis=1)

    last = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    return first, -last


def check_for_non_contiguous_spectra(database: database.Database) -> np.ndarray:
    """
    find spectra with non-contiguous values (go to zero in between)

    :param database:
    :type database: Database
    :returns:

    """

    ok_idx = np.ones(database.n_entries, dtype=bool)

    first, last = first_last_nonzero(database.values, axis=1)

    for i, datum in enumerate(database.values):

        if np.any(datum[first[i] : last[i]] == 0):

            ok_idx[i] = 0

    log.info(f"there were {(~ok_idx).sum()} dirty entries")

    return ~ok_idx
