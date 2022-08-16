import time
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from .logging import setup_logger

log = setup_logger(__name__)


@contextmanager
def open_database(file_name: str) -> h5py.File:

    block_file: Path = Path("HDF5_DATABSE_OPEN")

    while True:

        if not block_file.exists():

            # create the file to block

            block_file.touch()

            # check one more time that
            # the file is not accessed

            time.sleep(1)

            try:

                tmp = h5py.File(file_name, "r")

            except OSError:

                log.debug("file was already accessed")
                log.debug("will continue to wait")

            else:

                tmp.close()

                f = h5py.File(file_name, "a")

                log.debug("file is accessed!")

                break

        else:

            log.debug("file is open so we wait")

            time.sleep(np.random.uniform(3.0, 5.0))

    try:

        yield f

    finally:

        f.close()

        block_file.unlink()
