import time
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from .logging import setup_logger

log = setup_logger(__name__)


@contextmanager
def open_database(file_name: str, sim_id: int) -> h5py.File:

    block_file: Path = Path("HDF5_DATABASE_OPEN")

    while True:

        if not block_file.exists():

            # create the file to block

            block_file.touch()

            # check one more time that
            # the file is not accessed

            time.sleep(3)

            try:

                if Path(file_name).exists():

                    tmp = h5py.File(file_name, "r")

                else:

                    tmp = h5py.File(file_name, "a")

            except (OSError, BlockingIOError):

                log.debug("file was already accessed")
                log.debug(f"simulation {sim_id} will continue to wait")

            else:

                tmp.close()

                f = h5py.File(file_name, "a")

                log.debug(f"file is accessed by simulation {sim_id}!")

                break

        else:

            log.debug(f"file is open so simulation {sim_id} wait")

            time.sleep(np.random.uniform(3.0, 5.0))

    try:

        yield f

        log.debug(f"simulation {sim_id} finished storing")

    finally:

        log.debug(f"simulation {sim_id} is closing the file")

        f.close()

        time.sleep(2)

        if block_file.exists():

            block_file.unlink()

        else:

            log.error("the block file was supposed to exist but it did not!")

            raise RuntimeError(
                "the block file was supposed to exist but it did not!"
            )

        log.debug(f"simulation {sim_id} is unblocking")
