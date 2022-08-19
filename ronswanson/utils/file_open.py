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

            time.sleep(1)

            # try:

            #     if Path(file_name).exists():

            #         tmp = h5py.File(
            #             file_name,
            #             "r",
            #             libver='latest',
            #             swmr=True,
            #             # driver='core',
            #             # backing_store=True,
            #         )

            #         tmp.close()

            #     else:

            #         log.error("There is no database file to write to!")

            #         raise RuntimeError("There is no database file to write to!")

            # except (OSError, BlockingIOError) as e:

            #     log.debug(f"{e}")

            #     log.debug("file was already accessed")
            #     log.debug(f"simulation {sim_id} will continue to wait")

            if False:

                pass

            else:

                f = h5py.File(
                    file_name,
                    "a",
                    libver='latest',
                    swmr=True,
                )

                #                f.swmr_mode = True

                log.debug(f"file is accessed by simulation {sim_id}!")

                break

        else:

            log.debug(f"file is open so simulation {sim_id} waits")

            time.sleep(np.random.uniform(0.01, 3.0))

    try:

        yield f

        log.debug(f"simulation {sim_id} finished storing")

    finally:

        log.debug(f"simulation {sim_id} is closing the file")

        f.close()

        if block_file.exists():

            block_file.unlink()

        else:

            log.error("the block file was supposed to exist but it did not!")

            raise RuntimeError(
                "the block file was supposed to exist but it did not!"
            )

        log.debug(f"simulation {sim_id} is unblocking")


@contextmanager
def open_component_file(database_file_name: str, sim_id: int) -> h5py.File:

    p = Path(database_file_name)

    parent_dir = p.absolute().parent

    multi_file_dir: Path = parent_dir / Path(f"{p.stem}_store")

    if not multi_file_dir.exists():

        log.debug(f"creating the store directory: {multi_file_dir}")

        multi_file_dir.mkdir()

    this_file: Path = multi_file_dir / f"sim_store_{sim_id}.h5"

    if this_file.exists():

        log.error(f"{this_file} is getting ready to be written but it exists!")

        raise RuntimeError(
            f"{this_file} is getting ready to be written but it exists!"
        )

    f = h5py.File(str(this_file), "w")

    try:

        yield f

    finally:

        f.close()

        log.debug(f"closing {this_file}")
