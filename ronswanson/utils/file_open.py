import time
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from .logging import setup_logger
from .configuration import ronswanson_config

log = setup_logger(__name__)


@contextmanager
def open_component_file(database_file_name: str, sim_id: int) -> h5py.File:

    p = Path(database_file_name)

    if ronswanson_config.slurm.store_dir is None:

        parent_dir = p.absolute().parent

    else:

        parent_dir = Path(ronswanson_config.slurm.store_dir).absolute()

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
