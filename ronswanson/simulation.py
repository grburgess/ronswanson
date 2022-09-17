import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

import ronswanson.grids as grids
from ronswanson.utils.configuration import ronswanson_config

from .utils.file_open import open_component_file, open_database
from .utils.logging import setup_logger

log = setup_logger(__name__)


def file_is_open(file_name: str):

    try:
        tmp = h5py.File(file_name, "a")

        return False, tmp

    except OSError:

        return True, None


class Simulation(metaclass=ABCMeta):
    def __init__(
        self,
        simulation_id: int,
        parameter_set: Dict[str, float],
        energy_grid: grids.EnergyGrid,
        out_file: str,
        num_outputs: int = 1,
    ) -> None:

        """
        Generic simulation class

        :param simulation_id:
        :type simulation_id: int
        :param parameter_set:
        :type parameter_set: Dict[str, float]
        :param energy_grid:
        :type energy_grid: np.ndarray
        :param out_file:
        :type out_file: str
        :returns:
        """
        self._out_file: str = out_file
        self._parameter_set: Dict[str, float] = parameter_set
        self._simulation_id: int = simulation_id
        self._energy_grid: List[grids.EnergyGrid] = energy_grid
        self._num_outputs: int = num_outputs

        if not len(self._energy_grid) == self._num_outputs:

            log.error(f"requested number of outputs {self._num_outputs}")
            log.error(f"but only have {len(self._energy_grid)} energy grids")

            raise RuntimeError()

    def run(self) -> None:

        """
        run this simulation
        :returns:

        """

        # run the simulation

        log.debug(f"simulation {self._simulation_id} is now running")

        output: Dict[str, np.ndarray] = self._run_call()

        with open_component_file(self._out_file, self._simulation_id) as f:

            f.create_dataset(
                "parameters", data=np.array(list(self._parameter_set.values()))
            )

            for i in range(self._num_outputs):

                f.create_dataset(f"output_{i}", data=output[f"output_{i}"])

    @abstractmethod
    def _run_call(self) -> Dict[str, np.ndarray]:

        log.error("Attempting to use base class")

        raise RuntimeError()


def gather(file_name: str, current_size: int = 0, clean: bool = True) -> None:

    # gather the list of files

    p = Path(file_name)

    if ronswanson_config.slurm.store_dir is None:

        parent_dir = p.absolute().parent

    else:

        parent_dir = Path(ronswanson_config.slurm.store_dir).absolute()

    multi_file_dir: Path = parent_dir / Path(f"{p.stem}_store")

    files = multi_file_dir.glob("sim_store_*.h5")

    with h5py.File(file_name, "a") as database_file:

        for store in files:

            log.debug(f"reading: {store}")

            sim_id: int = int(
                re.match("^sim_store_(\d*).h5", str(store.name)).groups()[0]
            )

            with h5py.File(str(store), "r") as f:

                database_file["parameters"][current_size + sim_id] = f[
                    "parameters"
                ][()]

                for k, v in database_file["values"].items():

                    v[current_size + sim_id] = f[k][()]

            if clean:

                log.debug(f"removing: {store}")

                store.unlink()

    if clean:

        multi_file_dir.rmdir()
