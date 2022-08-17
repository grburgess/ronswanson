import time
from abc import ABCMeta, abstractmethod
from typing import Dict, List

import h5py
import numpy as np

import ronswanson.simulation_builder as sb

from .utils.logging import setup_logger
from .utils.file_open import open_database


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
        energy_grid: sb.EnergyGrid,
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
        self._energy_grid: List[sb.EnergyGrid] = energy_grid
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

        # while True:

        #     test, f = file_is_open(self._out_file)

        with open_database(self._out_file, self._simulation_id) as f:

            log.debug(f"simulation {self._simulation_id} is storing")

            # store the parameter names

            param_dataset: h5py.Dataset = f["parameters"]

            param_dataset.resize(
                (param_dataset.shape[0] + 1,) + param_dataset.shape[1:]
            )

            param_dataset[-1] = np.array(list(self._parameter_set.values()))

            values_group: h5py.Group = f["values"]

            for i in range(self._num_outputs):

                out_group: h5py.Group = values_group[f"output_{i}"]
                values_dataset: h5py.Dataset = out_group["values"]

                values_dataset.resize(
                    (values_dataset.shape[0] + 1,) + values_dataset.shape[1:]
                )

                values_dataset[-1] = output[f"output_{i}"]

    @abstractmethod
    def _run_call(self) -> Dict[str, np.ndarray]:

        log.error("Attempting to use base class")

        raise RuntimeError()
