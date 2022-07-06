import time
from abc import ABCMeta, abstractmethod

import h5py
import numpy as np

from .utils.logging import setup_logger

log = setup_logger(__name__)


def file_is_open(file_name: str):

    try:
        tmp = h5py.File(file_name, "w")

        return False, tmp

    except OSError:

        return True, _


class Simulation(metaclass=ABCMeta):
    def __init__(
        self, simulation_id: int, parameter_set: np.ndarray, out_file: str
    ) -> None:

        self._out_file: str = out_file
        self._parameter_set: np.ndarray = parameter_set
        self._simulation_id: int = simulation_id

    def run(self) -> None:

        """
        run this simulation
        :returns:

        """
        output: np.ndarray = self._run_call()

        while True:

            test, f = file_is_open(self._out_file)

            if test:

                # the file is already open so wait

                log.debug(
                    f"simulation {self._simulation_id} is waiting on file to be closed"
                )

                time.sleep(3)

            else:

                param_group: h5py.Group = f["paramaters"]

                number_of_entries: int = len(param_group.keys())

                new_key: int = number_of_entries + 1

                param_group.create_dataset(
                    f"{new_key}", data=self._parameter_set, compression="gzip"
                )

                values_group: h5py.Group = f["values"]

                values_group.create_dataset(
                    f"{new_key}", data=output, compression="gzip"
                )

                f.close()

    @abstractmethod
    def _run_call(self) -> np.ndarray:


        log.error("Attempting to use base class")

        raise RuntimeError()
