import time
from abc import ABCMeta, abstractmethod
from typing import Dict

import h5py
import numpy as np

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
        energy_grid: np.ndarray,
        out_file: str,
    ) -> None:

        self._out_file: str = out_file
        self._parameter_set: Dict[str, float] = parameter_set
        self._simulation_id: int = simulation_id
        self._energy_grid: np.ndarray = energy_grid

    def run(self) -> None:

        """
        run this simulation
        :returns:

        """

        # check if this already exists

        params = np.array(list(self._parameter_set.values()))

        run_flag: bool = True

        while True:

            test, f = file_is_open(self._out_file)

            if test:

                # the file is already open so wait

                log.debug(
                    f"simulation {self._simulation_id} is waiting on file to be closed"
                )

                time.sleep(np.random.randint(3, 5))

            else:

                if "parameters" not in f.keys():

                    # ok, this is a brand new file
                    log.debug("New database file")
                    break

                else:

                    for k, v in f["parameters"].items():

                        if np.alltrue(v[()] == params):

                            # this parameter set exists

                            run_flag = False

                            log.debug(f"parameters {v[()]} already exist!")

                            break

                f.close()

                break

        if not run_flag:

            log.debug(f"simulation {self._simulation_id} not running")

            return

        # run the simulation

        log.debug(f"simulation {self._simulation_id} is now running")

        output: np.ndarray = self._run_call()

        while True:

            test, f = file_is_open(self._out_file)

            if test:

                # the file is already open so wait

                log.debug(
                    f"simulation {self._simulation_id} is waiting on file to be closed"
                )

                time.sleep(np.random.randint(3, 5))

            else:

                log.debug(f"simulation {self._simulation_id} is storing")

                # store the parameter names

                if "parameter_names" not in f.keys():

                    p_name_group = f.create_group("parameter_names")

                    for i, name in enumerate(list(self._parameter_set.keys())):

                        p_name_group.attrs[f"par{i}"] = name

                # store the energy grid

                if "energy_grid" not in f.keys():

                    f.create_dataset(
                        "energy_grid",
                        data=self._energy_grid,
                        compression="gzip",
                    )

                if "parameters" not in f.keys():

                    f.create_group("parameters")

                param_group: h5py.Group = f["parameters"]

                number_of_entries: int = len(param_group.keys())

                new_key: int = number_of_entries

                params = np.array(list(self._parameter_set.values()))

                param_group.create_dataset(
                    f"{new_key}", data=params, compression="gzip"
                )

                if "values" not in f.keys():

                    f.create_group("values")

                values_group: h5py.Group = f["values"]

                values_group.create_dataset(
                    f"{new_key}", data=output, compression="gzip"
                )

                f.close()

                break

    @abstractmethod
    def _run_call(self) -> np.ndarray:

        log.error("Attempting to use base class")

        raise RuntimeError()
