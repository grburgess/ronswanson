import time
from abc import ABCMeta, abstractmethod
from typing import Dict, List

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
        self._energy_grid: List[np.ndarray] = energy_grid
        self._num_outputs: int = num_outputs

        if not len(self._energy_grid) == self._num_outputs:

            log.error(f"reuested number of outputs {self._num_outputs}")
            log.error(f"but only have {len(self._energy_grid)} energy grids")

            raise RuntimeError()

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

        output: Dict[str, np.ndarray] = self._run_call()

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

                    ene_grp = f.create_group("energy_grid")

                    for i, grid in enumerate(self._energy_grid):

                        ene_grp.create_dataset(
                            f"energy_grid_{i}",
                            data=grid,
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

                    val_grp: h5py.Group = f.create_group("values")

                    for i in range(self._num_outputs):

                        val_grp.create_group(f"output_{i}")

                values_group: h5py.Group = f["values"]

                for i in range(self._num_outputs):

                    out_group = values_group[f"output_{i}"]

                    out_group.create_dataset(
                        f"{new_key}",
                        data=output[f"output_{i}"],
                        compression="gzip",
                    )

                f.close()

                break

    @abstractmethod
    def _run_call(self) -> Dict[str, np.ndarray]:

        log.error("Attempting to use base class")

        raise RuntimeError()
