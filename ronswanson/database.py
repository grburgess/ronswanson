from collections import OrderedDict
from typing import Dict, List

import h5py
import numpy as np

from .simulation_builder import Parameter, ParameterGrid


class Database:
    def __init__(
        self,
        grid_points: Dict[str, np.ndarray],
        parameter_names: List[str],
        energy_grid: np.ndarray,
        values: Dict[str, np.ndarray],
    ) -> None:
        """
        Databse of parameters and simulated values

        :param grid_points:
        :type grid_points: Dict[str, np.ndarray]
        :param parameter_names:
        :type parameter_names: List[str]
        :param energy_grid:
        :type energy_grid: np.ndarray
        :param values:
        :type values: Dict[str, np.ndarray]
        :returns:

        """


        self._n_entries = len(values)

        self._n_parameters: int = len(parameter_names)

        self._parameter_names: List[str] = parameter_names

        self._energy_grid: np.ndarray = energy_grid

        # self._values: Dict[str, np.ndarray] = values
        # self._grid_points: Dict[str, np.ndarray] = grid_points

        self._values: np.ndarray = np.empty(
            (self._n_entries, len(self._energy_grid))
        )

        self._grid_points: np.ndarray = np.empty(
            (self._n_entries, self._n_parameters)
        )

        self._extract_parameter_values(values, grid_points)

    @property
    def n_entries(self) -> int:
        return self._n_entries

    @property
    def n_parameters(self) -> int:
        return self._n_parameters

    @property
    def parameter_ranges(self) -> Dict[str, np.ndarray]:
        return self._parameter_ranges



    def _extract_parameter_values(self, values, grid_points):

        """
        extract the values and parameter ranges for
        the tables

        :param values:
        :type values:
        :param grid_points:
        :type grid_points:
        :returns:

        """

        # extract the values

        for i in range(self._n_entries):

            self._values[i] = values[f"{i}"]
            self._grid_points[i] = grid_points[f"{i}"]

        self._parameter_ranges = OrderedDict()

        for i, par in enumerate(self._parameter_names):

            self._parameter_ranges[par] = np.sort(
                np.unique(self._grid_points[:, i])
            )

    @classmethod
    def from_file(cls, file_name: str) -> "Database":

        values = {}
        parameters = {}

        with h5py.File(file_name, "r") as f:

            energy_grid = f['energy_grid'][()]

            values_grp = f["values"]

            parameters_grp = f["parameters"]

            par_name_grp = f["parameter_names"]

            parameter_names = [
                par_name_grp.attrs[f"par{i}"]
                for i in range(len(par_name_grp.attrs))
            ]
            for key in values_grp.keys():

                values[key] = values_grp[key][()]
                parameters[key] = parameters_grp[key][()]

        return cls(parameters, parameter_names, energy_grid, values)

    def to_3ml(self):

        pass
