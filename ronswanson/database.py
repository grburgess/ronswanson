from collections import OrderedDict
from typing import Dict, List, Optional

import h5py
import numpy as np
from astromodels import TemplateModel, TemplateModelFactory

from .simulation_builder import Parameter, ParameterGrid
from .utils.logging import setup_logger

log = setup_logger(__name__)


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

    def _sub_selection(
        self,
        paramater_name: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> np.ndarray:

        if paramater_name not in self._parameter_names:

            log.error(f"{paramater_name} is not in {self._parameter_names}")

            raise AssertionError()

        par_idx: int = self._parameter_names.index(paramater_name)

        selection = np.ones(self._n_entries, dtype=bool)

        if vmin is not None:

            selection = (self._grid_points[..., par_idx] >= vmin) & selection

        if vmax is not None:

            selection = (self._grid_points[..., par_idx] <= vmax) & selection

        return selection

    def _parameter_sub_selection(
        self,
        paramater_name: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> np.ndarray:

        if paramater_name not in self._parameter_names:

            log.error(f"{paramater_name} is not in {self._parameter_names}")

            raise AssertionError()

        par_range = self._parameter_ranges[paramater_name]

        selection = np.ones(len(par_range), dtype=bool)

        if vmin is not None:

            selection = (par_range >= vmin) & selection

        if vmax is not None:

            selection = (par_range <= vmax) & selection

        return selection

    @classmethod
    def from_file(cls, file_name: str, output: int = 0) -> "Database":
        """
        open a database from a file.
        The output argument specifies which value to collect.
        For example, a simulation may save photon and electron
        distributions from a solution.
        """
        values = {}
        parameters = {}

        with h5py.File(file_name, "r") as f:

            energy_grid = f['energy_grid'][()]

            values_grp = f["values"][f"output_{output}"]

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

    def to_3ml(
        self,
        name: str,
        desc: str,
        overwrite: bool = False,
        **kwargs,
    ) -> TemplateModel:

        """
        construct a table model from the database.
        parameter sub-selections are passed as kwargs of
        dictionaries:

        selections = dict(param1=dict(vmin=1, vmax=2))

        :param name:
        :type name: str
        :param desc:
        :type desc: str
        :param overwrite:
        :type overwrite: bool
        :returns:

        """
        selection = np.ones(self._n_entries, dtype=bool)

        parameter_selection = {}

        for k, v in self._parameter_ranges.items():

            parameter_selection[k] = np.ones(len(v), dtype=bool)

        for k, v in kwargs.items():

            if k in self._parameter_names:

                vmin = None
                vmax = None

                if 'vmin' in v:

                    vmin = v['vmin']

                if 'vmax' in v:

                    vmax = v['vmax']

                selection = selection & self._sub_selection(k, vmin, vmax)

                parameter_selection[k] = parameter_selection[
                    k
                ] & self._parameter_sub_selection(k, vmin, vmax)

        # sub selections if any

        sub_grid = self._grid_points[selection, ...]
        sub_values = self._values[selection, ...]

        sub_parameter_ranges = {}

        for k, v in parameter_selection.items():

            sub_parameter_ranges[k] = self._parameter_ranges[k][
                parameter_selection[k]
            ]

        tmf = TemplateModelFactory(
            name, desc, self._energy_grid, self._parameter_names
        )

        for k, v in sub_parameter_ranges.items():

            tmf.define_parameter_grid(k, sub_parameter_ranges[k])

        for i in range(len(sub_values)):

            ### DO NOT SORT

            tmf.add_interpolation_data(
                sub_values[i],
                **{k: v for k, v in zip(self._parameter_names, sub_grid[i])},
            )

        tmf.save_data(overwrite=True)

        return TemplateModel(name)


__all__ = ["Database"]
