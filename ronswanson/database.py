from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import h5py
import numpy as np
from astromodels import TemplateModel, TemplateModelFactory
from astromodels.utils.logging import silence_console_log
from tqdm.auto import tqdm

from .utils.logging import setup_logger

log = setup_logger(__name__)


@dataclass(frozen=True)
class ValueContainer:
    params: np.ndarray
    values: np.ndarray


class Database:
    def __init__(
        self,
        grid_points: np.ndarray,
        parameter_names: List[str],
        energy_grid: np.ndarray,
        run_time: np.ndarray,
        values: np.ndarray,
        meta_data: Optional[Dict[str, np.ndarray]] = None,
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

        self._values: np.ndarray = values

        self._grid_points: np.ndarray = grid_points

        self._run_time: np.ndarray = run_time

        self._parameter_ranges: Dict[str, np.ndarray] = OrderedDict()

        for i, par in enumerate(self._parameter_names):

            self._parameter_ranges[par] = np.sort(
                np.unique(self._grid_points[:, i])
            )

        self._meta_data: Optional[Dict[str, np.ndarray]] = meta_data

    @property
    def n_entries(self) -> int:
        return self._n_entries

    @property
    def n_parameters(self) -> int:
        return self._n_parameters

    @property
    def parameter_ranges(self) -> Dict[str, np.ndarray]:
        return self._parameter_ranges

    @property
    def paramerter_names(self) -> List[str]:
        return self._parameter_names

    @property
    def run_time(self) -> np.ndarray:
        return self._run_time

    @property
    def energy_grid(self) -> np.ndarray:
        return self._energy_grid

    @property
    def meta_data(self) -> Optional[Dict[str, np.ndarray]]:
        return self._meta_data

    def at(self, i: int) -> ValueContainer:
        """
        return the parameters and values
        at an index
        """

        return ValueContainer(
            params=self._grid_points[i, :], values=self._values[i, :]
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

            energy_grid = f['energy_grid'][f'energy_grid_{output}'][()]

            values_grp = f["values"]

            run_time = f["run_time"][()]

            par_name_grp = f["parameter_names"]

            parameter_names = [
                par_name_grp.attrs[f"par{i}"]
                for i in range(len(par_name_grp.attrs))
            ]

            parameters = f['parameters'][()]

            values = values_grp[f'output_{output}'][()]

            meta_data = None

            if "meta" in list(f.keys()):

                meta_data = {}

                for k, v in f['meta'].items():

                    meta_data[k] = v[()]

        return cls(
            grid_points=parameters,
            parameter_names=parameter_names,
            energy_grid=energy_grid,
            values=values,
            run_time=run_time,
            meta_data=meta_data,
        )

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

        with silence_console_log():

            for i in tqdm(range(len(sub_values))):

                ### DO NOT SORT

                tmf.add_interpolation_data(
                    sub_values[i],
                    **{
                        k: v for k, v in zip(self._parameter_names, sub_grid[i])
                    },
                )

            tmf.save_data(overwrite=overwrite)

        return TemplateModel(name)


def merge_databases(
    *file_names: List[str], new_name: str = "merged_db.h5"
) -> None:

    """TODO describe function

    :param new_name:
    :type new_name: str
    :returns:

    """
    n_entries = 0

    for i, fname in enumerate(file_names):

        with h5py.File(fname, "r") as f:

            n_entries += f["parameters"].shape[0]

            if i == 0:

                par_name_grp = f["parameter_names"]

                parameter_names = [
                    par_name_grp.attrs[f"par{i}"]
                    for i in range(len(par_name_grp.attrs))
                ]

                energy_grids = []

                for _, v in f["energy_grid"].items():

                    energy_grids.append(v[()])

        n_outputs = len(energy_grids)

        n_parameters = len(parameter_names)

        parameters = np.zeros((n_entries, n_parameters))

        values = []

        for grid in energy_grids:

            values.append(np.zeros((n_entries, len(grid))))

        k = 0

        # extract all the information

        for i, fname in enumerate(file_names):

            with h5py.File(fname, "r") as f:

                n = f["parameters"].shape[0]

                parameter_names[k:n, :] = f["parameters"][()]

                for j, val in enumerate(values):

                    val[k:n] = f["values"][f"output_{j}"][()]

                k += n

        # create the new file

        with h5py.File(new_name, "w") as f:

            f.create_dataset("parameters", data=parameters)

            val_grp = f.create_group("values")
            energy_grid_grp = f.create_group("energy_grid")

            par_name_grp = f.create_group("parameters_names")
            for i in range(n_parameters):

                par_name_grp.attrs[f"par{i}"] = parameter_names[i]

            for i in range(n_outputs):

                val_grp.create_dataset(f"output_{i}", data=values[i])
                energy_grid_grp.create_dataset(
                    f"energy_grid_{i}", data=energy_grids[i]
                )


__all__ = ["Database"]
