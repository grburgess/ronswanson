import collections
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import astropy.units as u
import h5py
import numpy as np
import plotly.graph_objects as go
from astromodels import TemplateModel, TemplateModelFactory
from astromodels.functions.template_model import TemplateFile
from astromodels.utils import get_user_data_path
from astromodels.utils.logging import silence_console_log
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ronswanson.grids import Parameter, ParameterGrid
from ronswanson.utils.cartesian_product import cartesian_jit
from ronswanson.utils.color import Colors

from .utils.logging import setup_logger

log = setup_logger(__name__)


@dataclass(frozen=True)
class ValueContainer:
    params: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class SelectionContainer:
    sub_grid: np.ndarray
    sub_values: np.ndarray
    sub_range: Dict[str, np.ndarray]
    selection: np.ndarray


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

        for i, name in enumerate(self._parameter_names):

            if not isinstance(name, str):

                self._parameter_names[i] = name.decode()

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

        if np.any(~np.isfinite(self._values)):

            log.error("The table values contain non-finite values")
            log.info(
                "you can replace these by calling the replace_nan_inf_with() member"
            )

    @property
    def n_entries(self) -> int:
        """
        the number of entries in the database

        :returns:

        """

        return self._n_entries

    @property
    def n_parameters(self) -> int:
        """
        the number of parameters in the database

        :returns:

        """

        return self._n_parameters

    @property
    def grid_points(self) -> np.ndarray:
        return self._grid_points

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def parameter_ranges(self) -> Dict[str, np.ndarray]:
        return self._parameter_ranges

    @property
    def parameter_names(self) -> List[str]:
        """
        the names of the parameters
        """
        return self._parameter_names

    @property
    def run_time(self) -> np.ndarray:
        """
        an array of runtimes for all points in the grid

        :returns:

        """

        return self._run_time

    @property
    def energy_grid(self) -> np.ndarray:
        """
        The corresponding energy grid of the simulation

        :returns:

        """

        return self._energy_grid

    @property
    def energy_grid_nu(self) -> np.ndarray:

        return (self._energy_grid * u.keV).to("Hz", equivalencies = u.spectral())

    
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

    def to_hdf5(
        self, file_name: Union[str, Path], overwrite: bool = False
    ) -> None:

        path = Path(file_name).absolute()

        if path.exists() and (not overwrite):

            msg = f"{path} exists!"

            log.error(msg)

            raise RuntimeError(msg)

        with h5py.File(path.as_posix(), "w") as f:

            energy_grp: h5py.Group = f.create_group("energy_grid")

            energy_grp.create_dataset("energy_grid_0", data=self._energy_grid)

            values_grp = f.create_group("values")

            values_grp.create_dataset("output_0", data=self._values)

            par_name_grp = f.create_group("parameter_names")

            for i, name in enumerate(self._parameter_names):

                par_name_grp.attrs[f"par{i}"] = name

            f.create_dataset("parameters", data=self._grid_points)

            f.create_dataset("run_time", data=self._run_time)

            if self._meta_data is not None:

                meta_grp = f.create_group("meta")

                for k, v in self._meta_data.items():

                    meta_grp.create_dataset(k, data=v)

    def replace_nan_inf_with(self, value: float = 0.0) -> None:

        """
        Replace NaN and inf values with a float

        :param value:
        :type value: float
        :returns:

        """
        idx = np.isinf(self._values) | np.isnan(self._values)

        self._values[idx] = value

    def _get_sub_selection(
        self, selections_dict: Dict[str, Dict[str, float]]
    ) -> SelectionContainer:

        selection = np.ones(self._n_entries, dtype=bool)

        parameter_selection = {}

        for k, v in self._parameter_ranges.items():

            parameter_selection[k] = np.ones(len(v), dtype=bool)

        for k, v in selections_dict.items():

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

        return SelectionContainer(
            sub_grid=sub_grid,
            sub_values=sub_values,
            sub_range=sub_parameter_ranges,
            selection=selection,
        )

    def _get_sub_selection_via_index(
        self, selection_index: np.ndarray
    ) -> SelectionContainer:

        sub_grid = self._grid_points[selection_index, ...]
        sub_values = self._values[selection_index, ...]

        sub_parameter_ranges = {}

        return SelectionContainer(
            sub_grid=sub_grid,
            sub_values=sub_values,
            sub_range=sub_parameter_ranges,
            selection=selection_index,
        )

    # @classmethod
    # def create_sub_selected_database(self, **selection) -> "Database":

    #     sub_selection = self._get_sub_selection(selection)

    #     return Database()

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

        sub_selection = self._get_sub_selection(kwargs)

        tmf = TemplateModelFactory(
            name, desc, self._energy_grid, self._parameter_names
        )

        for k, v in sub_selection.sub_range.items():

            tmf.define_parameter_grid(k, v)

        with silence_console_log():

            for i in tqdm(
                range(len(sub_selection.sub_values)),
                desc="building table model",
                colour=Colors.YELLOW.value,
            ):

                ### DO NOT SORT

                tmf.add_interpolation_data(
                    sub_selection.sub_values[i],
                    **{
                        k: v
                        for k, v in zip(
                            self._parameter_names, sub_selection.sub_grid[i]
                        )
                    },
                )

            tmf.save_data(overwrite=overwrite)

        return TemplateModel(name)

    @classmethod
    def from_astromodels(cls, model_name: str) -> "Database":
        # Get the data directory

        data_dir_path: Path = get_user_data_path()

        # Sanitize the data file

        filename_sanitized = data_dir_path.absolute() / f"{model_name}.h5"

        if not filename_sanitized.exists():

            msg = f"The data file {filename_sanitized} does not exists. Did you use the TemplateFactory?"

            log.error(msg)

            raise RuntimeError(msg)

        # Open the template definition and read from it

        data_file: Path = filename_sanitized

        # use the file shadow to read

        template_file: TemplateFile = TemplateFile.from_file(
            filename_sanitized.as_posix()
        )

        parameters_grids = []

        for key in template_file.parameter_order:

            try:

                # sometimes this is
                # stored binary

                k = key.decode()

            except (AttributeError):

                # if not, then we
                # load as a normal str

                k = key

            parameters_grids.append(np.array(template_file.parameters[key]))

        parameter_grid_cart = cartesian_jit(parameters_grids)

        energies = template_file.energies

        shape = 1
        for dim in template_file.grid.shape[:-1]:
            shape *= dim

        values = template_file.grid.reshape(shape, template_file.grid.shape[-1])

        return cls(
            grid_points=parameter_grid_cart,
            parameter_names=template_file.parameter_order,
            energy_grid=energies,
            run_time=np.zeros(parameter_grid_cart.shape[0]),
            values=values,
        )

    def new_from_selections(
        self, selection_index: Optional[np.ndarray] = None, **selections
    ) -> "Database":

        if selection_index is None:

            selection_container: SelectionContainer = self._get_sub_selection(
                selections
            )

        else:

            selection_container: SelectionContainer = (
                self._get_sub_selection_via_index(selection_index)
            )

        return Database(
            selection_container.sub_grid,
            self.parameter_names,
            self.energy_grid,
            self._run_time[selection_container.selection],
            selection_container.sub_values,
        )

    def check_for_missing_parameters(
        self, parameter_grid: ParameterGrid, create_new_grid: bool = False
    ) -> None:

        """Search a parameter grid for missing values

        :param parameter_grid:
        :type parameter_grid: ParameterGrid
        :param create_new_grid:
        :type create_new_grid: bool
        :returns:

        """
        missing_parameters = []

        for i in tqdm(
            range(parameter_grid.n_points),
            desc="search through parameter grid",
            colour=Colors.BLUE.value,
        ):

            these_parameters = np.atleast_2d(
                parameter_grid.at_index(i, as_array=True)
            )

            # see if these parmeters exist!

            if (
                np.isclose(self._grid_points, these_parameters).all(-1).sum()
                == 0
            ):

                log.debug(f"MISSING:{these_parameters[0]}")

                missing_parameters.append(these_parameters[0].tolist())

        if len(missing_parameters) == 0:

            log.info("There are no missing parameters!")

        else:

            log.warning(
                f"There were {len(missing_parameters)} missing parameters!"
            )

            if create_new_grid:

                missing_parameters = np.array(missing_parameters)

                log.info("Creating a new grid")

                parameter_list = []

                for i, par_name in enumerate(parameter_grid.parameter_names):

                    par = Parameter(
                        name=par_name,
                        custom=True,
                        values=missing_parameters[:, i],
                    )

                    parameter_list.append(par)

                new_parameter_grid = ParameterGrid(
                    parameter_list, parameter_grid.energy_grid
                )

                new_parameter_grid.write("missing_parameter_grid.yml")

    def _parallel_coord_plot(
        self,
        parameter: np.ndarray,
        colorscale: str = "viridis",
        as_log: bool = False,
    ) -> None:

        dims = [
            dict(label=p, values=v)
            for p, v in zip(self._parameter_names, self._grid_points.T)
        ]

        if as_log:

            parameter = np.log10(parameter)

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=parameter,
                    colorscale=colorscale,
                    showscale=True,
                    cmin=parameter.min(),
                    cmax=parameter.max(),
                ),
                dimensions=dims,
                unselected=dict(line=dict(color='white', opacity=0.01)),
            )
        )

        fig.show()

    def plot_runtime(self, colorscale: str = "plasma"):

        """
        show a parallel plot of the run time

        :param colorscale:
        :type colorscale:
        :returns:

        """
        self._parallel_coord_plot(
            self._run_time, colorscale=colorscale, as_log=True
        )

    def plot_meta_data(
        self,
        meta_number: int = 0,
        colorscale: str = "plasma",
        as_log: bool = False,
    ):

        """

        parallel plot of the meta data

        :param meta_number:
        :type meta_number: int
        :param colorscale:
        :type colorscale: str
        :param as_log:
        :type as_log: bool
        :returns:

        """

        if self._meta_data is not None:

            self._parallel_coord_plot(
                self._meta_data[f"meta_{meta_number}"],
                colorscale=colorscale,
                as_log=as_log,
            )

        else:

            msg = "This database has no meta data"

            log.error(msg)

            raise RuntimeError(msg)


def update_database(
    database_filename: str,
    sim_number_to_replace: np.ndarray,
    sim_locations: str,
    create_backup: bool = True,
) -> None:

    """
    will replace the simulations (value arrays) of the indices supplied
    from the folder indicated. This is useful when certain runs of a
    simulation failed and one wants to replace those.

    Note: this requires that the sim ids of the replacement runs match those
    in the database!

    :param database_filename:
    :type database_filename: str
    :param sim_number_to_replace:
    :type sim_number_to_replace: np.ndarray
    :param sim_locations:
    :type sim_locations: str
    :returns:

    """
    import shutil

    database_path: Path = Path(database_filename).absolute()

    if create_backup:

        backup_file_name = (
            database_path.parent / f"{database_path.stem}_bkup.h5"
        )

        log.info(f"creating a backup as {backup_file_name}")

        shutil.copy(database_path, backup_file_name)

    with h5py.File(database_path.as_posix(), "r+") as f:

        n_output = 0
        for key in list(f.keys()):

            if "output_" in key:
                n_output += 1

        n_meta = 0
        for key in list(f.attrs.keys()):
            if "meta_" in key:
                n_meta += 1

        for idx in tqdm(sim_number_to_replace, desc="replaceing values"):

            with h5py.File(
                Path(sim_locations).absolute() / f"sim_store_{idx}.h5", "r"
            ) as r:

                for i in range(n_output):

                    f[f"values/output_{i}"][idx, :] = r[f"output_{i}"][()]

                for i in range(n_meta):

                    f[f"meta/meta_{i}"][idx, :] = r.attrs[f"meta_{i}"]

                    f["run_time"][idx] = r.attrs["run_time"]


def merge_outputs(
    *files_names: List[Union[str, Path]], out_file_name: Union[str, Path]
) -> None:

    with h5py.File(out_file_name, "w") as out_file:

        energy_grp: h5py.Group = out_file.create_group("energy_grid")
        values_grp = out_file.create_group("values")
        par_name_grp = out_file.create_group("parameter_names")

        for n_output, file_name in enumerate(files_names):

            with h5py.File(file_name, "r") as f:

                if n_output == 0:

                    for k, v in f["parameter_names"].attrs.items():

                        par_name_grp.attrs[k] = v

                    if "meta" in f.keys():

                        meta_grp = out_file.create_group("meta")

                        for key in list(f["meta"].keys()):

                            meta_grp.create_dataset(key, data=f[f"meta/{key}"])

                    out_file.create_dataset("run_time", data=f["run_time"])

                    out_file.create_dataset("parameters", data=f["parameters"])

                energy_grp.create_dataset(
                    f"energy_grid_{n_output}",
                    data=f["energy_grid/energy_grid_0"],
                )
                values_grp.create_dataset(
                    f"output_{n_output}", data=f["values/output_0"]
                )


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
