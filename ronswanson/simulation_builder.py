import itertools
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import yaml

from .script_generator import PythonGenerator, SLURMGenerator
from .simulation import Simulation

# from .utils.hdf5_utils import recursively_save_dict_contents_to_group
from .utils.logging import setup_logger

log = setup_logger(__name__)


@dataclass(frozen=True)
class EnergyGrid:

    vmin: Optional[float] = None
    vmax: Optional[float] = None
    scale: Optional[str] = None
    n_points: Optional[int] = None
    values: Optional[np.ndarray] = None
    custom: bool = False

    def __post_init__(self):

        if not self.custom:

            # we will build a grid

            if (self.vmin is None) or (self.vmax is None):

                log.error("non-custom grids must include vmin and vmax")

                raise AssertionError

            if self.scale is None:

                log.error(
                    "non-custom grids must include scale 'log' or 'linear'"
                )

                raise AssertionError

            else:

                if self.scale not in ['log', 'linear']:

                    log.error(
                        "non-custom grids must include scale 'log' or 'linear'"
                    )

                    raise AssertionError

            if self.n_points is None:

                log.error("non-custom grids must include n_points")

                raise AssertionError

        else:

            if self.values is None:

                log.error("custom grids must include values")

                raise AssertionError

    @property
    def grid(self) -> np.ndarray:

        if self.custom:

            return self.values

        else:

            if self.scale.lower() == 'log':

                return np.geomspace(self.vmin, self.vmax, self.n_points)

            else:

                return np.linspace(self.vmin, self.vmax, self.n_points)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Parameter":

        inputs = {}
        inputs["custom"] = d["custom"]

        if d["custom"]:

            inputs["values"] = np.array(d["values"])

        else:

            inputs["vmin"] = d["vmin"]
            inputs["vmax"] = d["vmax"]
            inputs["scale"] = d["scale"]
            inputs["n_points"] = d["n_points"]

        return cls(**inputs)

    def to_dict(self) -> Dict[str, Any]:

        out = dict(custom=self.custom)

        if self.custom:

            out["values"] = self.values.tolist()

        else:

            out["vmin"] = self.vmin
            out["vmax"] = self.vmax
            out["scale"] = self.scale
            out["n_points"] = self.n_points

        return out


@dataclass(frozen=True)
class Parameter:

    name: str
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    scale: Optional[str] = None
    n_points: Optional[int] = None
    values: Optional[np.ndarray] = None
    custom: bool = False
    #   grid: np.ndarray = field(init=False)

    def __post_init__(self):

        if not self.custom:

            # we will build a grid

            if (self.vmin is None) or (self.vmax is None):

                log.error("non-custom grids must include vmin and vmax")

                raise AssertionError

            if self.scale is None:

                log.error(
                    "non-custom grids must include scale 'log' or 'linear'"
                )

                raise AssertionError

            else:

                if self.scale not in ['log', 'linear']:

                    log.error(
                        "non-custom grids must include scale 'log' or 'linear'"
                    )

                    raise AssertionError

            if self.n_points is None:

                log.error("non-custom grids must include n_points")

                raise AssertionError

        else:

            if self.values is None:

                log.error("custom grids must include values")

                raise AssertionError

    @property
    def grid(self) -> np.ndarray:

        if self.custom:

            return self.values

        else:

            if self.scale.lower() == 'log':

                return np.geomspace(self.vmin, self.vmax, self.n_points)

            else:

                return np.linspace(self.vmin, self.vmax, self.n_points)

    @classmethod
    def from_dict(cls, name: str, d: Dict[str, Any]) -> "Parameter":

        inputs = {}
        inputs["custom"] = d["custom"]

        if d["custom"]:

            inputs["values"] = np.array(d["values"])

        else:

            inputs["vmin"] = d["vmin"]
            inputs["vmax"] = d["vmax"]
            inputs["scale"] = d["scale"]
            inputs["n_points"] = d["n_points"]

        return cls(name, **inputs)

    def to_dict(self) -> Dict[str, Any]:

        out = dict(custom=self.custom)

        if self.custom:

            out["values"] = self.values.tolist()

        else:

            out["vmin"] = self.vmin
            out["vmax"] = self.vmax
            out["scale"] = self.scale
            out["n_points"] = self.n_points

        return out


@dataclass(frozen=True)
class ParameterGrid:

    parameter_list: List[Parameter]
    energy_grid: EnergyGrid

    @property
    def n_points(self) -> int:

        n = 1
        for param in self.parameter_list:

            n *= len(param.grid)

        return n

    @property
    def n_parameters(self) -> int:

        return len(self.parameter_list)

    @classmethod
    def from_dict(cls, d: Dict[str, Dict[str, Any]]) -> "ParameterGrid":

        # make sure to sort so that we always have the same parameter
        # ordering

        energy_grid = EnergyGrid.from_dict(d.pop("energy_grid"))

        pars = list(d.keys())

        pars.sort()

        par_list = [
            Parameter.from_dict(par_name, d[par_name]) for par_name in pars
        ]

        return cls(par_list, energy_grid)

    @classmethod
    def from_yaml(cls, file_name: str) -> "ParameterGrid":

        with open(file_name, 'r') as f:

            inputs = yaml.load(stream=f, Loader=yaml.SafeLoader)

        return cls.from_dict(inputs)

    @property
    def parameter_names(self) -> List[str]:

        return [p.name for p in self.parameter_list]

    def to_dict(self) -> Dict[str, Dict[str, Any]]:

        out = {}

        for p in self.parameter_list:

            out[p.name] = p.to_dict()

        out['energy_grid'] = self.energy_grid.to_dict()

        return out

    def write(self, file_name: str) -> None:

        with open(file_name, "w") as f:

            yaml.dump(
                stream=f,
                data=self.to_dict(),
                default_flow_style=False,
                Dumper=yaml.SafeDumper,
            )

    # def to_hdf5_group(self, f: h5py.File) -> None:

    #     recursively_save_dict_contents_to_group(
    #         f, "parameter_grid", self.to_dict()
    #     )

    def at_index(self, i: int) -> Dict[str, float]:
        """
        return the ith set of parameters

        :param i:
        :type i: int
        :returns:

        """
        idx = 0

        for result in itertools.product(*[p.grid for p in self.parameter_list]):

            if i == idx:

                d = OrderedDict()

                for k, v in zip(self.parameter_names, result):

                    d[k] = v

                return d

            else:

                idx += 1


class SimulationBuilder:
    def __init__(
        self,
        parameter_grid: ParameterGrid,
        out_file: str,
        import_line: str,
        n_cores: int = 1,
        n_nodes: Optional[int] = None,
        linear_execution: bool = False,
    ):

        """
        The simulation builder class constructs the scripts
        needed for building the table model database

        :param parameter_grid:
        :type parameter_grid: ParameterGrid
        :param out_file:
        :type out_file: str
        :param import_line:
        :type import_line: str
        :param n_cores:
        :type n_cores: int
        :param n_nodes:
        :type n_nodes: Optional[int]
        :param linear_execution:
        :type linear_execution: bool
        :returns:

        """
        self._import_line: str = import_line

        self._n_cores: int = n_cores

        self._n_nodes: Optional[int] = n_nodes

        self._out_file: str = out_file

        self._base_dir: Path = Path(out_file).parent.absolute()

        self._linear_execution: bool = linear_execution

        # write out the parameter file

        self._parameter_file: Path = self._base_dir / "parameters.yml"

        parameter_grid.write(str(self._parameter_file))

        self._generate_python_script()

    def _generate_python_script(self) -> None:

        py_gen: PythonGenerator = PythonGenerator(
            "run_simulation.py",
            self._out_file,
            str(self._parameter_file),
            self._import_line,
            self._n_cores,
            self._n_nodes,
            self._linear_execution,
        )

        py_gen.write(str(self._base_dir))

    def _generate_slurm_script(self) -> None:

        pass
