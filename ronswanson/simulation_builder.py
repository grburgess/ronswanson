import itertools
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from .script_generator import PythonGenerator, SLURMGenerator

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
    energy_grid: List[EnergyGrid]

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

        is_multi_output: bool = False
        n_energy_grids = 1

        for k in d.keys():

            if "energy_grid" in k:

                if len(k.split("_")) == 3:

                    if is_multi_output:

                        # we have already detected one

                        n_energy_grids += 1

                    is_multi_output = True

        if not is_multi_output:

            energy_grid = [EnergyGrid.from_dict(d.pop("energy_grid"))]

        else:

            energy_grid = [
                EnergyGrid.from_dict(d.pop(f"energy_grid_{i}"))
                for i in range(n_energy_grids)
            ]

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

        if len(self.energy_grid) == 1:

            out['energy_grid'] = self.energy_grid[0].to_dict()

        else:

            for i, eg in enumerate(self.energy_grid):

                out[f"energy_grid_{i}"] = eg.to_dict()

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
    """
    The simulation builder class constructs the scripts
    needed for building the table model database

    """

    def __init__(
        self,
        parameter_grid: ParameterGrid,
        out_file: str,
        import_line: str,
        n_cores: int = 1,
        use_nodes: bool = False,
        runs_per_node: Optional[int] = None,
        linear_execution: bool = False,
        hrs: Optional[int] = None,
        min: Optional[int] = None,
        sec: Optional[int] = None,
    ):

        self._import_line: str = import_line

        self._n_cores: int = n_cores

        self._use_nodes: bool = use_nodes

        self._runs_per_node: Optional[int] = runs_per_node

        self._out_file: str = out_file

        self._base_dir: Path = Path(out_file).parent.absolute()

        self._linear_execution: bool = linear_execution

        self._hrs: Optional[int] = hrs

        self._min: Optional[int] = min

        self._sec: Optional[int] = sec

        # write out the parameter file

        self._parameter_file: Path = self._base_dir / "parameters.yml"

        parameter_grid.write(str(self._parameter_file))

        self._n_iterations: int = parameter_grid.n_points

        # if we are using nodes
        # we need to see how many
        # we need

        if self._use_nodes:

            self._compute_chunks()

        else:

            self._n_nodes: Optional[int] = None

        self._generate_python_script()

        if self._use_nodes:

            self._generate_slurm_script()

    def _compute_chunks(self) -> None:

        if self._runs_per_node is None:

            runs_per_node = 1

        else:

            runs_per_node = self._runs_per_node

        n_nodes = np.ceil(self._n_iterations / (self._n_cores * runs_per_node))

        if self._use_nodes:

            self._n_nodes = int(n_nodes)

            log.info(f"we will be using {self._n_nodes} nodes")

        # now generate the key files
        k = 0
        for i in range(self._n_nodes):
            output = []

            for j in range(self._n_cores * runs_per_node):

                if (k + j) < self._n_iterations:

                    output.append(k + j)

            with open(self._base_dir / f"key_file{i}.txt", "w") as f:

                for o in output:

                    f.write(f"{o}\n")

            k += self._n_cores * runs_per_node

    def _generate_python_script(self) -> None:

        py_gen: PythonGenerator = PythonGenerator(
            "run_simulation.py",
            self._out_file,
            str(self._parameter_file),
            self._base_dir,
            self._import_line,
            self._n_cores,
            self._n_nodes,
            self._linear_execution,
        )

        py_gen.write(str(self._base_dir))

    def _generate_slurm_script(self) -> None:

        slurm_gen: SLURMGenerator = SLURMGenerator(
            "run_simulation.sh",
            self._n_cores,
            self._n_nodes,
            self._hrs,
            self._min,
            self._sec,
        )

        slurm_gen.write(str(self._base_dir))
