from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from .simulation import Simulation
from .utils.logging import setup_logger

log = setup_logger(__name__)


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

            out["values"] = self.values

        else:

            out["vmin"] = self.vmin
            out["vmax"] = self.vmax
            out["scale"] = self.scale
            out["n_points"] = self.n_points

        return out


@dataclass(frozen=True)
class ParameterGrid:

    parameter_list: List[Parameter]

    @property
    def n_points(self) -> int:

        n = 1
        for param in self.parameter_list:

            n *= len(param.grid)

        return n

    @classmethod
    def from_dict(cls, d: Dict[str, Dict[str, Any]]) -> "ParameterGrid":

        # make sure to sort so that we always have the same parameter
        # ordering
        pars = list(d.keys())

        pars.sort()

        par_list = [
            Parameter.from_dict(par_name, d[par_name]) for par_name in pars
        ]

        return cls(par_list)

    @classmethod
    def from_yaml(cls, file_name: str) -> "ParameterGrid":

        with open(file_name, 'r') as f:

            inputs = yaml.load(stream=f, Loader=yaml.SafeLoader)

        return cls.from_dict(inputs)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:

        out = {}

        for p in self.parameter_list:

            out[p.name] = p.to_dict()

        return out

    def write(self, file_name: str) -> None:

        with open(file_name, "w") as f:

            yaml.dump(
                stream=f,
                data=self.to_dict(),
                default_flow_style=False,
                Dumper=yaml.SafeDumper,
            )


class SimulationBuilder:
    def __init__(
        self,
        parameter_grid: ParameterGrid,
        simulation_class: Simulation,
        n_cores: int = 1,
        n_nodes: Optional[int] = None,
    ):

        self._simulation_class: Simulation = simulation_class

        self._n_cores: int = n_cores

        self._n_nodes: Optional[int] = n_nodes

    def _generate_python_script(self) -> None:

        pass

    def _generate_slurm_script(self) -> None:

        pass
