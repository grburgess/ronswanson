import itertools
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

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
