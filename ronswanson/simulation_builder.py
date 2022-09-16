import itertools
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import yaml

from .grids import ParameterGrid
from .script_generator import (
    PythonGenerator,
    SLURMGatherGenerator,
    SLURMGenerator,
)
from .utils.logging import setup_logger

log = setup_logger(__name__)


@dataclass(frozen=True)
class SLURMTime:
    hrs: int = 0
    min: int = 10
    sec: int = 0


@dataclass
class JobConfig:
    n_cores_per_node: int
    time: SLURMTime


@dataclass
class GatherConfig(JobConfig):
    n_gather_per_core: int


@dataclass
class SimulationConfig(JobConfig):
    n_mp_jobs: int
    run_per_node: int = 1
    use_nodes: bool = True
    linear_execution: bool = False


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
        simulation_config: SimulationConfig,
        gather_config: Optional[GatherConfig] = None,
    ):

        self._has_complete_params: bool = False

        self._import_line: str = import_line

        self._simulation_config: SimulationConfig = simulation_config

        self._gather_config: Optional[GatherConfig] = gather_config

        self._out_file: str = out_file

        self._base_dir: Path = Path(out_file).parent.absolute()

        # write out the parameter file

        self._parameter_file: Path = self._base_dir / "parameters.yml"

        parameter_grid.write(str(self._parameter_file))

        self._n_iterations: int = parameter_grid.n_points

        self._current_database_size: int = 0

        self._initialize_database()

        self._check_completed()

        # if we are using nodes
        # we need to see how many
        # we need

        if self._simulation_config.use_nodes:

            self._compute_chunks()

        else:

            self._n_nodes: Optional[int] = None

            self._n_gather_nodes: Optional[int] = None

        self._generate_python_script()

        if self._simulation_config.use_nodes:

            self._generate_slurm_script()

    @classmethod
    def from_yaml(cls, file_name: str) -> "SimulationBuilder":

        with Path(file_name).open("r") as f:

            inputs = yaml.load(stream=f, Loader=yaml.SafeLoader)

        parameter_grid = ParameterGrid.from_yaml(inputs.pop("parameter_grid"))

        simulation_input = inputs.pop("simulation")

        if "time" in simulation_input:

            sim_time = SLURMTime(**simulation_input.pop("time"))

        else:

            sim_time = SLURMTime()

        simulation_config = SimulationConfig(time=sim_time, **simulation_input)

        gather_inputs = None

        if "gather" in inputs:

            gather_inputs = inputs.pop("gather")

            gather_time = SLURMTime(**gather_inputs.pop("time"))

            gather_config = GatherConfig(time=gather_time, **gather_inputs)

        return cls(
            parameter_grid=parameter_grid,
            simulation_config=simulation_config,
            gather_config=gather_config,
            **inputs,
        )

    def _initialize_database(self) -> None:

        if not Path(self._out_file).exists():

            with h5py.File(self._out_file, "w") as f:

                pg = ParameterGrid.from_yaml(self._parameter_file)

                # store the parameter names

                p_name_group = f.create_group("parameter_names")

                for i, name in enumerate(pg.parameter_names):

                    p_name_group.attrs[f"par{i}"] = name

                # store the energy grids

                ene_grp = f.create_group("energy_grid")

                for i, grid in enumerate(pg.energy_grid):

                    ene_grp.create_dataset(
                        f"energy_grid_{i}",
                        data=grid.grid,
                        compression="gzip",
                    )

                # create an empty group for the parameters

                f.create_dataset(
                    "parameters",
                    shape=(pg.n_points,) + np.array(pg.parameter_names).shape,
                    maxshape=(None,) + np.array(pg.parameter_names).shape,
                    #    compression="gzip",
                )

                val_grp: h5py.Group = f.create_group("values")

                # create an empty data set for the values

                for i in range(len(pg.energy_grid)):

                    grp = val_grp.create_group(f"output_{i}")

                    grp.create_dataset(
                        "values",
                        shape=(pg.n_points,) + pg.energy_grid[i].grid.shape,
                        maxshape=(None,) + pg.energy_grid[i].grid.shape,
                        # compression="gzip",
                    )

        else:

            # we need to resize the dataset

            with h5py.File(self._out_file, "a") as f:

                pg = ParameterGrid.from_yaml(self._parameter_file)

                dataset: h5py.Dataset = f["parameters"]

                self._current_database_size = dataset.shape[0]

                dataset.resize(
                    (self._current_database_size + pg.n_points,)
                    + dataset.shape[1:]
                )

                val_grp = f["values"]

                for i in range(len(pg.energy_grid)):

                    grp = val_grp[f"output_{i}"]

                    dataset: h5py.Dataset = grp["values"]

                    dataset.resize(
                        (self._current_database_size + pg.n_points,)
                        + dataset.shape[1:]
                    )

    def _check_completed(self) -> None:

        if Path(self._out_file).exists():

            with h5py.File(self._out_file, "r") as f:

                params = f["parameters"][()]

            out_file = self._base_dir / "completed_parameters.json"

            with out_file.open("w") as f:

                json.dump(params.tolist(), f)

            self._has_complete_params = True

    def _compute_chunks(self) -> None:

        if self._simulation_config.run_per_node == 1:

            runs_per_node = 1

            generator = range(self._simulation_config.n_mp_jobs)

            n_nodes = np.ceil(
                self._n_iterations / self._simulation_config.n_mp_jobs
            )

        else:

            runs_per_node = self._simulation_config.run_per_node

            generator = range(runs_per_node)

            n_nodes = np.ceil(self._n_iterations / runs_per_node)

        if self._simulation_config.use_nodes:

            self._n_nodes = int(n_nodes)

            log.info(f"we will be using {self._n_nodes} nodes")

        # now generate the key files
        k = 0

        key_out = {}

        for i in range(self._n_nodes):

            output = []

            for j in generator:

                if k < self._n_iterations:

                    output.append(k)
                    k += 1

            key_out[i] = output

        with open(self._base_dir / "key_file.json", "w") as f:

            json.dump(key_out, f)

        # now collect the gather information

        if self._simulation_config.use_nodes:

            self._n_gather_nodes = int(
                np.ceil(
                    self._n_iterations
                    / (
                        self._gather_config.n_cores_per_node
                        * self._gather_config.n_gather_per_core
                    )
                )
            )

            rank_list = {}
            n = 0

            log.debug(f"number nodes: {self._n_gather_nodes}")
            log.debug(
                f"total_ranks: {self._n_gather_nodes * self._gather_config.n_cores_per_node}"
            )
            log.debug(f"number iterations: {self._n_iterations}")

            for i in range(
                self._n_gather_nodes * self._gather_config.n_cores_per_node
            ):

                core_list = []

                for j in range(self._gather_config.n_gather_per_core):

                    if n < self._n_iterations:

                        core_list.append(n)

                        n += 1

                rank_list[i] = core_list

            with open(self._base_dir / "gather_file.json", "w") as f:

                json.dump(rank_list, f)

    def _generate_python_script(self) -> None:

        py_gen: PythonGenerator = PythonGenerator(
            "run_simulation.py",
            self._out_file,
            str(self._parameter_file),
            self._base_dir,
            self._import_line,
            self._simulation_config.n_mp_jobs,
            self._n_nodes,
            self._simulation_config.linear_execution,
            self._has_complete_params,
            self._current_database_size,
        )

        py_gen.write(str(self._base_dir))

    def _generate_slurm_script(self) -> None:

        slurm_gen: SLURMGenerator = SLURMGenerator(
            "run_simulation.sh",
            self._simulation_config.n_mp_jobs,
            self._simulation_config.n_cores_per_node,
            self._n_nodes,
            self._simulation_config.time.hrs,
            self._simulation_config.time.min,
            self._simulation_config.time.sec,
        )

        slurm_gen.write(str(self._base_dir))

        slurm_gen: SLURMGatherGenerator = SLURMGatherGenerator(
            "gather_results.sh",
            self._gather_config.n_cores_per_node,
            self._n_gather_nodes,
            self._gather_config.time.hrs,
            self._gather_config.time.min,
            self._gather_config.time.sec,
        )

        slurm_gen.write(str(self._base_dir))
