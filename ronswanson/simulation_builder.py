import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import yaml
from omegaconf import OmegaConf, MISSING

from .grids import ParameterGrid
from .script_generator import (
    PythonGatherGenerator,
    PythonGenerator,
    SLURMGatherGenerator,
    SLURMGenerator,
)
from .utils.logging import setup_logger

log = setup_logger(__name__)


@dataclass
class SLURMTime:
    hrs: int = 0
    min: int = 10
    sec: int = 0


@dataclass
class JobConfig:
    time: SLURMTime
    n_cores_per_node: int


@dataclass
class GatherConfig(JobConfig):
    n_gather_per_core: int


@dataclass
class SimulationConfig(JobConfig):
    n_mp_jobs: int
    run_per_node: Optional[int] = None
    use_nodes: bool = False
    max_nodes: Optional[int] = None
    linear_execution: bool = False


## structure for file


@dataclass
class JobConfigStructure:
    time: Optional[SLURMTime] = None
    n_cores_per_node: Optional[int] = None


@dataclass
class GatherConfigStructure(JobConfigStructure):
    n_gather_per_core: int = MISSING


@dataclass
class SimulationConfigStructure(JobConfigStructure):
    n_mp_jobs: int = MISSING
    run_per_node: Optional[int] = None
    use_nodes: bool = False
    max_nodes: Optional[int] = None
    linear_execution: bool = False


@dataclass
class YAMLStructure:
    import_line: str = MISSING
    parameter_grid: str = MISSING
    out_file: str = MISSING
    clean: bool = True
    simulation: SimulationConfigStructure = SimulationConfigStructure()
    gather: Optional[GatherConfigStructure] = None


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
        num_meta_parameters: Optional[int] = None,
        clean: bool = True,
    ):

        """TODO describe function

        :param parameter_grid:
        :type parameter_grid: ParameterGrid
        :param out_file:
        :type out_file: str
        :param import_line:
        :type import_line: str
        :param simulation_config:
        :type simulation_config: SimulationConfig
        :param gather_config:
        :type gather_config: Optional[GatherConfig]
        :param num_meta_parameters:
        :type num_meta_parameters: Optional[int]
        :param clean:
        :type clean: bool
        :returns:

        """
        self._has_complete_params: bool = False

        self._import_line: str = import_line

        self._simulation_config: SimulationConfig = simulation_config

        self._gather_config: Optional[GatherConfig] = gather_config

        self._out_file: str = out_file

        self._base_dir: Path = Path(out_file).parent.absolute()

        self._num_meta_parameters: Optional[int] = num_meta_parameters

        # write out the parameter file

        self._parameter_file: Path = self._base_dir / "parameters.yml"

        parameter_grid.write(str(self._parameter_file))

        self._n_outputs: int = len(parameter_grid.energy_grid)

        self._clean: bool = clean

        self._n_iterations: int = parameter_grid.n_points

        self._current_database_size: int = 0

        self._initialize_database()

        # if we are using nodes
        # we need to see how many
        # we need

        log.info(
            f"there are [bold bright_red]{self._n_iterations} iterations [/bold bright_red]"
        )

        if self._simulation_config.use_nodes:

            self._compute_chunks()

        else:

            self._n_nodes: Optional[int] = None

            self._n_gather_nodes: Optional[int] = None

        self._generate_python_script()

        if self._simulation_config.use_nodes:

            self._generate_slurm_script()

            output_dir = self._base_dir / "output"

            if not output_dir.exists():

                output_dir.mkdir()

                log.debug("created the output directory")

    @classmethod
    def from_yaml(cls, file_name: str) -> "SimulationBuilder":
        """
        Create a simulation setup from a yaml file.

        """

        # check the file structure

        structure = OmegaConf.structured(YAMLStructure)

        try:

            test_structure = OmegaConf.load(file_name)

            merged = OmegaConf.merge(structure, test_structure)

            OmegaConf.to_container(merged, throw_on_missing=True)

        except Exception as e:

            log.error(e)

            raise e

        with Path(file_name).open("r") as f:

            inputs = yaml.load(stream=f, Loader=yaml.SafeLoader)

        log.debug("reading setup inputs:")

        for k, v in inputs.items():

            log.debug(f"{k}: {v}")

        parameter_grid = ParameterGrid.from_yaml(inputs.pop("parameter_grid"))

        simulation_input = inputs.pop("simulation")

        if "time" in simulation_input:

            sim_time = SLURMTime(**simulation_input.pop("time"))

        else:

            sim_time = SLURMTime()

        if "n_cores_per_node" not in simulation_input:

            if "use_nodes" in simulation_input:

                if simulation_input["use_nodes"]:

                    log.warning(
                        "you are using nodes but did not specify the number of n_cores_per_node"
                    )
                    log.warning(
                        "the number of cores will be set to the number of multi-process jobs"
                    )

            simulation_input["n_cores_per_node"] = simulation_input["n_mp_jobs"]

        simulation_config = SimulationConfig(time=sim_time, **simulation_input)

        gather_config = None

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

                    val_grp.create_dataset(
                        f"output_{i}",
                        shape=(pg.n_points,) + pg.energy_grid[i].grid.shape,
                        maxshape=(None,) + pg.energy_grid[i].grid.shape,
                        # compression="gzip",
                    )

                f.create_dataset(
                    "run_time", shape=(pg.n_points,), maxshape=(None,)
                )

                if self._num_meta_parameters is not None:

                    meta_grp: h5py.Group = f.create_group("meta")

                    log.debug("detected meta parameters")

                    for i in range(self._num_meta_parameters):

                        meta_grp.create_dataset(
                            f"meta_{i}", shape=(pg.n_points,), maxshape=(None,)
                        )

        else:

            # we need to resize the dataset

            log.warning(
                f"There was already a database: [red]{self._out_file}[/red]"
            )

            self._check_completed()

            with h5py.File(self._out_file, "a") as f:

                pg = ParameterGrid.from_yaml(self._parameter_file)

                dataset: h5py.Dataset = f["parameters"]

                self._current_database_size = dataset.shape[0]

                log.warning(
                    f"The existing data base had {self._current_database_size} entries"
                )

                dataset.resize(
                    (self._current_database_size + pg.n_points,)
                    + dataset.shape[1:]
                )

                val_grp = f["values"]

                for i in range(len(pg.energy_grid)):

                    dataset: h5py.Dataset = val_grp[f"output_{i}"]

                    dataset.resize(
                        (self._current_database_size + pg.n_points,)
                        + dataset.shape[1:]
                    )

                dataset: h5py.Dataset = f["run_time"]

                dataset.resize((self._current_database_size + pg.n_points,))

                if self._num_meta_parameters is not None:

                    log.debug("detected meta parameters")

                    meta_grp = f["meta"]

                    for i in range(self._num_meta_parameters):

                        dataset: h5py.Dataset = meta_grp[f"meta_{i}"]

                        dataset.resize(
                            (self._current_database_size + pg.n_points,)
                        )

        self._n_outputs: int = len(pg.energy_grid)

    def _check_completed(self) -> None:

        if Path(self._out_file).exists():

            with h5py.File(self._out_file, "r") as f:

                params = f["parameters"][()]

            out_file = self._base_dir / "completed_parameters.json"

            with out_file.open("w") as f:

                json.dump(params.tolist(), f)

            self._has_complete_params = True

    def _compute_chunks(self) -> None:

        if self._simulation_config.run_per_node is None:

            log.debug("Each node will only execute the number of mp jobs")

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

            log.info(
                f"we will be using {self._n_nodes} nodes for the simulation"
            )

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

            log.info(f"the gather task will use: {self._n_gather_nodes} nodes")
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

        log.info(
            "[bold green blink]generated:[/bold green blink] run_simulation.py"
        )

    def _generate_slurm_script(self) -> None:

        multi_script: bool = False

        if self._simulation_config.max_nodes is not None:

            if self._n_nodes > self._simulation_config.max_nodes:

                log.debug("The number of reuested nodes is too large.")

                multi_script = True

                n_files = int(
                    np.ceil(self._n_nodes / self._simulation_config.max_nodes)
                )

                start = []
                stop = []
                current_number = 0

                for i in range(n_files):

                    start.append(current_number)

                    next_number = int(
                        (i + 1) * self._simulation_config.max_nodes
                    )

                    if next_number <= self._n_nodes:

                        stop.append(next_number)

                        current_number = next_number

                    else:

                        stop.append(self._n_nodes)

                        break

        if multi_script:

            for i, (a, b) in enumerate(zip(start, stop)):

                file_name = f"run_simulation_{i}.sh"

                slurm_gen: SLURMGenerator = SLURMGenerator(
                    file_name,
                    self._simulation_config.n_mp_jobs,
                    self._simulation_config.n_cores_per_node,
                    b,
                    self._simulation_config.time.hrs,
                    self._simulation_config.time.min,
                    self._simulation_config.time.sec,
                    node_start=a,
                )

                slurm_gen.write(str(self._base_dir))

                log.info(
                    f"[bold green blink]generated:[/bold green blink] {file_name}"
                )

        else:

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

            log.info(
                "[bold green blink]generated:[/bold green blink] run_simulations.sh"
            )

        slurm_gen: SLURMGatherGenerator = SLURMGatherGenerator(
            "gather_results.sh",
            self._gather_config.n_cores_per_node,
            self._n_gather_nodes,
            self._gather_config.time.hrs,
            self._gather_config.time.min,
            self._gather_config.time.sec,
        )

        slurm_gen.write(str(self._base_dir))

        log.info(
            "[bold green blink]generated:[/bold green blink] gather_results.sh"
        )

        python_gather_gen: PythonGatherGenerator = PythonGatherGenerator(
            "gather_results.py",
            database_file_name=self._out_file,
            current_size=self._current_database_size,
            n_outputs=self._n_outputs,
            clean=self._clean,
            num_meta_parameters=self._num_meta_parameters,
        )

        python_gather_gen.write(str(self._base_dir))

        log.info(
            "[bold green blink]generated:[/bold green blink] gather_results.py"
        )
