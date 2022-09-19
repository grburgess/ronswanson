from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ghost_writer import ScriptGenerator

from ronswanson.utils.logging import setup_logger

from .utils import ronswanson_config

log = setup_logger(__name__)


class PythonGenerator(ScriptGenerator):
    def __init__(
        self,
        file_name: str,
        database_file: str,
        parameter_file: str,
        base_dir: str,
        import_line: str,
        n_procs: int,
        n_nodes: Optional[int] = None,
        linear_exceution: bool = False,
        has_complete_params: bool = False,
        current_size: int = 0,
    ) -> None:

        """
        Generate the python script that will be run


        :param file_name:
        :type file_name: str
        :param database_file:
        :type database_file: str
        :param parameter_file:
        :type parameter_file: str
        :param base_dir:
        :type base_dir: str
        :param import_line:
        :type import_line: str
        :param n_procs:
        :type n_procs: int
        :param n_nodes:
        :type n_nodes: Optional[int]
        :param linear_exceution:
        :type linear_exceution: bool
        :returns:

        """
        self._import_line = import_line
        self._n_procs: int = n_procs
        self._n_nodes: Optional[int] = n_nodes
        self._parameter_file: str = parameter_file
        self._database_file: str = Path(database_file).absolute()
        self._base_dir: str = base_dir
        self._linear_execution: bool = linear_exceution
        self._has_complete_params: bool = has_complete_params
        self._current_size: int = current_size

        super().__init__(file_name)

    def _build_script(self) -> None:

        self._add_line(self._import_line)
        self._add_line("from joblib import Parallel, delayed")
        self._add_line("import json")
        self._add_line("import numpy as np")
        self._add_line("from tqdm.auto import tqdm")
        self._add_line("from ronswanson import ParameterGrid")
        self._add_line("from ronswanson.utils.logging import setup_logger")
        self._add_line("from ronswanson.simulation import gather")

        if self._n_nodes is not None:
            self._add_line("import sys")
            self._end_line()
            self._add_line("key_num = int(sys.argv[-1])")

        self._add_line("log = setup_logger(__name__)")

        self._end_line()

        if self._has_complete_params:

            self._add_line("with open('completed_parameters.json', 'r') as f:")
            self._add_line("complete_params = json.load(f)", indent_level=1)

        self._add_line(
            f"pg = ParameterGrid.from_yaml('{self._parameter_file}')"
        )

        self._add_line("def func(i):")
        self._add_line("params = pg.at_index(i)", indent_level=1)

        if self._has_complete_params:

            self._add_line("for p in complete_params:", indent_level=1)
            self._add_line(
                "if np.alltrue(np.array(p) == params):", indent_level=2
            )
            self._add_line(
                "log.debug('parameters already exists in file!')",
                indent_level=3,
            )
            self._add_line("return", indent_level=3)

        self._add_line(
            f"simulation = Simulation(i, params, pg.energy_grid,'{self._database_file}')",
            indent_level=1,
        )
        self._add_line("simulation.run()", indent_level=1)

        if self._n_nodes is None:

            self._add_line("iteration = [i for i in range(0, pg.n_points)]")

        else:

            self._add_line(
                f"with open(f'{self._base_dir}/key_file.json','r') as f:"
            )

            self._add_line("keys = json.load(f)[str(key_num)]", indent_level=1)

            self._add_line("iteration = [int(x) for x in keys]", indent_level=1)

            pass

        if self._linear_execution:

            # just do a straight for loop

            self._add_line("for i in tqdm(iteration):")
            self._add_line("func(i)", indent_level=1)

            self._add_line(
                f"gather('{self._database_file}', {self._current_size}, clean=True)"
            )

        else:

            # use joblib

            if self._n_nodes is not None:

                self._add_line(
                    f"Parallel(n_jobs={self._n_procs})(delayed(func)(i) for i in iteration)"
                )

            else:

                self._add_line(
                    f"Parallel(n_jobs={self._n_procs})(delayed(func)(i) for i in tqdm(iteration, colour='#FC0A5A'))"
                )

                self._add_line(
                    f"gather('{self._database_file}', {self._current_size}, clean=True)"
                )


class PythonGatherGenerator(ScriptGenerator):
    def __init__(
        self,
        file_name: str,
        database_file_name: str,
        current_size: int,
        n_outputs: int,
        num_meta_parameters: Optional[int] = None,
        clean: bool = True,
    ) -> None:

        self._database_file_name: str = database_file_name
        self._current_size: int = current_size
        self._n_outputs: int = n_outputs
        self._clean: bool = clean
        self._num_meta_parmeters: Optional[int] = num_meta_parameters

        super().__init__(file_name)

    def _build_script(self) -> None:
        self._add_line('import json')
        self._add_line('from mpi4py import MPI')
        self._end_line()
        self._add_line(
            'from ronswanson.utils.configuration import ronswanson_config'
        )
        self._end_line()
        self._add_line('import h5py')
        self._add_line('import sys')
        self._add_line('from pathlib import Path')
        self._end_line()
        self._end_line()
        self._add_line('rank = MPI.COMM_WORLD.rank')
        self._end_line()
        self._add_line('with open("gather_file.json", "r") as f:')
        self._end_line()
        self._add_line(
            'sim_ids = [int(x) for x in json.load(f)[str(rank)]]',
            indent_level=1,
        )
        self._end_line()
        self._end_line()
        self._add_line(f'p = Path("{self._database_file_name}").absolute()')
        self._end_line()
        self._end_line()
        self._add_line(
            'database = h5py.File(str(p), "a", driver="mpio", comm=MPI.COMM_WORLD)'
        )
        self._end_line()
        self._end_line()
        self._end_line()
        self._add_line('if ronswanson_config.slurm.store_dir is None:')
        self._end_line()
        self._add_line('parent_dir = p.absolute().parent', indent_level=1)
        self._end_line()
        self._add_line('else:')
        self._end_line()
        self._add_line(
            'parent_dir = Path(ronswanson_config.slurm.store_dir).absolute()',
            indent_level=1,
        )
        self._end_line()
        self._add_line(
            'multi_file_dir: Path = parent_dir / Path(f"{p.stem}_store")'
        )
        self._end_line()
        self._end_line()
        self._add_line(f'current_size = {self._current_size}')
        self._end_line()
        self._end_line()
        self._add_line('db_params = database["parameters"]')
        self._add_line('db_run_time = database["run_time"]')
        self._end_line()
        self._add_line('vals = database["values"]')

        if self._num_meta_parmeters is not None:

            self._add_line("meta = database['meta']")

            for i in range(self._num_meta_parmeters):

                self._add_line(f'meta_{i} = meta["meta_{i}"]')

        self._end_line()

        for i in range(self._n_outputs):

            self._add_line(f'output_{i} = vals["output_{i}"]')
        self._end_line()

        self._end_line()
        self._end_line()
        self._add_line('for sim_id in sim_ids:')
        self._end_line()
        self._add_line(
            'this_file: Path = multi_file_dir / f"sim_store_{sim_id}.h5"',
            indent_level=1,
        )
        self._end_line()
        self._add_line('index = int(current_size + sim_id)', indent_level=1)
        self._end_line()
        self._add_line('with h5py.File(this_file, "r") as f:', indent_level=1)
        self._end_line()
        self._add_line(
            'db_params[index, :] = f["parameters"][()]', indent_level=2
        )
        self._end_line()

        self._add_line(
            'db_run_time[index] = f.attrs["run_time"]', indent_level=2
        )

        for i in range(self._n_outputs):

            self._add_line(
                f'output_{i}[index, :] = f["output_{i}"][()]', indent_level=2
            )

        if self._num_meta_parmeters is not None:

            for i in range(self._num_meta_parmeters):

                self._add_line(
                    f'meta_{i}[index] = f.attrs["meta_{i}"]', indent_level=2
                )

        if self._clean:

            self._add_line("this_file.unlink()", indent_level=1)

        self._end_line()
        self._end_line()
        self._end_line()
        self._add_line('f.close()')


class SLURMGenerator(ScriptGenerator):
    def __init__(
        self,
        file_name: str,
        n_procs: int,
        n_procs_to_use: int,
        n_nodes: int,
        hrs: int,
        min: int,
        sec: int,
        node_start: int = 0,
    ) -> None:

        self._n_procs: int = n_procs
        self._n_procs_to_use: int = n_procs_to_use
        self._n_nodes: int = n_nodes
        self._node_start: int = node_start
        self._hrs: int = hrs
        self._min: int = min
        self._sec: int = sec

        super().__init__(file_name)

    def _build_script(self) -> None:

        self._add_line("#!/bin/bash")
        self._add_line("")

        if self._node_start < (self._n_nodes - 1):

            self._add_line(
                f"#SBATCH --array={self._node_start}-{self._n_nodes-1} #generate array"
            )

        self._add_line("#SBATCH -o ./output/%A_%a.out      #output file")
        self._add_line("#SBATCH -e ./output/%A_%a.err      #error file")
        self._add_line("#SBATCH -D ./                      #working directory")
        self._add_line("#SBATCH -J grid_mp                 #job name")
        self._add_line("#SBATCH -N 1               ")
        self._add_line("#SBATCH --ntasks-per-node=1")
        self._add_line(f"#SBATCH --cpus-per-task={self._n_procs_to_use}")
        self._add_line(
            f"#SBATCH --time={str(self._hrs).zfill(2)}:{str(self._min).zfill(2)}:{str(self._sec).zfill(2)}"
        )
        self._add_line("#SBATCH --mail-type=ALL ")

        self._add_line(
            f"#SBATCH --mail-user={ronswanson_config.slurm.user_email}"
        )
        self._add_line("")

        self._add_line("module purge")

        if ronswanson_config.slurm.modules is not None:

            for m in ronswanson_config.slurm.modules:

                self._add_line(f"module load {m}")

        self._end_line()

        self._add_line("#add HDF5 library path to ld path")
        self._add_line("export LD_LIBRARY_PATH=$HDF5_HOME/lib:$LD_LIBRARY_PATH")

        self._add_line(
            f"srun {ronswanson_config.slurm.python} run_simulation.py ${{SLURM_ARRAY_TASK_ID}}"
        )


class SLURMGatherGenerator(ScriptGenerator):
    def __init__(
        self,
        file_name: str,
        n_procs: int,
        n_nodes: int,
        hrs: int,
        min: int,
        sec: int,
    ) -> None:

        self._n_procs: int = n_procs
        self._n_nodes: int = n_nodes

        self._hrs: int = hrs
        self._min: int = min
        self._sec: int = sec

        super().__init__(file_name)

    def _build_script(self) -> None:

        self._add_line("#!/bin/bash")
        self._add_line("")
        self._add_line("#SBATCH -o ./output/%A.out      #output file")
        self._add_line("#SBATCH -e ./output/%A.err      #error file")
        self._add_line("#SBATCH -D ./                      #working directory")
        self._add_line("#SBATCH -J gather_out                 #job name")
        self._add_line(f"#SBATCH -N {self._n_nodes}               ")
        self._add_line(f"#SBATCH --ntasks-per-node={self._n_procs}")

        self._add_line(
            f"#SBATCH --time={str(self._hrs).zfill(2)}:{str(self._min).zfill(2)}:{str(self._sec).zfill(2)}"
        )
        self._add_line("#SBATCH --mail-type=ALL ")

        self._add_line(
            f"#SBATCH --mail-user={ronswanson_config.slurm.user_email}"
        )
        self._add_line("")

        self._add_line("module purge")

        if ronswanson_config.slurm.mpi_modules is not None:

            for m in ronswanson_config.slurm.mpi_modules:

                self._add_line(f"module load {m}")

        self._end_line()

        self._add_line(
            f"srun {ronswanson_config.slurm.python} gather_results.py"
        )
