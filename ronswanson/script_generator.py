from pathlib import Path
from typing import Optional


class ScriptGenerator:
    def __init__(self, file_name: str) -> None:

        self._file_name: str = file_name
        self._output: str = ""
        self._build_script()

    def _build_script(self) -> None:
        pass

    def _add_line(self, line: str, indent_level: int = 0) -> None:

        for i in range(indent_level):

            self._output += "\t"

        # add the line

        self._output += line

        # close the line
        self._end_line()

    def _end_line(self):

        self._output += "\n"

    def write(self, directory: str = ".") -> None:

        out_file: Path = Path(directory) / self._file_name

        with out_file.open("w") as f:

            f.write(self._output)


class PythonGenerator(ScriptGenerator):
    def __init__(
        self,
        file_name: str,
        database_file: str,
        parameter_file: str,
        import_line: str,
        n_procs: int,
        n_nodes: Optional[int] = None,
        linear_exceution: bool = False,
    ) -> None:

        self._import_line = import_line
        self._n_procs: int = n_procs
        self._n_nodes: Optional[int] = n_nodes
        self._parameter_file: str = parameter_file
        self._database_file: str = Path(database_file).absolute()
        self._linear_execution: bool = linear_exceution

        super().__init__(file_name)

    def _build_script(self) -> None:

        self._add_line(self._import_line)
        self._add_line("from joblib import Parallel, delayed")
        self._add_line("from ronswanson import ParameterGrid")
        self._end_line()

        self._add_line(
            f"pg = ParameterGrid.from_yaml('{self._parameter_file}')"
        )

        self._add_line("def func(i):")
        self._add_line("params = pg.at_index(i)", indent_level=1)
        self._add_line(
            f"simulation = Simulation(i, params, pg.energy_grid.grid,'{self._database_file}')",
            indent_level=1,
        )
        self._add_line("simulation.run()", indent_level=1)

        if self._n_nodes is None:

            self._add_line("iteration = [i for i in range(0, pg.n_points)]")

        else:

            pass

        if self._linear_execution:

            # just do a straight for loop

            self._add_line("for i in iteration:")
            self._add_line("func(i)", indent_level=1)

        else:

            # use joblib

            self._add_line(
            f"Parallel(n_jobs={self._n_procs})(delayed(func)(i) for i in iteration)"
        )


class SLURMGenerator(ScriptGenerator):
    def __init__(self, file_name: str) -> None:
        super().__init__(file_name)
