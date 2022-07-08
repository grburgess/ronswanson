import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ronswanson import database
from ronswanson.simulation_builder import (
    Parameter,
    ParameterGrid,
    SimulationBuilder,
)
from ronswanson.utils.package_data import get_path_of_data_file


def test_script_gen_linear():

    database_file = Path("database.h5")

    if database_file.exists():

        database_file.unlink()

    file_name = get_path_of_data_file("test_params.yml")

    pg = ParameterGrid.from_yaml(file_name)

    sb = SimulationBuilder(
        pg,
        "database.h5",
        "from ronswanson.band_simulation import BandSimulation as Simulation",
        n_cores=8,
        linear_execution=True,
    )

    os.system("python3 run_simulation.py")

    database_file = Path("database.h5")

    assert database_file.exists()
    database_file.unlink()


def test_script_gen_parallel():

    database_file = Path("database.h5")

    if database_file.exists():

        database_file.unlink()

    file_name = get_path_of_data_file("test_params.yml")

    pg = ParameterGrid.from_yaml(file_name)

    sb = SimulationBuilder(
        pg,
        "database.h5",
        "from ronswanson.band_simulation import BandSimulation as Simulation",
        n_cores=8,
        linear_execution=True,
    )

    os.system("python3 run_simulation.py")

    database_file = Path("database.h5")

    assert database_file.exists()
    database_file.unlink()
