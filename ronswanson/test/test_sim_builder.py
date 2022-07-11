import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ronswanson.database import Database
from ronswanson.simulation_builder import (
    Parameter,
    ParameterGrid,
    SimulationBuilder,
)
from ronswanson.utils.logging import update_logging_level
from ronswanson.utils.package_data import get_path_of_data_file


def test_script_gen_nodes():

    update_logging_level("ERROR")

    database_file = Path("database.h5")

    if database_file.exists():

        database_file.unlink()

    file_name = get_path_of_data_file("test_params.yml")

    pg = ParameterGrid.from_yaml(file_name)

    sb = SimulationBuilder(
        pg,
        "database.h5",
        "from ronswanson.band_simulation import BandSimulation as Simulation",
        n_cores=72,
        use_nodes=True,
    )


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

    update_logging_level("ERROR")

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
        linear_execution=False,
    )

    os.system("python3 run_simulation.py")

    database_file = Path("database.h5")

    db = Database.from_file(str(database_file))

    assert db.n_entries == 10 * 10 * 10
    assert db.n_parameters == 3
    for k, v in db.parameter_ranges.items():

        if k == "epeak":

            assert len(v) == 10

        else:

            assert len(v) == 10

    assert database_file.exists()
    database_file.unlink()


def test_adding_params():

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
        linear_execution=False,
    )

    os.system("python3 run_simulation.py")

    database_file = Path("database.h5")

    db = Database.from_file(str(database_file))

    assert db.n_entries == 10 * 10 * 10
    assert db.n_parameters == 3
    for k, v in db.parameter_ranges.items():

        if k == "epeak":

            assert len(v) == 10

        else:

            assert len(v) == 10

    file_name = get_path_of_data_file("test_addition_params.yml")

    pg = ParameterGrid.from_yaml(file_name)

    sb = SimulationBuilder(
        pg,
        "database.h5",
        "from ronswanson.band_simulation import BandSimulation as Simulation",
        n_cores=8,
        linear_execution=False,
    )

    os.system("python3 run_simulation.py")

    database_file = Path("database.h5")

    db = Database.from_file(str(database_file))

    assert db.n_entries == 10 * 10 * 12

    assert db.n_parameters == 3

    for k, v in db.parameter_ranges.items():

        if k == "epeak":

            assert len(v) == 12

        else:

            assert len(v) == 10

    assert database_file.exists()
    database_file.unlink()
