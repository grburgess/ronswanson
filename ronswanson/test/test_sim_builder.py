import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from ronswanson.database import Database
from ronswanson.grids import Parameter, ParameterGrid
from ronswanson.simulation_builder import SimulationBuilder
from ronswanson.utils.logging import update_logging_level
from ronswanson.utils.package_data import get_path_of_data_file
from omegaconf.errors import MissingMandatoryValue


update_logging_level("DEBUG")


def test_bad_sim_build():

    file_name = get_path_of_data_file("bad_sim_build.yaml")

    with pytest.raises(MissingMandatoryValue):

        SimulationBuilder.from_yaml(file_name=str(file_name))

    file_name = get_path_of_data_file("bad_slurm_build.yaml")

    with pytest.raises(MissingMandatoryValue):

        SimulationBuilder.from_yaml(file_name=str(file_name))


def test_script_gen_nodes(node_script):

    database_file = Path("database_node.h5")

    if database_file.exists():

        database_file.unlink()

    sb = SimulationBuilder.from_yaml(node_script)

    # delete crap

    slurm_script = Path("run_simulation.sh")

    assert slurm_script.exists()

    database_file.unlink()


def test_script_gen_linear(linear_script):

    database_file = Path("database_lin.h5")

    if database_file.exists():

        database_file.unlink()

    sb = SimulationBuilder.from_yaml(linear_script)

    os.system("python3 run_simulation.py")

    database_file = Path("database_lin.h5")

    assert database_file.exists()

    database_file.unlink()

    assert not Path("database_lin_store").exists()


def test_script_gen_parallel(parallel_script):

    update_logging_level("ERROR")

    database_file = Path("database_para.h5")

    if database_file.exists():

        database_file.unlink()

    sb = SimulationBuilder.from_yaml(parallel_script)

    os.system("python3 run_simulation.py")

    database_file = Path("database_para.h5")

    db = Database.from_file(str(database_file))

    assert db.n_entries == 10 * 5 * 5
    assert db.n_parameters == 3

    for k, v in db.parameter_ranges.items():

        if k == "epeak":

            assert len(v) == 10

        else:

            assert len(v) == 5

    assert database_file.exists()
    database_file.unlink()


def test_adding_params(parallel_script, parallel_add_script):

    database_file = Path("database_para.h5")

    if database_file.exists():

        database_file.unlink()

    sb = SimulationBuilder.from_yaml(parallel_script)

    os.system("python3 run_simulation.py")

    database_file = Path("database_para.h5")

    db = Database.from_file(str(database_file))

    assert db.n_entries == 10 * 5 * 5
    assert db.n_parameters == 3
    for k, v in db.parameter_ranges.items():

        if k == "epeak":

            assert len(v) == 10

        else:

            assert len(v) == 5

    sb = SimulationBuilder.from_yaml(parallel_add_script)

    assert Path("completed_parameters.json").exists()

    os.system("python3 run_simulation.py")

    database_file = Path("database_para.h5")

    db = Database.from_file(str(database_file))

    assert db.n_entries == 5 * 5 * 12

    assert db.n_parameters == 3

    for k, v in db.parameter_ranges.items():

        if k == "epeak":

            assert len(v) == 12

        else:

            assert len(v) == 5

    assert database_file.exists()

    database_file.unlink()
