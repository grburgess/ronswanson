import os
import shutil
from glob import glob
from pathlib import Path

import pytest
import yaml

from ronswanson.database import Database
from ronswanson.utils.package_data import get_path_of_data_file


@pytest.fixture(scope="session")
def database():

    db = Database.from_file(get_path_of_data_file("test_database.h5"))

    yield db

    p = Path("~/.astromodels/data/ron.h5").expanduser()

    p.unlink()


@pytest.fixture(scope="session")
def node_script():

    out = {}

    out[
        "import_line"
    ] = "from ronswanson.band_simulation import BandSimulation as Simulation"
    out["parameter_grid"] = str(get_path_of_data_file("test_params.yml"))
    out["out_file"] = "database_node.h5"

    simulation = {}
    simulation["n_mp_jobs"] = 72
    simulation["n_cores_per_node"] = 72
    simulation["use_nodes"] = True

    gather = {}
    gather["n_gather_per_core"] = 100
    gather["n_cores_per_node"] = 70
    gather["time"] = dict(hrs=1)

    out["simulation"] = simulation
    out["gather"] = gather

    file_name = str(Path("node_build.yml").absolute())

    with Path(file_name).open("w") as f:

        yaml.dump(out, stream=f, Dumper=yaml.SafeDumper)

    yield file_name

    Path(file_name).unlink()

    for x in Path(".").glob("*.sh"):

        x.unlink()

    Path("output").rmdir()

    if Path("gather_file.json").exists():

        Path("gather_file.json").unlink()

    if Path("key_file.json").exists():

        Path("key_file.json").unlink()


@pytest.fixture(scope="function")
def linear_script():

    out = {}

    out[
        "import_line"
    ] = "from ronswanson.band_simulation import BandSimulation as Simulation"
    out["parameter_grid"] = str(get_path_of_data_file("test_params.yml"))
    out["out_file"] = "database_lin.h5"

    simulation = {}
    simulation["n_mp_jobs"] = 8
    simulation["linear_execution"] = True

    out["simulation"] = simulation

    file_name = str(Path("linear_build.yml").absolute())

    with Path(file_name).open("w") as f:

        yaml.dump(out, stream=f, Dumper=yaml.SafeDumper)

    yield file_name

    Path(file_name).unlink()

    if Path("gather_file.json").exists():

        Path("gather_file.json").unlink()

    if Path("gather_results.py").exists():

        Path("gather_results.py").unlink()

    if Path("key_file.json").exists():

        Path("key_file.json").unlink()


@pytest.fixture(scope="function")
def parallel_script():

    out = {}

    out[
        "import_line"
    ] = "from ronswanson.band_simulation import BandSimulation as Simulation"
    out["parameter_grid"] = str(get_path_of_data_file("test_params.yml"))
    out["out_file"] = "database_para.h5"

    simulation = {}
    simulation["n_mp_jobs"] = 8

    out["simulation"] = simulation

    file_name = str(Path("parallel_build.yml").absolute())

    with Path(file_name).open("w") as f:

        yaml.dump(out, stream=f, Dumper=yaml.SafeDumper)

    yield file_name

    Path(file_name).unlink()

    if Path("gather_file.json").exists():

        Path("gather_file.json").unlink()

    if Path("key_file.json").exists():

        Path("key_file.json").unlink()


@pytest.fixture(scope="function")
def parallel_add_script():

    out = {}

    out[
        "import_line"
    ] = "from ronswanson.band_simulation import BandSimulation as Simulation"
    out["parameter_grid"] = str(
        get_path_of_data_file("test_addition_params.yml")
    )
    out["out_file"] = "database_para.h5"

    simulation = {}
    simulation["n_mp_jobs"] = 8

    out["simulation"] = simulation

    file_name = str(Path("parallel_add_build.yml").absolute())

    with Path(file_name).open("w") as f:

        yaml.dump(out, stream=f, Dumper=yaml.SafeDumper)

    yield file_name

    Path(file_name).unlink()

    if Path("gather_file.json").exists():

        Path("gather_file.json").unlink()

    if Path("key_file.json").exists():

        Path("key_file.json").unlink()
