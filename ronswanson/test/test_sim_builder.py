import numpy as np
import pytest

from ronswanson.simulation_builder import (
    Parameter,
    ParameterGrid,
    SimulationBuilder,
)
from ronswanson.utils.package_data import get_path_of_data_file


def test_script_gen():

    file_name = get_path_of_data_file("test_params.yml")

    pg = ParameterGrid.from_yaml(file_name)

    sb = SimulationBuilder(
        pg,
        "database.h5",
        "from ronswanson.band import BandSimulation",
        n_cores=8,
    )
