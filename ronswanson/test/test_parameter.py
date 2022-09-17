import numpy as np
import pytest

from ronswanson.grids import Parameter, ParameterGrid
from ronswanson.utils.package_data import get_path_of_data_file


def test_constructor():

    p = Parameter("p", vmin=1, vmax=10, scale="linear", n_points=10)

    assert len(p.grid) == 10

    p = Parameter("p", values=np.linspace(1, 10, 10), custom=True)

    assert len(p.grid) == 10

    with pytest.raises(AssertionError):

        p = Parameter("p", vmin=1, vmax=10)

    with pytest.raises(AssertionError):

        p = Parameter("p", scale='linear')

    with pytest.raises(AssertionError):

        p = Parameter("p", vmin=1, vmax=10, scale="wrong", n_points=10)

    with pytest.raises(AssertionError):

        p = Parameter("p", custom=True)


def test_parameter_grid():

    file_name = get_path_of_data_file("test_params.yml")

    pg = ParameterGrid.from_yaml(file_name)

    assert len(pg.parameter_list) == 3
    assert len(pg.parameter_list[0].grid) == 5
    assert len(pg.parameter_list[1].grid) == 5
    assert len(pg.parameter_list[2].grid) == 10

    assert pg.n_points == 10 * 5 * 5
