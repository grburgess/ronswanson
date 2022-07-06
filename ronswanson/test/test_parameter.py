import numpy as np
import pytest

from ronswanson.simulation_builder import Parameter


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
