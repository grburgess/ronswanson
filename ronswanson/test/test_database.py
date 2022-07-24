from ronswanson.database import Database
from astromodels import Band
import numpy as np


def test_database(database: Database):

    assert database.n_parameters == 3
    assert database.n_entries == 10 * 10 * 10

    assert database._grid_points.shape == (10 * 10 * 10, 3)

    assert len(database._parameter_names) == 3

    tm = database.to_3ml("ron", "ron")

    selections = {}
    selections["alpha"] = dict(vmin=-0.5)
    selections["epeak"] = dict(vmin=300, vmax=800)

    tm = database.to_3ml("ron", "ron", overwrite=True, **selections)


def test_table_model(database: Database):

    tm = database.to_3ml("ron", "ron", overwrite=True)

    b = Band()

    b.alpha = -0.6
    b.beta = -2.5
    b.xp = 250
    b.K = 1.00

    tm.alpha = -0.6
    tm.beta = -2.5
    tm.epeak = 250

    ene = np.geomspace(10, 1000, 25)

    np.testing.assert_allclose(b(ene), tm(ene), rtol=1e-1)

    b.alpha = -1
    b.beta = -2.0
    b.xp = 300
    b.K = 1.00

    tm.alpha = -1
    tm.beta = -2
    tm.epeak = 300

    np.testing.assert_allclose(b(ene), tm(ene), rtol=1e-1)
