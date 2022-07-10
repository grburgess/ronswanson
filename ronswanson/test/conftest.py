import logging
import os
import shutil
from glob import glob
from pathlib import Path

import pytest
from ronswanson.utils.package_data import get_path_of_data_file
from ronswanson.database import Database


@pytest.fixture(scope="session")
def database():

    db = Database.from_file(get_path_of_data_file("test_database.h5"))

    yield db

    p = Path("~/.astromodels/data/ron.h5").expanduser()

    p.unlink()
