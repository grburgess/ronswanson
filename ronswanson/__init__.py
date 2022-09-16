# -*- coding: utf-8 -*-

"""Top-level package for ronswanson."""

__author__ = """J. Michael Burgess"""
__email__ = 'jburgess@mpe.mpg.de'

from ._version import get_versions
from .database import Database
from .simulation import Simulation
from .simulation_builder import SimulationBuilder
from .grids import ParameterGrid
from .utils import ronswanson_config, show_configuration
from .utils.logging import update_logging_level

__all__ = [
    "Simulation",
    "ParameterGrid",
    "SimulationBuilder",
    "Database",
    "update_logging_level",
    "ronswanson_config",
    "show_configuration",
]


__version__ = get_versions()['version']
del get_versions
