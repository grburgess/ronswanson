# -*- coding: utf-8 -*-

"""Top-level package for ronswanson."""

__author__ = """J. Michael Burgess"""
__email__ = 'jburgess@mpe.mpg.de'

from ._version import get_versions
from .simulation import Simulation
from .simulation_builder import ParameterGrid, SimulationBuilder

__version__ = get_versions()['version']
del get_versions
