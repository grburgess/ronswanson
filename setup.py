#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import versioneer
from setuptools import setup


setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    py_modules=['scripts'],
    entry_points={
        'console_scripts': [
            'simulation_build = scripts.simulation_build:simulation_build',
            'examine_simulation = scripts.check:examine_simulation',
            'examine_simulation_detailed = scripts.check:examine_simulation_detailed'
        ],
    },
    #        package_data={"": extra_files},
)
