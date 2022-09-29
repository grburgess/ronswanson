#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import versioneer
from setuptools import setup


setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    entry_points={
        'console_scripts': [
            'simulation_build = scripts.simulation_build:simulation_build',
        ],
    },
    #        package_data={"": extra_files},
)
