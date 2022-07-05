#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import versioneer
from setuptools import setup

# # Create list of data files
# def find_data_files(directory):

#     paths = []

#     for (path, directories, filenames) in os.walk(directory):

#         for filename in filenames:

#             paths.append(os.path.join("..", path, filename))

#     return paths


#extra_files = find_data_files("ronswanson/data")

setup(

    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    #        package_data={"": extra_files},
)
