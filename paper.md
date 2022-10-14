---
title: 'ronswanson: Building Table Models for 3ML'
tags:
  - Python
  - astronomy
  - spectral fitting
  - interpolation
authors:
  - name: J. Michael Burgess
    orcid: 0000-0003-3345-9515
    affiliation: "1"
bibliography: paper.bib
date: "13 October 2022"
---

# Summary

`ronswanson` provides a simple to use framework for building so-called table or
template models for `astromodels`[@astromodels] the modeling package for
multi-messenger astrophysical data analysis framework, `3ML`[@threeml]. This
allows for the construction of fast evaluating interpolation table of expensive
computer simulations of physical models which can then be fitted to data in a
reasonable amount of time. While `3ML` and `astromodels` provide factories for
building table models, the construction of pipelines for models that must be run
on HPC systems can be cumbersome. `ronswanson` removes this complexity with a
simple, reproducible templating system. Users can easily prototype their
pipeline on multi-core workstations and then switch to a multi-node HPC
system. `ronswanson` auto generates all python and SLURM[@slurm] scripts required for
the construction of the table model.



# Statement of need

Spatio-spectral fitting of astrophysical data many times requires using complex
physical models whose output is that of an computationally expensive
simulation. In these situations, the evaluation of the likelihood is intractable
even on HPC systems. To circumvent this issue, one can create a so-called
template or table model by evaluating the simulation on a grid of its
parameters, and use interpolation on the output which allows for the simulated
model to compared with data via the likelihood in a reasonable amount of
time. `ronswanson` builds table models for `astromodels`, the modeling language
of the multi-messenger data analysis framework `3ML`. `astromodels` stores its
table models as HDF5[@hdf5] files. While `astromodels` provides user friendly factories
for constructing table models, the workflow for using these factories on desktop
workstations or HPC systems can be complex. However, these workflows are easily
abstracted to a templating system that can be user-friendly and reproducible.


# Procedure

![Logo of ronswanson](docs/media/logo_sq.png)

Once the user selects a simulation from which they would like to create a table
model, the first task is to create an interface class that tells `ronswanson`
how to run the simulation and collect it's output. This is achieved by
inheriting a class from the package called `Simulation` and defining it's
virtual `run` member function. With this function, the user specifies how the
model parameters for each point in the simulation grid are fed to the simulation
software. The possibly multiple outputs from the simulation are passed to a
dictionary where each key separates the different outputs from each
other. Finally, this dictionary is returned from the `run` function. This is all
the programming that is required as `ronswanson` uses this subclass to run the
simulation on the user's specified architecture.

With the interface to the simulation defined, the user must specify the grid of
parameters on which to compute the output of the simulation. This is achieved by
specifying the grid points of each parameter in a YAML file. Parameter grids can
either be custom, or specified with ranges and a number of evaluation
points. Additionally, the energy grid corresponding to the evaluation of each of
the simulation outputs must be specified in this file. The final step is to
create a YAML configuration file telling `ronswanson` how to create the
table. This includes specifying the name of the output HDF5 database, where to
find the simulation subclass created in the first step, the name of the
parameter YAML file, and details on the compute architecture on which the
simulation grid is to be run.

With these two configuration files defined, the user runs the command line
program `simulation_build` on the main configuration file. This automatically
generates all the required python and SLURM scripts required for the
construction of the table model. If running on a workstation, the user then
executes the `run_simulation.py` script. If, instead, the simulation is run on
an HPC cluster, the user runs `sbatch run_simulation.sh`. In the case of running
on an HPC system, the final step to build the database requires running `sbatch
gather_results.sh` which uses MPI[@mpi] to gather the individual pieces of the
simulations into the main database.

The created HDF5 database can be loaded with utilities in `ronswanson` to then
construct an table model in the `astromodels` format. This intermediate step
allows the user to select subsets of the parameters from which to construct the
table model. This is useful as large interpolation tables can consume a lot of
computer memory and it is possible that certain fits may only need a limited
parameter range. Additionally, utilities are provided that allow adding
parameter sets onto the primary database to extend the interpolation
range. Moreover, the database stores information such as the runtime of each
grid point of the simulation. Utilities are provided to view these
metadata. More details and examples can be found in the
[documentation](http://jmichaelburgess.com/ronswanson/index.html)

# Acknowledgments

This project was inspired by earlier works of Elisa Schoesser and Francesco
Berlato.

# References
