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
affiliations:
 - name: Max Planck Institute for Extraterrestrial Physics, Giessenbachstrasse, 85748 Garching, Germany
   index: 1
date: "13 October 2022"
---

# Summary

`ronswanson` provides a simple-to-use framework for building so-called table or
template models for `astromodels`[@astromodels] the modeling package for
multi-messenger astrophysical data-analysis framework, `3ML`[@threeml]. With
astromodels and 3ML one can build the interpolation table of a physical model
result of an expensive computer simulation. This then allow to fastly
re-evaluate the model several times, for example while fitting it to a
dataset. While `3ML` and `astromodels` provide factories for building table
models, the construction of pipelines for models that must be run on
high-performance computing (HPC) systems can be cumbersome. `ronswanson` removes
this complexity with a simple, reproducible templating system. Users can easily
prototype their pipeline on multi-core workstations and then switch to a
multi-node HPC system. `ronswanson` ronswanson automatically generates all
python and SLURM scripts to scale the execution of 3ML with the astromodel's
table models on a HPC system.



# Statement of need

Spatio-spectral fitting of astrophysical data might require the iterative
evaluation of a complex physical model obtained from a computationally expensive
simulation. In these situations, the evaluation of the likelihood is intractable
even on HPC systems. To circumvent this issue, one can create a so-called
template or table model by evaluating the simulation on a grid of its
parameters, and use interpolation on the output which allows for the simulated
model to compared with data via the likelihood in a reasonable amount of
time. Several spectral fitting packages (e.g. `XSPEC`[@xspec], `3ML`,
`gammapy`[@gammapy, @acero_fabio_2023_7734804]) implement frameworks that allow for the reading in of these
template models in various file formats. However, there is no framework that
assists in uniformly performing the task of generating the data from which these
templates are built. `ronswanson` builds table models for `astromodels`, the
modeling language of the multi-messenger data analysis framework `3ML` in an
attempt to solve this problem. `astromodels` stores its table models as
HDF5[@hdf5] files. While `astromodels` provides user friendly factories for
constructing table models, the workflow for using these factories on desktop
workstations or HPC systems can be complex. However, these workflows are easily
abstracted to a templating system that can be user-friendly and reproducible.


# Procedure

Once the user selects a simulation from which they would like to create a table
model, the first task is to create an interface class that tells `ronswanson`
how to run the simulation and collect its output. This is achieved by inheriting
a class from the package called `Simulation` and defining its virtual `run`
member function. With this function, the user specifies how the model parameters
for each point in the simulation grid are fed to the simulation software. The
possibly multiple outputs from the simulation are passed to a dictionary where
each key separates the different outputs from each other. Finally, this
dictionary is returned from the `run` function. This is all the programming that
is required as `ronswanson` uses this subclass to run the simulation on the
user's specified architecture.

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
construct an table model in the `astromodels` format ([see here for
details](https://threeml.readthedocs.io/en/stable/notebooks/spectral_models.html#Template-(Table)-Models)). This
intermediate step allows the user to select subsets of the parameters from which
to construct the table model. This is useful as large interpolation tables can
consume a lot of computer memory and it is possible that certain fits may only
need a limited parameter range. Additionally, utilities are provided that allow
adding parameter sets onto the primary database to extend the interpolation
range. Moreover, the database stores information such as the runtime of each
grid point of the simulation. Utilities are provided to view these
metadata. With future interfacing of `3ML` and `gammapy`, these table models can
even be used to fit data from optical to very high energy gamma-rays. More
details and examples can be found in the
[documentation](http://jmichaelburgess.com/ronswanson/index.html).

# Acknowledgments

This project was inspired by earlier works of Elisa Schoesser and Francesco
Berlato.

# References
