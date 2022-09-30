---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Intro

So you need to build a table model? 3ML via astromodels provides you with nice
facilities to accomplish this task. But often, we need to interface with
computationally expensive simulations and require many runs of these
simulations. This is a very generic task and Ron Swanson wants to make things as
simple as possible for you. Making things complicated is annoying.

![alt text](https://raw.githubusercontent.com/grburgess/ronswanson/master/docs/media/mad.jpg)


## Example with a Band function

Let's say we want to make a table model from a Band function.

We pick a parameter grid and a grid of energies for our simulation. We can enter
these in a YAML file:

```yaml
alpha:
  custom: no
  vmin: -1
  vmax: 0
  scale: "linear"
  n_points: 10
  
beta:
  custom: no
  vmin: -3
  vmax: -2
  scale: "linear"
  n_points: 10
  
epeak:
  custom: yes
  values: [50., 69.74 , 97.29, 135.72, 189.32, 264.097, 368.4, 513.9, 716.87, 1000.]

energy_grid:
  custom: no
  vmin: 10
  vmax: 1000
  scale: "log"
  n_points: 50


```

As can be seen, we can specify the parameter/energy grids ourselves, or we can
specify their ranges and let it be done for us.

It is possible that a simulation outputs more than one type of array (photons,
electrons, neutrinos, etc.). In this case, each output may have its own energy
grid. These can be specified as `energy_grid_0`,
`energy_grid_1`...`energy_grid_n`. More on how to grab the output from
these below.





### The Simulation class

Now we need to make a class for the simulation. We will inherit from the
simulation class and specify a `_run_call` function that tells the program how
to run the simulation for a given set of parameters. This function **must**
return a dictionary of arrays of photon / particle fluxes for the given
energies. The keys of the dictionary should be `output_0`,
`output_1`...`output_n` for each type of output corresponding to the
energy grids above.



```python
from typing import Dict

import numpy as np
from astromodels import Band
import ronswanson as dukesilver


class BandSimulation(dukesilver.Simulation):
    def __init__(
        self,
        simulation_id: int,
        parameter_set: Dict[str, float],
        energy_grid: np.ndarray,
        out_file: str,
    ) -> None:
        super().__init__(simulation_id, parameter_set, energy_grid, out_file)

    def _run_call(self) -> np.ndarray:

        b = Band(
            K=1,
            alpha=self._parameter_set["alpha"],
            beta=self._parameter_set["beta"],
            xp=self._parameter_set["epeak"],
        )

        return dict(output_0=b(self._energy_grid[0].grid))

```


### The Simulation Builder

Now we need to tell the simulation builder a few things so it can construct our
files for us. We have stored this YAML file in the repo itself. You should use
your own!

The `SimulationBuilder` class takes a parameter grid, the name of the file that
will be created, the import line for the custom simulation class, the number of
cores and nodes to execute on.

We configure this with a YAML file.

```yaml
# the import line to get our simulation
# we ALWAYS import as Simulation
import_line: "from ronswanson.band_simulation import BandSimulation as Simulation"

# the parameter grid 
parameter_grid: "test_params.yml"

# name of our database
out_file: database.h5

# clean out the simulation directory after
# the run. It is defaulted to yes
clean: yes

simulation:
  
  # number of multi-process jobs
  n_mp_jobs: 8




```

```python
import ronswanson

sb = ronswanson.SimulationBuilder.from_yaml("sim_build.yml")
```

However, you can easily do this from the command line:

```
> simulation_build sim_build.yml
```



Now a python file will be written to the disk which you can run to create your
simulation runs. we can have a look at the file.

```
from ronswanson.band_simulation import BandSimulation as Simulation
from joblib import Parallel, delayed
from ronswanson import ParameterGrid

pg = ParameterGrid.from_yaml('/Users/jburgess/coding/projects/ronswanson/parameters.yml')
def func(i):
	params = pg.at_index(i)
	simulation = Simulation(i, params, pg.energy_grid.grid,'/Users/jburgess/coding/projects/ronswanson/database.h5')
	simulation.run()

iteration = [i for i in range(0, pg.n_points)]

Parallel(n_jobs=8)(delayed(func)(i) for i in iteration)

```

<!-- #region --> Now this simply uses `joblib` to farm out the iterations over
the parameter combinations. If iterations are to also be divided across HPC
nodes, the python script will be modified and an associated `SLURM` script will
be generated.

<!-- #region -->
#### SLURM and advanced options

Configuring for SLURM and SBATCH systems is similar, but there are a few more options. `ronswanson` will set up bash scripts that will submit jobs to complete the simulations and then gather them into a database.

Here is an example script:

```yaml

# the import line to get our simulation
# we ALWAYS import as Simulation
import_line: "from ronswanson.band_simulation import BandSimulation as Simulation"

# the parameter grid 
parameter_grid: "test_params.yml"

# name of our database
out_file: database.h5

simulation:
  
  # number of multi-process jobs PER node
  n_mp_jobs: 9
  
  # number of cpus to request per node (leave room for threads)
  n_cores_per_node: 72
  
  # number of runs per node
  # here, the 9 mp jobs will take turns with these 500 runs
  run_per_node: 500
  
  # the switch to say we are performing SLURM job
  use_nodes: yes
  
  # optional maximum number of nodes to request
  # if more than this are required, multiple
  # submission scripts are generated
  max_nodes: 300

  # the max run time for each job in the array
  time:
    hrs: 10
    min: 30
    sec: 0

gather:

  # after the simulations run
  # you submit and MPI job that collects the simualtions
  
  # number of simulations to collect per MPI rank
  n_gather_per_core: 100
  
  # number of cpus per node
  n_cores_per_node: 70

  # maximum job time
  time:
    hrs: 1
    min: 0
    sec: 0

```


Additional configuration of SLURM jobs can be handle with the `ronswanson` configuration.
<!-- #endregion -->

### The Database



Upon running the script, an HDF5 database of the runs is created which contains
all the information needed to build a table model in `3ML`.  <!-- #endregion -->

```python
from ronswanson.utils.package_data import get_path_of_data_file
from ronswanson import Database

```

```python
db = Database.from_file(get_path_of_data_file("test_database.h5"))
db.parameter_ranges
db.plot_runtime()
```

Now we can use the database to construct and save a table model for `3ML`

```python
table_model = db.to_3ml("my_model", "a new model for 3ML")
table_model
```

```python
import matplotlib.pyplot as plt
from astromodels import Band

%matplotlib inline
```

We can compare our table model with the Band function.

```python
import numpy as np
ene = np.geomspace(10,1000,100)

b = Band()
b.alpha = -.6
b.beta = -2.5
b.xp = 250
b.K=1.05

table_model.alpha=-.6
table_model.beta = -2.5
table_model.epeak = 250



fig, ax = plt.subplots()

ax.loglog(ene, ene**2 * table_model(ene), color='#33FFC4',lw=3,label="table")
ax.loglog(ene, ene**2 * b(ene),color='#C989FB',lw=3, label="band")

ax.set(xlabel="energy", ylabel="vFv")

ax.legend()

```

<!-- #region -->
Great! That was way easier than programming everything yourself.

![alt text](https://raw.githubusercontent.com/grburgess/ronswanson/master/docs/media/happy.jpeg)


Suppose we did not want to use all the values in the parameter ranges we have simulated. Bigger interpolation tables take up memory when fitting. 

We can select a subset of the parameter ranges when building the table
<!-- #endregion -->

```python
selection  = {}
selection['alpha'] = dict(vmax=0)
selection['epeak'] = dict(vmin=200, vmax=700)

table_model_small = db.to_3ml("my_model_small", "a new model for 3ML", **selection)
table_model_small
```

Awesome! Now go enjoy your weekend.

![alt text](https://raw.githubusercontent.com/grburgess/ronswanson/master/docs/media/enjoy.jpg)


## User configuration

A simple YAML configuration is stored in `~/.config/ronswanson/ronswanson_config.yml`. It allows for configuring the log as well as putting default SLURM configuration parameters.

An example:

```yaml

logging:
  'on': on
  level: INFO
slurm:

  # where to send SLURM emails
  user_email: my_email.com
  
  # modules to be loaded for MPI jobs (gather script)
  mpi_modules: ['intel/21.4.0', 'impi/2021.4',
  'anaconda/3/2021.11','hdf5-mpi/1.12.2', 'mpi4py/3.0.3', 'h5py-mpi/2.10']
  
  # modules to load for simulation jobs
  modules: ['intel/21.4.0', 'impi/2021.4', 'anaconda/3/2021.11','hdf5-serial/1.12.2']
  
  # the python binary for running the simulation jobs
  python: "python3"
  
  # where to store the simulations before
  # database creation (default will be the directory where the code is run)
  store_dir: /ptmp/jburgess

```

The configuration can be modified on the fly.

```python
from ronswanson import ronswanson_config, show_configuration
```

```python
show_configuration()
```

```python
ronswanson_config.slurm.user_email
```

```python
ronswanson_config.slurm.user_email = "workemail@email.com"
```

```python
show_configuration()
```

```python

```
