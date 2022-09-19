from pathlib import Path
import yaml
from ronswanson.utils.package_data import get_path_of_data_file

test_params = str(get_path_of_data_file("test_params.yml"))

out = {}

out[
    "import_line"
] = "from ronswanson.band_simulation import BandSimulation as Simulation"
out["parameter_grid"] = str(get_path_of_data_file("test_params.yml"))
out["out_file"] = "database_para.h5"

simulation = {}
simulation["n_mp_jobs"] = 8

out["simulation"] = simulation

file_name = str(Path("sim_build.yml").absolute())

with Path(file_name).open("w") as f:

    yaml.dump(out, stream=f, Dumper=yaml.SafeDumper)
