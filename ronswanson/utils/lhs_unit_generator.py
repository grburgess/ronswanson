import h5py
from scipy.stats import qmc


def generate_lhs_unit_cube(
    n_parameters: int, n_samples: int, file_name: str
) -> None:
    """
    generate a until latin hypercube for faster generation
    of points to be scaled later

    :param n_parameters:
    :type n_parameters: int
    :param n_samples:
    :type n_samples: int
    :param file_name:
    :type file_name: str
    :returns:

    """


    sampling = qmc.LatinHypercube(d=n_parameters, optimization="random-cd")

    samples = sampling.random(n_samples)

    with h5py.File(file_name, "w") as f:

        f.create_dataset("lhs_points", data=samples, compression="gzip")
