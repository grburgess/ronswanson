import itertools

import re
from pathlib import Path
from typing import List

import numpy as np
import plotille
from tqdm.auto import tqdm

from ronswanson.grids import ParameterGrid
from ronswanson.utils.color import Colors
from ronswanson.utils.logging import setup_logger

from .configuration import ronswanson_config

log = setup_logger(__name__)


def check_complete_ids(database_file_name: str) -> List[int]:

    p = Path(database_file_name)

    if ronswanson_config.slurm.store_dir is None:

        parent_dir = p.absolute().parent

    else:

        parent_dir = Path(ronswanson_config.slurm.store_dir).absolute()

    multi_file_dir: Path = parent_dir / Path(f"{p.stem}_store")

    finished_ids: List[int] = []

    for fname in tqdm(
        list(multi_file_dir.glob("sim_store_*.h5")),
        desc="checking existing files",
        colour=Colors.GREEN.value,
    ):

        sim_id = int(re.search("sim_store_(\d*).h5", str(fname)).groups()[0])

        finished_ids.append(sim_id)

    finished_ids = np.array(finished_ids)

    return finished_ids


def int_formatter(val, chars, delta, left):
    return '{:{}{}}'.format(int(val), '<' if left else '>', chars)


def make_fig(
    this_name: str,
    other_name: str,
    finished_values: np.ndarray,
    unfinished_values: np.ndarray,
    width: int,
) -> str:

    fig = plotille.Figure()

    fig.register_label_formatter(float, int_formatter)
    fig.color_mode = 'rgb'
    fig.width = int(np.ceil(width))
    fig.height = 10
    fig.x_label = this_name
    fig.y_label = other_name

    fig.scatter(
        unfinished_values[:, 0],
        unfinished_values[:, 1],
        lc="FE093A",
        marker="o",
    )

    fig.scatter(
        finished_values[:, 0],
        finished_values[:, 1],
        lc="2DFF96",
        marker="o",
    )

    print(fig.show())

    return " "


def examine_parameter(
    database_file_name: str,
    parameter_grid_file_name: str,
    parameter_to_check: str,
) -> None:

    finished_ids = check_complete_ids(database_file_name)

    pg = ParameterGrid.from_yaml(parameter_grid_file_name)

    if parameter_to_check not in pg.parameter_names:

        msg = f"{parameter_to_check} is not a valid parameter name"

        log.error(msg)

        raise RuntimeError(msg)

    idx = pg.parameter_names.index(parameter_to_check)

    lines: List[str] = []

    fig_width = int(np.ceil(100.0 / (pg.n_parameters - 1)))
    fig_width = 40

    plt_idx = 0
    for jdx, other_name in enumerate(pg.parameter_names):

        # do not check against yourself
        if idx != jdx:

            finished_values = []
            unfinished_values = []

            k = 0

            for result in itertools.product(
                *[p.grid for p in pg.parameter_list]
            ):

                this_param = result[idx]
                other_param = result[jdx]

                if k in finished_ids:

                    finished_values.append([this_param, other_param])

                else:

                    unfinished_values.append([this_param, other_param])

                k += 1

            finished_values = np.array(finished_values)
            unfinished_values = np.array(unfinished_values)

            line = make_fig(
                parameter_to_check,
                other_name=other_name,
                finished_values=finished_values,
                unfinished_values=unfinished_values,
                width=fig_width,
            )

            lines.append(line)

        # for line in lines:

        #     print(' '.join(line))
