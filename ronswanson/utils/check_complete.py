import itertools
import re
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.colors as mcolors
import matplotlib
import numpy as np
import plotille
from tqdm.auto import tqdm

from ronswanson.grids import ParameterGrid
from ronswanson.utils.color import Colors
from ronswanson.utils.colormap_generator import get_continuous_cmap
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

    n_points = pg.n_points

    for jdx, other_name in enumerate(pg.parameter_names):

        # do not check against yourself
        if idx != jdx:

            finished_values = []
            unfinished_values = []

            k = 0

            for i in range(n_points):

                result = pg.full_grid[i]

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


def make_fig_detailed(
    this_name: str,
    other_name: str,
    this_param_values: np.ndarray,
    other_param_values: np.ndarray,
    count_dict: Dict[str, Dict[str, int]],
    n_points: int,
    width: int,
    colormap,
) -> None:

    fig = plotille.Figure()

    fig.register_label_formatter(float, int_formatter)
    fig.color_mode = 'rgb'
    fig.width = int(np.ceil(width))
    fig.height = 10
    fig.x_label = this_name
    fig.y_label = other_name

    total_other = n_points / (len(this_param_values) * len(other_name))

    for result in itertools.product(this_param_values, other_param_values):

        a, b = result

        count = 0

        if a in count_dict:

            if b in count_dict[a]:

                count = count_dict[a][b]

        color = mcolors.to_hex(colormap(float(count / total_other)))[1:]

        fig.scatter(
            [a],
            [b],
            lc=color,
            marker="o",
        )

    print(fig.show())


def examine_parameter_detailed(
    database_file_name: str,
    parameter_grid_file_name: str,
    parameter_to_check: str,
    colormap: Optional[str] = None,
) -> None:

    if colormap is None:

        cmap = get_continuous_cmap(['#FE093A', '#FFFF53', '#2DFF96'])

    else:

        cmap = matplotlib.cm.get_cmap(colormap)

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

    n_points = pg.n_points

    for jdx, other_name in enumerate(pg.parameter_names):

        counts = np.zeros((len(finished_ids), 2))

        # do not check against yourself
        if idx != jdx:

            finished_values = []
            unfinished_values = []

            for i, j in enumerate(finished_ids):

                result = pg.full_grid[j]

                this_param = result[idx]
                other_param = result[jdx]

                counts[i] = np.array([this_param, other_param])

            unique, counts = np.unique(counts, axis=0, return_counts=True)

            count_dict = {}

            for i, u in enumerate(unique):

                if u[0] in count_dict:

                    count_dict[u[0]][u[1]] = counts[i]

                else:

                    inner_dict = {}
                    inner_dict[u[1]] = counts[i]
                    count_dict[u[0]] = inner_dict

            finished_values = np.array(finished_values)
            unfinished_values = np.array(unfinished_values)

            make_fig_detailed(
                parameter_to_check,
                other_name=other_name,
                this_param_values=pg.parameter_list[idx].grid,
                other_param_values=pg.parameter_list[jdx].grid,
                count_dict=count_dict,
                n_points=n_points,
                width=fig_width,
                colormap=cmap,
            )
