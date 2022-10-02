import click

from ronswanson.utils.check_complete import (
    examine_parameter,
    examine_parameter_detailed,
)


@click.command()
@click.argument('database', nargs=1)
@click.argument('parameter_grid', nargs=1)
@click.argument('parameter', nargs=1)
def examine_simulation(
    database: str, parameter_grid: str, parameter: str
) -> None:

    examine_parameter(database, parameter_grid, parameter)


@click.command()
@click.argument('database', nargs=1)
@click.argument('parameter_grid', nargs=1)
@click.argument('parameter', nargs=1)
@click.option('--colormap', default=None)
def examine_simulation_detailed(
        database: str, parameter_grid: str, parameter: str, colormap
) -> None:

    examine_parameter_detailed(database, parameter_grid, parameter, colormap)
