import click

from ronswanson.utils.check_complete import examine_parameter


@click.command()
@click.argument('database', nargs=1)
@click.argument('parameter_grid', nargs=1)
@click.argument('parameter', nargs=1)
def examine_simulation(
    database: str, parameter_grid: str, parameter: str
) -> None:

    examine_parameter(database, parameter_grid, parameter)
