import click

from ronswanson import SimulationBuilder

@click.command()
@click.argument('file_name')
def simulation_build(file_name: str) -> None:
    SimulationBuilder.from_yaml(file_name)
