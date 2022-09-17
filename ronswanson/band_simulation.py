from typing import Dict

import numpy as np
from astromodels import Band

from ronswanson.grids import EnergyGrid

from .simulation import Simulation


class BandSimulation(Simulation):
    def __init__(
        self,
        simulation_id: int,
        parameter_set: Dict[str, float],
        energy_grid: EnergyGrid,
        out_file: str,
    ) -> None:
        super().__init__(simulation_id, parameter_set, energy_grid, out_file)

    def _run_call(self) -> Dict[str, np.ndarray]:

        b = Band(
            K=1,
            alpha=self._parameter_set["alpha"],
            beta=self._parameter_set["beta"],
            xp=self._parameter_set["epeak"],
        )

        return dict(output_0=b(self._energy_grid[0].grid))
