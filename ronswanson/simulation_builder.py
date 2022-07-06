from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .utils.logging import setup_logger

log = setup_logger(__name__)


@dataclass(frozen=True)
class Parameter:

    name: str
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    scale: Optional[str] = None
    n_points: Optional[int] = None
    values: Optional[np.ndarray] = None
    custom: bool = False
    #   grid: np.ndarray = field(init=False)

    def __post_init__(self):

        if not self.custom:

            # we will build a grid

            if (self.vmin is None) or (self.vmax is None):

                log.error("non-custom grids must include vmin and vmax")

                raise AssertionError

            if self.scale is None:

                log.error(
                    "non-custom grids must include scale 'log' or 'linear'"
                )

                raise AssertionError

            else:

                if self.scale not in ['log', 'linear']:

                    log.error(
                        "non-custom grids must include scale 'log' or 'linear'"
                    )

                    raise AssertionError

            if self.n_points is None:

                log.error("non-custom grids must include n_points")

                raise AssertionError

        else:

            if self.values is None:

                log.error("custom grids must include values")

                raise AssertionError

    @property
    def grid(self) -> np.ndarray:

        if self.custom:

            return self.values

        else:

            if self.scale.lower() == 'log':

                return np.geomspace(self.vmin, self.vmax, self.n_points)

            else:

                return np.linspace(self.vmin, self.vmax, self.n_points)
