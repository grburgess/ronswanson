from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from omegaconf import OmegaConf

# Path to configuration

_config_path = Path("~/.config/ronswanson/").expanduser()

_config_name = Path("ronswanson_config.yml")

_config_file = _config_path / _config_name

# Define structure of configuration with dataclasses


@dataclass
class Logging:

    on: bool = True
    level: str = "WARNING"


@dataclass
class SLURM:
    user_email: str = "user@email.com"
    modules: Optional[List[str]] = None


@dataclass
class RonSwansonConfig:

    logging: Logging = Logging()
    slurm: SLURM = SLURM()


# Read the default config
ronswanson_config: RonSwansonConfig = OmegaConf.structured(RonSwansonConfig)

# Merge with local config if it exists
if _config_file.is_file():

    _local_config = OmegaConf.load(_config_file)

    ronswanson_config: RonSwansonConfig = OmegaConf.merge(
        ronswanson_config, _local_config
    )

# Write defaults if not
else:

    # Make directory if needed
    _config_path.mkdir(parents=True, exist_ok=True)

    with _config_file.open("w") as f:

        OmegaConf.save(config=ronswanson_config, f=f.name)
