from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from rich.tree import Tree

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
    mpi_modules: Optional[List[str]] = None
    python: str = "python"
    store_dir: Optional[str] = None


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


def recurse_dict(d, tree) -> None:

    for k, v in d.items():

        if (type(v) == dict) or isinstance(v, DictConfig):

            branch = tree.add(
                k, guide_style="bold medium_orchid", style="bold medium_orchid"
            )

            recurse_dict(v, branch)

        else:

            tree.add(
                f"{k}: [blink cornflower_blue]{v}",
                guide_style="medium_spring_green",
                style="medium_spring_green",
            )

    return


def show_configuration() -> Tree:

    tree = Tree(
        "config", guide_style="bold medium_orchid", style="bold medium_orchid"
    )

    recurse_dict(ronswanson_config, tree)

    return tree
