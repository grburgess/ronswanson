from pathlib import Path
from shutil import copyfile

import pkg_resources


def get_path_of_data_dir() -> Path:
    """
    get the path of the package data directory

    :returns:

    """
    file_path: str = pkg_resources.resource_filename("ronswanson", "data")

    return Path(file_path)


def get_path_of_data_file(data_file: str) -> Path:
    """
    get the path of a dat file

    :param data_file: name of the data file
    :type data_file: str
    :returns:

    """
    file_path: Path = get_path_of_data_dir() / data_file

    return file_path


def copy_package_data(data_file: str) -> None:
    """
    copy file from the package data directory
    to the current directory

    :param data_file: the name of the file
    :type data_file: str
    :returns:

    """
    data_file_path: Path = get_path_of_data_file(data_file)
    copyfile(data_file_path, f"./{data_file}")


def get_path_of_log_dir() -> Path:
    """
    return the path of the logging directory

    :returns:

    """
    p: Path = Path("~/.log/ronswanson").expanduser()

    if not p.exists():

        p.mkdir(parents=True)

    return p


def get_path_of_log_file(log_file: str) -> Path:
    """
    returns the path of a log file

    :param log_file: the name of the log file
    :type log_file: str
    :returns:

    """
    return get_path_of_log_dir() / log_file


def get_path_of_user_config() -> Path:
    """
    get the path to the user configuration

    :returns:

    """
    config_path: Path = Path().home() / ".config" / "ronswanson"

    if not config_path.exists():

        config_path.mkdir(parents=True)

    return config_path


__all__ = [
    "get_path_of_data_file",
    "get_path_of_data_dir",
    "get_path_of_user_config",
]
