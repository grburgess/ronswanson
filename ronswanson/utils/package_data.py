from pathlib import Path

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


# def get_path_of_user_config() -> Path:
#     """
#     get the path to the user configuration

#     :returns:

#     """
#     config_path: Path = Path().home() / ".config" / "ronswanson"

#     if not config_path.exists():

#         config_path.mkdir(parents=True)

#     return config_path


__all__ = [
    "get_path_of_data_file",
    "get_path_of_data_dir",
]
