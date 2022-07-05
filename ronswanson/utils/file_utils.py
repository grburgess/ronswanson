import os
import shutil
import tempfile
from builtins import str
from contextlib import contextmanager
from pathlib import Path


def sanitize_filename(filename, abspath: bool = False) -> Path:

    path: Path = Path(filename)

    sanitized = path.expanduser()

    if abspath:

        return sanitized.absolute()

    else:

        return sanitized


def file_existing_and_readable(filename) -> bool:

    sanitized_filename: Path = sanitize_filename(filename)

    return sanitized_filename.is_file()


def fits_file_existing_and_readable(filename) -> bool:
    """
    checks if a FITS file exists ignoring extension ({})
    info

    """
    base_filename = str(filename).split("{")[0]

    return file_existing_and_readable(base_filename)


def path_exists_and_is_directory(path) -> bool:

    sanitized_path: Path = sanitize_filename(path, abspath=True)

    return sanitized_path.is_dir()


def if_directory_not_existing_then_make(directory) -> None:
    """
    If the given directory does not exists, then make it

    :param directory: directory to check or make
    :return: None
    """

    sanitized_directory: Path = sanitize_filename(directory)

    try:

        sanitized_directory.mkdir(parents=True, exist_ok=False)

    except (FileExistsError):

        # should add logging here!

        pass


@contextmanager
def temporary_directory(prefix="", within_directory=None):
    """
    This context manager creates a temporary directory in the most secure possible way (with no race condition), and
    removes it at the end.

    :param prefix: the directory name will start with this prefix, if specified
    :param within_directory: create within a specific directory (assumed to exist). Otherwise, it will be created in the
    default system temp directory (/tmp in unix)
    :return: the absolute pathname of the provided directory
    """

    directory = tempfile.mkdtemp(prefix=prefix, dir=within_directory)

    yield directory

    try:

        shutil.rmtree(directory)

    except:

        custom_warnings.warn(
            "Couldn't remove temporary directory %s" % directory)


@contextmanager
def within_directory(directory):

    path: Path = Path(directory)

    assert path.is_dir(), f"path {path} does not exist!"

    current_dir: Path = Path(".")

    os.chdir(path)
    try:

        yield

    finally:

        os.chdir(current_dir)
