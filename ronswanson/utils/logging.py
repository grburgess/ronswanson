import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from .configuration import ronswanson_config


class LogFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno != self.__level


_console_formatter = logging.Formatter(
    ' %(message)s',
    datefmt="%H:%M:%S",
)


_theme = {}

# Banner
_theme["h1"] = "deep_sky_blue3"
_theme["status.spinner"] = "cyan2"
_theme["status.text"] = "deep_sky_blue4"
_theme["repr.filename"] = "blue"
_theme["repr.number"] = "white"
_theme["repr.path"] = "grey37"
_theme["repr.str"] = "grey37"
_theme["repr.tag_name"] = "white"
_theme["repr.url"] = "not bold not italic underline grey84"
_theme["log.time"] = "green1"
_theme["log.message"] = "bold grey78"
_theme["logging.level.debug"] = "blue_violet"
_theme["logging.level.error"] = "blink bold bright_red"
_theme["logging.level.info"] = "medium_spring_green"
_theme["logging.level.warning"] = "blink medium_orchid"


mytheme = Theme(_theme)


console = Console(theme=mytheme)

ronswanson_console_log_handler = RichHandler(
    level=ronswanson_config.logging.level,
    rich_tracebacks=True,
    markup=True,
    console=console,
)
ronswanson_console_log_handler.setFormatter(_console_formatter)

warning_filter = LogFilter(logging.WARNING)


def silence_warnings():
    """
    supress warning messages in console and file usr logs
    """

    ronswanson_console_log_handler.addFilter(warning_filter)


def activate_warnings():
    """
    supress warning messages in console and file usr logs
    """

    ronswanson_console_log_handler.removeFilter(warning_filter)


def update_logging_level(level):

    ronswanson_console_log_handler.setLevel(level)


def setup_logger(name):

    # A logger with name name will be created
    # and then add it to the print stream
    log = logging.getLogger(name)

    # this must be set to allow debug messages through
    log.setLevel(logging.DEBUG)

    if ronswanson_config.logging.on:

        # add the handlers
        log.addHandler(ronswanson_console_log_handler)

    # we do not want to duplicate teh messages in the parents
    log.propagate = False

    return log
