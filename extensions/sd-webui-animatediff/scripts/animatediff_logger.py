import copy
import logging
import sys

from modules import shared


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


# Create a new logger
logger_animatediff = logging.getLogger("AnimateDiff")
logger_animatediff.propagate = False

# Add handler if we don't have one.
if not logger_animatediff.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger_animatediff.addHandler(handler)

# Configure logger
loglevel_string = getattr(shared.cmd_opts, "animatediff_loglevel", "ERROR")
loglevel = getattr(logging, loglevel_string.upper(), None)
logger_animatediff.setLevel(loglevel)
