# init for hera_stats
from . import automate, average, flag, noise, plotting, shuffle, split, \
              stats, testing, version
from . import jackknives, plots # Deprecating these
from .jkset import JKSet

from hera_pspec.container import PSpecContainer

__version__ = version.version
