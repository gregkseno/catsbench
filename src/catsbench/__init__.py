from importlib.metadata import version, PackageNotFoundError

from .benchmarks.hd import BenchmarkHD, BenchmarkHDConfig
from .benchmarks.image import BenchmarkImage, BenchmarkImageConfig


try:
    __version__ = version("catsbench")
except PackageNotFoundError:
    __version__ = "0.0.0"
    