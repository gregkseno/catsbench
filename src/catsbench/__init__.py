from importlib.metadata import version, PackageNotFoundError

from .benchmarks.hd import BenchmarkHD, BenchmarkHDConfig


try:
    __version__ = version("catsbench")
except PackageNotFoundError:
    __version__ = "0.0.0"
    