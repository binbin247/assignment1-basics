import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
    print(__version__)
except importlib.metadata.PackageNotFoundError:
    pass
