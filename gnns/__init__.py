__version__ = "0.1.0"

try:
    from .model import HeteroGNN
except ImportError:
    HeteroGNN = None
    __all__ = []
else:
    __all__ = ["HeteroGNN"]
