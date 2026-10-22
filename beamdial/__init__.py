"""BeamDial: pick Beam's reasoning level per request, pay for max thinking only when it helps."""
from .client import BeamDial, DialResult
from .router import LEVELS, Router

__all__ = ["BeamDial", "DialResult", "Router", "LEVELS"]
__version__ = "0.1.0"
