"""peakpredict — athletics peak-performance prediction platform.

Three loosely-coupled components share one installable package so that the
contract modules in `common` (schemas, normalization, event maps, IO) have a
single implementation imported by every component — no drift between training
(pipeline) and inference (dashboard).
"""

__version__ = "0.1.0"
