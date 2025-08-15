#src/__init__.py
"""
src package initializer.

This file marks the 'src' directory as a Python package so that modules inside
can be imported using:
    from src.utils import get_env_var

It can also be used to define package-wide variables or imports if needed.
"""

__all__ = ["utils", "logger", "preprocessing", "inference_realtime", "inference_batch", "api"]
__version__ = "1.0.0"
