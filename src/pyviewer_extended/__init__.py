"""
PyViewer Extended
=================

On the shoulders of giants, this package adds some additional features to the original PyViewer package.
"""

# ----------------------------------------------------------------------------
# Check Python version

import sys

if sys.version_info < (3, 11):
    raise ImportError('Python 3.11 or higher is required.')

# ----------------------------------------------------------------------------
# Check the version of this package

import importlib.metadata

try:
    __version__ = importlib.metadata.version('pyviewer_extended')
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"

# ----------------------------------------------------------------------------
# Import modules

from .multi_textures_viewer import MultiTexturesDockingViewer, dockable

__all__ = [
    '__version__',
    'MultiTexturesDockingViewer',
    'dockable',
]
