"""
Remote
======

Remote single image viewer module

LICENSE
-------

This module includes modified codes from:

- Erik Härkönen's 'PyViewer' library (licensed under CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/): https://github.com/harskish/pyviewer.git
"""

# ----------------------------------------------------------------------------
# Check dependent library

import importlib
import os

dependencies = ['fastapi', 'httpx', 'radpsd', 'uvicorn']
err_msg = ''

for module_name in dependencies:
    try:
        _ = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        err_msg += f' - {module_name}' + os.linesep

if len(err_msg) > 0:
    err_msg = f'Failed to import \'{__name__}\' because the following libraries are missing:' + os.linesep + err_msg
    raise ImportError(err_msg)

# ----------------------------------------------------------------------------
