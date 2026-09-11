"""Where the data lives.

The code in this package is self-contained; the data it reads is not. Every script resolves
cache/, results/, ablation/ and runs/ against one root, found here. Set SCOPE_ROOT to point at
a checkout explicitly; otherwise the nearest ancestor of this package holding a data directory
wins, and the working directory is the last resort. Nothing is hardcoded to one machine.
"""

from __future__ import annotations

import os
from pathlib import Path

_DATA_MARKERS = ("cache", "runs", "results", "ablation")


def find_project_root() -> Path:
    """Resolve the data root: SCOPE_ROOT, else the nearest ancestor holding a data directory."""
    override = os.environ.get("SCOPE_ROOT", "")
    if override:
        return Path(override).resolve()
    package_parent = Path(__file__).resolve().parent.parent
    for candidate in (package_parent, *package_parent.parents):
        if any((candidate / marker).is_dir() for marker in _DATA_MARKERS):
            return candidate
    return Path.cwd()


PROJECT_ROOT = find_project_root()
