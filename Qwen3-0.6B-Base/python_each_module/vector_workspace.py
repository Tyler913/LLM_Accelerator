from __future__ import annotations

import os
from pathlib import Path


SIM_VECTOR_DIR_ENV = "QMAP_SIM_VECTOR_DIR"


def resolve_sim_vector_dir(repo_root: Path) -> Path:
    override = os.environ.get(SIM_VECTOR_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return repo_root / "FPGA_Project" / "sim" / "vectors"
