from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

from .params import SiC4H, Gate

ROOT = Path(__file__).resolve().parents[1]


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads 1.0e18 or 1e-4 as float (plain PyYAML reads them as strings)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."))


def load_config(path: str | Path = ROOT / "configs" / "baseline.yaml") -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=_Loader)


def material_from(cfg: dict) -> SiC4H:
    m = SiC4H()
    for k, v in (cfg.get("material") or {}).items():
        if hasattr(m, k):
            setattr(m, k, v)
    return m


def gate_from(cfg: dict) -> Gate:
    return Gate(**(cfg.get("gate") or {}))


def deep(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
