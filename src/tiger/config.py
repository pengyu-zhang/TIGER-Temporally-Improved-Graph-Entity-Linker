"""YAML config loading with dotted-path CLI overrides."""

import copy

import yaml


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        for item in overrides:
            key, _, value = item.partition("=")
            if not _:
                raise ValueError(f"Override must look like key=value, got: {item!r}")
            _set_dotted(cfg, key.strip(), yaml.safe_load(value))
    return cfg


def _set_dotted(cfg: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def merge_config(base: dict, extra: dict) -> dict:
    """Recursively merge `extra` on top of `base` (returns a new dict)."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
