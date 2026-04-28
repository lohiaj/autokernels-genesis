"""
_config.py -- single-source-of-truth loader for harness.toml.

Every project-specific or platform-specific value is in harness.toml. This
module loads it once at import time and exposes typed accessor functions.

Design rules:
  - Backwards-compatible: if harness.toml is missing, every getter falls back
    to the Genesis-on-MI300X defaults that were hardcoded before this module
    existed. Code that doesn't load harness.toml still works.
  - No magic: every value the harness uses comes from one of the get_*()
    functions below. Grep for `cfg.get_` to see every consumer.
  - No mutation: load once, return cached dict. Callers must not mutate.

Usage:
    from _config import cfg
    h100_ref = cfg.get_reference_value()         # 794280.0
    nsteps   = cfg.get_default_e2e_nsteps()      # 500
    image    = cfg.get_container_image()         # "genesis:amd-integration"
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

# tomllib is stdlib on 3.11+. Fall back to `tomli` on 3.10.
# If neither is available, _load() silently falls back to _DEFAULTS, which means
# the harness still works unchanged for the Genesis-on-MI300X baseline use case.
try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

_HERE = Path(__file__).resolve().parent
_TOML_PATH = _HERE / "harness.toml"


# ---------------------------------------------------------------------------
# Defaults -- mirror the pre-config Genesis-on-MI300X hardcoded values exactly
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "reference": {
        "metric_name": "e2e_throughput",
        "metric_unit": "env*steps/s",
        "higher_is_better": True,
        "reference_label": "H100",
        "reference_value": 794280.0,
    },
    "bench": {
        "script": "/work/bench_mi300.py",
        "default_e2e_nsteps": 500,
        "default_prof_nsteps": 100,
        "default_n_envs": 8192,
        "default_precision": "32",
        "default_trials": 3,
        "metric_json_keys": ["throughput", "env_steps_per_sec"],
        "wall_json_keys": ["wall_time_s", "wall"],
        "timeouts": {
            "correctness_s": 600,
            "untraced_s": 600,
            "traced_s": 900,
        },
    },
    "container": {
        "name_template": "ak-gpu{gpu_id}",
        "image": "genesis:amd-integration",
        "runs_dir": "/work/runs",
        "caches_to_wipe": [
            "/root/.cache/quadrants",
            "/root/.cache/mesa_shader_cache",
        ],
        "env": [
            "GS_FAST_MATH=0",
            "PYOPENGL_PLATFORM=egl",
            "EGL_PLATFORM=surfaceless",
            "PYGLET_HEADLESS=true",
        ],
    },
    "profiler": {
        "command": "rocprofv3 --stats --kernel-trace -f csv -d . -o traced --",
        "stats_csv_suffix": "_kernel_stats.csv",
        "exclude_kernel_re": "runtime_initialize_rand_states",
    },
    "host": {
        "required_dirs": ["~/work/Genesis", "~/work/quadrants", "~/work/newton-assets"],
        "project_src": "~/work/Genesis",
        "gpu_arch": "gfx942",
    },
    "correctness": {
        "pytest_dir": "/src/Genesis",
        "pytest_extra_env": "GS_FAST_MATH=0",
        "pytest_args": "-v -n 0 --forked -m required",
    },
    "gpu": {
        "read_perf_command": "rocm-smi -d {gpu_id} --showperflevel",
        "pin_command": "rocm-smi -d {gpu_id} --setperflevel high",
        "restore_command": "rocm-smi -d {gpu_id} --setperflevel {prev}",
        "prev_level_regex": r"Performance Level[^:]*:\s*(\w+)",
    },
}


_warned_no_toml = False


@lru_cache(maxsize=1)
def _load() -> dict:
    """Load harness.toml once. Falls back to _DEFAULTS if file missing OR
    if no TOML parser is installed (Python 3.10 without `tomli`)."""
    global _warned_no_toml
    if not _TOML_PATH.exists():
        return _DEFAULTS
    if tomllib is None:
        if not _warned_no_toml:
            print(f"_config: WARN: harness.toml present but no TOML parser available "
                  f"(install `tomli` on Python 3.10, or use Python 3.11+); "
                  f"falling back to built-in defaults", file=sys.stderr)
            _warned_no_toml = True
        return _DEFAULTS
    try:
        with open(_TOML_PATH, "rb") as f:
            loaded = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"_config: WARN: failed to read {_TOML_PATH}: {e}; using defaults",
              file=sys.stderr)
        return _DEFAULTS
    return _deep_merge(_DEFAULTS, loaded)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge: override wins on leaves, defaults fill gaps."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Typed accessors
# ---------------------------------------------------------------------------

class _Cfg:
    """Thin wrapper exposing typed getters. Use the singleton `cfg` below."""

    # Reference / metric ----------------------------------------------------
    @staticmethod
    def get_metric_name() -> str:        return _load()["reference"]["metric_name"]
    @staticmethod
    def get_metric_unit() -> str:        return _load()["reference"]["metric_unit"]
    @staticmethod
    def higher_is_better() -> bool:      return bool(_load()["reference"]["higher_is_better"])
    @staticmethod
    def get_reference_label() -> str:    return _load()["reference"]["reference_label"]
    @staticmethod
    def get_reference_value() -> float:  return float(_load()["reference"]["reference_value"])

    # Bench -----------------------------------------------------------------
    @staticmethod
    def get_bench_script() -> str:       return _load()["bench"]["script"]
    @staticmethod
    def get_default_e2e_nsteps() -> int: return int(_load()["bench"]["default_e2e_nsteps"])
    @staticmethod
    def get_default_prof_nsteps() -> int:return int(_load()["bench"]["default_prof_nsteps"])
    @staticmethod
    def get_default_n_envs() -> int:     return int(_load()["bench"]["default_n_envs"])
    @staticmethod
    def get_default_precision() -> str:  return str(_load()["bench"]["default_precision"])
    @staticmethod
    def get_default_trials() -> int:     return int(_load()["bench"]["default_trials"])
    @staticmethod
    def get_metric_json_keys() -> list[str]: return list(_load()["bench"]["metric_json_keys"])
    @staticmethod
    def get_wall_json_keys() -> list[str]:   return list(_load()["bench"]["wall_json_keys"])
    @staticmethod
    def get_correctness_timeout_s() -> int: return int(_load()["bench"]["timeouts"]["correctness_s"])
    @staticmethod
    def get_untraced_timeout_s() -> int:    return int(_load()["bench"]["timeouts"]["untraced_s"])
    @staticmethod
    def get_traced_timeout_s() -> int:      return int(_load()["bench"]["timeouts"]["traced_s"])

    # Container -------------------------------------------------------------
    @staticmethod
    def get_container_name(gpu_id: str | int) -> str:
        return _load()["container"]["name_template"].format(gpu_id=gpu_id)
    @staticmethod
    def get_container_image() -> str:    return _load()["container"]["image"]
    @staticmethod
    def get_runs_dir() -> str:           return _load()["container"]["runs_dir"]
    @staticmethod
    def get_caches_to_wipe() -> list[str]: return list(_load()["container"]["caches_to_wipe"])
    @staticmethod
    def get_container_env() -> list[str]: return list(_load()["container"]["env"])
    @staticmethod
    def get_container_env_str() -> str:
        """Joined as 'A=1 B=2 C=3' for shell prefix use."""
        return " ".join(_load()["container"]["env"])

    # Profiler --------------------------------------------------------------
    @staticmethod
    def get_profiler_command() -> str:   return _load()["profiler"]["command"]
    @staticmethod
    def get_stats_csv_suffix() -> str:   return _load()["profiler"]["stats_csv_suffix"]
    @staticmethod
    def get_exclude_kernel_re() -> str:  return _load()["profiler"]["exclude_kernel_re"]

    # Host ------------------------------------------------------------------
    @staticmethod
    def get_required_dirs() -> list[str]:
        return [os.path.expanduser(p) for p in _load()["host"]["required_dirs"]]
    @staticmethod
    def get_project_src_default() -> str:
        return os.path.expanduser(_load()["host"]["project_src"])
    @staticmethod
    def get_gpu_arch() -> str:           return _load()["host"]["gpu_arch"]

    # Correctness -----------------------------------------------------------
    @staticmethod
    def get_pytest_dir() -> str:         return _load()["correctness"]["pytest_dir"]
    @staticmethod
    def get_pytest_extra_env() -> str:   return _load()["correctness"]["pytest_extra_env"]
    @staticmethod
    def get_pytest_args() -> str:        return _load()["correctness"]["pytest_args"]

    # GPU clock pinning -----------------------------------------------------
    @staticmethod
    def get_gpu_read_perf_command(gpu_id: str | int) -> str:
        return _load()["gpu"]["read_perf_command"].format(gpu_id=gpu_id)
    @staticmethod
    def get_gpu_pin_command(gpu_id: str | int) -> str:
        return _load()["gpu"]["pin_command"].format(gpu_id=gpu_id)
    @staticmethod
    def get_gpu_restore_command(gpu_id: str | int, prev: str) -> str:
        return _load()["gpu"]["restore_command"].format(gpu_id=gpu_id, prev=prev)
    @staticmethod
    def get_prev_level_regex() -> str:   return _load()["gpu"]["prev_level_regex"]


cfg = _Cfg()
