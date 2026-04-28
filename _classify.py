"""
_classify.py -- single source of truth for hypothesis classification.

Both `summarize.py` (per-campaign learning.md) and `global_log.py` (cross-agent
digest) need to bucket experiment descriptions into hypothesis classes
("block_dim", "fuse", "hoist", ...) so the agent can see per-class success rates
and dead-ends. Without classification, the digest tables are useless.

Patterns are loaded in this order, first hit wins:

  1. `kernels/<campaign>/classes.json` if it exists -- per-campaign override
  2. Built-in DEFAULT_PATTERNS below -- a generic GPU-kernel-tuning vocabulary

A team retargeting at a non-Genesis project (e.g. Composable Kernel, AITER)
should write their own `classes.json` per campaign with project-specific terms
(tile_size, mfma_layout, async_copy, etc.). The default patterns mostly cover
generic kernel-tuning vocabulary (block_dim, fuse, hoist, layout, async,
prefetch, atomic) and degrade gracefully -- unmatched descriptions land in
"other", which is correct (uninformative) rather than misleading.

Schema for `kernels/<campaign>/classes.json`:

    {
        "comment": "<free-text purpose>",
        "patterns": [
            ["<class_name>", "<regex>"],
            ...
        ]
    }

Patterns are case-insensitive. First matching pattern wins. Order matters --
put the more specific patterns first.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KERNELS = _HERE / "kernels"


# ---------------------------------------------------------------------------
# Default patterns (generic GPU-kernel-tuning vocabulary)
# ---------------------------------------------------------------------------
# These are the fallback when a campaign has no classes.json. They cover the
# classes we used in the Genesis-on-MI300X work; most of them generalize to
# any GPU-kernel project (block_dim, fuse, hoist, async, prefetch, atomic),
# a few are Genesis-DSL flavored (vec3 matches "vec3 layout"; bitfield, AABB).
# Override per-campaign if your project uses different terminology.

DEFAULT_PATTERNS: list[tuple[str, str]] = [
    ("block_dim",  r"\bblock[_-]?dim\b|\bbd\s*=\s*\d|\bblock\s*size\b"),
    ("fuse",       r"\bfus(e|ed|ing)\b|\bmerge\s+(loops?|kernels?)\b"),
    ("hoist",      r"\bhoist(ed|ing)?\b|\bcommon\s+sub|\binvariant\b"),
    ("inline",     r"\binlin(e|ed|ing)\b"),
    ("atomic",     r"\batomic"),
    ("swap",       r"\bswap\b|\breorder\b"),
    ("simplify",   r"\bsimplif|\bdelete|\bdedup|\bremove\s+(redundant|unused)|\bO\(1\)"),
    ("async",      r"\basync\b|\bglobal[_-]?load[_-]?lds\b|\boverlap\b"),
    ("prefetch",   r"\bprefetch"),
    ("layout",     r"\b(soa|aos)\b|\blayout\b|\bvec3\b|\bbitfield\b|\bpack\b"),
    ("scheduling", r"\boccupancy\b|\bwave(s|fronts?)?\b|\bxcd\b|\bpersistent\b|\bstream[_-]?k\b"),
    ("memory",     r"\blds\b|\bbank\s+conflict\b|\bvgpr\b|\bagpr\b|\bregister\b|\bspill\b"),
    ("algorithm",  r"\balgorithm\b|\bdecomp(ose|osition)?\b|\brefactor\b|\bstructur"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _load_patterns(campaign: str | None) -> tuple[tuple[str, re.Pattern], ...]:
    """Return compiled (name, regex) pairs for `campaign`. Falls back to
    DEFAULT_PATTERNS if no per-campaign classes.json exists.

    Cached per-campaign so repeated calls (every classify() invocation) don't
    re-read the JSON.
    """
    raw: list[tuple[str, str]] = []
    if campaign:
        path = _KERNELS / campaign / "classes.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                raw = [(str(name), str(pat)) for name, pat in data.get("patterns", [])]
            except (OSError, ValueError, TypeError):
                raw = []
    if not raw:
        raw = DEFAULT_PATTERNS
    return tuple((name, re.compile(pat, re.I)) for name, pat in raw)


def classify(description: str | None, campaign: str | None = None) -> str:
    """Return the hypothesis class name for `description`. First match wins.
    Returns 'unknown' for empty input, 'other' for no match."""
    if not description:
        return "unknown"
    for name, rx in _load_patterns(campaign):
        if rx.search(description):
            return name
    return "other"


def known_classes(campaign: str | None = None) -> list[str]:
    """All defined class names for the campaign, in priority order."""
    return [name for name, _ in _load_patterns(campaign)]
