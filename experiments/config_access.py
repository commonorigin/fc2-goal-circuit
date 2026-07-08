"""Strict config accessor — the single enforcement point for "zero hardcoded values".

(config-driven; config.yaml is the single
source of truth). Design principle 4 in CLAUDE.md: every parameter comes from
``configs/config.yaml``; no value lives in code as a silent fallback.

Policy: STRICT FAIL-FAST. ``require()`` raises ``ConfigKeyError`` when a key is
absent instead of substituting a code-level default. A run launched against an
incomplete config therefore fails loudly at construction time (the constructor
reads all of its parameters up front) rather than silently driving the
experiment with a hardcoded number.
"""
from __future__ import annotations

from typing import Any


class ConfigKeyError(KeyError):
    """A required ``configs/config.yaml`` key is missing."""


def require(cfg: dict[str, Any], dotted_key: str) -> Any:
    """Return the value at ``dotted_key`` (e.g. ``"world_model.training.gamma"``).

    Walks the nested-dict config along the dotted path. Raises
    :class:`ConfigKeyError` naming the exact path segment that is missing — there
    is NO fallback default, because config is the single source of truth.

    Component.

    Args:
        cfg: parsed ``configs/config.yaml`` (a nested ``dict``).
        dotted_key: dot-separated path into ``cfg``.

    Returns:
        The value stored at ``dotted_key``.

    Raises:
        ConfigKeyError: if any segment of the path is absent or a non-leaf
            segment is not a mapping.
    """
    node: Any = cfg
    parts = dotted_key.split(".")
    for i, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            so_far = ".".join(parts[: i + 1])
            raise ConfigKeyError(
                f"Required config key '{dotted_key}' is missing at '{so_far}'. "
                f"Add it to configs/config.yaml — strict fail-fast policy means "
                f"there is no code-level fallback"
            )
        node = node[part]
    return node


def require_all(cfg: dict[str, Any], dotted_keys: list[str]) -> dict[str, Any]:
    """Resolve a list of dotted keys at once; raise on the FIRST missing one.

    Convenience for constructors / ``main()`` entry points that want to validate
    every parameter they depend on at startup. Returns a flat dict keyed by the
    leaf name (last path segment). Component.

    Args:
        cfg: parsed config dict.
        dotted_keys: paths to resolve.

    Returns:
        ``{leaf_name: value}`` for each requested key.

    Raises:
        ConfigKeyError: on the first missing key.
    """
    return {key.split(".")[-1]: require(cfg, key) for key in dotted_keys}


def _smoke_test() -> None:
    """Self-test the strict accessor."""
    cfg = {"a": {"b": {"c": 7, "flag": False}}, "top": 1}
    assert require(cfg, "top") == 1
    assert require(cfg, "a.b.c") == 7
    assert require(cfg, "a.b.flag") is False  # falsy values resolve, not "missing"
    assert require_all(cfg, ["top", "a.b.c"]) == {"top": 1, "c": 7}
    for bad in ("missing", "a.x", "a.b.c.d", "top.x"):
        try:
            require(cfg, bad)
        except ConfigKeyError as exc:
            assert bad.split(".")[0] in str(exc) or "missing" in str(exc).lower()
        else:  # pragma: no cover
            raise AssertionError(f"expected ConfigKeyError for {bad!r}")
    print("OK: config_access strict accessor smoke passed "
          "(require, require_all, falsy-resolves, 4 missing-key raises).")


if __name__ == "__main__":
    _smoke_test()
