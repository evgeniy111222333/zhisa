"""Feature registry: declarative feature definitions and grouping.

Each feature is described by a :class:`FeatureDefinition` that carries its
name, group, version, compute function, lookback requirement, and
dependencies.  The :class:`FeatureRegistry` stores these definitions
and supports lookup by name or group.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import pandas as pd


@dataclass
class FeatureDefinition:
    """A single feature (or feature-set) definition.

    Attributes:
        name: Unique feature name (e.g. ``"rsi_14"``).
        group: Logical group (e.g. ``"indicators"``).
        version: Monotonically increasing version number.
            Bump to force re-materialisation.
        compute_fn: A callable ``(df: DataFrame) -> Series | DataFrame``.
            *df* will always have at least the ``dependencies`` columns
            and at least ``lookback`` rows of history preceding the
            requested range.
        lookback: Minimum number of past rows the function needs to
            produce correct output (e.g. ``14`` for RSI-14).
        dependencies: Column names from the raw OHLCV data that the
            function requires.
        description: Human-readable docstring.
        output_columns: If the compute function returns a DataFrame
            (multiple columns), list the column names here.
    """

    name: str
    group: str
    version: int = 1
    compute_fn: Callable[[pd.DataFrame], Union[pd.Series, pd.DataFrame]] = field(repr=False, default=lambda df: pd.Series(dtype=float))
    lookback: int = 0
    dependencies: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    description: str = ""
    output_columns: Optional[List[str]] = None

    @property
    def is_multi_output(self) -> bool:
        return self.output_columns is not None and len(self.output_columns) > 1


class FeatureRegistryError(Exception):
    """Raised for registry-related errors."""


class FeatureRegistry:
    """An in-memory registry of :class:`FeatureDefinition` instances.

    Thread-safe for reads; not designed for concurrent writes.
    """

    def __init__(self) -> None:
        self._defs: Dict[str, FeatureDefinition] = {}
        self._groups: Dict[str, List[str]] = {}

    # ── Registration ─────────────────────────────────────────

    def register(self, defn: FeatureDefinition) -> None:
        """Register a feature definition (overwrites on name collision)."""
        self._defs[defn.name] = defn
        self._groups.setdefault(defn.group, [])
        if defn.name not in self._groups[defn.group]:
            self._groups[defn.group].append(defn.name)

    def register_many(self, defns: Sequence[FeatureDefinition]) -> None:
        """Register multiple feature definitions at once."""
        for d in defns:
            self.register(d)

    def unregister(self, name: str) -> None:
        """Remove a feature definition by name."""
        if name not in self._defs:
            raise FeatureRegistryError(f"Feature not registered: {name!r}")
        group = self._defs[name].group
        del self._defs[name]
        if group in self._groups and name in self._groups[group]:
            self._groups[group].remove(name)
            if not self._groups[group]:
                del self._groups[group]

    # ── Lookup ───────────────────────────────────────────────

    def get(self, name: str) -> FeatureDefinition:
        """Get a feature definition by name.

        Raises:
            FeatureRegistryError: If the feature is not registered.
        """
        if name not in self._defs:
            raise FeatureRegistryError(
                f"Feature not registered: {name!r}. "
                f"Available: {sorted(self._defs.keys())}"
            )
        return self._defs[name]

    def has(self, name: str) -> bool:
        return name in self._defs

    def list_features(self, group: Optional[str] = None) -> List[str]:
        """List registered feature names, optionally filtered by group."""
        if group is not None:
            return list(self._groups.get(group, []))
        return sorted(self._defs.keys())

    def list_groups(self) -> List[str]:
        """List all feature groups."""
        return sorted(self._groups.keys())

    def get_group(self, group: str) -> List[FeatureDefinition]:
        """Get all definitions in a group."""
        names = self._groups.get(group, [])
        return [self._defs[n] for n in names]

    def max_lookback(self, features: Optional[List[str]] = None) -> int:
        """Return the maximum lookback across the given features (or all)."""
        if features is None:
            defs = list(self._defs.values())
        else:
            defs = [self.get(n) for n in features]
        if not defs:
            return 0
        return max(d.lookback for d in defs)

    def versions_fingerprint(self, features: Optional[List[str]] = None) -> str:
        """Return a version fingerprint for cache invalidation.

        The fingerprint changes whenever any of the selected features'
        versions change.  The result is a short hex digest that is safe
        for use in filenames on all operating systems.
        """
        import hashlib
        if features is None:
            features = sorted(self._defs.keys())
        else:
            features = sorted(features)
        raw = "|".join(f"{n}:v{self._defs[n].version}" for n in features if n in self._defs)
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @property
    def size(self) -> int:
        return len(self._defs)

    def __len__(self) -> int:
        return self.size

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __repr__(self) -> str:
        return f"FeatureRegistry({self.size} features, groups={self.list_groups()})"
