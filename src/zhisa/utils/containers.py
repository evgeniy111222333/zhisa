"""Small typed container utilities used across packages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Config:
    """A plain, dict-like config container with attribute access.

    Used both as the result of `load_config` and as a stand-alone
    base for typed config sections (see `EnvConfig`, etc.).
    """

    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __iter__(self):
        return iter(self.data)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)
