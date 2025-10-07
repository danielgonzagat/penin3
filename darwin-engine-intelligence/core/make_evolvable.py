"""
Simple decorator to mark classes as evolvable without changing behavior.
Other systems can detect `__evolvable__` metadata and attach hooks.
"""
from typing import TypeVar

_T = TypeVar("_T", bound=type)


def make_evolvable(cls: _T) -> _T:
    """Decorator that marks a class as evolvable.

    It adds a `__evolvable__ = True` attribute and returns the class unchanged.
    """
    try:
        setattr(cls, "__evolvable__", True)
    except Exception:
        # Best-effort; do not fail decoration
        pass
    return cls
