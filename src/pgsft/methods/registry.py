"""Fail-closed public objective registry."""

from __future__ import annotations

from collections.abc import Callable

from pgsft.methods.base import MethodObjective
from pgsft.methods.dft import DFTObjective
from pgsft.methods.sft import SFTObjective


METHOD_NAMES = ("sft", "dft", "sft_base_kl", "psft", "pg_sft")


def _deferred(module: str, symbol: str) -> MethodObjective:
    imported = __import__(module, fromlist=[symbol])
    return getattr(imported, symbol)()


_FACTORIES: dict[str, Callable[[], MethodObjective]] = {
    "sft": SFTObjective,
    "dft": DFTObjective,
    "sft_base_kl": lambda: _deferred(
        "pgsft.methods.sft_base_kl", "SFTBaseKLObjective"
    ),
    "psft": lambda: _deferred("pgsft.methods.psft", "PSFTObjective"),
    "pg_sft": lambda: _deferred("pgsft.methods.pg_sft", "PGSFTObjective"),
}


def get_method(name: str) -> MethodObjective:
    try:
        factory = _FACTORIES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown method {name!r}; expected one of {METHOD_NAMES}"
        ) from error
    return factory()
