from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from functools import lru_cache
from typing import Iterator

from nsdl.profiles.base import BenchmarkProfile, RuntimeMappingProxy


DEFAULT_BENCHMARK_PROFILE = "viki_l2"
_REGISTERED_PROFILES: dict[str, BenchmarkProfile] = {}
_LAZY_PROFILE_MODULES = {
    "mat_thor": "nsdl.profiles.mat_thor",
}


@lru_cache(maxsize=1)
def _profile_registry() -> dict[str, BenchmarkProfile]:
    from nsdl.profiles.viki_l2 import VIKI_L2_PROFILE

    registry = {
        DEFAULT_BENCHMARK_PROFILE: VIKI_L2_PROFILE,
    }
    registry.update(_REGISTERED_PROFILES)
    return registry


BENCHMARK_PROFILES = RuntimeMappingProxy(_profile_registry)

_ACTIVE_PROFILE: ContextVar[BenchmarkProfile | None] = ContextVar(
    "nsdl_active_profile",
    default=None,
)


def get_benchmark_profile(name: str) -> BenchmarkProfile:
    registry = _profile_registry()
    if name not in registry and name in _LAZY_PROFILE_MODULES:
        from importlib import import_module

        import_module(_LAZY_PROFILE_MODULES[name])
        registry = _profile_registry()
    try:
        return registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(set(registry) | set(_LAZY_PROFILE_MODULES)))
        raise KeyError(f"Unknown benchmark profile {name!r}. Available: {available}") from exc


def get_default_profile() -> BenchmarkProfile:
    return get_benchmark_profile(DEFAULT_BENCHMARK_PROFILE)


def get_active_profile() -> BenchmarkProfile:
    return _ACTIVE_PROFILE.get() or get_default_profile()


def register_benchmark_profile(name: str, profile: BenchmarkProfile) -> None:
    _REGISTERED_PROFILES[name] = profile
    _profile_registry.cache_clear()


def set_active_profile(profile: BenchmarkProfile) -> Token[BenchmarkProfile | None]:
    return _ACTIVE_PROFILE.set(profile)


def reset_active_profile(token: Token[BenchmarkProfile | None]) -> None:
    _ACTIVE_PROFILE.reset(token)


@contextmanager
def use_benchmark_profile(profile: BenchmarkProfile) -> Iterator[BenchmarkProfile]:
    token = set_active_profile(profile)
    try:
        yield profile
    finally:
        reset_active_profile(token)


def __getattr__(name: str):
    if name == "VIKI_L2_PROFILE":
        return get_benchmark_profile(DEFAULT_BENCHMARK_PROFILE)
    if name == "MAT_THOR_PROFILE":
        return get_benchmark_profile("mat_thor")
    raise AttributeError(name)


__all__ = [
    "BENCHMARK_PROFILES",
    "DEFAULT_BENCHMARK_PROFILE",
    "MAT_THOR_PROFILE",
    "VIKI_L2_PROFILE",
    "get_active_profile",
    "get_benchmark_profile",
    "get_default_profile",
    "register_benchmark_profile",
    "reset_active_profile",
    "set_active_profile",
    "use_benchmark_profile",
]
