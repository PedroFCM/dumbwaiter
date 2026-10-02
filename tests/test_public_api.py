"""Guards on the package's public surface.

`__all__` is hand-maintained, so it drifts silently: a name gets removed and the export
stays, or a new type is added and never exported. Both are caught here rather than by a
user's ImportError.
"""

from __future__ import annotations

import importlib

import pytest

import dumbwaiter


class TestExports:
    @pytest.mark.parametrize("name", dumbwaiter.__all__)
    def test_every_exported_name_exists(self, name: str):
        assert hasattr(dumbwaiter, name), f"__all__ lists {name!r}, which is not importable"

    def test_star_import_matches_all(self):
        # Exercises the real import path rather than reading __all__ back to itself.
        namespace: dict[str, object] = {}
        exec("from dumbwaiter import *", namespace)
        namespace.pop("__builtins__", None)  # injected by exec, not by the package
        assert set(namespace) == set(dumbwaiter.__all__)

    def test_all_is_sorted(self):
        # Keeps diffs on this list readable as it grows.
        assert list(dumbwaiter.__all__) == sorted(dumbwaiter.__all__)

    def test_all_has_no_duplicates(self):
        assert len(dumbwaiter.__all__) == len(set(dumbwaiter.__all__))

    @pytest.mark.parametrize("module_name", ["config", "errors", "pricing", "types"])
    def test_public_types_are_reexported_from_the_root(self, module_name: str):
        # Users should never need to import from a private submodule path.
        module = importlib.import_module(f"dumbwaiter.{module_name}")
        public = {
            name
            for name, obj in vars(module).items()
            if not name.startswith("_")
            and isinstance(obj, type)
            and obj.__module__ == module.__name__
        }
        missing = public - set(dumbwaiter.__all__)
        assert not missing, (
            f"defined in dumbwaiter.{module_name} but not exported: {sorted(missing)}"
        )


class TestVersion:
    def test_version_is_exposed(self):
        assert isinstance(dumbwaiter.__version__, str)

    def test_version_matches_installed_metadata(self):
        from importlib.metadata import version

        assert dumbwaiter.__version__ == version("dumbwaiter")
