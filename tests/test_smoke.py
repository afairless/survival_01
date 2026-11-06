"""Smoke tests: the dev tooling, markers and package imports work."""

import importlib
import shutil


def test_src_package_is_importable() -> None:
    importlib.import_module("src.dp_weibull")


def test_dev_tools_are_installed() -> None:
    for tool in ("pytest", "ruff", "pyright"):
        assert shutil.which(tool), f"{tool} is not installed in the dev environment"
