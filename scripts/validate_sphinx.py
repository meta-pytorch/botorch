#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import os
import pkgutil
import re
from typing import Set


# Paths are relative to top-level botorch directory (passed as arg below)
SPHINX_RST_PATH = os.path.join("sphinx", "source")
BOTORCH_LIBRARY_PATH = "botorch"

# Regex for automodule directive used in Sphinx docs
AUTOMODULE_REGEX = re.compile(r"\.\. automodule:: ([\.\w]*)")

# The top-level modules in botorch not to be validated
EXCLUDED_MODULES = {"version"}


def parse_rst(rst_filename: str) -> Set[str]:
    """Extract automodule directives from rst."""
    ret = set()
    with open(rst_filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            name = AUTOMODULE_REGEX.findall(line)
            if name:
                ret.add(name[0])
    return ret


def validate_complete_sphinx(path_to_botorch: str) -> None:
    """Validate that Sphinx-based API documentation is complete.

    - Every top-level module (e.g., acquisition, models, etc.) should have a
        corresponding .rst sphix source file in sphinx/source.
    - Every single non-package (i.e. py file) module should be included in an
        .rst file ``automodule::`` directive. Sphinx will then automatically
        include all members from the module in the documentation.

    Note: this function does not validate any documentation, only its presence.

    Args:
        path_to_botorch: the path to the top-level botorch directory (directory
            that includes botorch library, sphinx, website, etc.).
    """
    # Load top-level modules used in botorch (e.g., acquisition, models)
    # Exclude auxiliary packages
    modules = {
        modname
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=[BOTORCH_LIBRARY_PATH], onerror=lambda x: None
        )
        if modname not in EXCLUDED_MODULES
    }

    # Load all rst files (these contain the documentation for Sphinx)
    rstpath = os.path.join(path_to_botorch, SPHINX_RST_PATH)
    rsts = {f.replace(".rst", "") for f in os.listdir(rstpath) if f.endswith(".rst")}

    # Verify that all top-level modules have a corresponding rst
    missing_rsts = modules.difference(rsts)
    if not len(missing_rsts) == 0:
        raise RuntimeError(
            f"""Not all modules have corresponding rst:
            {missing_rsts}
            Please add them to the appropriate rst file in {SPHINX_RST_PATH}.
            """
        )

    # Track all modules that are not in docs (so can print all)
    modules_not_in_docs = []

    # Iterate over top-level modules
    for module in modules.intersection(rsts):
        # Parse rst & extract all modules use automodule directive
        modules_in_rst = parse_rst(os.path.join(rstpath, module + ".rst"))

        # Extract all non-package modules
        for _importer, modname, ispkg in pkgutil.walk_packages(
            path=[
                os.path.join(BOTORCH_LIBRARY_PATH, module)
            ],  # botorch.__path__[0], module),
            prefix="botorch." + module + ".",
            onerror=lambda x: None,
        ):
            if not ispkg and ".tests" not in modname and modname not in modules_in_rst:
                modules_not_in_docs.append(modname)

    if not len(modules_not_in_docs) == 0:
        raise RuntimeError(f"Not all modules are documented: {modules_not_in_docs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate that Sphinx documentation is complete."
    )
    parser.add_argument(
        "-p",
        "--path",
        metavar="path",
        required=True,
        help="Path to the top-level botorch directory.",
    )
    args = parser.parse_args()
    validate_complete_sphinx(args.path)
