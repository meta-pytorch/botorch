# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import mdformat  # noqa: F401
    import nbformat
except ImportError as e:  # pragma: no cover
    # The converter needs the `tutorials` extras, which the `test` extras used
    # by the main test workflow do not include.
    raise unittest.SkipTest(f"Notebook converter dependencies missing: {e}")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/convert_ipynb_to_mdx.py"
SPEC = importlib.util.spec_from_file_location("notebook_converter_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class TestNotebookLinks(unittest.TestCase):
    def test_core_and_community_links_preserve_exact_tag(self) -> None:
        metadata = {"id": "example_notebook", "title": "Example"}
        with patch.object(converter, "get_current_git_tag", return_value="v0.18.1"):
            for community, folder in (
                (False, "tutorials"),
                (True, "notebooks_community"),
            ):
                with self.subTest(community=community):
                    buttons = converter.create_buttons(metadata, is_community=community)
                    path = (
                        f"meta-pytorch/botorch/blob/v0.18.1/{folder}/"
                        "example_notebook/example_notebook.ipynb"
                    )
                    self.assertIn(f'githubUrl="https://github.com/{path}"', buttons)
                    self.assertIn(
                        f'colabUrl="https://colab.research.google.com/github/{path}"',
                        buttons,
                    )

    def test_untagged_branch_links_to_main(self) -> None:
        error = subprocess.CalledProcessError(128, ["git", "describe"])
        with patch.object(converter.subprocess, "check_output", side_effect=error):
            buttons = converter.create_buttons({"id": "example"}, is_community=True)
        self.assertIn("/blob/main/notebooks_community/example/example.ipynb", buttons)
        self.assertNotIn("/tutorials/", buttons)

    def test_tag_output_and_path_component_encoding(self) -> None:
        with patch.object(
            converter.subprocess, "check_output", return_value=b"release/0.18\n"
        ):
            self.assertEqual(converter.get_current_git_tag(), "release/0.18")
            buttons = converter.create_buttons({"id": "example"})
        self.assertIn("/blob/release%2F0.18/tutorials/example/example.ipynb", buttons)

    def test_transform_passes_category_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "example.ipynb"
            nbformat.write(
                nbformat.v4.new_notebook(
                    cells=[nbformat.v4.new_markdown_cell("# A minimal notebook")]
                ),
                path,
            )
            with (
                patch.object(converter, "NOTEBOOKS_COMMUNITY_DIR", root / "community"),
                patch.object(converter, "TUTORIALS_DIR", root / "tutorials"),
                patch.object(converter, "get_current_git_tag", return_value=None),
            ):
                for community, destination, category in (
                    (False, "tutorials", "tutorials"),
                    (True, "community", "notebooks_community"),
                ):
                    with self.subTest(community=community):
                        mdx = converter.transform_notebook(
                            path,
                            {"id": "example", "title": "Passed metadata title"},
                            is_community=community,
                        )
                        self.assertIn("title: Passed metadata title", mdx)
                        self.assertIn(
                            f"/blob/main/{category}/example/example.ipynb", mdx
                        )
                        self.assertTrue(
                            (root / destination / "example/index.mdx").is_file()
                        )
