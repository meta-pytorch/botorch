# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Union

import mdformat
import nbformat
import pandas as pd
from nbformat.notebooknode import NotebookNode

SCRIPTS_DIR = Path(__file__).parent.resolve()
LIB_DIR = SCRIPTS_DIR.parent.resolve()

WEBSITE_DIR = LIB_DIR.joinpath("website")
DOCS_DIR = LIB_DIR.joinpath("docs")
TUTORIALS_DIR = DOCS_DIR.joinpath("tutorials")
NOTEBOOKS_COMMUNITY_DIR = DOCS_DIR.joinpath("notebooks_community")

ORGANIZATION = "pytorch"
PROJECT = "botorch"

# Data display priority. Below lists the priority for displaying data from cell outputs.
# Cells can output many different items, and some will output a fallback display, e.g.
# text/plain if text/html is not working. The below priorities help ensure the output in
# the MDX file shows the best representation of the cell output.
priorities = [
    "error",
    "text/markdown",
    "image/png",  # matplotlib output.
    "application/vnd.jupyter.widget-view+json",  # tqdm progress bars.
    "application/vnd.plotly.v1+json",  # Plotly
    "text/html",
    "stream",
    "text/plain",
]


def load_tutorial_metadata() -> list[dict[str, str]]:
    """
    Load the metadata and list of notebooks that are to be converted to MDX.

    Args:
        None

    Returns:
        list[dict[str, str]]: A list of metadata needed to convert notebooks
            to MDX. Only those notebooks that are listed in the `tutorials.json` file
            will be included in the Docusaurus MDX output.
    """
    tutorials_json_path = WEBSITE_DIR.joinpath("tutorials.json")
    with tutorials_json_path.open("r") as f:
        tutorials_data = json.load(f)
    # flatten tutorial config, we handle the nested structure in sidebars.js
    tutorial_configs = [
        config for category in tutorials_data.values() for config in category
    ]
    return tutorial_configs


def load_notebooks_community_metadata() -> list[dict[str, str]]:
    """
    Load the metadata and list of community notebooks that are to be converted to MDX.

    Args:
        None

    Returns:
        list[dict[str, str]]: A list of metadata needed to convert notebooks
            to MDX. Only those notebooks that are listed in the
            `notebooks_community.json` file will be included in the
            Docusaurus MDX output.
    """
    json_path = WEBSITE_DIR.joinpath("notebooks_community.json")
    with json_path.open("r") as f:
        configs = json.load(f)
    return configs


def load_notebook(path: Path) -> NotebookNode:
    """
    Load the given notebook into memory.

    Args:
        path (Path): Path to the Jupyter notebook.

    Returns:
        NotebookNode: `nbformat` object, which contains all the notebook cells in it.
    """
    with path.open("r") as f:
        nb_str = f.read()
        nb = nbformat.reads(nb_str, nbformat.NO_CONVERT)
    return nb


def create_folders(path: Path, is_community: bool) -> tuple[str, Path]:
    """
    Create asset folders for the notebook.

    Args:
        path: Path to the Jupyter notebook.
        is_community: Whether the notebook is a community notebook (False for tutorial).

    Returns:
        Tuple[str, Path]: Returns a tuple with the filename to use for the MDX file
            and the path for the MDX assets folder.
    """
    folder_name = path.stem
    filename = "".join([token.title() for token in folder_name.split("_")])
    base_folder_dir = NOTEBOOKS_COMMUNITY_DIR if is_community else TUTORIALS_DIR
    folder = base_folder_dir.joinpath(folder_name)
    assets_folder = folder / "assets"
    img_folder = assets_folder / "img"
    plot_data_folder = assets_folder / "plot_data"
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    if not img_folder.exists():
        img_folder.mkdir(parents=True, exist_ok=True)
    if not plot_data_folder.exists():
        plot_data_folder.mkdir(parents=True, exist_ok=True)
    return filename, assets_folder


def create_frontmatter(path: Path, nb_metadata: dict[str, dict[str, str]]) -> str:
    """
    Create frontmatter for the resulting MDX file.

    The frontmatter is the data between the `---` lines in an MDX file.

    Args:
        path (Path): Path to the Jupyter notebook.
        nb_metadata (Dict[str, Dict[str, str]]): The metadata associated with the given
            notebook. Metadata is defined in the `tutorials.json` file.

    Returns:
        str: MDX formatted frontmatter.
    """
    # Add the frontmatter to the MDX string. This is the part between the `---` lines
    # that define the tutorial sidebar_label information.
    frontmatter_delimiter = ["---"]
    frontmatter = [
        f'"{key}": "{value}"'
        for key, value in {
            "title": metadata["title"],
            "sidebar_label": metadata["title"],
        }.items()
    ]
    frontmatter = "\n".join(frontmatter_delimiter + frontmatter + frontmatter_delimiter)
    mdx = mdformat.text(frontmatter, options={"wrap": 88}, extensions={"myst"})
    return f"{mdx}\n"


def create_imports() -> str:
    """
    Create the imports needed for displaying buttons, and interactive plots in MDX.

    Returns:
        str: MDX formatted imports.
    """
    link_btn = "@site/src/components/LinkButtons.jsx"
    cell_out = "@site/src/components/CellOutput.jsx"
    plot_out = "@site/src/components/Plotting.jsx"
    imports = f'import LinkButtons from "{link_btn}";\n'
    imports += f'import CellOutput from "{cell_out}";\n'
    imports += f'import {{PlotlyFigure}} from "{plot_out}";\n'
    return f"{imports}\n"


def get_current_git_tag() -> str | None:
    """
    Retrieve the current Git tag if the current commit is tagged.

    Returns:
        Optional[str]: The current Git tag as a string if available, otherwise None.
    """
    try:
        tag = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--exact-match"], stderr=subprocess.STDOUT
            )
            .strip()
            .decode("utf-8")
        )
        return tag
    except subprocess.CalledProcessError:
        return None


def create_buttons(
    nb_metadata: dict[str, dict[str, str]],
) -> str:
    """
    Create buttons that link to Colab and GitHub for the tutorial.

    Args:
        nb_metadata (Dict[str, Dict[str, str]]): Metadata for the tutorial.

    Returns:
        str: MDX formatted buttons.
    """
    version = get_current_git_tag() or "main"
    github_path = (
        f"{ORGANIZATION}/{PROJECT}/blob/{version}/tutorials/"
        f"{nb_metadata['id']}/{nb_metadata['id']}.ipynb"
    )
    github_url = f"https://github.com/{github_path}"
    colab_url = f"https://colab.research.google.com/github/{github_path}"
    return f'<LinkButtons\n  githubUrl="{github_url}"\n  colabUrl="{colab_url}"\n/>\n\n'


def handle_image_attachments(
    markdown: str,
    attachments: dict[str, dict[str, str]],
) -> str:
    """
    Image attachments are stored in the notebook cell's "attachments" field in base64
    format with their associated mime_type and referenced in the markdown via
    attachment name.

    The pattern we search for in the Markdown is
    `![alt_text](attachment:attachment_name title)` with three groups:

    - group 1 = alt_text (optional)
    - group 2 = attachment_name
    - group 3 = title (optional)

    To represent this in MD we replace the attachment reference with the base64 encoded
    string as `![{alt_text}](data:{mime_type};base64,{img_as_base64})`

    Args:
        markdown (str): The markdown content containing image attachments.
        attachments (Dict[str, Dict[str, str]]): A dictionary of attachments with their
            corresponding MIME types and base64 encoded data.

    Returns:
        str: The markdown content with images converted to base64 format.
    """
    markdown_image_pattern = re.compile(
        r"""!\[([^\]]*)\]\(attachment:(.*?)\s*?(\".*\")?\)"""
    )
    # go through searches in reverse order so that each replacement doesn't affect the
    # start/end indices for the next replacements
    searches = reversed(list(re.finditer(markdown_image_pattern, markdown)))
    for search in searches:
        alt_text, attachment_name, _ = search.groups()
        mime_type, base64 = next(iter(attachments[attachment_name].items()))
        start, end = search.span()
        markdown = (
            markdown[:start]
            + generate_img_base64_md(base64, mime_type, alt_text)
            + markdown[end:]
        )
    return markdown


def handle_image_paths_found_in_markdown(
    markdown: str,
    new_img_dir: Path,
    nb_path: Path,
) -> str:
    """
    Update image paths in the Markdown, and copy the image to the docs location.

    The pattern we search for in the Markdown is
    ``![alt-text](path/to/image.png "title")`` with two groups:

    - group 1 = path/to/image.png
    - group 2 = "title"

    We explicitly exclude matching if the path starts with `attachment:` as this
    indicates that the image is embedded as a base64 attachment not a file path.

    The first group (the path to the image from the original notebook) will be replaced
    with ``assets/img/{name}`` where the name is `image.png` from the example above. The
    original image will also be copied to the new location
    ``{new_img_dir}/assets/img/{name}``, which can be directly read into the MDX file.

    Args:
        markdown (str): Markdown where we look for Markdown flavored images.
        new_img_dir (Path): Path where images are copied to for display in the
            MDX file.
        lib_dir (Path): The location for the repo.
        nb_path (Path): The location for the notebook.

    Returns:
        str: The original Markdown with new paths for images.
    """
    markdown_image_pattern = re.compile(
        r"""!\[[^\]]*\]\((?!attachment:)(.*?)(?=\"|\))(\".*\")?\)"""
    )
    searches = list(re.finditer(markdown_image_pattern, markdown))

    # Return the given Markdown if no images are found.
    if not searches:
        return markdown

    # Process searches in reverse order so that each replacement doesn't affect the
    # start/end indices for the next replacements
    for search in reversed(searches):
        # Find the old image path and replace it with the new one.
        old_path, _ = search.groups()
        # Get the span of the old_path within the full match
        start, end = search.span(1)

        old_path = Path(old_path)
        name = old_path.name.strip()
        new_path = f"assets/img/{name}"

        # Replace the old path with the new path in the markdown
        markdown = markdown[:start] + new_path + markdown[end:]

        # Copy the original image to the new location.
        if old_path.exists():
            # resolves if an absolute path is used
            old_img_path = old_path
        else:
            # fall back to path relative to the notebook
            old_img_path = (nb_path.parent / old_path).resolve()
        new_img_path = str(new_img_dir / name)
        shutil.copy(str(old_img_path), new_img_path)

    return markdown


def transform_style_attributes(markdown: str) -> str:
    """
    Convert HTML style attributes to something React can consume.

    Args:
        markdown (str): Markdown where we look for HTML style attributes.

    Returns:
        str: The original Markdown with new React style attributes.
    """
    # Finds all instances of `style="attr: value; ..."`.
    token = "style="
    pattern = re.compile(f"""{token}["'`]([^"]*)["'`]""")
    found_patterns = re.findall(pattern, markdown)
    if not found_patterns:
        return markdown

    for found_pattern in found_patterns:
        # Step 1: splits "attr: value; ..." to
        #                ["attr: value", ..."].
        step1 = [token.strip() for token in found_pattern.split(";") if token]

        # Step 2: splits ["attr: value", ...] to
        #                [["attr", "value"], ...].
        step2 = [[token.strip() for token in tokens.split(":")] for tokens in step1]

        # Step 3: converts [["attr", "value"], ...] to
        #                  '{"attr": "value", ...}'.
        step3 = json.dumps(dict(step2))

        # Step 4 wraps the JSON object in {}, so we end up with a string of the form;
        #        '{{"attr": "value", ...}}'.
        step4 = f"{{{step3}}}"

        # Step 5 replaces the old style data with the React style data, and clean the
        #        string for inclusion in the final Markdown.
        markdown = markdown.replace(found_pattern, step4)
        markdown = markdown.replace('"{{', "{{").replace('}}"', "}}")
    return markdown


def sanitize_mdx(mdx: str) -> str:
    """
    Sanitize the given MDX string.

    Args:
        mdx (str): MDX string to sanitize.

    Returns:
        str: Sanitized MDX string.
    """
    # Remove some lingering HTML tags that break MDX.
    mdx = mdx.replace("<markdowncell>", "")
    mdx = mdx.replace("<TODO>", "")
    mdx = mdx.replace("<br>", "<br />")
    # Remove any HTML comments from the Markdown. They are fine to keep in the
    # notebooks, but are not really useful in the MDX.
    mdx = re.sub("(<!--.*?-->)", "", mdx, flags=re.DOTALL)
    # "\" Escape braces to make the text MDX compatible.
    mdx = re.sub("([^\\\\])([{}])", "\\g<1>\\\\\\g<2>", mdx)

    # -- KaTeX --
    # Make sure $$ symbols are not escaped and include line breaks.
    mdx = re.sub(
        "\\\\?\\$\\\\?\\$((?:.|\n)*?)\\\\?\\$\\\\?\\$", "\n$$\n\\g<1>\n$$\n", mdx
    )
    # Escaping braces causes issues in math blocks, unescape them.
    mdx = re.sub(
        "\\$?\\$(.|\n)*?\\$\\$?",
        lambda match: match[0].replace("\\{", "{").replace("\\}", "}"),
        mdx,
    )

    return mdx


def get_source(cell: NotebookNode) -> str:
    """
    Extract the source code from a Jupyter notebook cell. The source is
    characteristically multi-line strings but may be stored as a list of strings, in
    which case we join them together.
    https://ipython.readthedocs.io/en/3.x/notebook/nbformat.html

    Args:
        cell (NotebookNode): A Jupyter notebook cell object.

    Returns:
        str: The source code as a string.
    """
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source


def handle_markdown_cell(
    cell: NotebookNode,
    new_img_dir: Path,
    nb_path: Path,
) -> str:
    """
    Handle the given Jupyter Markdown cell and convert it to MDX.

    Args:
        cell (NotebookNode): Jupyter Markdown cell object.
        new_img_dir (Path): Path where images are copied to for display in the
            Markdown cell.
        lib_dir (Path): The location for the library.
        nb_path (Path): The location for the notebook.

    Returns:
        str: Transformed Markdown object suitable for inclusion in MDX.
    """
    markdown = get_source(cell)

    # Handle the different ways images are included in the Markdown.
    markdown = handle_image_paths_found_in_markdown(markdown, new_img_dir, nb_path)
    markdown = handle_image_attachments(markdown, cell.get("attachments", {}))

    markdown = sanitize_mdx(markdown)
    mdx = mdformat.text(markdown, options={"wrap": 88}, extensions={"myst"})

    # We will attempt to handle inline style attributes written in HTML by converting
    # them to something React can consume. This has to be done after the
    # mdformat.text() step since mdformat complains about the converted styles.
    mdx = transform_style_attributes(mdx)

    return f"{mdx}\n"


def handle_cell_input(cell: NotebookNode, language: str) -> str:
    """
    Create a Markdown cell block using the given cell source, and the language.

    The given language will determine cell input syntax styles. Docusaurus uses Prism as
    the syntax highlighter, https://prismjs.com. See the Docusaurus documentation for
    more information on code blocks
    https://docusaurus.io/docs/markdown-features/code-blocks.

    Args:
        cell (NotebookNode): A notebook cell.
        language (str): Language specifier for syntax highlighting.

    Returns:
        str: Code block formatted Markdown string.
    """
    cell_source = get_source(cell)
    return f"```{language}\n{cell_source}\n```\n\n"


def generate_img_base64_md(
    img_as_base64: int | str | NotebookNode,
    mime_type: int | str | NotebookNode,
    alt_text: str = "",
) -> str:
    """
    Generate a markdown image tag from a base64 encoded image.

    Args:
        img_as_base64 (int | str | NotebookNode): The base64 encoded image data.
        mime_type (int | str | NotebookNode): The MIME type of the image.
        alt_text (str, optional): The alternative text for the image. Defaults to an
            empty string.

    Returns:
        str: A markdown formatted image tag.
    """
    return f"![{alt_text}](data:{mime_type};base64,{img_as_base64})"


def handle_image(
    values: list[dict[str, int | str | NotebookNode]],
) -> list[tuple[int, str]]:
    """
    Convert embedded images to string MDX can consume.

    Args:
        values (List[Dict[str, Union[int, str, NotebookNode]]]): Bokeh tagged cell
            outputs.

    Returns:
        List[Tuple[int, str]]: A list of tuples, where the first entry in the tuple is
            the index where the output occurred from the cell, and the second entry of
            the tuple is the MDX formatted string.
    """
    output = []
    for value in values:
        index = value["index"]
        mime_type = value["mime_type"]
        img = value["data"]
        output.append((index, f"{generate_img_base64_md(img, mime_type)}\n\n"))
    return output


def handle_markdown(
    values: list[dict[str, int | str | NotebookNode]],
) -> list[tuple[int, str]]:
    """
    Convert and format Markdown for MDX.

    Args:
        values (List[Dict[str, Union[int, str, NotebookNode]]]): Bokeh tagged cell
            outputs.

    Returns:
        List[Tuple[int, str]]: A list of tuples, where the first entry in the tuple is
            the index where the output occurred from the cell, and the second entry of
            the tuple is the MDX formatted string.
    """
    output = []
    for value in values:
        index = int(value["index"])
        markdown = str(value["data"])
        markdown = mdformat.text(markdown, options={"wrap": 88}, extensions={"myst"})
        output.append((index, f"{markdown}\n\n"))
    return output


def handle_pandas(
    values: list[dict[str, int | str | NotebookNode]],
) -> list[tuple[int, str]]:
    """
    Handle how to display pandas DataFrames.

    There is a scoped style tag in the DataFrame output that uses the class name
    `dataframe` to style the output. We will use this token to determine if a pandas
    DataFrame is being displayed.

    Args:
        values (List[Dict[str, Union[int, str, NotebookNode]]]): Bokeh tagged cell
            outputs.

    Returns:
        List[Tuple[int, str]]: A list of tuples, where the first entry in the tuple is
            the index where the output occurred from the cell, and the second entry of
            the tuple is the MDX formatted string.
    """
    output = []
    for value in values:
        index = int(value["index"])
        data = str(value["data"])
        if 'class="dataframe"' not in data:
            # pd.read_html() raises an error if there's no dataframe.
            continue
        df = pd.read_html(io.StringIO(data), flavor="lxml")
        if len(df) == 0:
            continue
        # NOTE: The return is a list of dataframes and we only care about the first
        #       one.
        md_df = df[0]
        for column in md_df.columns:
            if column.startswith("Unnamed"):
                md_df.rename(columns={column: ""}, inplace=True)
        # Remove the index if it is just a range, and output to markdown.
        mdx = ""
        if isinstance(md_df.index, pd.RangeIndex):
            mdx = md_df.to_markdown(index=False)
        elif not isinstance(md_df.index, pd.RangeIndex):
            mdx = md_df.to_markdown()
        output.append((index, f"\n{sanitize_mdx(mdx)}\n\n"))
    return output


def handle_plain(
    values: list[dict[str, int | str | NotebookNode]],
) -> list[tuple[int, str]]:
    """
    Handle how to plain cell output should be displayed in MDX.

    Stdout streams are chunked during execution, we merge adjacent streams here into
    single cell output blocks.

    Args:
        values (List[Dict[str, Union[int, str, NotebookNode]]]): Bokeh tagged cell
            outputs.

    Returns:
        List[Tuple[int, str]]: A list of tuples, where the first entry in the tuple is
            the index where the output occurred from the cell, and the second entry of
            the tuple is the MDX formatted string.
    """
    output = []
    adjacent_outputs = []
    previous_index = -1

    def append_to_output() -> None:
        if not adjacent_outputs:
            return
        adjacent_outputs_str = "\n".join(adjacent_outputs)
        output.append(
            (
                previous_index,
                "\n".join(
                    [
                        "<CellOutput>",
                        "{",
                        f"`{adjacent_outputs_str}`",
                        "}",
                        "</CellOutput>\n\n",
                    ]
                ),
            ),
        )

    for value in values:
        index = int(value["index"])
        data = str(value["data"])
        data = [line.strip() for line in data.splitlines() if line]
        data = [datum for datum in data if datum]
        if data:
            data = "\n".join([line for line in str(value["data"]).splitlines() if line])
            # Remove backticks to make the text MDX compatible.
            data = data.replace("`", "")
            if previous_index in [-1, index - 1]:
                # store in cache until we reach nonconsecutive index
                adjacent_outputs.append(data)
            else:
                # flush cache to output and start a new one
                append_to_output()
                adjacent_outputs = [data]
            previous_index = index
    # flush the remaining cache to output
    append_to_output()
    return output


def handle_plotly(
    values: list[dict[str, int | str | NotebookNode]],
    plot_data_folder: Path,
) -> list[tuple[int, str]]:
    """
    Convert Plotly outputs to MDX.

    Args:
        values (List[Dict[str, Union[int, str, NotebookNode]]]): Bokeh tagged cell
            outputs.
        plot_data_folder (Path): Path to the folder where plot data should be
            stored.

    Returns:
        List[Tuple[int, str]]: A list of tuples, where the first entry in the tuple is
            the index where the output occurred from the cell, and the second entry of
            the tuple is the MDX formatted string.
    """
    output = []
    for value in values:
        index = value["index"]
        data = value["data"]
        file_name = str(uuid.uuid4())
        file_path = plot_data_folder / f"{file_name}.json"
        path_to_data = f"./assets/plot_data/{file_name}.json"
        output.append(
            (index, f"<PlotlyFigure data={{require('{path_to_data}')}} />\n\n"),
        )
        with file_path.open("w") as f:
            json.dump(data, f, indent=2)
    return output


def handle_tqdm(
    values: list[dict[str, int | str | NotebookNode]],
) -> list[tuple[int, str]]:
    """
    Handle the output of tqdm.

    tqdm will be displayed as separate CellOutput React components if we do not
    aggregate them all into a single CellOutput object, which is what this method does.

    Args:
        values (List[Dict[str, Union[int, str, NotebookNode]]]): Bokeh tagged cell
            outputs.

    Returns:
        List[Tuple[int, str]]: A list of tuples, where the first entry in the tuple is
            the index where the output occurred from the cell, and the second entry of
            the tuple is the MDX formatted string.
    """
    output = sorted(values, key=lambda item: item["index"])
    index = int(output[0]["index"])
    md = "\n".join([str(item["data"]) for item in output if item["data"]])
    return [(index, f"<CellOutput>\n{{\n  `{md}`\n}}\n</CellOutput>\n\n")]


CELL_OUTPUTS_TO_PROCESS = dict[
    str,
    list[dict[str, Union[int, str, NotebookNode]]],
]


def aggregate_mdx(
    cell_outputs_to_process: CELL_OUTPUTS_TO_PROCESS,
    plot_data_folder: Path,
) -> str:
    """
    Aggregate the `cell_outputs_to_process` into MDX.

    Args:
        cell_outputs_to_process (CELL_OUTPUTS_TO_PROCESS): A dictionary of cell outputs
            that need further processing.
        plot_data_folder (Path): Path to where plot data should be stored for the
            tutorial.

    Returns:
        str: MDX formatted string.
    """
    processed_mdx = []
    for key, values in cell_outputs_to_process.items():
        if not values:
            continue
        if key == "image":
            processed_mdx.extend(handle_image(values))
        if key == "markdown":
            processed_mdx.extend(handle_markdown(values))
        if key == "pandas":
            processed_mdx.extend(handle_pandas(values))
        if key == "plain":
            processed_mdx.extend(handle_plain(values))
        if key == "plotly":
            processed_mdx.extend(handle_plotly(values, plot_data_folder))
        if key == "tqdm":
            processed_mdx.extend(handle_tqdm(values))

    # Ensure the same ordering of the MDX happens as was found in the original cell
    # output.
    processed_mdx = sorted(processed_mdx, key=lambda item: item[0])
    mdx = "\n".join([item[1] for item in processed_mdx])
    return mdx


def prioritize_dtypes(
    cell_outputs: list[NotebookNode],
) -> tuple[list[list[str]], list[bool]]:
    """
    Prioritize cell output data types.

    Args:
        cell_outputs (List[NotebookNode]): A list of cell outputs.

    Returns:
        Tuple[List[List[str]], List[bool]]: Return two items in the tuple; the first is
            a list of prioritized data types and the second is a list boolean values
            associated with the cell output having Plotly information in it or not.
    """
    cell_output_dtypes = [
        (
            list(cell_output["data"].keys())
            if "data" in cell_output
            else [cell_output["output_type"]]
        )
        for cell_output in cell_outputs
    ]
    prioritized_cell_output_dtypes = [
        sorted(
            set(dtypes).intersection(set(priorities)),
            key=lambda dtype: priorities.index(dtype),
        )
        for dtypes in cell_output_dtypes
    ]
    prioritized_cell_output_dtypes = [
        [str(item) for item in items] for items in prioritized_cell_output_dtypes
    ]
    plotly_flags = [
        any(["plotly" in output for output in outputs])
        for outputs in cell_output_dtypes
    ]
    return prioritized_cell_output_dtypes, plotly_flags


def aggregate_images_and_plotly(
    prioritized_data_dtype: str,
    cell_output: NotebookNode,
    data: NotebookNode,
    plotly_flags: list[bool],
    cell_outputs_to_process: CELL_OUTPUTS_TO_PROCESS,
    i: int,
) -> None:
    """
    Aggregates images or Plotly cell outputs into an appropriate bucket.

    Args:
        prioritized_data_dtype (str): The prioritized cell output data type.
        cell_output (NotebookNode): The actual cell output from the notebook.
        data (NotebookNode): The data of the cell output.
        plotly_flags (List[bool]): True if a Plotly plot was found in the cell outputs
            else False.
        cell_outputs_to_process (CELL_OUTPUTS_TO_PROCESS): Dictionary containing
            aggregated cell output objects.
        i (int): Index for the cell output in the list of cell output objects.

    Returns:
        None: Does not return anything, instead adds values to the
            cell_outputs_to_process if applicable.
    """
    if not plotly_flags[i]:
        cell_outputs_to_process["image"].append(
            {"index": i, "data": data, "mime_type": prioritized_data_dtype},
        )
    # Plotly outputs a static image, but we can use the JSON in the cell
    # output to create interactive plots using a React component.
    if plotly_flags[i]:
        data = cell_output["data"]["application/vnd.plotly.v1+json"]
        cell_outputs_to_process["plotly"].append({"index": i, "data": data})


def aggregate_plain_output(
    prioritized_data_dtype: str,
    cell_output: NotebookNode,
    data: NotebookNode,
    cell_outputs_to_process: CELL_OUTPUTS_TO_PROCESS,
    i: int,
) -> None:
    """
    Aggregate plain text cell outputs together.

    Args:
        prioritized_data_dtype (str): The prioritized cell output data type.
        cell_output (NotebookNode): The actual cell output from the notebook.
        data (NotebookNode): The data of the cell output.
        cell_outputs_to_process (CELL_OUTPUTS_TO_PROCESS): Dictionary containing
            aggregated cell output objects.
        i (int): Index for the cell output in the list of cell output objects.

    Returns:
        None: Does not return anything, instead adds values to the
            cell_outputs_to_process if applicable.
    """
    # Ignore error outputs.
    if "name" in cell_output and cell_output["name"] == "stderr":
        pass
    # Ignore matplotlib legend text output.
    if prioritized_data_dtype == "text/plain" and "matplotlib" in data:
        pass
    cell_outputs_to_process["plain"].append({"index": i, "data": data})


def aggregate_output_types(cell_outputs: list[NotebookNode]) -> CELL_OUTPUTS_TO_PROCESS:
    """
    Aggregate cell outputs into a dictionary for further processing.

    Args:
        cell_outputs (List[NotebookNode]): List of cell outputs.

    Returns:
        CELL_OUTPUTS_TO_PROCESS: Dictionary containing aggregated cell output objects.
    """
    # We will use the below cell output data types for prioritizing the output shown in
    # the MDX file.
    prioritized_cell_output_dtypes, plotly_flags = prioritize_dtypes(cell_outputs)

    cell_outputs_to_process = {
        "image": [],
        "markdown": [],
        "pandas": [],
        "plain": [],
        "plotly": [],
        "tqdm": [],
    }
    for i, cell_output in enumerate(cell_outputs):
        prioritized_data_dtype = prioritized_cell_output_dtypes[i][0]

        # If there is no `data` key in the cell_output, then it may be an error that
        # needs to be handled. Even if it is not an error, the data is stored in a
        # different key if no `data` key is found.
        data = (
            cell_output["data"][prioritized_data_dtype]
            if "data" in cell_output
            else cell_output["text"]
            if "text" in cell_output
            else cell_output["evalue"]
        )
        image_check = (
            prioritized_data_dtype.startswith("image")
            or "plotly" in prioritized_data_dtype
        )
        if image_check:
            aggregate_images_and_plotly(
                prioritized_data_dtype,
                cell_output,
                data,
                plotly_flags,
                cell_outputs_to_process,
                i,
            )
        plain_check = prioritized_data_dtype in ["text/plain", "stream", "error"]
        if plain_check:
            aggregate_plain_output(
                prioritized_data_dtype,
                cell_output,
                data,
                cell_outputs_to_process,
                i,
            )
        if prioritized_data_dtype == "text/markdown":
            cell_outputs_to_process["markdown"].append({"index": i, "data": data})
        if "dataframe" in data:
            cell_outputs_to_process["pandas"].append({"index": i, "data": data})
        if prioritized_data_dtype == "application/vnd.jupyter.widget-view+json":
            data = cell_output["data"]["text/plain"]
            cell_outputs_to_process["tqdm"].append({"index": i, "data": data})

    return cell_outputs_to_process


def handle_cell_outputs(cell: NotebookNode, plot_data_folder: Path) -> str:
    """
    Handle cell outputs and convert to MDX.

    Args:
        cell (NotebookNode): The cell where the outputs need converting.
        plot_data_folder (Path): Path to the folder where plot data should be
            stored.

    Returns:
        str: MDX formatted cell output.
    """
    mdx = ""

    # Return an empty string if there are no actual cell outputs.
    cell_outputs = cell.get("outputs", [])
    if not cell_outputs:
        return mdx

    # We will loop over all cell outputs and bucket them into the appropriate key in the
    # dictionary below for further processing. Doing it in this way helps aggregate like
    # outputs together e.g. tqdm outputs.
    cell_outputs_to_process = aggregate_output_types(cell_outputs)

    # Now we process all aggregated cell outputs into a single output for the type.
    md = aggregate_mdx(cell_outputs_to_process, plot_data_folder)
    return md


def handle_code_cell(cell: NotebookNode, plot_data_folder: Path) -> str:
    """
    Handle code cells in Jupyter notebooks and convert them to MDX.

    Args:
        cell (NotebookNode): A Jupyter notebook cell that contains code.
        plot_data_folder (Path): Path to the folder where plot data should be
            stored.

    Returns:
        str: MDX formatted code cell.
    """
    cell_input_mdx = handle_cell_input(cell, "python")
    cell_output_mdx = handle_cell_outputs(cell, plot_data_folder)
    return cell_input_mdx + cell_output_mdx


def transform_notebook(path: Path, nb_metadata: object, is_community: bool) -> str:
    """
    Transform a notebook located at the given path into MDX.

    Args:
        path: Path to the Jupyter notebook tutorial.
        nb_metadata: Metadata for the notebook to be converted.
        is_community: Whether the notebook is a community notebook (False for tutorial).

    Returns:
        str: MDX formatted string.
    """
    _, assets_folder = create_folders(path=path, is_community=is_community)
    img_folder = assets_folder / "img"
    plot_data_folder = assets_folder / "plot_data"
    save_folder = assets_folder.joinpath("..").resolve()
    nb = load_notebook(path)
    mdx = ""
    mdx += create_frontmatter(path, nb_metadata)
    mdx += create_imports()
    mdx += create_buttons(nb_metadata)
    for cell in nb["cells"]:
        cell_type = cell["cell_type"]

        # Handle a Markdown cell.
        if cell_type == "markdown":
            mdx += handle_markdown_cell(cell, img_folder, path)

        # Handle a code cell.
        if cell_type == "code":
            mdx += handle_code_cell(cell, plot_data_folder)

    # Write the MDX file to disk.
    save_path = save_folder / "index.mdx"
    with save_path.open("w") as f:
        f.write(mdx)

    # Return the string for debugging purposes.
    return mdx


def clean_up_directories() -> None:
    """
    Delete output from previously converted notebooks. Particularly useful for
    removing plot data since those filenames are randomly generated and won't
    be replaced with future runs.

    Returns:
        None: Does not return anything.
    """
    for dir_ in [TUTORIALS_DIR, NOTEBOOKS_COMMUNITY_DIR]:
        if dir_.exists():
            # We intentionally leave the static `index.mdx` file in place since
            # that is not autogenerated.
            for item in os.scandir(dir_):
                if item.is_dir():
                    shutil.rmtree(item.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert tutorial notebooks into mdx files."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output from previously converted notebooks.",
    )
    args = parser.parse_args()

    tutorials_metadata = load_tutorial_metadata()
    notebooks_community_metadata = load_notebooks_community_metadata()
    print("--------------------------------------------")
    print("Converting tutorial notebooks into mdx files")
    print("--------------------------------------------")
    if args.clean:
        clean_up_directories()
    for metadata in tutorials_metadata:
        path = (
            LIB_DIR / "tutorials" / metadata["id"] / (metadata["id"] + ".ipynb")
        ).resolve()
        print(f"{path.stem}")
        mdx = transform_notebook(path=path, nb_metadata=metadata, is_community=False)
    print("--------------------------------------------")
    print("Converting community notebooks into mdx files")
    print("--------------------------------------------")
    for metadata in notebooks_community_metadata:
        path = (
            LIB_DIR
            / "notebooks_community"
            / metadata["id"]
            / (metadata["id"] + ".ipynb")
        ).resolve()
        print(f"{path.stem}")
        mdx = transform_notebook(path=path, nb_metadata=metadata, is_community=True)
    print("")
