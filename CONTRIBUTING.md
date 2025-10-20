# Contributing to BoTorch

We want to make contributing to BoTorch is as easy and transparent as possible.

## Development installation

To get the development installation with all the necessary dependencies for
linting, testing, and building the documentation, run the following:

```bash
git clone https://github.com/meta-pytorch/botorch.git
cd botorch
pip install -e ".[dev]"
```

## Our Development Process

#### Code Style

BoTorch uses [ufmt](https://github.com/omnilib/ufmt) to enforce consistent code
formatting (based on [black](https://github.com/ambv/black)) and import sorting
(based on [µsort](https://github.com/facebook/usort)) across the code base.
Install via `pip install ufmt`, and auto-format and auto-sort by running

```bash
ufmt format .
```

from the repository root.

#### Flake8 linting

BoTorch uses `flake8` for linting. To run the linter locally, install `flake8`
via `pip install flake8`, and then run

```bash
flake8 .
```

from the repository root.

#### Pre-commit hooks

Contributors can use [pre-commit](https://pre-commit.com/) to run `ufmt` and
`flake8` as part of the commit process. To install the hooks, install `pre-commit`
via `pip install pre-commit` and run `pre-commit install` from the repository
root.

#### Docstring formatting

BoTorch uses
[Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
docstrings. To make sure documentation is rendered correctly, we require that
every `__init__` function contains an `Args:` block. We use the
`flake8-docstrings` plugin to check this - install via
`pip install flake8-docstrings` and run `flake8` as above to check.

#### Type Hints

BoTorch is fully typed using python 3.10+
[type hints](https://www.python.org/dev/peps/pep-0484/). We expect any
contributions to also use proper type annotations. While we currently do not
enforce full consistency of these in our continuous integration test, you should
strive to type check your code locally. For this we recommend using
[pyre](https://pyre-check.org/).

#### Unit Tests

To run the unit tests, you can either use `pytest` (if installed):

```bash
pytest -ra
```

or python's `unittest`:

```bash
python -m unittest
```

To get coverage reports we recommend using the `pytest-cov` plugin:

```bash
pytest -ra --cov=. --cov-report term-missing
```

#### Documentation

BoTorch's website is also open source, and is part of this very repository (the
code can be found in the [website](/website/) folder). It is built using
[Docusaurus](https://docusaurus.io/), and consists of three main elements:

1. The documentation in Docusaurus itself (if you know Markdown, you can already
   contribute!). This lives in the [docs](/docs/).
2. The API reference, auto-generated from the docstrings using
   [Sphinx](http://www.sphinx-doc.org), and embedded into the Docusaurus
   website. The sphinx .rst source files for this live in
   [sphinx/source](/sphinx/source/).
3. The Jupyter notebook tutorials, parsed and converted to MDX, and embedded into the
   Docusaurus website. These live in [tutorials](/tutorials/).

To build the documentation you will need [Node](https://nodejs.org/en/) >= 8.x
and [Yarn](https://yarnpkg.com/en/) >= 1.5.

The following command will both build the docs and serve the site locally:

```bash
./scripts/build_docs.sh
```

See the [website/README.md](website/README.md) for more details.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you have added code that should be tested, add unit tests. In other words,
   add unit tests.
3. If you have changed APIs, update the documentation. Make sure the
   documentation builds.
4. Ensure the test suite passes.
5. Make sure your code passes both `ufmt` and `flake8` formatting checks.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

### Community Contributions (Beta)

In order to reduce the maintenance work required from our team, while continuing
to support contributions from BoTorch community, we're trying out a revised
support model for community contributions.

Contributing new methods & notebooks: We're asking our contributors to add these
under `botorch_community`, `notebooks_community` & `test_community`, and their
help in maintaining added code going forward. The maintenance expectations
include keeping the code up to date against any deprecations in BoTorch and
dependencies, fixing any breakages, and help in responding to issues &
discussions from other users. We will notify the contributors of any maintenance
needs and expect a resolution of the issues within 90 days. We may decide to
move any contribution into core BoTorch at a future date, at which point our
team will assume all maintenance responsibility.

Fixes & improvements to core BoTorch: We appreciate the support from our
community and continue welcoming PRs with bug-fixes & improvements to core
BoTorch.

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects. You can
complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License

By contributing to BoTorch, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
