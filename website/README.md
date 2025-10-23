The Botorch website was created with [Docusaurus](https://docusaurus.io/), with some customization to support tutorials, and supplemented with Sphinx for API documentation.

## Dependencies

Ensure necessary dependencies are installed (ideally to your virtual env):
```bash
pip install -e ".[tutorials]"
```

## Building (all-in-one)

For convenience we provide a single shell script to convert the tutorials and build the website in one command. Must be executed from the repository root.
```bash
./scripts/build_docs.sh
```

To also execute the tutorials, add the `-t` flag.
To generate a static build add the `-b` flag.

`-h` for all options.


## Building (manually)

### Notebooks

Tutorials can be executed locally using the following script. This is optional for locally building the website and is slow.
```bash
python3 scripts/run_tutorials.py -w .
```

We convert tutorial notebooks to MDX for embedding as docs. This needs to be done before serving the website and can be done by running this script from the project root:

```bash
python3 scripts/convert_ipynb_to_mdx.py --clean
```

### Docusaurus
You need [Node](https://nodejs.org/en/) >= 18.x and
[Yarn](https://yarnpkg.com/en/) in order to build the Botorch website.

Switch to the `website` dir from the project root and start the server:
```bash
cd website
yarn install
yarn start
```

Open http://localhost:3000 (if doesn't automatically open).

Anytime you change the contents of the page, the page should auto-update.

> [!NOTE]
> You may need to switch to the "Next" version of the website documentation to see your latest changes.

### Sphinx
Sphinx is used to generate an API reference from the source file docstrings. In production we use [ReadTheDocs](https://botorch.readthedocs.io/en/stable/index.html) to build and host these docs, but they can also be built locally for testing.
```sh
cd sphinx/
make html
```

The build output is in `sphinx/build/html/` but Sphinx does not provide a server. Here's a serving example using Python:

```sh
cd sphinx/build/html/
python3 -m http.server 8000
```


## Publishing

The site is hosted on GitHub pages, automatically deployed using the Github [deploy-pages](https://github.com/actions/deploy-pages) action - see the
[config file](https://github.com/meta-pytorch/botorch/blob/main/.github/workflows/publish_website.yml) for details.
