<a href="https://botorch.org">
  <img width="350" src="https://botorch.org/img/botorch_logo_lockup.png" alt="BoTorch Logo" />
</a>

<hr/>

[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://opensource.fb.com/support-ukraine)
[![Lint](https://github.com/meta-pytorch/botorch/workflows/Lint/badge.svg)](https://github.com/meta-pytorch/botorch/actions?query=workflow%3ALint)
[![Test](https://github.com/meta-pytorch/botorch/workflows/Test/badge.svg)](https://github.com/meta-pytorch/botorch/actions?query=workflow%3ATest)
[![Docs](https://github.com/meta-pytorch/botorch/workflows/Docs/badge.svg)](https://github.com/meta-pytorch/botorch/actions?query=workflow%3ADocs)
[![Nightly](https://github.com/meta-pytorch/botorch/actions/workflows/nightly.yml/badge.svg)](https://github.com/meta-pytorch/botorch/actions?query=workflow%3ANightly)
[![Codecov](https://img.shields.io/codecov/c/github/meta-pytorch/botorch.svg)](https://codecov.io/github/meta-pytorch/botorch)

[![PyPI](https://img.shields.io/pypi/v/botorch.svg)](https://pypi.org/project/botorch)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


BoTorch is a library for Bayesian Optimization built on PyTorch.

*BoTorch is currently in beta and under active development!*


#### Why BoTorch ?
BoTorch
* Provides a modular and easily extensible interface for composing Bayesian
  optimization primitives, including probabilistic models, acquisition functions,
  and optimizers.
* Harnesses the power of PyTorch, including auto-differentiation, native support
  for highly parallelized modern hardware (e.g. GPUs) using device-agnostic code,
  and a dynamic computation graph.
* Supports Monte Carlo-based acquisition functions via the
  [reparameterization trick](https://arxiv.org/abs/1312.6114), which makes it
  straightforward to implement new ideas without having to impose restrictive
  assumptions about the underlying model.
* Enables seamless integration with deep and/or convolutional architectures in PyTorch.
* Has first-class support for state-of-the art probabilistic models in
  [GPyTorch](http://www.gpytorch.ai/), including support for multi-task Gaussian
  Processes (GPs) deep kernel learning, deep GPs, and approximate inference.


#### Target Audience

The primary audience for hands-on use of BoTorch are researchers and
sophisticated practitioners in Bayesian Optimization and AI.
We recommend using BoTorch as a low-level API for implementing new algorithms
for [Ax](https://ax.dev). Ax has been designed to be an easy-to-use platform
for end-users, which at the same time is flexible enough for Bayesian
Optimization researchers to plug into for handling of feature transformations,
(meta-)data management, storage, etc.
We recommend that end-users who are not actively doing research on Bayesian
Optimization simply use Ax.


## Installation

**Installation Requirements**
- Python >= 3.11
- PyTorch >= 2.0.1
- gpytorch >= 1.14
- linear_operator >= 0.6
- pyro-ppl >= 1.8.4
- scipy
- multiple-dispatch


### Option 1: Installing the latest release

The latest release of BoTorch is easily installed via `pip`:

```bash
pip install botorch
```

_Note_: Make sure the `pip` being used is actually the one from the newly created
Conda environment. If you're using a Unix-based OS, you can use `which pip` to check.

BoTorch [stopped publishing](https://github.com/meta-pytorch/botorch/discussions/2613#discussion-7431533)
an official Anaconda package to the `pytorch` channel after the 0.12 release. However,
users can still use the package published to the `conda-forge` channel and install botorch via

```bash
conda install botorch -c gpytorch -c conda-forge
```

### Option 2: Installing from latest main branch

If you would like to try our bleeding edge features (and don't mind potentially
running into the occasional bug here or there), you can install the latest
development version directly from GitHub. You may also want to install the
current `gpytorch` and `linear_operator` development versions:
```bash
pip install --upgrade git+https://github.com/cornellius-gp/linear_operator.git
pip install --upgrade git+https://github.com/cornellius-gp/gpytorch.git
pip install --upgrade git+https://github.com/meta-pytorch/botorch.git
```

### Option 3: Editable/dev install

If you want to [contribute](CONTRIBUTING.md) to BoTorch, you will want to install editably so that you can change files and have the
changes reflected in your local install.

If you want to install the current `gpytorch` and `linear_operator` development versions, as in Option 2, do that
before proceeding.

#### Option 3a: Bare-bones editable install

```bash
git clone https://github.com/meta-pytorch/botorch.git
cd botorch
pip install -e .
```

#### Option 3b: Editable install with development and tutorials dependencies

```bash
git clone https://github.com/meta-pytorch/botorch.git
cd botorch
pip install -e ".[dev, tutorials]"
```

* `dev`: Specifies tools necessary for development
  (testing, linting, docs building; see [Contributing](#contributing) below).
* `tutorials`: Also installs all packages necessary for running the tutorial notebooks.
* You can also install either the dev or tutorials dependencies without installing both, e.g. by changing the last command to `pip install -e ".[dev]"`.

## Getting Started

Here's a quick run down of the main components of a Bayesian optimization loop.
For more details see our [Documentation](https://botorch.org/docs/introduction) and the
[Tutorials](https://botorch.org/docs/tutorials).

1. Fit a Gaussian Process model to data
  ```python
  import torch
  from botorch.models import SingleTaskGP
  from botorch.models.transforms import Normalize
  from botorch.fit import fit_gpytorch_mll
  from gpytorch.mlls import ExactMarginalLogLikelihood

  # Double precision is highly recommended for GPs.
  # See https://github.com/meta-pytorch/botorch/discussions/1444
  train_X = torch.rand(10, 2, dtype=torch.double) * 2
  Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)  # explicit output dimension
  Y += 0.1 * torch.rand_like(Y)

  gp = SingleTaskGP(
      train_X=train_X,
      train_Y=Y,
      input_transform=Normalize(d=2),
  )
  mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
  fit_gpytorch_mll(mll)
  ```

2. Construct an acquisition function
  ```python
  from botorch.acquisition import LogExpectedImprovement

  logEI = LogExpectedImprovement(model=gp, best_f=Y.max())
  ```

3. Optimize the acquisition function
  ```python
  from botorch.optim import optimize_acqf

  bounds = torch.stack([torch.zeros(2), torch.ones(2)]).to(torch.double)
  candidate, acq_value = optimize_acqf(
      logEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
  )
  ```


## Citing BoTorch

If you use BoTorch, please cite the following paper:
> [M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020.](https://arxiv.org/abs/1910.06403)

```
@inproceedings{balandat2020botorch,
  title={{BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization}},
  author={Balandat, Maximilian and Karrer, Brian and Jiang, Daniel R. and Daulton, Samuel and Letham, Benjamin and Wilson, Andrew Gordon and Bakshy, Eytan},
  booktitle = {Advances in Neural Information Processing Systems 33},
  year={2020},
  url = {http://arxiv.org/abs/1910.06403}
}
```

See [here](https://botorch.org/docs/papers) for an incomplete selection of peer-reviewed papers that build off of BoTorch.


## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.


## License
BoTorch is MIT licensed, as found in the [LICENSE](LICENSE) file.
