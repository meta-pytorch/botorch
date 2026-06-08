---
id: introduction
title: Introduction
---

BoTorch (pronounced "bow-torch" / ˈbō-tȯrch) is a library for
[Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization)
research built on top of [PyTorch](https://pytorch.org/), and is part of the
PyTorch ecosystem. Read the [BoTorch paper](https://arxiv.org/abs/1910.06403)
[^BoTorch] for a detailed exposition.

Bayesian Optimization (BayesOpt) is an established technique for sequential
optimization of costly-to-evaluate black-box functions. It can be applied to a
wide variety of problems, including hyperparameter optimization for machine
learning algorithms, A/B testing, as well as many scientific and engineering
problems.

BoTorch is best used in tandem with [Ax](https://ax.dev), Facebook's open-source
adaptive experimentation platform, which provides an easy-to-use interface for
defining, managing and running sequential experiments, while handling
(meta-)data management, transformations, and systems integration. Users who just
want an easy-to-use suite for Bayesian Optimization
[should start with Ax](https://ax.dev/docs/bayesopt).


## Why BoTorch?

### Improved Developer Efficiency

BoTorch provides a modular and easily extensible interface for composing
Bayesian Optimization primitives, including probabilistic models, acquisition
functions, and optimizers.

It significantly improves developer efficiency by utilizing quasi-Monte-Carlo
acquisition functions (by way of the "re-parameterization trick"
[^AutoEncVarBayes], [^ReparamAcq]), which makes it straightforward to implement
new ideas without having to impose restrictive assumptions about the underlying
model. Specifically, it avoids pen and paper math to derive analytic expressions
for acquisition functions and their gradients.
More importantly, it opens the door for novel approaches that do not admit
analytic solutions, including batch acquisition functions and proper handling of
rich multi-output models with multiple correlated outcomes.

BoTorch follows the same modular design philosophy as PyTorch, which makes it
very easy for users to swap out or rearrange individual components in order to
customize all aspects of their algorithm, thereby empowering researchers to do
state-of-the art research on modern Bayesian Optimization methods.


### State-of-the-art Modeling

Bayesian Optimization traditionally relies heavily on Gaussian Process (GP)
models, which provide well-calibrated uncertainty estimates. BoTorch provides
first-class support for state-of-the art probabilistic models in
[GPyTorch](https://gpytorch.ai), a library for efficient and scalable GPs
implemented in PyTorch (and to which the BoTorch authors have significantly
contributed).
This includes support for multi-task GPs, deep kernel learning, deep GPs, and
approximate inference. This enables using GP models for problems that have
traditionally not been amenable to Bayesian Optimization techniques.

In addition, BoTorch's lightweight APIs are model-agnostic (they can for example
work with [Pyro](http://pyro.ai) models), and support optimization of
acquisition functions over any kind of posterior distribution, as long as it can
be sampled from.


### Harnessing the Features of PyTorch

Built on PyTorch, BoTorch takes advantage of auto-differentiation, native
support for highly parallelized modern hardware (such as GPUs) using
device-agnostic code, and a dynamic computation graph that facilitates
interactive development.

BoTorch's modular design allows for a great deal of modeling flexibility for
including deep and/or convolutional architectures through seamless integration
with generic PyTorch modules. Importantly, working full-stack in python allows
back-propagating gradients through the full composite model, in turn enabling
joint training of GP and Neural Network modules, and end-to-end gradient-based
optimization of acquisition functions operating on differentiable models.


### Bridging the Gap Between Research and Production

BoTorch implements modular building blocks for modern Bayesian Optimization.
It bridges the gap between research and production by being a very flexible
research framework, but at the same time, a reliable, production-grade
implementation that integrates well with other higher-level platforms,
specifically [Ax](https://ax.dev).


## Target Audience

The primary audience for hands-on use of BoTorch are researchers and
sophisticated practitioners in Bayesian Optimization and AI.

We recommend using BoTorch as a low-level API for implementing new algorithms
for Ax. Ax has been designed to be an easy-to-use platform for end-users, which
at the same time is flexible enough for Bayesian Optimization researchers to
plug into for handling of feature transformations, (meta-)data management,
storage, etc. See [Using BoTorch with Ax](botorch_and_ax) for more details.

We recommend that end-users who are not actively doing research on Bayesian
Optimization simply use Ax.


[^BoTorch]: M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson,
and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.
Advances in Neural Information Processing Systems 33, 2020.
[pdf](https://arxiv.org/abs/1910.06403)

[^AutoEncVarBayes]: D. P. Kingma and M. Welling. Auto-Encoding Variational Bayes.
ArXiv e-prints, [arXiv:1312.6114](https://arxiv.org/abs/1312.6114), Dec 2013.

[^ReparamAcq]: J. T. Wilson, R. Moriconi, F. Hutter, and M. P. Deisenroth.
The reparameterization trick for acquisition functions. ArXiv e-prints,
[arXiv:1712.00424](https://arxiv.org/abs/1712.00424), Dec 2017.
