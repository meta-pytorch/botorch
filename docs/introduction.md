---
id: introduction
title: Introduction
---

botorch (pronounced like blow-torch) is a library for Bayesian
Optimization research built on top of PyTorch, and is part of the PyTorch
ecosystem.  Botorch is best used in tandem with Ax, Facebook's open-source adaptive experimentation platform, which provides an easy-to-use Bayesian optimization interface while handling various experiment and data management, transformations, and systems integration.

Bayesian Optimization is an established technique for sequential optimization
of costly-to-evaluate black-box functions. It can be applied to a wide variety
of problems, including machine learning (tuning algorithms' hyper-parameters),
A/B testing, as well as scientific and engineering problems.

## Why botorch

### Improved developer efficiency

Being built on top of PyTorch, botorch harnesses the power of native GPU
acceleration and auto-differentiation, while allowing for a great deal of modeling flexibility
for deep architectures through seamless integration with generic PyTorch modules.
By doing so, it significantly improves developer efficiency, as new model types and acqusition functions can be implemented without the need to derive analytical gradients by hand.

### State-of-the-art modeling

Bayesian Optimization traditionally relies heavily on Gaussian Process (GP)
models, and as such, we provide first-class support for [GPyTorch](https://gpytorch.ai/),
a scalable package for GPs and Bayesian deep learning implemented in PyTorch.
Botorch's basic API is model-agnostic, and can be used with any kind of probabilistic model,
such as those written in [Pyro](http://pyro.ai/).  This opens the door for tackling problems that have traditionally not been
amenable to Bayesian Optimization.

### Bridging the gap between research and production

botorch implements modular building blocks for modern Bayesian optimization.
It aims to bridge the gap between research and productionn by being a very flexible
research framework, but at the same time, a reliable, production-grade
implementation that integrates well with platforms like Facebook Ax (TODO: add link). Finally, botorch strives for excellent code quality by enforcing comprehensive unit test coverage, full type annotations, and strict style rules.

## Target Audience

The primary audience for hands-on use of botorch are researchers and
sophisticated practitioners in Bayesian optimization and AI.
We recommend using botorch as a low-level API for implementing new algorithms for Ax (TODO, add link to Ax website), an open-source platform for sequential experimentation.  Ax has been designed to be an easy-to-use platform for end-users, which is at the same time, flexible enough for Bayesian optimization researchers to plug into for handling of feature transformations, data management, etc (TODO: link to botorch-Ax tutorial).  We recommend that end-users of Bayesian optimization who are not actively doing research in the field simply use Ax.