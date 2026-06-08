---
id: posteriors
title: Posteriors
---

A BoTorch `Posterior` object is a layer of
abstraction that separates the specific model used from the evaluation (and
subsequent optimization) of acquisition functions.
In the simplest case, a posterior is a lightweight wrapper around an explicit
distribution object from `torch.distributions` (or `gpytorch.distributions`).
However, a BoTorch `Posterior` can be any distribution (even an implicit one),
so long as one can sample from that distribution. For example, a posterior could
be represented implicitly by some base distribution mapped through a neural network.

While the analytic acquisition functions assume that the posterior is a
multivariate Gaussian, the Monte-Carlo (MC) based acquisition functions do not make any
assumptions about the underlying distribution. Rather, the MC-based acquisition
functions only require that the posterior can generate samples through an `rsample`
method. As long as the posterior implements the [`Posterior`](https://botorch.readthedocs.io/en/latest/posteriors.html#botorch.posteriors.posterior.Posterior)
interface, it can be used with an MC-based acquisition function. In addition, note that
gradient-based acquisition function optimization requires the ability to back-propagate
gradients through the MC samples.

For GP models based on GPyTorch for which the posterior distribution is a
multivariate Gaussian,
[`GPyTorchPosterior`](https://botorch.readthedocs.io/en/latest/posteriors.html#botorch.posteriors.gpytorch.GPyTorchPosterior) should be used.
