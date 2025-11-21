---
id: objectives
title: Objectives
---


In BoTorch, an *objective* is a module that allows for convenient transformation
of model outputs into a scalar function to be optimized.
Typical use cases for this are the scalarization of outputs for a multi-output
model (see e.g. [^RandScal]), or optimization subject to outcome constraints,
which can be achieved by weighting the objective by the probability of
feasibility [^NoisyEI].

When using classical analytic formulations of acquisition functions, one needs
to be careful that the transformation results in a posterior distribution of the
transformed outputs that still satisfies the assumptions of the analytic
formulation. For instance, to use standard Expected Improvement on a transformed
output of a model, the transformation needs to be affine (because Gaussians are
closed under affine transformations).
When using MC-based acquisition functions, however, fewer assumptions are
required, and one can apply general transformations to the model outputs with
relative impunity so long gradients can be back-propagated through the
transformation.

All BoTorch objectives are derived from
[`MCAcquisitionObjective`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.objective.MCAcquisitionObjective).
BoTorch implements several MC-based objectives, including
[`LinearMCObjective`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.objective.LinearMCObjective) for linear
combinations of model outputs, and
[`ConstrainedMCObjective`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.objective.ConstrainedMCObjective) for
constrained objectives (using a sigmoid approximation for the constraints).


## Using custom objectives

### Utilizing GenericMCObjective

The [`GenericMCObjective`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.objective.GenericMCObjective) allows
simply using a generic callable to implement an ad-hoc objective. The callable
is expected to map a `sample_shape x batch_shape x q x o`-dimensional tensor of
posterior samples and an (optional) `batch_shape x q x d`-dimensional tensor of
inputs to a `sample_shape x batch_shape x q`-dimensional tensor of sampled
objective values.

For instance, say you have a multi-output model with $o=2$ outputs, and you want
to optimize a $obj(y) = 1 - \|y - y_0\|_2$, where $y_0 \in \mathbb{R}^2$.
For this you would use the following custom objective (here we can ignore the
inputs $X$ as the objective does not depend on it):
```python
obj = lambda xi, X: 1 - torch.linalg.norm(xi - y_0, dim=-1)
mc_objective = GenericMCObjective(obj)
```

### Implementing a custom objective module

Instead of using `GenericMCObjective`, you can also implement your own
`MCAcquisitionObjective` modules to make them easier to re-use, or support
more complex logic. The only thing required to implement
is a `forward` method that takes in a
`sample_shape x batch_shape x q x o`-dimensional tensor of
posterior samples and maps it to a
`sample_shape x batch_shape x q`-dimensional tensor of sampled objective values.

A custom objective module of the above example would be
```python
class MyCustomObjective(MCAcquisitionObjective):

    def forward(self, samples, X=None):
      return 1 - torch.linalg.norm(samples - y_0, dim=-1)
```


[^RandScal]: B. Paria, K. Kandasamy, and B. Póczos. A Flexible Multi-Objective
Bayesian Optimization Approach using Random Scalarizations. ArXiv, 2018.

[^NoisyEI]: B. Letham, B. Karrer, G. Ottoni and Bakshy, E. Constrained Bayesian
Optimization with Noisy Experiments. Bayesian Analysis, 2018.
