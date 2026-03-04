---
id: design_philosophy
title: Design Philosophy
---

## Main Design Tenets

BoTorch adheres to the following main design tenets:
* **Modularity & Simplicity**
  * Make it easy for researchers to develop and implement new ideas by following
    a modular design philosophy & making heavy use of auto-differentiation. Most
    BoTorch components are `torch.nn.Module` instances, so that users familiar
    with PyTorch can easily implement new differentiable
    components.
  * Facilitate model-agnostic Bayesian Optimization by maintaining lightweight
    APIs and first-class support for Monte-Carlo-based acquisition functions.

* **Performance & Scalability**
  * Achieve high levels of performance across different platforms with
    device-agnostic code by using highly parallelized batch operations.
  * Expand the applicability of Bayesian Optimization to very large problems by
    harnessing scalable modeling frameworks such as
    [GPyTorch](https://gpytorch.ai).


## Parallelism Through Batched Computations

Batching (as in batching data or batching computations) is a central component
to all modern deep learning platforms and plays a critical role in the design of
BoTorch. Examples of batch computations in BoTorch include:

1. A batch of candidate points $X$ to be evaluated in parallel on the black-box
   function we are trying optimize. In BoTorch, we refer to this kind of batch
   as a **"q-batch"**.
2. A batch of q-batches to be evaluated in parallel on the surrogate model of
   the black-box function. These facilitate fast evaluation on modern hardware
   such as GPUs and multi-core CPUs with advanced instruction sets (e.g. AVX).
   In BoTorch, we refer to a batch of this type as **"t-batch"** (as in
   "torch-batch").
3. A **batched** surrogate **model**, each batch of which models a different
   output (which is useful for multi-objective Bayesian Optimization). This kind
   of batching also aims to exploit modern hardware architecture.

Note that none of these notions of batch pertains to the batching of *training
data*, which is commonly done in training Neural Network models (sometimes
called "mini-batching"). BoTorch aims to be agnostic with regards to the
particular model used - so while model fitting may indeed be performed via
stochastic gradient descent using mini-batch training, BoTorch itself abstracts
away from this.

For an in-depth look at the different batch notions in BoTorch, take a look at
the [Batching in BoTorch](batching) section.


## Optimizing Acquisition Functions

While BoTorch tries to align as closely as possible with PyTorch when possible,
optimization of acquisition functions requires a somewhat different approach.
We now describe this discrepancy and explain in detail why we made this design
decision.

In PyTorch, modules typically map (batches of) data to an output, where the
mapping is parameterized by the parameters of the modules (often the weights
of a Neural Network). Fitting the model means optimizing some loss (which is
defined with respect to the underlying distribution of the data).
As this distribution is unknown, one cannot directly evaluate this function.
Instead, one considers the empirical loss function, i.e. the loss evaluated on
all data available. In typical machine learning model training, a stochastic
version of the empirical loss, obtained by "mini-batching" the data, is
optimized using stochastic optimization algorithms.

In BoTorch, [`AcquisitionFunction`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.acquisition.AcquisitionFunction)
modules map an input design $X$ to the acquisition function value. Optimizing
the acquisition function means optimizing the output over the possible values of
$X$. If the acquisition function is deterministic, then so is the optimization
problem.

For large Neural Network models, the number of optimization variables is very
high, and can be in the hundreds of thousands or even millions of parameters.
The resulting optimization problem is often solved using first-order
stochastic gradient descent algorithms (e.g. SGD and its many variants).
Many of these are implemented in the `torch.optim` module. The typical way of
optimizing a model with these algorithms is by extracting the module's
parameters (e.g. using `parameters()`), and writing a manual optimization loop
that calls `step()` on a torch `Optimizer` object.

Optimizing acquisition functions is different since the problem dimensionality
is often much smaller. Indeed, optimizing over $q$ design points in a
$d$-dimensional feature space results in $qd$ scalar parameters to optimize
over. Both $q$ and $d$ are often quite small, and hence so is the dimensionality
of the problem.
Moreover, the optimization problem can be cast as a deterministic one (either
because an analytic acquisition function is used, or because the
reparameterization trick is employed to render the Monte-Carlo-based evaluation
of the acquisition function deterministic in terms of the input tensor $X$).
As a result, optimization algorithms that are typically inadmissible for
problems such as training Neural Networks become promising alternatives to
standard first-order methods. In particular, this includes quasi-second order
methods (such as L-BFGS or SLSQP) that approximate local curvature of the
acquisition function by using past gradient information.
These methods are currently not well supported in the `torch.optim` package,
which is why BoTorch provides a custom interface that wraps the optimizers from
the `scipy.optimize` module.
