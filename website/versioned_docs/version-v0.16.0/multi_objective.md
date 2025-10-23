---
id: multi_objective
title: Multi-Objective Bayesian Optimization
---

BoTorch provides first-class support for Multi-Objective (MO) Bayesian
Optimization (BO) including implementations of
[`qLogNoisyExpectedHypervolumeImprovement`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.multi_objective.logei.qLogNoisyExpectedHypervolumeImprovement)
(qLogNEHVI)[^qNEHVI][^LogEI],
[`qLogExpectedHypervolumeImprovement`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.multi_objective.logei.qLogExpectedHypervolumeImprovement)
(qLogEHVI),
[`qLogNParEGO`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.multi_objective.parego.qLogNParEGO)[^qNEHVI],
and analytic
[`ExpectedHypervolumeImprovement`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.multi_objective.analytic.ExpectedHypervolumeImprovement)
(EHVI) with gradients via auto-differentiation acquisition functions[^qEHVI].

The goal in MOBO is learn the _Pareto front_: the set of optimal trade-offs,
where an improvement in one objective means deteriorating another objective.
Botorch provides implementations for a number of acquisition functions
specifically for the multi-objective scenario, as well as generic interfaces for
implemented new multi-objective acquisition functions.

## Multi-Objective Acquisition Functions

MOBO leverages many advantages of BoTorch to make provide practical algorithms
for computationally intensive and analytically intractable problems. For
example, analytic EHVI has no known analytical gradient for when there are more
than two objectives, but BoTorch computes analytic gradients for free via
auto-differentiation, regardless of the number of objectives [^qEHVI].

For analytic and MC-based MOBO acquisition functions such as qLogNEHVI,
qLogEHVI, and `qLogNParEGO`, BoTorch leverages GPU acceleration and quasi-second
order methods for acquisition optimization for efficient computation and
optimization in many practical scenarios [^qNEHVI][^qEHVI]. The MC-based
acquisition functions support using the sample average approximation for rapid
convergence [^BoTorch].

All analytic MO acquisition functions derive from
[`MultiObjectiveAnalyticAcquisitionFunction`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.multi_objective.base.MultiObjectiveAnalyticAcquisitionFunction)
and all MC-based acquisition functions derive from
[`MultiObjectiveMCAcquisitionFunction`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.multi_objective.base.MultiObjectiveMCAcquisitionFunction).
These abstract classes easily integrate with BoTorch's standard optimization
machinery.

`qLogNParEGO` supports optimization via random scalarizations. In the batch
setting, it uses a new random scalarization for each candidate [^qEHVI].
Candidates are selected in a sequential greedy fashion, each with a different
scalarization, via the
[`optimize_acqf_list`](https://botorch.readthedocs.io/en/latest/optim.html#botorch.optim.optimize.optimize_acqf_list)
function.

For a more in-depth example using these acquisition functions, check out the
[Multi-Objective Bayesian Optimization tutorial notebook](tutorials/multi_objective_bo).

## Multi-Objective Utilities

BoTorch provides several utility functions for evaluating performance in MOBO
including a method for computing the Pareto front
[`is_non_dominated`](https://botorch.readthedocs.io/en/latest/utils.html#botorch.utils.multi_objective.pareto.is_non_dominated)
and efficient box decomposition algorithms for efficiently partitioning the the
space dominated
[`DominatedPartitioning`](https://botorch.readthedocs.io/en/latest/utils.html#botorch.utils.multi_objective.box_decompositions.dominated.DominatedPartitioning)
or non-dominated
[`NonDominatedPartitioning`](https://botorch.readthedocs.io/en/latest/utils.html#botorch.utils.multi_objective.box_decompositions.non_dominated.NondominatedPartitioning)
by the Pareto frontier into axis-aligned hyperrectangular boxes. For exact box
decompositions, BoTorch uses a two-step approach similar to that in [^Yang2019],
where (1) Algorithm 1 from [Lacour17]_ is used to find the local lower bounds
for the maximization problem and (2) the local lower bounds are used as the
Pareto frontier for the minimization problem, and [Lacour17]_ is applied again
to partition the space dominated by that Pareto frontier. Approximate box
decompositions are also supported using the algorithm from [^Couckuyt2012]. See
Appendix F.4 in [^qEHVI] for an analysis of approximate vs exact box
decompositions with EHVI. These box decompositions (approximate or exact) can
also be used to efficiently compute hypervolumes.

Additionally, variations on ParEGO can be trivially implemented using an
augmented Chebyshev scalarization as the objective with an EI-type
single-objective acquisition function such as
[`qLogNoisyExpectedImprovement`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.logei.qLogNoisyExpectedImprovement).
The
[`get_chebyshev_scalarization`](https://botorch.readthedocs.io/en/latest/utils.html#botorch.utils.multi_objective.scalarization.get_chebyshev_scalarization)
convenience function generates these scalarizations.

[^qNEHVI]:
    S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of
    Multiple Noisy Objectives with Expected Hypervolume Improvement. Advances in
    Neural Information Processing Systems 34, 2021.
    [paper](https://arxiv.org/abs/2105.08195)

[^LogEI]:
    S. Ament, S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy. Unexpected
    Improvements to Expected Improvement for Bayesian Optimization. Advances in
    Neural Information Processing Systems 36, 2023.
    [paper](https://arxiv.org/abs/2310.20708) "Log" variances of acquisition
    functions, such as
    [`qLogNoisyExpectedHypervolumeImprovement`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.multi_objective.logei.qLogNoisyExpectedHypervolumeImprovement),
    offer improved numerics compared to older counterparts such as
    [`qNoisyExpectedHypervolumeImprovement`](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.multi_objective.monte_carlo.qNoisyExpectedHypervolumeImprovement).

[^qEHVI]:
    S. Daulton, M. Balandat, and E. Bakshy. Differentiable Expected Hypervolume
    Improvement for Parallel Multi-Objective Bayesian Optimization. Advances in
    Neural Information Processing Systems 33, 2020.
    [paper](https://arxiv.org/abs/2006.05078)

[^BoTorch]:
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson,
    and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian
    Optimization. Advances in Neural Information Processing Systems 33, 2020.
    [paper](https://arxiv.org/abs/1910.06403)

[^Yang2019]:
    K. Yang, M. Emmerich, A. Deutz, et al. Efficient computation of expected
    hypervolume improvement using box decomposition algorithms. J Glob Optim
    75, 2019. [paper](https://arxiv.org/abs/1904.12672)

[^Lacour17]:
    R. Lacour, K. Klamroth, C. Fonseca. A box decomposition algorithm to compute
    the hypervolume indicator. Computers & Operations Research, Volume 79, 2017.
    [paper](https://www.sciencedirect.com/science/article/pii/S0305054816301538)

[^Couckuyt2012]:
    I. Couckuyt, D. Deschrijver and T. Dhaene. Towards Efficient Multiobjective
    Optimization: Multiobjective statistical criterions. IEEE Congress on
    Evolutionary Computation, Brisbane, QLD, 2012.
    [paper](https://ieeexplore.ieee.org/document/6256586)
