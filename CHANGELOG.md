# Changelog

The release log for BoTorch.

## [0.15.1] -- Aug 12, 2025
This is a compatibility release, coming only one week after 0.15.0.

#### New features
* Enable optimizing a sequence of acquisition functions in `optimize_acqf` (#2931).

## [0.15.0] -- Aug 5, 2025

#### New Features
* NP Regression Model w/ LIG Acquisition (#2683).
* Fully Bayesian Matern GP with dimension scaling prior (#2855).
* Option for input warping in non-linear fully Bayesian GPs (#2858).
* Support for `condition_on_observations` in `FullyBayesianMultiTaskGP` (#2871).
* Improvements to `optimize_acqf_mixed_alternating`:
  * Support categoricals in alternating optimization (#2866).
  * Batch mixed optimization (#2895).
  * Non-equidistant discrete dimensions for `optimize_acqf_mixed_alternating` (#2923).
  * Update syntax for categoricals in `optimize_acqf_mixed_alternating` (#2942).
  * Equality constraints for `optimize_acqf_mixed_alternating` (#2944).
* Multi-output acquisition functions and related utilities:
  * Multi-Output Acquisition Functions (#2935).
  * Utility for greedily selecting an approximate hypervolume maximizing subset (#2936).
  * Update optimize with NSGA-II (#2937).
  * Add utility for running pymoo NSGA-II (#2868).
* Batched L-BFGS-B for more efficient acquisition function optimization (#2870, #2892).
* Pathwise Thompson sampling for ensemble models (#2877).
* ROBOT tutorial notebook (#2883).
* Add community notebooks to the botorch.org website (#2913).

#### Bug Fixes
* Fix model paths in prior fitted networks (#2843).
* Fix a bug where input transforms were not applied in fully Bayesian models in train mode (#2859).
* Fix local `Y` vs global `Y_Train` in `generate_batch` function in TURBO tutorial (#2862).
* Fix CUDA support for `FullyBayesianMTGP` (#2875).
* Fix edge case with NaNs in `is_non_dominated` (#2925).
* Normalize for correct fidelity in `qLowerBoundMaxValueEntropy` (#2930).
* Bug: Botorch_community `VBLLModel` posterior doesn't work with single value tensor (#2929).
* Fix variance shape bug in Riemann posterior (#2939).
* Fix input constructor for `LogProbabilityOfFeasibility` (#2945).
* Fix `AugmentedRosenbrock` problem and expand testing for optimizers (#2950).

#### Other Changes
* Improved documentation for `optimize_acqf` (#2865).
* Fully Bayesian Multi-Task GP cleanup (#2869).
* `average_over_ensemble_models` decorator for acquisition functions (#2873).
* Changes to I-BNN tutorial (#2889).
* Allow batched fixed features in gen_candidates_scipy and gen_candidates_torch (#2893)
* Refactor of `MultiTask` / `FullyBayesianMultiTaskGP` to use `ProductKernel` and `IndexKernel` (#2908).
* Various changes to PFNs to improve Ax compatibility (#2915, #2940).
* Eliminate expensive indexing in `separate_mtmvn` (#2920).
* Added reset method to `StoppingCriterion` (#2927).
* Simplify closure dispatch (#2947).
* Add BaseTestProblem.is_minimization_problem property (#2949).
* Simplify NdarrayOptimizationClosure (#2951).


## [0.14.0] -- May 6, 2025

#### Highlights
* Prior Fitted Network (PFN) surrogate model integration (#2784).
* Variational Bayesian last-layer models as surrogate `Model`s (#2754).
* Probabilities of feasibility for classifier-based constraints in acquisition functions (#2776).

#### New Features
* Helper for evaluating feasibility of candidate points (#2733).
  * Check for feasibility in `gen_candidates_scipy` and error out for infeasible candidates (#2737).
  * Return a feasible candidate if there is one and `return_best_only=True` (#2778).
* Allow for observation noise without provided `evaluation_mask` mask in `ModelListGP` (#2735).
* Implement incremental `qLogNEI` via `incremental` argument to `qLogNoisyExpectedImprovement` (#2760).
* Add utility for computing AIC/BIC/MLL from a model (#2785).
* New test functions:
  * Multi-fidelity test functions with discrete fidelities (#2796).
  * Keane bump function (#2802).
  * Mixed Ackley test function (#2830).
  * LABS test function (#2832).
* Add parameter types to test functions to support problems defined in mixed / discrete spaces (#2809).
  * Add input validation to test functions (#2829).
* Add `[q]LogProbabilityOfFeasibility` acquisition functions (#2815).

#### Bug Fixes
* Remove hard-coded `dtype` from `best_f` buffers (#2725).
* Fix `dtype/nan` issue in `StratifiedStandardize` (#2757).
* Properly handle observed noise in `AdditiveMapSaasSingleTaskGP` with outcome transforms (#2763).
* Do not count STOPPED (due to specified budget) as a model fitting failure (#2767).
* Ensure that `initialize_q_batch` always includes the maximum value when called in batch mode (#2773).
* Fix posterior with observation noise in batched MTGP models (#2782).
* Detach tensor in `gen_candidates_scipy` to avoid test failure due to new warning (#2797).
* Fix batch computation in Pivoted Cholesky (#2823).

#### Other Changes
* Add optimal values for synthetic contrained optimization problems (#2730).
  * Update `max_hv` and reference point for Penicillin problem (#2771).
  * Add optimal value to SpeedReducer problem (#2799).
* Update `nonlinear_constraint_is_feasible` to return a boolean tensor (#2731).
* Restructure sampling methods for info-theoretic acquisition functions (#2753).
* Prune baseline points in `qLogNEI` by default (#2762).
* Misc updates to MES-based acqusition functions (#2769).
* Pass option to reset submodules in train method for fully Bayesian models (#2783).
* Put outcome transforms into train mode in model constructors (#2817).
* `LogEI`: select `cache_root` based on model support (#2820).
* Remove Ax dependency from BoTorch tutorials and reference Ax tutorials instead (#2839).

#### Deprecations and removals
* Remove deprecated `gp_sampling` module (#2768).
* Remove `qMultiObjectiveMaxValueEntropy` acquisition function (#2800).
* Remove model converters (#2801).


## [0.13.0] -- Feb 3, 2025

#### Highlights
* BoTorch website has been upgraded to utilize Docusaurus v3, with the API
  reference being hosted by ReadTheDocs. The tutorials now expose an option to
  open with Colab, for easy access to a runtime with modifiable tutorials.
  The old versions of the website can be found at archive.botorch.org (#2653).
* `RobustRelevancePursuitSingleTaskGP`, a robust Gaussian process model that adaptively identifies
  outliers and leverages Bayesian model selection ([paper](https://arxiv.org/pdf/2410.24222)) (#2608, #2690, #2707).
* `LatentKroneckerGP`, a scalable model for data on partially observed grids, like the joint modeling
  of hyper-parameters and partially completed learning curves in AutoML ([paper](https://arxiv.org/pdf/2410.09239)) (#2647).
* Add MAP-SAAS model, which utilizes the sparse axis-aligned subspace priors
  ([paper](https://proceedings.mlr.press/v161/eriksson21a/eriksson21a.pdf)) with MAP model fitting (#2694).

#### Compatibility
* Require GPyTorch==1.14 and linear_operator==0.6 (#2710).
* Remove support for anaconda (official package) (#2617).
* Remove `mpmath` dependency pin (#2640).
* Updates to optimization routines to support SciPy>1.15:
  * Use `threadpoolctl` in `minimize_with_timeout` to prevent CPU oversubscription (#2712).
  * Update optimizer output parsing to make model fitting compatible with SciPy>1.15 (#2667).

#### New Features
* Add support for priors in OAK Kernel (#2535).
* Add `BatchBroadcastedTransformList`, which broadcasts a list of `InputTransform`s over batch shapes (#2558).
* `InteractionFeatures` input transform (#2560).
* Implement `percentile_of_score`, which takes inputs `data` and `score`, and returns the percentile of
  values in `data` that are below `score` (#2568).
* Add `optimize_acqf_mixed_alternating`, which supports optimization over mixed discrete & continuous spaces (#2573).
* Add support for `PosteriorTransform` to `get_optimal_samples` and `optimize_posterior_samples` (#2576).
* Support inequality constraints & `X_avoid` in `optimize_acqf_discrete` (#2593).
* Add ability to mix batch initial conditions and internal IC generation (#2610).
* Add `qPosteriorStandardDeviation` acquisition function (#2634).
* TopK downselection for initial batch generation. (#2636).
* Support optimization over mixed spaces in `optimize_acqf_homotopy` (#2639).
* Add `InfeasibilityError` exception class (#2652).
* Support `InputTransform`s in `SparseOutlierLikelihood` and `get_posterior_over_support` (#2659).
* `StratifiedStandardize` outcome transform (#2671).
* Add `center` argument to `Normalize` (#2680).
* Add input normalization step in `Warp` input transform (#2692).
* Support mixing fully Bayesian & `SingleTaskGP` models in `ModelListGP` (#2693).
* Add abstract fully Bayesian GP class and fully Bayesian linear GP model (#2696, #2697).
* Tutorial on BO constrained by probability of classification model (#2700).

#### Bug Fixes
* Fix error in decoupled_mobo tutorial due to torch/numpy issues (#2550).
* Raise error for MTGP in `batch_cross_validation` (#2554).
* Fix `posterior` method in `BatchedMultiOutputGPyTorchModel` for tracing JIT (#2592).
* Replace hard-coded double precision in test_functions with default dtype (#2597).
* Remove `as_tensor` argument of `set_tensors_from_ndarray_1d` (#2615).
* Skip fixed feature enumerations in `optimize_acqf_mixed` that can't satisfy the parameter constraints (#2614).
* Fix `get_default_partitioning_alpha` for >7 objectives (#2646).
* Fix random seed handling in `sample_hypersphere` (#2688).
* Fix bug in `optimize_objective` with fixed features (#2691).
* `FullyBayesianSingleTaskGP.train` should not return `None` (#2702).

#### Other Changes
* More efficient sampling from `KroneckerMultiTaskGP` (#2460).
* Update `HigherOrderGP` to use new priors & standardize outcome transform by default (#2555).
* Update `initialize_q_batch` methods to return both candidates and the corresponding acquisition values (#2571).
* Update optimization documentation with LogEI insights (#2587).
* Make all arguments in `optimize_acqf_homotopy` explicit (#2588).
* Introduce `trial_indices` argument to `SupervisedDataset` (#2595).
* Make optimizers raise an error when provided negative indices for fixed features (#2603).
* Make input transforms `Module`s by default (#2607).
* Reduce memory usage in `ConstrainedMaxPosteriorSampling` (#2622).
* Add `clone` method to datasets (#2625).
* Add support for continuous relaxation within `optimize_acqf_mixed_alternating` (#2635).
* Update indexing in `qLogNEI._get_samples_and_objectives` to support multiple input batches (#2649).
* Pass `X` to `OutcomeTransform`s (#2663).
* Use mini-batches when evaluating candidates within `optimize_acqf_discrete_local_search` (#2682).

#### Deprecations
* Remove `HeteroskedasticSingleTaskGP` (#2616).
* Remove `FixedNoiseDataset` (#2626).
* Remove support for legacy format non-linear constraints (#2627).
* Remove `maximize` option from information theoretic acquisition functions (#2590).


## [0.12.0] -- Sep 17, 2024

#### Major changes
* Update most models to use dimension-scaled log-normal hyperparameter priors by
  default, which makes performance much more robust to dimensionality. See
  discussion #2451 for details. The only models that are _not_ changed are those
  for fully Bayesian models and `PairwiseGP`; for models that utilize a
  composite kernel, such as multi-fidelity/task/context, this change only
  affects the base kernel (#2449, #2450, #2507).
* Use `Standarize` by default in all the models using the upgraded priors. In
  addition to reducing the amount of boilerplate needed to initialize a model,
  this change was motivated by the change to default priors, because the new
  priors will work less well when data is not standardized. Users who do not
  want to use transforms should explicitly pass in `None` (#2458, #2532).

#### Compatibility
* Unpin NumPy (#2459).
* Require PyTorch>=2.0.1, GPyTorch==1.13, and linear_operator==0.5.3 (#2511).

#### New features
* Introduce `PathwiseThompsonSampling` acquisition function (#2443).
* Enable `qBayesianActiveLearningByDisagreement` to accept a posterior
  transform, and improve its implementation (#2457).
* Enable `SaasPyroModel` to sample via NUTS when training data is empty (#2465).
* Add multi-objective `qBayesianActiveLearningByDisagreement` (#2475).
* Add input constructor for `qNegIntegratedPosteriorVariance` (#2477).
* Introduce `qLowerConfidenceBound` (#2517).
* Add input constructor for `qMultiFidelityHypervolumeKnowledgeGradient` (#2524).
* Add `posterior_transform` to `ApproximateGPyTorchModel.posterior` (#2531).

#### Bug fixes
* Fix `batch_shape` default in `OrthogonalAdditiveKernel` (#2473).
* Ensure all tensors are on CPU in `HitAndRunPolytopeSampler` (#2502).
* Fix duplicate logging in `generation/gen.py` (#2504).
* Raise exception if `X_pending` is set on the underlying `AcquisitionFunction`
  in prior-guided `AcquisitionFunction` (#2505).
* Make affine input transforms error with data of incorrect dimension, even in
 eval mode (#2510).
* Use fidelity-aware `current_value` in input constructor for `qMultiFidelityKnowledgeGradient` (#2519).
* Apply input transforms when computing MLL in model closures (#2527).
* Detach `fval` in `torch_minimize` to remove an opportunity for memory leaks
  (#2529).

#### Documentation
* Clarify incompatibility of inter-point constraints with `get_polytope_samples`
  (#2469).
* Update tutorials to use the log variants of EI-family acquisition functions,
  don't make tutorials pass `Standardize` unnecessarily, and other
  simplifications and cleanup (#2462, #2463, #2490, #2495, #2496, #2498, #2499).
* Remove deprecated `FixedNoiseGP` (#2536).

#### Other changes
* More informative warnings about failure to standardize or normalize data
  (#2489).
* Suppress irrelevant warnings in `qHypervolumeKnowledgeGradient` helpers
  (#2486).
* Cleaner `botorch/acquisition/multi_objective` directory structure (#2485).
* With `AffineInputTransform`, always require data to have at least two
 dimensions (#2518).
* Remove deprecated argument `data_fidelity` to `SingleTaskMultiFidelityGP` and
  deprecated model `FixedNoiseMultiFidelityGP` (#2532).
* Raise an `OptimizationGradientError` when optimization produces NaN gradients (#2537).
* Improve numerics by replacing `torch.log(1 + x)` with `torch.log1p(x)`
  and `torch.exp(x) - 1` with `torch.special.expm1` (#2539, #2540, #2541).


## [0.11.3] -- Jul 22, 2024

#### Compatibility
* Pin NumPy to <2.0 (#2382).
* Require GPyTorch 1.12 and LinearOperator 0.5.2 (#2408, #2441).

#### New features
* Support evaluating posterior predictive in `MultiTaskGP` (#2375).
* Infinite width BNN kernel (#2366) and the corresponding tutorial (#2381).
* An improved elliptical slice sampling implementation (#2426).
* Add a helper for producing a `DeterministicModel` using a Matheron path (#2435).

#### Deprecations and Deletions
* Stop allowing some arguments to be ignored in acqf input constructors (#2356).
* Reap deprecated `**kwargs` argument from `optimize_acqf` variants (#2390).
* Delete `DeterministicPosterior` and `DeterministicSampler` (#2391, #2409, #2410).
* Removed deprecated `CachedCholeskyMCAcquisitionFunction` (#2399).
* Deprecate model conversion code (#2431).
* Deprecate `gp_sampling` module in favor of pathwise sampling (#2432).

#### Bug Fixes
* Fix observation noise shape for batched models (#2377).
* Fix `sample_all_priors` to not sample one value for all lengthscales (#2404).
* Make `(Log)NoisyExpectedImprovement` create a correct fantasy model with
  non-default `SingleTaskGP` (#2414).

#### Other Changes
* Various documentation improvements (#2395, #2425, #2436, #2437, #2438).
* Clean up `**kwargs` arguments in `qLogNEI` (#2406).
* Add a `NumericsWarning` for Legacy EI implementations (#2429).


## [0.11.2] -- Jul 22, 2024

See 0.11.3 release. This release failed due to mismatching GPyTorch and LinearOperator versions.


## [0.11.1] -- Jun 11, 2024

#### New Features
* Implement `qLogNParEGO` (#2364).
* Support picking best of multiple fit attempts in `fit_gpytorch_mll` (#2373).

#### Deprecations
* Many functions that used to silently ignore arbitrary keyword arguments will now
raise an exception when passed unsupported arguments (#2327, #2336).
* Remove `UnstandardizeMCMultiOutputObjective` and `UnstandardizePosteriorTransform` (#2362).

#### Bug Fixes
* Remove correlation between the step size and the step direction in `sample_polytope` (#2290).
* Fix pathwise sampler bug (#2337).
* Explicitly check timeout against `None` so that `0.0` isn't ignored (#2348).
* Fix boundary handling in `sample_polytope` (#2353).
* Avoid division by zero in `normalize` & `unnormalize` when lower & upper bounds are equal (#2363).
* Update `sample_all_priors` to support wider set of priors (#2371).

#### Other Changes
* Clarify `is_non_dominated` behavior with NaN (#2332).
* Add input constructor for `qEUBO` (#2335).
* Add `LogEI` as a baseline in the `TuRBO` tutorial (#2355).
* Update polytope sampling code and add thinning capability (#2358).
* Add initial objective values to initial state for sample efficiency (#2365).
* Clarify behavior on standard deviations with <1 degree of freedom (#2357).


## [0.11.0] -- May 1, 2024

#### Compatibility
* Reqire Python >= 3.10 (#2293).

#### New Features
* SCoreBO and Bayesian Active Learning acquisition functions (#2163).

#### Bug Fixes
* Fix non-None constraint noise levels in some constrained test problems (#2241).
* Fix inverse cost-weighted utility behaviour for non-positive acquisition values (#2297).

#### Other Changes
* Don't allow unused keyword arguments in `Model.construct_inputs` (#2186).
* Re-map task values in MTGP if they are not contiguous integers starting from zero (#2230).
* Unify `ModelList` and `ModelListGP` `subset_output` behavior (#2231).
* Ensure `mean` and `interior_point` of `LinearEllipticalSliceSampler` have correct shapes (#2245).
* Speed up task covariance of `LCEMGP` (#2260).
* Improvements to `batch_cross_validation`, support for model init kwargs (#2269).
* Support custom `all_tasks` for MTGPs (#2271).
* Error out if scipy optimizer does not support bounds / constraints (#2282).
* Support diagonal covariance root with fixed indices for `LinearEllipticalSliceSampler` (#2283).
* Make `qNIPV` a subclass of `AcquisitionFunction` rather than `AnalyticAcquisitionFunction` (#2286).
* Increase code-sharing of `LCEMGP` & define `construct_inputs` (#2291).

#### Deprecations
* Remove deprecated args from base `MCSampler` (#2228).
* Remove deprecated `botorch/generation/gen/minimize` (#2229).
* Remove `fit_gpytorch_model` (#2250).
* Remove `requires_grad_ctx` (#2252).
* Remove `base_samples` argument of `GPyTorchPosterior.rsample` (#2254).
* Remove deprecated `mvn` argument to `GPyTorchPosterior` (#2255).
* Remove deprecated `Posterior.event_shape` (#2320).
* Remove `**kwargs` & deprecated `indices` argument of `Round` transform (#2321).
* Remove `Standardize.load_state_dict` (#2322).
* Remove `FixedNoiseMultiTaskGP` (#2323).


## [0.10.0] -- Feb 26, 2024

#### New Features
* Introduce updated guidelines and a new directory for community contributions (#2167).
* Add `qEUBO` preferential acquisition function (#2192).
* Add Multi Information Source Augmented GP (#2152).

#### Bug Fixes
* Fix `condition_on_observations` in fully Bayesian models (#2151).
* Fix for bug that occurs when splitting single-element bins, use default BoTorch kernel for BAxUS. (#2165).
* Fix a bug when non-linear constraints are used with `q > 1` (#2168).
* Remove unsupported `X_pending` from `qMultiFidelityLowerBoundMaxValueEntropy` constructor (#2193).
* Don't allow `data_fidelities=[]` in `SingleTaskMultiFidelityGP` (#2195).
* Fix `EHVI`, `qEHVI`, and `qLogEHVI` input constructors (#2196).
* Fix input constructor for `qMultiFidelityMaxValueEntropy` (#2198).
* Add ability to not deduplicate points in `_is_non_dominated_loop` (#2203).

#### Other Changes
* Minor improvements to `MVaR` risk measure (#2150).
* Add support for multitask models to `ModelListGP` (#2154).
* Support unspecified noise in `ContextualDataset` (#2155).
* Update `HVKG` sampler to reflect the number of model outputs (#2160).
* Release restriction in `OneHotToNumeric` that the categoricals are the trailing dimensions (#2166).
* Standardize broadcasting logic of `q(Log)EI`'s `best_f` and `compute_best_feasible_objective` (#2171).
* Use regular inheritance instead of dispatcher to special-case `PairwiseGP` logic (#2176).
* Support `PBO` in `EUBO`'s input constructor (#2178).
* Add `posterior_transform` to `qMaxValueEntropySearch`'s input constructor (#2181).
* Do not normalize or standardize dimension if all values are equal (#2185).
* Reap deprecated support for objective with 1 arg in `GenericMCObjective` (#2199).
* Consistent signature for `get_objective_weights_transform` (#2200).
* Update context order handling in `ContextualDataset` (#2205).
* Update contextual models for use in MBM (#2206).
* Remove `(Identity)AnalyticMultiOutputObjective` (#2208).
* Reap deprecated support for `soft_eval_constraint` (#2223). Please use `botorch.utils.sigmoid` instead.

#### Compatibility
* Pin `mpmath <= 1.3.0` to avoid CI breakages due to removed modules in the
  latest alpha release (#2222).


## [0.9.5] -- Dec 8, 2023

#### New features

Hypervolume Knowledge Gradient (HVKG):
* Add `qHypervolumeKnowledgeGradient`, which seeks to maximize the difference in hypervolume of the hypervolume-maximizing set of a fixed size after conditioning the unknown observation(s) that would be received if X were evaluated (#1950, #1982, #2101).
* Add tutorial on decoupled Multi-Objective Bayesian Optimization (MOBO) with HVKG (#2094).

Other new features:
* Add `MultiOutputFixedCostModel`, which is useful for decoupled scenarios where the objectives have different costs (#2093).
* Enable `q > 1` in acquisition function optimization when nonlinear constraints are present (#1793).
* Support different noise levels for different outputs in test functions (#2136).

#### Bug fixes
* Fix fantasization with a `FixedNoiseGaussianLikelihood` when `noise` is known and `X` is empty (#2090).
* Make `LearnedObjective` compatible with constraints in acquisition functions regardless of `sample_shape` (#2111).
* Make input constructors for `qExpectedImprovement`, `qLogExpectedImprovement`, and `qProbabilityOfImprovement` compatible with `LearnedObjective` regardless of `sample_shape` (#2115).
* Fix handling of constraints in `qSimpleRegret` (#2141).

#### Other changes
* Increase default sample size for `LearnedObjective` (#2095).
* Allow passing in `X` with or without fidelity dimensions in `project_to_target_fidelity` (#2102).
* Use full-rank task covariance matrix by default in SAAS MTGP (#2104).
* Rename `FullyBayesianPosterior` to `GaussianMixturePosterior`; add `_is_ensemble` and `_is_fully_bayesian` attributes to `Model` (#2108).
* Various improvements to tutorials including speedups, improved explanations, and compatibility with newer versions of libraries.


## [0.9.4] - Nov 6, 2023

#### Compatibility
* Re-establish compatibility with PyTorch 1.13.1 (#2083).


## [0.9.3] - Nov 2, 2023

### Highlights
* Additional "Log" acquisition functions for multi-objective optimization with better numerical behavior, which often leads to significantly improved BO performance over their non-"Log" counterparts:
  * `qLogEHVI` (#2036).
  * `qLogNEHVI` (#2045, #2046, #2048, #2051).
  * Support fully Bayesian models with `LogEI`-type acquisition functions (#2058).
* `FixedNoiseGP` and `FixedNoiseMultiFidelityGP` have been deprecated, their functionalities merged into `SingleTaskGP` and `SingleTaskMultiFidelityGP`, respectively (#2052, #2053).
* Removed deprecated legacy model fitting functions: `numpy_converter`, `fit_gpytorch_scipy`, `fit_gpytorch_torch`, `_get_extra_mll_args` (#1995, #2050).

#### New Features
* Support multiple data fidelity dimensions in `SingleTaskMultiFidelityGP` and (deprecated) `FixedNoiseMultiFidelityGP` models (#1956).
* Add `logsumexp` and `fatmax` to handle infinities and control asymptotic behavior in "Log" acquisition functions (#1999).
* Add outcome and feature names to datasets, implement `MultiTaskDataset` (#2015, #2019).
* Add constrained Hartmann and constrained Gramacy synthetic test problems (#2022, #2026, #2027).
* Support observed noise in `MixedSingleTaskGP` (#2054).
* Add `PosteriorStandardDeviation` acquisition function (#2060).

#### Bug fixes
* Fix input constructors for `qMaxValueEntropy` and `qMultiFidelityKnowledgeGradient` (#1989).
* Fix precision issue that arises from inconsistent data types in `LearnedObjective` (#2006).
* Fix fantasization with `FixedNoiseGP` and outcome transforms and use `FantasizeMixin` (#2011).
* Fix `LearnedObjective` base sample shape (#2021).
* Apply constraints in `prune_inferior_points` (#2069).
* Support non-batch evaluation of `PenalizedMCObjective` (#2073).
* Fix `Dataset` equality checks (#2077).

#### Other changes
* Don't allow unused `**kwargs` in input_constructors except for a defined set of exceptions (#1872, #1985).
* Merge inferred and fixed noise LCE-M models (#1993).
* Fix import structure in `botorch.acquisition.utils` (#1986).
* Remove deprecated functionality: `weights` argument of `RiskMeasureMCObjective` and `squeeze_last_dim` (#1994).
* Make `X`, `Y`, `Yvar` into properties in datasets (#2004).
* Make synthetic constrained test functions subclass from `SyntheticTestFunction` (#2029).
* Add `construct_inputs` to contextual GP models `LCEAGP` and `SACGP` (#2057).


## [0.9.2] - Aug 10, 2023

#### Bug fixes
* Hot fix (#1973) for a few issues:
  * A naming mismatch between Ax's modular `BotorchModel` and the BoTorch's acquisition input constructors, leading to outcome constraints in Ax not being used with single-objective acquisition functions in Ax's modular `BotorchModel`. The naming has been updated in Ax and consistent naming is now used in input constructors for single and multi-objective acquisition functions in BoTorch.
  * A naming mismatch in the acquisition input constructor `constraints` in `qNoisyLogExpectedImprovement`, which kept constraints from being used.
  * A bug in `compute_best_feasible_objective` that could lead to `-inf` incumbent values.
* Fix setting seed in `get_polytope_samples` (#1968)

#### Other changes
* Merge `SupervisedDataset` and `FixedNoiseDataset` (#1945).
* Constrained tutorial updates (#1967, #1970).
* Resolve issues with missing pytorch binaries with py3.11 on Mac (#1966).


## [0.9.1] - Aug 1, 2023

* Require linear_operator == 0.5.1 (#1963).


## [0.9.0] - Aug 1, 2023

#### Compatibility
* Require Python >= 3.9.0 (#1924).
* Require PyTorch >= 1.13.1 (#1960).
* Require linear_operator == 0.5.0 (#1961).
* Require GPyTorch == 1.11 (#1961).

#### Highlights
* Introduce `OrthogonalAdditiveKernel` (#1869).
* Speed up LCE-A kernel by over an order of magnitude (#1910).
* Introduce `optimize_acqf_homotopy`, for optimizing acquisition functions with homotopy (#1915).
* Introduce `PriorGuidedAcquisitionFunction` (PiBO) (#1920).
* Introduce `qLogExpectedImprovement`, which provides more accurate numerics than `qExpectedImprovement` and can lead to significant optimization improvements (#1936).
* Similarly, introduce `qLogNoisyExpectedImprovement`, which is analogous to `qNoisyExpectedImprovement` (#1937).

#### New Features
* Add constrained synthetic test functions `PressureVesselDesign`, `WeldedBeam`, `SpeedReducer`, and `TensionCompressionString` (#1832).
* Support decoupled fantasization (#1853) and decoupled evaluations in cost-aware utilities (#1949).
* Add `PairwiseBayesianActiveLearningByDisagreement`, an active learning acquisition function for PBO and BOPE (#1855).
* Support custom mean and likelihood in `MultiTaskGP` (#1909).
* Enable candidate generation (via `optimize_acqf`) with both `non_linear_constraints` and `fixed_features` (#1912).
* Introduce `L0PenaltyApproxObjective` to support L0 regularization (#1916).
* Enable batching in `PriorGuidedAcquisitionFunction` (#1925).

#### Other changes
* Deprecate `FixedNoiseMultiTaskGP`; allow `train_Yvar` optionally in `MultiTaskGP` (#1818).
* Implement `load_state_dict` for SAAS multi-task GP (#1825).
* Improvements to `LinearEllipticalSliceSampler` (#1859, #1878, #1879, #1883).
* Allow passing in task features as part of X in MTGP.posterior (#1868).
* Improve numerical stability of log densities in pairwise GPs (#1919).
* Python 3.11 compliance (#1927).
* Enable using constraints with `SampleReducingMCAcquisitionFunction`s when using `input_constructor`s and `get_acquisition_function` (#1932).
* Enable use of `qLogExpectedImprovement` and `qLogNoisyExpectedImprovement` with Ax (#1941).

#### Bug Fixes
* Enable pathwise sampling modules to be converted to GPU  (#1821).
* Allow `Standardize` modules to be loaded once trained (#1874).
* Fix memory leak in Inducing Point Allocators (#1890).
* Correct einsum computation in `LCEAKernel` (#1918).
* Properly whiten bounds in MVNXPB (#1933).
* Make `FixedFeatureAcquisitionFunction` convert floats to double-precision tensors rather than single-precision (#1944).
* Fix memory leak in `FullyBayesianPosterior` (#1951).
* Make `AnalyticExpectedUtilityOfBestOption` input constructor work correctionly with multi-task GPs (#1955).


## [0.8.5] - May 8, 2023

#### New Features
* Support inferred noise in `SaasFullyBayesianMultiTaskGP` (#1809).

#### Other Changes
* More informative error message when `Standardize` has wrong batch shape (#1807).
* Make GIBBON robust to numerical instability (#1814).
* Add `sample_multiplier` in EUBO's `acqf_input_constructor` (#1816).

#### Bug Fixes
* Only do checks for `_optimize_acqf_sequential_q` when it will be used (#1808).
* Fix an issue where `PairwiseGP` comparisons might be implicitly modified (#1811).


## [0.8.4] - Apr 24, 2023

#### Compatibility
* Require GPyTorch == 1.10 and linear_operator == 0.4.0 (#1803).

#### New Features
* Polytope sampling for linear constraints along the q-dimension (#1757).
* Single-objective joint entropy search with additional conditioning, various improvements to entropy-based acquisition functions (#1738).

#### Other changes
* Various updates to improve numerical stability of `PairwiseGP` (#1754, #1755).
* Change batch range for `FullyBayesianPosterior` (1176a38352b69d01def0a466233e6633c17d6862, #1773).
* Make `gen_batch_initial_conditions` more flexible (#1779).
* Deprecate `objective` in favor of `posterior_transform` for `MultiObjectiveAnalyticAcquisitionFunction` (#1781).
* Use `prune_baseline=True` as default for `qNoisyExpectedImprovement` (#1796).
* Add `batch_shape` property to `SingleTaskVariationalGP` (#1799).
* Change minimum inferred noise level for `SaasFullyBayesianSingleTaskGP` (#1800).

#### Bug fixes
* Add `output_task` to `MultiTaskGP.construct_inputs` (#1753).
* Fix custom bounds handling in test problems (#1760).
* Remove incorrect `BotorchTensorDimensionWarning` (#1790).
* Fix handling of non-Container-typed positional arguments in `SupervisedDatasetMeta` (#1663).


## [0.8.3] - Mar 15, 2023

#### New Features
* Add BAxUS tutorial (#1559).

#### Other changes
* Various improvements to tutorials (#1703, #1706, #1707, #1708, #1710, #1711, #1718, #1719, #1739, #1740, #1742).
* Allow tensor input for `integer_indices` in `Round` transform (#1709).
* Expose `cache_root` in qNEHVI input constructor (#1730).
* Add `get_init_args` helper to `Normalize` & `Round` transforms (#1731).
* Allowing custom dimensionality and improved gradient stability in `ModifiedFixedSingleSampleModel` (#1732).

#### Bug fixes
* Improve batched model handling in `_verify_output_shape` (#1715).
* Fix qNEI with Derivative Enabled BO (#1716).
* Fix `get_infeasible_cost` for objectives that require X (#1721).


## [0.8.2] - Feb 23, 2023

#### Compatibility
* Require PyTorch >= 1.12 (#1699).

#### New Features
* Introduce pathwise sampling API for efficiently sampling functions from (approximate) GP priors and posteriors (#1463).
* Add `OneHotToNumeric` input transform (#1517).
* Add `get_rounding_input_transform` utility for constructing rounding input transforms (#1531).
* Introduce `EnsemblePosterior` (#1636).
* Inducing Point Allocators for Sparse GPs (#1652).
* Pass `gen_candidates` callable in `optimize_acqf` (#1655).
* Adding `logmeanexp` and `logdiffexp` numerical utilities (#1657).

#### Other changes
* Warn if inoperable keyword arguments are passed to optimizers (#1421).
* Add `BotorchTestCase.assertAllClose` (#1618).
* Add `sample_shape` property to `ListSampler` (#1624).
* Do not filter out `BoTorchWarning`s by default (#1630).
* Introduce a `DeterministicSampler` (#1641).
* Warn when optimizer kwargs are being ignored in BoTorch optim utils `_filter_kwargs` (#1645).
* Don't use `functools.lru_cache` on methods (#1650).
* More informative error when someone adds a module without updating the corresponding rst file (#1653).
* Make indices a buffer in `AffineInputTransform` (#1656).
* Clean up `optimize_acqf` and `_make_linear_constraints` (#1660, #1676).
* Support NaN `max_reference_point` in `infer_reference_point` (#1671).
* Use `_fast_solves` in `HOGP.posterior` (#1682).
* Approximate qPI using `MVNXPB` (#1684).
* Improve filtering for `cache_root` in `CachedCholeskyMCAcquisitionFunction` (#1688).
* Add option to disable retrying on optimization warning (#1696).

#### Bug fixes
* Fix normalization in Chebyshev scalarization (#1616).
* Fix `TransformedPosterior` missing batch shape error in `_update_base_samples` (#1625).
* Detach `coefficient` and `offset` in `AffineTransform` in eval mode (#1642).
* Fix pickle error in `TorchPosterior` (#1644).
* Fix shape error in `optimize_acqf_cyclic` (#1648).
* Fixed bug where `optimize_acqf` didn't work with different batch sizes (#1668).
* Fix EUBO optimization error when two Xs are identical (#1670).
* Bug fix: `_filter_kwargs` was erroring when provided a function without a `__name__` attribute (#1678).


## [0.8.1] - Jan 5, 2023

### Highlights
* This release includes changes for compatibility with the newest versions of linear_operator and gpytorch.
* Several acquisition functions now have "Log" counterparts, which provide better
numerical behavior for improvement-based acquisition functions in areas where the probability of
improvement is low. For example, `LogExpectedImprovement` (#1565) should behave better than
`ExpectedImprovement`. These new acquisition functions are
    * `LogExpectedImprovement` (#1565).
    * `LogNoisyExpectedImprovement` (#1577).
    * `LogProbabilityOfImprovement` (#1594).
    * `LogConstrainedExpectedImprovement` (#1594).
* Bug fix: Stop `ModelListGP.posterior` from quietly ignoring `Log`, `Power`, and `Bilog` outcome transforms (#1563).
* Turn off `fast_computations` setting in linear_operator by default (#1547).

#### Compatibility
* Require linear_operator == 0.3.0 (#1538).
* Require pyro-ppl >= 1.8.4 (#1606).
* Require gpytorch == 1.9.1 (#1612).

#### New Features
* Add `eta` to `get_acquisition_function` (#1541).
* Support 0d-features in `FixedFeatureAcquisitionFunction` (#1546).
* Add timeout ability to optimization functions (#1562, #1598).
* Add `MultiModelAcquisitionFunction`, an abstract base class for acquisition functions that require multiple types of models (#1584).
* Add `cache_root` option for qNEI in `get_acquisition_function` (#1608).

#### Other changes
* Docstring corrections (#1551, #1557, #1573).
* Removal of `_fit_multioutput_independent` and `allclose_mll` (#1570).
* Better numerical behavior for fully Bayesian models (#1576).
* More verbose Scipy `minimize` failure messages (#1579).
* Lower-bound noise in`SaasPyroModel` to avoid Cholesky errors (#1586).

#### Bug fixes
* Error rather than failing silently for NaN values in box decomposition (#1554).
* Make `get_bounds_as_ndarray` device-safe (#1567).


## [0.8.0] - Dec 6, 2022

### Highlights
This release includes some backwards incompatible changes.
* Refactor `Posterior` and `MCSampler` modules to better support non-Gaussian distributions in BoTorch (#1486).
    * Introduced a `TorchPosterior` object that wraps a PyTorch `Distribution` object and makes it compatible with the rest of `Posterior` API.
    * `PosteriorList` no longer accepts Gaussian base samples. It should be used with a `ListSampler` that includes the appropriate sampler for each posterior.
    * The MC acquisition functions no longer construct a Sobol sampler by default. Instead, they rely on a `get_sampler` helper, which dispatches an appropriate sampler based on the posterior provided.
    * The `resample` and `collapse_batch_dims` arguments to `MCSampler`s have been removed. The `ForkedRNGSampler` and `StochasticSampler` can be used to get the same functionality.
    * Refer to the PR for additional changes. We will update the website documentation to reflect these changes in a future release.
* #1191 refactors much of `botorch.optim` to operate based on closures that abstract
away how losses (and gradients) are computed. By default, these closures are created
using multiply-dispatched factory functions (such as `get_loss_closure`), which may be
customized by registering methods with an associated dispatcher (e.g. `GetLossClosure`).
Future releases will contain tutorials that explore these features in greater detail.

#### New Features
* Add mixed optimization for list optimization (#1342).
* Add entropy search acquisition functions (#1458).
* Add utilities for straight-through gradient estimators for discretization functions (#1515).
* Add support for categoricals in Round input transform and use STEs (#1516).
* Add closure-based optimizers (#1191).

#### Other Changes
* Do not count hitting maxiter as optimization failure & update default maxiter (#1478).
* `BoxDecomposition` cleanup (#1490).
* Deprecate `torch.triangular_solve` in favor of `torch.linalg.solve_triangular` (#1494).
* Various docstring improvements (#1496, #1499, #1504).
* Remove `__getitem__` method from `LinearTruncatedFidelityKernel` (#1501).
* Handle Cholesky errors when fitting a fully Bayesian model (#1507).
* Make eta configurable in `apply_constraints` (#1526).
* Support SAAS ensemble models in RFFs (#1530).
* Deprecate `botorch.optim.numpy_converter` (#1191).
* Deprecate `fit_gpytorch_scipy` and `fit_gpytorch_torch` (#1191).

#### Bug Fixes
* Enforce use of float64 in `NdarrayOptimizationClosure` (#1508).
* Replace deprecated np.bool with equivalent bool (#1524).
* Fix RFF bug when using FixedNoiseGP models (#1528).


## [0.7.3] - Nov 10, 2022

### Highlights
* #1454 fixes a critical bug that affected multi-output `BatchedMultiOutputGPyTorchModel`s that were using a `Normalize` or `InputStandardize` input transform and trained using `fit_gpytorch_model/mll` with `sequential=True` (which was the default until 0.7.3). The input transform buffers would be reset after model training, leading to the model being trained on normalized input data but evaluated on raw inputs. This bug had been affecting model fits since the 0.6.5 release.
* #1479 changes the inheritance structure of `Model`s in a backwards-incompatible way. If your code relies on `isinstance` checks with BoTorch `Model`s, especially `SingleTaskGP`, you should revisit these checks to make sure they still work as expected.

#### Compatibility
* Require linear_operator == 0.2.0 (#1491).

#### New Features
* Introduce `bvn`, `MVNXPB`, `TruncatedMultivariateNormal`, and `UnifiedSkewNormal` classes / methods (#1394, #1408).
* Introduce `AffineInputTransform` (#1461).
* Introduce a `subset_transform` decorator to consolidate subsetting of inputs in input transforms (#1468).

#### Other Changes
* Add a warning when using float dtype (#1193).
* Let Pyre know that `AcquisitionFunction.model` is a `Model` (#1216).
* Remove custom `BlockDiagLazyTensor` logic when using `Standardize` (#1414).
* Expose `_aug_batch_shape` in `SaasFullyBayesianSingleTaskGP` (#1448).
* Adjust `PairwiseGP` `ScaleKernel` prior (#1460).
* Pull out `fantasize` method into a `FantasizeMixin` class, so it isn't so widely inherited (#1462, #1479).
* Don't use Pyro JIT by default , since it was causing a memory leak (#1474).
* Use `get_default_partitioning_alpha` for NEHVI input constructor (#1481).

#### Bug Fixes
* Fix `batch_shape` property of `ModelListGPyTorchModel` (#1441).
* Tutorial fixes (#1446, #1475).
* Bug-fix for Proximal acquisition function wrapper for negative base acquisition functions (#1447).
* Handle `RuntimeError` due to constraint violation while sampling from priors (#1451).
* Fix bug in model list with output indices (#1453).
* Fix input transform bug when sequentially training a `BatchedMultiOutputGPyTorchModel` (#1454).
* Fix a bug in `_fit_multioutput_independent` that failed mll comparison (#1455).
* Fix box decomposition behavior with empty or None `Y` (#1489).


## [0.7.2] - Sep 27, 2022

#### New Features
* A full refactor of model fitting methods (#1134).
  * This introduces a new `fit_gpytorch_mll` method that multiple-dispatches
    on the model type. Users may register custom fitting routines for different
    combinations of MLLs, Likelihoods, and Models.
  * Unlike previous fitting helpers, `fit_gpytorch_mll` does **not** pass
   `kwargs` to `optimizer` and instead introduces an optional `optimizer_kwargs`
    argument.
  * When a model fitting attempt fails, `botorch.fit` methods restore modules to their
    original states.
  * `fit_gpytorch_mll` throws a `ModelFittingError` when all model fitting attempts fail.
  * Upon returning from `fit_gpytorch_mll`, `mll.training` will be `True` if fitting failed
    and `False` otherwise.
* Allow custom bounds to be passed in to `SyntheticTestFunction` (#1415).

#### Deprecations
* Deprecate weights argument of risk measures in favor of a `preprocessing_function` (#1400),
* Deprecate `fit_gyptorch_model`; to be superseded by `fit_gpytorch_mll`.

#### Other Changes
* Support risk measures in MOO input constructors (#1401).

#### Bug Fixes
* Fix fully Bayesian state dict loading when there are more than 10 models (#1405).
* Fix `batch_shape` property of `SaasFullyBayesianSingleTaskGP` (#1413).
* Fix `model_list_to_batched` ignoring the `covar_module` of the input models (#1419).


## [0.7.1] - Sep 13, 2022

#### Compatibility
* Pin GPyTorch >= 1.9.0 (#1397).
* Pin linear_operator == 0.1.1 (#1397).

#### New Features
* Implement `SaasFullyBayesianMultiTaskGP` and related utilities (#1181, #1203).

#### Other Changes
* Support loading a state dict for `SaasFullyBayesianSingleTaskGP` (#1120).
* Update `load_state_dict` for `ModelList` to support fully Bayesian models (#1395).
* Add `is_one_to_many` attribute to input transforms (#1396).

#### Bug Fixes
* Fix `PairwiseGP` on GPU (#1388).


## [0.7.0] - Sep 7, 2022

#### Compatibility
* Require python >= 3.8 (via #1347).
* Support for python 3.10 (via #1379).
* Require PyTorch >= 1.11 (via (#1363).
* Require GPyTorch >= 1.9.0 (#1347).
  * GPyTorch 1.9.0 is a major refactor that factors out the lazy tensor
  functionality into a new `LinearOperator` library, which required
  a number of adjustments to BoTorch (#1363, #1377).
* Require pyro >= 1.8.2 (#1379).

#### New Features
* Add ability to generate the features appended in the `AppendFeatures` input
transform via a generic callable (#1354).
* Add new synthetic test functions for sensitivity analysis (#1355, #1361).

#### Other Changes
* Use `time.monotonic()` instead of `time.time()` to measure duration (#1353).
* Allow passing `Y_samples` directly in `MARS.set_baseline_Y` (#1364).

#### Bug Fixes
* Patch `state_dict` loading for `PairwiseGP` (#1359).
* Fix `batch_shape` handling in `Normalize` and `InputStandardize` transforms (#1360).


## [0.6.6] - Aug 12, 2022

#### Compatibility
* Require GPyTorch >= 1.8.1 (#1347).

#### New Features
* Support batched models in `RandomFourierFeatures` (#1336).
* Add a `skip_expand` option to `AppendFeatures` (#1344).

#### Other Changes
* Allow `qProbabilityOfImprovement` to use batch-shaped `best_f` (#1324).
* Make `optimize_acqf` re-attempt failed optimization runs and handle optimization
errors in `optimize_acqf` and `gen_candidates_scipy` better (#1325).
* Reduce memory overhead in `MARS.set_baseline_Y` (#1346).

#### Bug Fixes
* Fix bug where `outcome_transform` was ignored for `ModelListGP.fantasize` (#1338).
* Fix bug causing `get_polytope_samples` to sample incorrectly when variables
live in multiple dimensions (#1341).

#### Documentation
* Add more descriptive docstrings for models (#1327, #1328, #1329, #1330) and for other
classes (#1313).
* Expanded on the model documentation at [botorch.org/docs/models](https://botorch.org/docs/models) (#1337).

## [0.6.5] - Jul 15, 2022

#### Compatibility
* Require PyTorch >=1.10 (#1293).
* Require GPyTorch >=1.7 (#1293).

#### New Features
* Add MOMF (Multi-Objective Multi-Fidelity) acquisition function (#1153).
* Support `PairwiseLogitLikelihood` and modularize `PairwiseGP` (#1193).
* Add in transformed weighting flag to Proximal Acquisition function (#1194).
* Add `FeasibilityWeightedMCMultiOutputObjective` (#1202).
* Add outcome_transform to `FixedNoiseMultiTaskGP` (#1255).
* Support Scalable Constrained Bayesian Optimization (#1257).
* Support `SaasFullyBayesianSingleTaskGP` in `prune_inferior_points` (#1260).
* Implement MARS as a risk measure (#1303).
* Add MARS tutorial (#1305).

#### Other Changes
* Add `Bilog` outcome transform (#1189).
* Make `get_infeasible_cost` return a cost value for each outcome (#1191).
* Modify risk measures to accept `List[float]` for weights (#1197).
* Support `SaasFullyBayesianSingleTaskGP` in prune_inferior_points_multi_objective (#1204).
* BotorchContainers and BotorchDatasets: Large refactor of the original `TrainingData` API to allow for more diverse types of datasets (#1205, #1221).
* Proximal biasing support for multi-output `SingleTaskGP` models (#1212).
* Improve error handling in `optimize_acqf_discrete` with a check that `choices` is non-empty  (#1228).
* Handle `X_pending` properly in `FixedFeatureAcquisition` (#1233, #1234).
* PE and PLBO support in Ax (#1240, #1241).
* Remove `model.train` call from `get_X_baseline` for better caching (#1289).
* Support `inf` values in `bounds` argument of `optimize_acqf` (#1302).

#### Bug Fixes
* Update `get_gp_samples` to support input / outcome transforms (#1201).
* Fix cached Cholesky sampling in `qNEHVI` when using `Standardize` outcome transform (#1215).
* Make `task_feature` as required input in `MultiTaskGP.construct_inputs` (#1246).
* Fix CUDA tests (#1253).
* Fix `FixedSingleSampleModel` dtype/device conversion (#1254).
* Prevent inappropriate transforms by putting input transforms into train mode before converting models (#1283).
* Fix `sample_points_around_best` when using 20 dimensional inputs or `prob_perturb` (#1290).
* Skip bound validation in `optimize_acqf` if inequality constraints are specified (#1297).
* Properly handle RFFs when used with a `ModelList` with individual transforms (#1299).
* Update `PosteriorList` to support deterministic-only models and fix `event_shape` (#1300).

#### Documentation
* Add a note about observation noise in the posterior in `fit_model_with_torch_optimizer` notebook (#1196).
* Fix custom botorch model in Ax tutorial to support new interface (#1213).
* Update MOO docs (#1242).
* Add SMOKE_TEST option to MOMF tutorial (#1243).
* Fix `ModelListGP.condition_on_observations`/`fantasize` bug (#1250).
* Replace space with underscore for proper doc generation (#1256).
* Update PBO tutorial to use EUBO (#1262).


## [0.6.4] - Apr 21, 2022

#### New Features
* Implement `ExpectationPosteriorTransform` (#903).
* Add `PairwiseMCPosteriorVariance`, a cheap active learning acquisition function (#1125).
* Support computing quantiles in the fully Bayesian posterior, add `FullyBayesianPosteriorList` (#1161).
* Add expectation risk measures (#1173).
* Implement Multi-Fidelity GIBBON (Lower Bound MES) acquisition function (#1185).

#### Other Changes
* Add an error message for one shot acquisition functions in `optimize_acqf_discrete` (#939).
* Validate the shape of the `bounds` argument in `optimize_acqf` (#1142).
* Minor tweaks to `SAASBO` (#1143, #1183).
* Minor updates to tutorials (24f7fda7b40d4aabf502c1a67816ac1951af8c23, #1144, #1148, #1159, #1172, #1180).
* Make it easier to specify a custom `PyroModel` (#1149).
* Allow passing in a `mean_module` to `SingleTaskGP/FixedNoiseGP` (#1160).
* Add a note about acquisitions using gradients to base class (#1168).
* Remove deprecated `box_decomposition` module (#1175).

#### Bug Fixes
* Bug-fixes for `ProximalAcquisitionFunction` (#1122).
* Fix missing warnings on failed optimization in `fit_gpytorch_scipy` (#1170).
* Ignore data related buffers in `PairwiseGP.load_state_dict` (#1171).
* Make `fit_gpytorch_model` properly honor the `debug` flag (#1178).
* Fix missing `posterior_transform` in `gen_one_shot_kg_initial_conditions` (#1187).


## [0.6.3] - Mar 28, 2022

#### New Features
* Implement SAASBO - `SaasFullyBayesianSingleTaskGP` model for sample-efficient high-dimensional Bayesian optimization (#1123).
* Add SAASBO tutorial (#1127).
* Add `LearnedObjective` (#1131), `AnalyticExpectedUtilityOfBestOption` acquisition function (#1135), and a few auxiliary classes to support Bayesian optimization with preference exploration (BOPE).
* Add BOPE tutorial (#1138).

#### Other Changes
* Use `qKG.evaluate` in `optimize_acqf_mixed` (#1133).
* Add `construct_inputs` to SAASBO (#1136).

#### Bug Fixes
* Fix "Constraint Active Search" tutorial (#1124).
* Update "Discrete Multi-Fidelity BO" tutorial (#1134).


## [0.6.2] - Mar 9, 2022

#### New Features
* Use `BOTORCH_MODULAR` in tutorials with Ax (#1105).
* Add `optimize_acqf_discrete_local_search` for discrete search spaces (#1111).

#### Bug Fixes
* Fix missing `posterior_transform` in qNEI and `get_acquisition_function` (#1113).


## [0.6.1] - Feb 28, 2022

#### New Features
* Add `Standardize` input transform (#1053).
* Low-rank Cholesky updates for NEI (#1056).
* Add support for non-linear input constraints (#1067).
* New MOO problems: MW7 (#1077), disc brake (#1078), penicillin (#1079), RobustToy (#1082), GMM (#1083).

#### Other Changes
* Support multi-output models in MES using `PosteriorTransform` (#904).
* Add `Dispatcher` (#1009).
* Modify qNEHVI to support deterministic models (#1026).
* Store tensor attributes of input transforms as buffers (#1035).
* Modify NEHVI to support MTGPs (#1037).
* Make `Normalize` input transform input column-specific (#1047).
* Improve `find_interior_point` (#1049).
* Remove deprecated `botorch.distributions` module (#1061).
* Avoid costly application of posterior transform in Kronecker & HOGP models (#1076).
* Support heteroscedastic perturbations in `InputPerturbations` (#1088).

#### Performance Improvements
* Make risk measures more memory efficient (#1034).

#### Bug Fixes
* Properly handle empty `fixed_features` in optimization (#1029).
* Fix missing weights in `VaR` risk measure (#1038).
* Fix `find_interior_point` for negative variables & allow unbounded problems (#1045).
* Filter out indefinite bounds in constraint utilities (#1048).
* Make non-interleaved base samples use intuitive shape (#1057).
* Pad small diagonalization with zeros for `KroneckerMultitaskGP` (#1071).
* Disable learning of bounds in `preprocess_transform` (#1089).
* Fix `gen_candidates_torch` (4079164489613d436d19c7b2df97677d97dfa8dc).
* Catch runtime errors with ill-conditioned covar (#1095).
* Fix `compare_mc_analytic_acquisition` tutorial (#1099).


## [0.6.0] - Dec 8, 2021

#### Compatibility
* Require PyTorch >=1.9 (#1011).
* Require GPyTorch >=1.6 (#1011).

#### New Features
* New `ApproximateGPyTorchModel` wrapper for various (variational) approximate GP models (#1012).
* New `SingleTaskVariationalGP` stochastic variational Gaussian Process model (#1012).
* Support for Multi-Output Risk Measures (#906, #965).
* Introduce `ModelList` and `PosteriorList` (#829).
* New Constraint Active Search tutorial (#1010).
* Add additional multi-objective optimization test problems (#958).

#### Other Changes
* Add `covar_module` as an optional input of `MultiTaskGP` models (#941).
* Add `min_range` argument to `Normalize` transform to prevent division by zero (#931).
* Add initialization heuristic for acquisition function optimization that samples around best points (#987).
* Update initialization heuristic to perturb a subset of the dimensions of the best points if the dimension is > 20 (#988).
* Modify `apply_constraints` utility to work with multi-output objectives (#994).
* Short-cut `t_batch_mode_transform` decorator on non-tensor inputs (#991).

#### Performance Improvements
* Use lazy covariance matrix in `BatchedMultiOutputGPyTorchModel.posterior` (#976).
* Fast low-rank Cholesky updates for `qNoisyExpectedHypervolumeImprovement` (#747, #995, #996).

#### Bug Fixes
* Update error handling to new PyTorch linear algebra messages (#940).
* Avoid test failures on Ampere devices (#944).
* Fixes to the `Griewank` test function (#972).
* Handle empty base_sample_shape in `Posterior.rsample` (#986).
* Handle `NotPSDError` and hitting `maxiter` in `fit_gpytorch_model` (#1007).
* Use TransformedPosterior for subclasses of GPyTorchPosterior (#983).
* Propagate `best_f` argument to `qProbabilityOfImprovement` in input constructors (f5a5f8b6dc20413e67c6234e31783ac340797a8d).


## [0.5.1] - Sep 2, 2021

#### Compatibility
* Require GPyTorch >=1.5.1 (#928).

#### New Features
* Add `HigherOrderGP` composite Bayesian Optimization tutorial notebook (#864).
* Add Multi-Task Bayesian Optimziation tutorial (#867).
* New multi-objective test problems from (#876).
* Add `PenalizedMCObjective` and `L1PenaltyObjective` (#913).
* Add a `ProximalAcquisitionFunction` for regularizing new candidates towards previously generated ones (#919, #924).
* Add a `Power` outcome transform (#925).

#### Bug Fixes
* Batch mode fix for `HigherOrderGP` initialization (#856).
* Improve `CategoricalKernel` precision (#857).
* Fix an issue with `qMultiFidelityKnowledgeGradient.evaluate` (#858).
* Fix an issue with transforms with `HigherOrderGP`. (#889)
* Fix initial candidate generation when parameter constraints are on different device (#897).
* Fix bad in-place op in `_generate_unfixed_lin_constraints` (#901).
* Fix an input transform bug in `fantasize` call (#902).
* Fix outcome transform bug in `batched_to_model_list` (#917).

#### Other Changes
* Make variance optional for `TransformedPosterior.mean` (#855).
* Support transforms in `DeterministicModel` (#869).
* Support `batch_shape` in `RandomFourierFeatures` (#877).
* Add a `maximize` flag to `PosteriorMean` (#881).
* Ignore categorical dimensions when validating training inputs in `MixedSingleTaskGP` (#882).
* Refactor `HigherOrderGPPosterior` for memory efficiency (#883).
* Support negative weights for minimization objectives in `get_chebyshev_scalarization` (#884).
* Move `train_inputs` transforms to `model.train/eval` calls (#894).


## [0.5.0] - Jun 29, 2021

#### Compatibility
* Require PyTorch >=1.8.1 (#832).
* Require GPyTorch >=1.5 (#848).
* Changes to how input transforms are applied: `transform_inputs` is applied in `model.forward` if the model is in `train` mode, otherwise it is applied in the `posterior` call (#819, #835).

#### New Features
* Improved multi-objective optimization capabilities:
  * `qNoisyExpectedHypervolumeImprovement` acquisition function that improves on `qExpectedHypervolumeImprovement` in terms of tolerating observation noise and speeding up computation for large `q`-batches (#797, #822).
  * `qMultiObjectiveMaxValueEntropy` acqusition function (913aa0e510dde10568c2b4b911124cdd626f6905, #760).
  * Heuristic for reference point selection (#830).
  * `FastNondominatedPartitioning` for Hypervolume computations (#699).
  * `DominatedPartitioning` for partitioning the dominated space (#726).
  * `BoxDecompositionList` for handling box decompositions of varying sizes (#712).
  * Direct, batched dominated partitioning for the two-outcome case (#739).
  * `get_default_partitioning_alpha` utility providing heuristic for selecting approximation level for partitioning algorithms (#793).
  * New method for computing Pareto Frontiers with less memory overhead (#842, #846).
* New `qLowerBoundMaxValueEntropy` acquisition function (a.k.a. GIBBON), a lightweight variant of Multi-fidelity Max-Value Entropy Search using a Determinantal Point Process approximation (#724, #737, #749).
* Support for discrete and mixed input domains:
  * `CategoricalKernel` for categorical inputs (#771).
  * `MixedSingleTaskGP` for mixed search spaces (containing both categorical and ordinal parameters) (#772, #847).
  * `optimize_acqf_discrete` for optimizing acquisition functions over fully discrete domains (#777).
  * Extend `optimize_acqf_mixed` to allow batch optimization (#804).
* Support for robust / risk-aware optimization:
  * Risk measures for robust / risk-averse optimization (#821).
  * `AppendFeatures` transform (#820).
  * `InputPerturbation` input transform for for risk averse BO with implementation errors (#827).
  * Tutorial notebook for Bayesian Optimization of risk measures (#823).
  * Tutorial notebook for risk-averse Bayesian Optimization under input perturbations (#828).
* More scalable multi-task modeling and sampling:
  * `KroneckerMultiTaskGP` model for efficient multi-task modeling for block-design settings (all tasks observed at all inputs) (#637).
  * Support for transforms in Multi-Task GP models (#681).
  * Posterior sampling based on Matheron's rule for Multi-Task GP models (#841).
* Various changes to simplify and streamline integration with Ax:
  * Handle non-block designs in `TrainingData` (#794).
  * Acquisition function input constructor registry (#788, #802, #845).
* Random Fourier Feature (RFF) utilties for fast (approximate) GP function sampling (#750).
* `DelaunayPolytopeSampler` for fast uniform sampling from (simple) polytopes (#741).
* Add `evaluate` method to `ScalarizedObjective` (#795).

#### Bug Fixes
* Handle the case when all features are fixed in `optimize_acqf` (#770).
* Pass `fixed_features` to initial candidate generation functions (#806).
* Handle batch empty pareto frontier in `FastPartitioning` (#740).
* Handle empty pareto set in `is_non_dominated` (#743).
* Handle edge case of no or a single observation in `get_chebyshev_scalarization` (#762).
* Fix an issue in `gen_candidates_torch` that caused problems with acqusition functions using fantasy models (#766).
* Fix `HigherOrderGP` `dtype` bug (#728).
* Normalize before clamping in `Warp` input warping transform (#722).
* Fix bug in GP sampling (#764).

#### Other Changes
* Modify input transforms to support one-to-many transforms (#819, #835).
* Make initial conditions for acquisition function optimization honor parameter constraints (#752).
* Perform optimization only over unfixed features if `fixed_features` is passed (#839).
* Refactor Max Value Entropy Search Methods (#734).
* Use Linear Algebra functions from the `torch.linalg` module (#735).
* Use PyTorch's `Kumaraswamy` distribution (#746).
* Improved capabilities and some bugfixes for batched models (#723, #767).
* Pass `callback` argument to `scipy.optim.minimize` in `gen_candidates_scipy` (#744).
* Modify behavior of `X_pending` in in multi-objective acqusiition functions (#747).
* Allow multi-dimensional batch shapes in test functions (#757).
* Utility for converting batched multi-output models into batched single-output models (#759).
* Explicitly raise `NotPSDError` in `_scipy_objective_and_grad` (#787).
* Make `raw_samples` optional if `batch_initial_conditions` is passed (#801).
* Use powers of 2 in qMC docstrings & examples (#812).


## [0.4.0] - Feb 23, 2021

#### Compatibility
* Require PyTorch >=1.7.1 (#714).
* Require GPyTorch >=1.4 (#714).

#### New Features
* `HigherOrderGP` - High-Order Gaussian Process (HOGP) model for
  high-dimensional output regression (#631, #646, #648, #680).
* `qMultiStepLookahead` acquisition function for general look-ahead
  optimization approaches (#611, #659).
* `ScalarizedPosteriorMean` and `project_to_sample_points` for more
  advanced MFKG functionality (#645).
* Large-scale Thompson sampling tutorial (#654, #713).
* Tutorial for optimizing mixed continuous/discrete domains (application
  to multi-fidelity KG with discrete fidelities) (#716).
* `GPDraw` utility for sampling from (exact) GP priors (#655).
* Add `X` as optional arg to call signature of `MCAcqusitionObjective` (#487).
* `OSY` synthetic test problem (#679).

#### Bug Fixes
* Fix matrix multiplication in `scalarize_posterior` (#638).
* Set `X_pending` in `get_acquisition_function` in `qEHVI` (#662).
* Make contextual kernel device-aware (#666).
* Do not use an `MCSampler` in `MaxPosteriorSampling` (#701).
* Add ability to subset outcome transforms (#711).

#### Performance Improvements
* Batchify box decomposition for 2d case (#642).

#### Other Changes
* Use scipy distribution in MES quantile bisect (#633).
* Use new closure definition for GPyTorch priors (#634).
* Allow enabling of approximate root decomposition in `posterior` calls (#652).
* Support for upcoming 21201-dimensional PyTorch `SobolEngine` (#672, #674).
* Refactored various MOO utilities to allow future additions (#656, #657, #658, #661).
* Support input_transform in PairwiseGP (#632).
* Output shape checks for t_batch_mode_transform (#577).
* Check for NaN in `gen_candidates_scipy` (#688).
* Introduce `base_sample_shape` property to `Posterior` objects (#718).


## [0.3.3] - Dec 8, 2020

Contextual Bayesian Optimization, Input Warping, TuRBO, sampling from polytopes.

#### Compatibility
* Require PyTorch >=1.7 (#614).
* Require GPyTorch >=1.3 (#614).

#### New Features
* Models (LCE-A, LCE-M and SAC ) for Contextual Bayesian Optimziation (#581).
   * Implements core models from:
     [High-Dimensional Contextual Policy Search with Unknown Context Rewards using Bayesian Optimization](https://proceedings.neurips.cc/paper/2020/hash/faff959d885ec0ecf70741a846c34d1d-Abstract.html).
      Q. Feng, B. Letham, H. Mao, E. Bakshy. NeurIPS 2020.
    * See Ax for usage of these models.
* Hit and run sampler for uniform sampling from a polytope (#592).
* Input warping:
  * Core functionality (#607).
  * Kumaraswamy Distribution (#606).
  * Tutorial (8f34871652042219c57b799669a679aab5eed7e3).
* TuRBO-1 tutorial (#598).
  * Implements the method from [Scalable Global Optimization via
Local Bayesian Optimization](https://proceedings.neurips.cc/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf).
    D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, M. Poloczek. NeurIPS 2019.

#### Bug fixes
* Fix bounds of `HolderTable` synthetic function (#596).
* Fix `device` issue in MOO tutorial (#621).

#### Other changes
* Add `train_inputs` option to `qMaxValueEntropy` (#593).
* Enable gpytorch settings to override BoTorch defaults for `fast_pred_var` and `debug` (#595).
* Rename `set_train_data_transform` -> `preprocess_transform` (#575).
* Modify `_expand_bounds()` shape checks to work with >2-dim bounds (#604).
* Add `batch_shape` property to models (#588).
* Modify `qMultiFidelityKnowledgeGradient.evaluate()` to work with `project`, `expand` and `cost_aware_utility` (#594).
* Add list of papers using BoTorch to website docs (#617).


## [0.3.2] - Oct 23, 2020

Maintenance Release

#### New Features
* Add `PenalizedAcquisitionFunction` wrapper (#585)
* Input transforms
  * Reversible input transform (#550)
  * Rounding input transform (#562)
  * Log input transform (#563)
* Differentiable approximate rounding for integers (#561)

#### Bug fixes
* Fix sign error in UCB when `maximize=False` (a4bfacbfb2109d3b89107d171d2101e1995822bb)
* Fix batch_range sample shape logic (#574)

#### Other changes
* Better support for two stage sampling in preference learning
  (0cd13d0cb49b1ac8d0971e42f1f0e9dd6126fd9a)
* Remove noise term in `PairwiseGP` and add `ScaleKernel` by default (#571)
* Rename `prior` to `task_covar_prior` in `MultiTaskGP` and `FixedNoiseMultiTaskGP`
  (8e42ea82856b165a7df9db2a9b6f43ebd7328fc4)
* Support only transforming inputs on training or evaluation (#551)
* Add `equals` method for `InputTransform` (#552)


## [0.3.1] - Sep 15, 2020

Maintenance Release

#### New Features
* Constrained Multi-Objective tutorial (#493)
* Multi-fidelity Knowledge Gradient tutorial (#509)
* Support for batch qMC sampling (#510)
* New `evaluate` method for `qKnowledgeGradient` (#515)

#### Compatibility
* Require PyTorch >=1.6 (#535)
* Require GPyTorch >=1.2 (#535)
* Remove deprecated `botorch.gen module` (#532)

#### Bug fixes
* Fix bad backward-indexing of task_feature in `MultiTaskGP` (#485)
* Fix bounds in constrained Branin-Currin test function (#491)
* Fix max_hv for C2DTLZ2 and make Hypervolume always return a float (#494)
* Fix bug in `draw_sobol_samples` that did not use the proper effective dimension (#505)
* Fix constraints for `q>1` in `qExpectedHypervolumeImprovement` (c80c4fdb0f83f0e4f12e4ec4090d0478b1a8b532)
* Only use feasible observations in partitioning for `qExpectedHypervolumeImprovement`
  in `get_acquisition_function` (#523)
* Improved GPU compatibility for `PairwiseGP` (#537)

#### Performance Improvements
* Reduce memory footprint in `qExpectedHypervolumeImprovement` (#522)
* Add `(q)ExpectedHypervolumeImprovement` to nonnegative functions
  [for better initialization] (#496)

#### Other changes
* Support batched `best_f` in `qExpectedImprovement` (#487)
* Allow to return full tree of solutions in `OneShotAcquisitionFunction` (#488)
* Added `construct_inputs` class method to models to programmatically construct the
  inputs to the constructor from a standardized `TrainingData` representation
  (#477, #482, 3621198d02195b723195b043e86738cd5c3b8e40)
* Acquisition function constructors now accept catch-all `**kwargs` options
  (#478, e5b69352954bb10df19a59efe9221a72932bfe6c)
* Use `psd_safe_cholesky` in `qMaxValueEntropy` for better numerical stabilty (#518)
* Added `WeightedMCMultiOutputObjective` (81d91fd2e115774e561c8282b724457233b6d49f)
* Add ability to specify `outcomes` to all multi-output objectives (#524)
* Return optimization output in `info_dict` for `fit_gpytorch_scipy` (#534)
* Use `setuptools_scm` for versioning (#539)


## [0.3.0] - July 6, 2020

Multi-Objective Bayesian Optimization

#### New Features
* Multi-Objective Acquisition Functions (#466)
  * q-Expected Hypervolume Improvement
  * q-ParEGO
  * Analytic Expected Hypervolume Improvement with auto-differentiation
* Multi-Objective Utilities (#466)
  * Pareto Computation
  * Hypervolume Calculation
  * Box Decomposition algorithm
* Multi-Objective Test Functions (#466)
  * Suite of synthetic test functions for multi-objective, constrained optimization
* Multi-Objective Tutorial (#468)
* Abstract ConstrainedBaseTestProblem (#454)
* Add optimize_acqf_list method for sequentially, greedily optimizing 1 candidate
  from each provided acquisition function (d10aec911b241b208c59c192beb9e4d572a092cd)

#### Bug fixes
* Fixed re-arranging mean in MultiTask MO models (#450).

#### Other changes
* Move gpt_posterior_settings into models.utils (#449)
* Allow specifications of batch dims to collapse in samplers (#457)
* Remove outcome transform before model-fitting for sequential model fitting
  in MO models (#458)


## [0.2.5] - May 14, 2020

Bugfix Release

#### Bug fixes
* Fixed issue with broken wheel build (#444).

#### Other changes
* Changed code style to use absolute imports throughout (#443).


## [0.2.4] - May 12, 2020

Bugfix Release

#### Bug fixes
* There was a mysterious issue with the 0.2.3 wheel on pypi, where part of the
  `botorch/optim/utils.py` file was not included, which resulted in an `ImportError` for
  many central components of the code. Interestingly, the source dist (built with the
  same command) did not have this issue.
* Preserve order in ChainedOutcomeTransform (#440).

#### New Features
* Utilities for estimating the feasible volume under outcome constraints (#437).


## [0.2.3] - Apr 27, 2020

Pairwise GP for Preference Learning, Sampling Strategies.

#### Compatibility
* Require PyTorch >=1.5 (#423).
* Require GPyTorch >=1.1.1 (#425).

#### New Features
* Add `PairwiseGP` for preference learning with pair-wise comparison data (#388).
* Add `SamplingStrategy` abstraction for sampling-based generation strategies, including
  `MaxPosteriorSampling` (i.e. Thompson Sampling) and `BoltzmannSampling` (#218, #407).

#### Deprecations
* The existing `botorch.gen` module is moved to `botorch.generation.gen` and imports
  from `botorch.gen` will raise a warning (an error in the next release) (#218).

#### Bug fixes
* Fix & update a number of tutorials (#394, #398, #393, #399, #403).
* Fix CUDA tests (#404).
* Fix sobol maxdim limitation in `prune_baseline` (#419).

#### Other changes
* Better stopping criteria for stochastic optimization (#392).
* Improve numerical stability of `LinearTruncatedFidelityKernel` (#409).
* Allow batched `best_f` in `qExpectedImprovement` and `qProbabilityOfImprovement`
  (#411).
* Introduce new logger framework (#412).
* Faster indexing in some situations (#414).
* More generic `BaseTestProblem` (9e604fe2188ac85294c143d249872415c4d95823).


## [0.2.2] - Mar 6, 2020

Require PyTorch 1.4, Python 3.7 and new features for active learning,
multi-fidelity optimization, and a number of bug fixes.

#### Compatibility
* Require PyTorch >=1.4 (#379).
* Require Python >=3.7 (#378).

#### New Features
* Add `qNegIntegratedPosteriorVariance` for Bayesian active learning (#377).
* Add `FixedNoiseMultiFidelityGP`, analogous to `SingleTaskMultiFidelityGP` (#386).
* Support `scalarize_posterior` for m>1 and q>1 posteriors (#374).
* Support `subset_output` method on multi-fidelity models (#372).
* Add utilities for sampling from simplex and hypersphere (#369).

#### Bug fixes
* Fix `TestLoader` local test discovery (#376).
* Fix batch-list conversion of `SingleTaskMultiFidelityGP` (#370).
* Validate tensor args before checking input scaling for more
  informative error messaages (#368).
* Fix flaky `qNoisyExpectedImprovement` test (#362).
* Fix test function in closed-loop tutorial (#360).
* Fix num_output attribute in BoTorch/Ax tutorial (#355).

#### Other changes
* Require output dimension in `MultiTaskGP` (#383).
* Update code of conduct (#380).
* Remove deprecated `joint_optimize` and `sequential_optimize` (#363).


## [0.2.1] - Jan 15, 2020

Minor bug fix release.

#### New Features
* Add a static method for getting batch shapes for batched MO models (#346).

#### Bug fixes
* Revamp qKG constructor to avoid issue with missing objective (#351).
* Make sure MVES can support sampled costs like KG (#352).

#### Other changes
* Allow custom module-to-array handling in fit_gpytorch_scipy (#341).


## [0.2.0] - Dec 20, 2019

Max-value entropy acquisition function, cost-aware / multi-fidelity optimization,
subsetting models, outcome transforms.

#### Compatibility
* Require PyTorch >=1.3.1 (#313).
* Require GPyTorch >=1.0 (#342).

#### New Features
* Add cost-aware KnowledgeGradient (`qMultiFidelityKnowledgeGradient`) for
  multi-fidelity optimization (#292).
* Add `qMaxValueEntropy` and `qMultiFidelityMaxValueEntropy` max-value entropy
  search acquisition functions (#298).
* Add `subset_output` functionality to (most) models (#324).
* Add outcome transforms and input transforms (#321).
* Add `outcome_transform` kwarg to model constructors for automatic outcome
  transformation and un-transformation (#327).
* Add cost-aware utilities for cost-sensitive acquisiiton functions (#289).
* Add `DeterminsticModel` and `DetermisticPosterior` abstractions (#288).
* Add `AffineFidelityCostModel` (f838eacb4258f570c3086d7cbd9aa3cf9ce67904).
* Add `project_to_target_fidelity` and `expand_trace_observations` utilties for
  use in multi-fidelity optimization (1ca12ac0736e39939fff650cae617680c1a16933).

#### Performance Improvements
* New `prune_baseline` option for pruning `X_baseline` in
  `qNoisyExpectedImprovement` (#287).
* Do not use approximate MLL computation for deterministic fitting (#314).
* Avoid re-evaluating the acquisition function in `gen_candidates_torch` (#319).
* Use CPU where possible in `gen_batch_initial_conditions` to avoid memory
  issues on the GPU (#323).

#### Bug fixes
* Properly register `NoiseModelAddedLossTerm` in `HeteroskedasticSingleTaskGP`
  (671c93a203b03ef03592ce322209fc5e71f23a74).
* Fix batch mode for `MultiTaskGPyTorchModel` (#316).
* Honor `propagate_grads` argument in `fantasize` of `FixedNoiseGP` (#303).
* Properly handle `diag` arg in `LinearTruncatedFidelityKernel` (#320).

#### Other changes
* Consolidate and simplify multi-fidelity models (#308).
* New license header style (#309).
* Validate shape of `best_f` in `qExpectedImprovement` (#299).
* Support specifying observation noise explicitly for all models (#256).
* Add `num_outputs` property to the `Model` API (#330).
* Validate output shape of models upon instantiating acquisition functions (#331).

#### Tests
* Silence warnings outside of explicit tests (#290).
* Enforce full sphinx docs coverage in CI (#294).


## [0.1.4] - Oct 1, 2019

Knowledge Gradient acquisition function (one-shot), various maintenance

#### Breaking Changes
* Require explicit output dimensions in BoTorch models (#238)
* Make `joint_optimize` / `sequential_optimize` return acquisition function
  values (#149) [note deprecation notice below]
* `standardize` now works on the second to last dimension (#263)
* Refactor synthetic test functions (#273)

#### New Features
* Add `qKnowledgeGradient` acquisition function (#272, #276)
* Add input scaling check to standard models (#267)
* Add `cyclic_optimize`, convergence criterion class (#269)
* Add `settings.debug` context manager (#242)

#### Deprecations
* Consolidate `sequential_optimize` and `joint_optimize` into `optimize_acqf`
  (#150)

#### Bug fixes
* Properly pass noise levels to GPs using a `FixedNoiseGaussianLikelihood` (#241)
  [requires gpytorch > 0.3.5]
* Fix q-batch dimension issue in `ConstrainedExpectedImprovement`
  (6c067185f56d3a244c4093393b8a97388fb1c0b3)
* Fix parameter constraint issues on GPU (#260)

#### Minor changes
* Add decorator for concatenating pending points (#240)
* Draw independent sample from prior for each hyperparameter (#244)
* Allow `dim > 1111` for `gen_batch_initial_conditions` (#249)
* Allow `optimize_acqf` to use `q>1` for `AnalyticAcquisitionFunction` (#257)
* Allow excluding parameters in fit functions (#259)
* Track the final iteration objective value in `fit_gpytorch_scipy` (#258)
* Error out on unexpected dims in parameter constraint generation (#270)
* Compute acquisition values in gen_ functions w/o grad (#274)

#### Tests
* Introduce BotorchTestCase to simplify test code (#243)
* Refactor tests to have monolithic cuda tests (#261)


## [0.1.3] - Aug 9, 2019

Compatibility & maintenance release

#### Compatibility
* Updates to support breaking changes in PyTorch to boolean masks and tensor
  comparisons (#224).
* Require PyTorch >=1.2 (#225).
* Require GPyTorch >=0.3.5 (itself a compatibility release).

#### New Features
* Add `FixedFeatureAcquisitionFunction` wrapper that simplifies optimizing
  acquisition functions over a subset of input features (#219).
* Add `ScalarizedObjective` for scalarizing posteriors (#210).
* Change default optimization behavior to use L-BFGS-B by for box constraints
  (#207).

#### Bug fixes
* Add validation to candidate generation (#213), making sure constraints are
  strictly satisfied (rater than just up to numerical accuracy of the optimizer).

#### Minor changes
* Introduce `AcquisitionObjective` base class (#220).
* Add propagate_grads context manager, replacing the `propagate_grads` kwarg in
  model `posterior()` calls (#221)
* Add `batch_initial_conditions` argument to `joint_optimize()` for
  warm-starting the optimization (ec3365a37ed02319e0d2bb9bea03aee89b7d9caa).
* Add `return_best_only` argument to `joint_optimize()` (#216). Useful for
  implementing advanced warm-starting procedures.


## [0.1.2] - July 9, 2019

Maintenance release

#### Bug fixes
* Avoid [PyTorch bug]((https://github.com/pytorch/pytorch/issues/22353)
  resulting in bad gradients on GPU by requiring GPyTorch >= 0.3.4
* Fixes to resampling behavior in MCSamplers (#204)

#### Experimental Features
* Linear truncated kernel for multi-fidelity bayesian optimization (#192)
* SingleTaskMultiFidelityGP for GP models that have fidelity parameters (#181)


## [0.1.1] - June 27, 2019

API updates, more robust model fitting

#### Breaking changes
* rename `botorch.qmc` to `botorch.sampling`, move MC samplers from
  `acquisition.sampler` to `botorch.sampling.samplers` (#172)

#### New Features
* Add `condition_on_observations` and `fantasize` to the Model level API (#173)
* Support pending observations generically for all `MCAcqusitionFunctions` (#176)
* Add fidelity kernel for training iterations/training data points (#178)
* Support for optimization constraints across `q`-batches (to support things like
  sample budget constraints) (2a95a6c3f80e751d5cf8bc7240ca9f5b1529ec5b)
* Add ModelList <-> Batched Model converter (#187)
* New test functions
    * basic: `neg_ackley`, `cosine8`, `neg_levy`, `neg_rosenbrock`, `neg_shekel`
      (e26dc7576c7bf5fa2ba4cb8fbcf45849b95d324b)
    * for multi-fidelity BO: `neg_aug_branin`, `neg_aug_hartmann6`,
      `neg_aug_rosenbrock` (ec4aca744f65ca19847dc368f9fee4cc297533da)

#### Improved functionality:
* More robust model fitting
    * Catch gpytorch numerical issues and return `NaN` to the optimizer (#184)
    * Restart optimization upon failure by sampling hyperparameters from their prior (#188)
    * Sequentially fit batched and `ModelListGP` models by default (#189)
    * Change minimum inferred noise level (e2c64fef1e76d526a33951c5eb75ac38d5581257)
* Introduce optional batch limit in `joint_optimize` to increases scalability of
  parallel optimization (baab5786e8eaec02d37a511df04442471c632f8a)
* Change constructor of `ModelListGP` to comply with GPyTorch’s `IndependentModelList`
  constructor (a6cf739e769c75319a67c7525a023ece8806b15d)
* Use `torch.random` to set default seed for samplers (rather than `random`) to
  making sampling reproducible when setting `torch.manual_seed`
  (ae507ad97255d35f02c878f50ba68a2e27017815)

####  Performance Improvements
* Use `einsum` in `LinearMCObjective` (22ca29535717cda0fcf7493a43bdf3dda324c22d)
* Change default Sobol sample size for `MCAquisitionFunctions` to be base-2 for
  better MC integration performance (5d8e81866a23d6bfe4158f8c9b30ea14dd82e032)
* Add ability to fit models in `SumMarginalLogLikelihood` sequentially (and make
  that the default setting) (#183)
* Do not construct the full covariance matrix when computing posterior of
  single-output BatchedMultiOutputGPyTorchModel (#185)

#### Bug fixes
* Properly handle observation_noise kwarg for BatchedMultiOutputGPyTorchModels (#182)
* Fix a issue where `f_best` was always max for NoisyExpectedImprovement
  (de8544a75b58873c449b41840a335f6732754c77)
* Fix bug and numerical issues in `initialize_q_batch`
  (844dcd1dc8f418ae42639e211c6bb8e31a75d8bf)
* Fix numerical issues with `inv_transform` for qMC sampling (#162)

#### Other
* Bump GPyTorch minimum requirement to 0.3.3



## [0.1.0] - April 30, 2019

First public beta release.
