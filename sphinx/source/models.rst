.. role:: hidden
    :class: hidden-section


botorch.models
========================================================
.. automodule:: botorch.models


Model APIs
-------------------------------------------

Base Model API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.model
    :members:

GPyTorch Model API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.gpytorch
    :members:

Deterministic Model API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.deterministic
    :members:

Ensemble Model API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.ensemble
    :members:


Models
-------------------------------------------

Additive GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.additive_gp
    :members:

Cost Models (for cost-aware optimization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.cost
    :members:

Contextual GP Models with Aggregate Rewards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.contextual
    :members:

Contextual GP Models with Context Rewards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.contextual_multioutput
    :members:

Fully Bayesian GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.fully_bayesian
    :members:

Fully Bayesian Multitask GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.fully_bayesian_multitask
    :members:

GP Regression Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.gp_regression
    :members:

GP Regression Models for Mixed Parameter Spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.gp_regression_mixed
    :members:

Higher Order GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.higher_order_gp
    :members:

Latent Kronecker GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.latent_kronecker_gp
    :members:

Model List GP Regression Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.model_list_gp_regression
    :members:

Multitask GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.multitask
    :members:

Heterogeneous Search Space Multitask GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.heterogeneous_mtgp
    :members:

Multi-Fidelity GP Regression Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.gp_regression_fidelity
    :members:

Pairwise GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.pairwise_gp
    :members:

Relevance Pursuit Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.relevance_pursuit
    :members:

Sparse Axis-Aligned Subspaces (SAAS) GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.map_saas
    :members:

Variational GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.robust_relevance_pursuit_model
    :members:

.. automodule:: botorch.models.approximate_gp
    :members:

Model Components
-------------------------------------------

Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.kernels.categorical
.. autoclass:: CategoricalKernel

.. automodule:: botorch.models.kernels.downsampling
.. autoclass:: DownsamplingKernel

.. automodule:: botorch.models.kernels.exponential_decay
.. autoclass:: ExponentialDecayKernel

.. automodule:: botorch.models.kernels.infinite_width_bnn
.. autoclass:: InfiniteWidthBNNKernel

.. automodule:: botorch.models.kernels.linear_truncated_fidelity
.. autoclass:: LinearTruncatedFidelityKernel

.. automodule:: botorch.models.kernels.contextual_lcea
.. autoclass:: LCEAKernel

.. automodule:: botorch.models.kernels.contextual_sac
.. autoclass:: SACKernel

.. automodule:: botorch.models.kernels.orthogonal_additive_kernel
.. autoclass:: OrthogonalAdditiveKernel

.. automodule:: botorch.models.kernels.positive_index
.. autoclass:: PositiveIndexKernel

.. automodule:: botorch.models.kernels.heterogeneous_multitask
.. autoclass:: MultiTaskConditionalKernel

Likelihoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.likelihoods.pairwise
    :members:

.. automodule:: botorch.models.likelihoods.sparse_outlier_noise
    :members:

Transforms
-------------------------------------------

Outcome Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.transforms.outcome
    :members:

Input Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.transforms.input
    :members:

Transform Factory Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.transforms.factory
    :members:

Transform Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.transforms.utils
    :members:


Utilities
-------------------------------------------

GPyTorch Module Constructors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.utils.gpytorch_modules
    :members:


Inducing Point Allocators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.utils.inducing_point_allocators
    :members:
    :private-members: _pivoted_cholesky_init

Other Utilties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.utils.assorted
    :members:
