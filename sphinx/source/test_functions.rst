.. role:: hidden
    :class: hidden-section

botorch.test_functions
========================================================
.. automodule:: botorch.test_functions


Abstract Test Function API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.test_functions.base
    :members:

Synthetic Test Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The last axis of the input tensor holds the coordinates of each point. For a single-objective function with ``d`` input dimensions, an input of shape ``(..., d)`` produces one value per point with shape ``(...)``. The same shape convention applies to ``evaluate_true(X)``.

For example, evaluate three points with the two-dimensional Ackley function.

.. doctest::

    >>> import torch
    >>> from botorch.test_functions import Ackley
    >>> ackley = Ackley(dim=2)
    >>> points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    >>> ackley(points).shape
    torch.Size([3])

For a surface plot, stack the coordinates along the last axis to evaluate a whole grid in one call.

.. doctest::

    >>> x = torch.linspace(-5.0, 5.0, 5)
    >>> y = torch.linspace(-5.0, 5.0, 7)
    >>> grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    >>> grid = torch.stack((grid_x, grid_y), dim=-1)
    >>> grid.shape
    torch.Size([5, 7, 2])
    >>> ackley(grid).shape
    torch.Size([5, 7])

.. automodule:: botorch.test_functions.synthetic
    :members:

Multi-Fidelity Synthetic Test Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.test_functions.multi_fidelity
    :members:

Multi-Objective Synthetic Test Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.test_functions.multi_objective
    :members:

Multi-Objective Multi-Fidelity Synthetic Test Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.test_functions.multi_objective_multi_fidelity
    :members:

Sensitivity Analysis Test Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.test_functions.sensitivity_analysis
    :members:

Utilities For Test Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.test_functions.utils
    :members:
