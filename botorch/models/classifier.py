# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Classifier-based models for constraint boundaries and deterministic feasibility.

These models wrap scikit-learn and XGBoost classifiers as BoTorch deterministic models,
enabling them to be used for modeling binary constraints, feasibility, and other
discontinuous outputs where traditional GP models fail due to smoothness assumptions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from botorch.models.deterministic import GenericDeterministicModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch import Tensor
from xgboost import XGBClassifier


class RandomForestClassifierModel(GenericDeterministicModel):
    """
    Random Forest classifier wrapped as a BoTorch deterministic model.

    This model is suitable for modeling hard constraint boundaries and binary
    feasibility without smoothness assumptions. It creates step functions at
    decision boundaries rather than smooth transitions.

    Random Forests are particularly effective for:
    - Hard constraint boundaries (no smoothness assumptions)
    - Deterministic failure regions with cliff-like transitions
    - Discontinuous binary outputs (feasible/infeasible)
    - Robustness to outliers and noisy labels

    Example:
        >>> from botorch.models.classifier import RandomForestClassifierModel
        >>> from botorch.utils.datasets import SupervisedDataset
        >>> import torch
        >>>
        >>> X = torch.randn(100, 5)
        >>> y = torch.randint(0, 2, (100, 1), dtype=torch.float64)
        >>> dataset = SupervisedDataset(X=X, Y=y)
        >>>
        >>> model_inputs = RandomForestClassifierModel.construct_inputs(
        ...     training_data=dataset,
        ...     n_estimators=100,
        ...     max_depth=10
        ... )
        >>> model = RandomForestClassifierModel(**model_inputs)
        >>> test_X = torch.randn(10, 5)
        >>> probs = model.posterior(test_X).mean  # Returns P(y=1|X)
    """

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        num_outputs: int = 1,
        input_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RandomForestClassifierModel.

        Args:
            f: Prediction function.
            num_outputs: Number of outputs (always 1 for binary classification).
            input_transform: Accepted for API compatibility but not used.
                GenericDeterministicModel does not support input transforms.
            **kwargs: Additional arguments that are ignored.
        """
        super().__init__(f=f, num_outputs=num_outputs)

    @classmethod
    def construct_inputs(
        cls,
        training_data: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Construct inputs for RandomForestClassifierModel from training data.

        Args:
            training_data: SupervisedDataset with X (features) and Y (labels).
                Y should be binary (0 or 1) with shape (n, 1) or (n,).
            **kwargs: Model options including:
                - n_estimators: Number of trees (default: 10)
                - max_depth: Maximum tree depth (default: 5)
                - min_samples_split: Minimum samples to split (default: 5)
                - class_weight: Class weights (default: "balanced")
                - max_features: Features per split (default: "sqrt")

        Returns:
            Dictionary with 'f' (prediction function) and 'num_outputs' (1).
            Can be passed to RandomForestClassifierModel() as **kwargs.

        Example:
            >>> model_inputs = RandomForestClassifierModel.construct_inputs(
            ...     training_data=dataset,
            ...     n_estimators=200,
            ...     max_depth=15,
            ...     min_samples_split=10
            ... )
            >>> model = RandomForestClassifierModel(**model_inputs)
        """
        n_estimators = kwargs.pop("n_estimators", 10)
        max_depth = kwargs.pop("max_depth", 5)
        min_samples_split = kwargs.pop("min_samples_split", 5)
        class_weight = kwargs.pop("class_weight", "balanced")
        max_features = kwargs.pop("max_features", "sqrt")

        # Train RandomForest classifier
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            max_features=max_features,
            **kwargs,
        )

        X_np = training_data.X.detach().cpu().numpy()
        y_np = training_data.Y.detach().cpu().numpy()

        # Flatten y if needed (sklearn expects 1D labels)
        if y_np.ndim > 1:
            y_np = y_np.ravel()

        clf.fit(X_np, y_np)

        def predict_proba_fn(X: Tensor) -> Tensor:
            """Predict class probabilities for the positive class."""
            # Store original shape to restore later
            original_shape = X.shape
            # Reshape to 2D: (batch_size * n_samples, n_features)
            # sklearn expects 2D input: (n_samples, n_features)
            X_2d = X.reshape(-1, X.shape[-1])
            X_np_pred = X_2d.detach().cpu().numpy()

            # Get probabilities for positive class (class=1)
            probs_np = clf.predict_proba(X_np_pred)[:, 1]

            probs = torch.from_numpy(probs_np).to(dtype=X.dtype, device=X.device)

            # BoTorch models expect: batch_shape x n x d input ->
            # batch_shape x n x 1 output
            probs = probs.reshape(*original_shape[:-1], 1)

            return probs

        return {"f": predict_proba_fn, "num_outputs": 1}


class SVCClassifierModel(GenericDeterministicModel):
    """
    Support Vector Classifier wrapped as a BoTorch deterministic model.

    SVC is particularly effective for:
    - Smooth decision boundaries with sharp transitions
    - Non-axis-aligned constraint boundaries
    - Small to medium-sized datasets (scales O(n^2) to O(n^3))
    - High-dimensional spaces with proper kernel selection

    Example:
        >>> from botorch.models.classifier import SVCClassifierModel
        >>> from botorch.utils.datasets import SupervisedDataset
        >>> import torch
        >>>
        >>> X = torch.randn(100, 5)
        >>> y = torch.randint(0, 2, (100, 1), dtype=torch.float64)
        >>> dataset = SupervisedDataset(X=X, Y=y)
        >>>
        >>> model_inputs = SVCClassifierModel.construct_inputs(
        ...     training_data=dataset,
        ...     kernel="rbf",
        ...     C=1.0
        ... )
        >>> model = SVCClassifierModel(**model_inputs)
        >>> test_X = torch.randn(10, 5)
        >>> probs = model.posterior(test_X).mean  # Returns P(y=1|X)
    """

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        num_outputs: int = 1,
        input_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SVCClassifierModel.

        Args:
            f: Prediction function.
            num_outputs: Number of outputs (always 1 for binary classification).
            input_transform: Accepted for API compatibility but not used.
                GenericDeterministicModel does not support input
                transforms.
            **kwargs: Additional arguments that are ignored.
        """
        super().__init__(f=f, num_outputs=num_outputs)

    @classmethod
    def construct_inputs(
        cls,
        training_data: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Construct inputs for SVCClassifierModel from training data.

        Args:
            training_data: SupervisedDataset with X (features) and Y (labels).
                Y should be binary (0 or 1) with shape (n, 1) or (n,).
        **kwargs: Model options including:
                - kernel: Kernel type (default: "rbf").
                  Options: "rbf", "linear", "poly"
                - C: Regularization parameter (default: 1.0).
                  Higher = tighter fit
                - gamma: Kernel coefficient (default: "scale").
                  Options: "scale", "auto", float
                - class_weight: Class weights (default: "balanced")

        Returns:
            Dictionary with 'f' (prediction function) and 'num_outputs' (1).
            Can be passed to SVCClassifierModel() as **kwargs.

        Example:
            >>> model_inputs = SVCClassifierModel.construct_inputs(
            ...     training_data=dataset,
            ...     kernel="rbf",
            ...     C=10.0,
            ...     gamma="scale"
            ... )
            >>> model = SVCClassifierModel(**model_inputs)
        """
        # Extract hyperparameters from kwargs
        kernel = kwargs.pop("kernel", "rbf")
        C = kwargs.pop("C", 1.0)
        gamma = kwargs.pop("gamma", "scale")
        class_weight = kwargs.pop("class_weight", "balanced")
        probability = kwargs.pop("probability", True)

        # Train SVC classifier
        clf = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            **kwargs,
        )

        X_np = training_data.X.detach().cpu().numpy()
        y_np = training_data.Y.detach().cpu().numpy()

        # Flatten y if needed
        if y_np.ndim > 1:
            y_np = y_np.ravel()

        clf.fit(X_np, y_np)

        def predict_proba_fn(X: Tensor) -> Tensor:
            """Predict class probabilities for the positive class."""
            original_shape = X.shape
            # Reshape to 2D: (batch_size * n_samples, n_features)
            # sklearn expects 2D input: (n_samples, n_features)
            X_2d = X.reshape(-1, X.shape[-1])
            X_np_pred = X_2d.detach().cpu().numpy()

            # Get probabilities for positive class (class=1)
            probs_np = clf.predict_proba(X_np_pred)[:, 1]

            probs = torch.from_numpy(probs_np).to(dtype=X.dtype, device=X.device)

            # Reshape to match original input's batch structure
            # BoTorch models expect: batch_shape x n x d input ->
            # batch_shape x n x 1 output
            probs = probs.reshape(*original_shape[:-1], 1)

            return probs

        return {"f": predict_proba_fn, "num_outputs": 1}


class XGBoostClassifierModel(GenericDeterministicModel):
    """
    XGBoost classifier wrapped as a BoTorch deterministic model.

    This model is particularly effective for complex, non-linear decision boundaries.
    XGBoost uses gradient boosting to create powerful ensemble models.

    XGBoost is particularly effective for:
    - Complex, highly non-linear decision boundaries
    - Capturing feature interactions automatically
    - Large datasets with efficient training
    - Handling missing values natively

    Example:
        >>> from botorch.models.classifier import XGBoostClassifierModel
        >>> from botorch.utils.datasets import SupervisedDataset
        >>> import torch
        >>>
        >>> X = torch.randn(100, 5)
        >>> y = torch.randint(0, 2, (100, 1), dtype=torch.float64)
        >>> dataset = SupervisedDataset(X=X, Y=y)
        >>>
        >>> model_inputs = XGBoostClassifierModel.construct_inputs(
        ...     training_data=dataset,
        ...     n_estimators=100,
        ...     max_depth=6
        ... )
        >>> model = XGBoostClassifierModel(**model_inputs)
        >>> test_X = torch.randn(10, 5)
        >>> probs = model.posterior(test_X).mean  # Returns P(y=1|X)
    """

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        num_outputs: int = 1,
        input_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize XGBoostClassifierModel.

        Args:
            f: Prediction function.
            num_outputs: Number of outputs (always 1 for binary classification).
            input_transform: Accepted for API compatibility but not used.
                GenericDeterministicModel does not support input transforms.
            **kwargs: Additional arguments that are ignored.
        """
        super().__init__(f=f, num_outputs=num_outputs)

    @classmethod
    def construct_inputs(
        cls,
        training_data: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Construct inputs for XGBoostClassifierModel from training data.

        Args:
            training_data: SupervisedDataset with X (features) and Y (labels).
                Y should be binary (0 or 1) with shape (n, 1) or (n,).
            **kwargs: Model options including:
                - n_estimators: Number of boosting rounds (default: 1-)
                - max_depth: Maximum tree depth (default: 5)
                - learning_rate: Boosting learning rate (default: 0.1)
                - eval_metric: Evaluation metric (default: "logloss")

        Returns:
            Dictionary with 'f' (prediction function) and 'num_outputs' (1).
            Can be passed to XGBoostClassifierModel() as **kwargs.

        Example:
            >>> model_inputs = XGBoostClassifierModel.construct_inputs(
            ...     training_data=dataset,
            ...     n_estimators=200,
            ...     max_depth=8,
            ...     learning_rate=0.05
            ... )
            >>> model = XGBoostClassifierModel(**model_inputs)
        """
        # Extract hyperparameters from kwargs
        n_estimators = kwargs.pop("n_estimators", 10)
        max_depth = kwargs.pop("max_depth", 5)
        learning_rate = kwargs.pop("learning_rate", 0.1)
        eval_metric = kwargs.pop("eval_metric", "logloss")
        use_label_encoder = kwargs.pop("use_label_encoder", False)

        # Train XGBoost classifier
        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric=eval_metric,
            use_label_encoder=use_label_encoder,
            **kwargs,
        )

        X_np = training_data.X.detach().cpu().numpy()
        y_np = training_data.Y.detach().cpu().numpy()

        # Flatten y if needed
        if y_np.ndim > 1:
            y_np = y_np.ravel()

        clf.fit(X_np, y_np)

        def predict_proba_fn(X: Tensor) -> Tensor:
            """Predict class probabilities for the positive class."""
            original_shape = X.shape
            # Reshape to 2D: (batch_size * n_samples, n_features)
            # xgboost expects 2D input: (n_samples, n_features)
            X_2d = X.reshape(-1, X.shape[-1])
            X_np_pred = X_2d.detach().cpu().numpy()

            # Get probabilities for positive class (class=1)
            probs_np = clf.predict_proba(X_np_pred)[:, 1]

            # Convert back to tensor with same dtype and device as input
            probs = torch.from_numpy(probs_np).to(dtype=X.dtype, device=X.device)

            # Reshape to match original input's batch structure
            # BoTorch models expect: batch_shape x n x d input ->
            # batch_shape x n x 1 output
            probs = probs.reshape(*original_shape[:-1], 1)

            return probs

        return {"f": predict_proba_fn, "num_outputs": 1}
