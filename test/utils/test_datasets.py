#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from itertools import product

import torch
from botorch.exceptions.errors import InputDataError, UnsupportedError
from botorch.utils.containers import DenseContainer, SliceContainer
from botorch.utils.datasets import (
    ContextualDataset,
    MultiTaskDataset,
    RankingDataset,
    SupervisedDataset,
)
from botorch.utils.testing import BotorchTestCase
from torch import rand, randperm, Size, stack, Tensor, tensor


def make_dataset(
    num_samples: int = 3,
    d: int = 2,
    m: int = 1,
    has_yvar: bool = False,
    feature_names: list[str] | None = None,
    outcome_names: list[str] | None = None,
    batch_shape: torch.Size | None = None,
) -> SupervisedDataset:
    feature_names = feature_names or [f"x{i}" for i in range(d)]
    outcome_names = outcome_names or [f"y{i}" for i in range(m)]
    batch_shape = batch_shape or torch.Size()
    return SupervisedDataset(
        X=rand(*batch_shape, num_samples, d),
        Y=rand(*batch_shape, num_samples, m),
        Yvar=rand(*batch_shape, num_samples, m) if has_yvar else None,
        feature_names=feature_names,
        outcome_names=outcome_names,
    )


def make_contextual_dataset(
    has_yvar: bool = False, contextual_outcome: bool = False
) -> tuple[ContextualDataset, list[SupervisedDataset]]:
    num_contexts = 3
    feature_names = [f"x_c{i}" for i in range(num_contexts)]
    parameter_decomposition = {
        "context_2": ["x_c2"],
        "context_1": ["x_c1"],
        "context_0": ["x_c0"],
    }
    context_buckets = list(parameter_decomposition.keys())
    if contextual_outcome:
        context_outcome_list = [f"y:context_{i}" for i in range(num_contexts)]
        metric_decomposition = {f"{c}": [f"y:{c}"] for c in context_buckets}

        dataset_list2 = [
            make_dataset(
                d=1 * num_contexts,
                has_yvar=has_yvar,
                feature_names=feature_names,
                outcome_names=[context_outcome_list[0]],
            )
        ]
        for mname in context_outcome_list[1:]:
            dataset_list2.append(
                SupervisedDataset(
                    X=dataset_list2[0].X,
                    Y=rand(dataset_list2[0].Y.size()),
                    Yvar=rand(dataset_list2[0].Yvar.size()) if has_yvar else None,
                    feature_names=feature_names,
                    outcome_names=[mname],
                )
            )
        context_dt = ContextualDataset(
            datasets=dataset_list2,
            parameter_decomposition=parameter_decomposition,
            metric_decomposition=metric_decomposition,
        )
        return context_dt, dataset_list2
    dataset_list1 = [
        make_dataset(
            d=num_contexts,
            has_yvar=has_yvar,
            feature_names=feature_names,
            outcome_names=["y"],
        )
    ]
    context_dt = ContextualDataset(
        datasets=dataset_list1,
        parameter_decomposition=parameter_decomposition,
    )
    return context_dt, dataset_list1


class TestDatasets(BotorchTestCase):
    def test_supervised(self):
        # Generate some data
        n_rows = 3
        X = rand(n_rows, 2)
        Y = rand(n_rows, 1)
        feature_names = ["x1", "x2"]
        outcome_names = ["y"]
        group_indices = tensor(range(n_rows))

        # Test `__init__`
        dataset = SupervisedDataset(
            X=X,
            Y=Y,
            feature_names=feature_names,
            outcome_names=outcome_names,
            group_indices=group_indices,
        )
        self.assertIsInstance(dataset.X, Tensor)
        self.assertIsInstance(dataset._X, Tensor)
        self.assertIsInstance(dataset.Y, Tensor)
        self.assertIsInstance(dataset._Y, Tensor)
        self.assertEqual(dataset.feature_names, feature_names)
        self.assertEqual(dataset.outcome_names, outcome_names)
        self.assertTrue(torch.equal(dataset.group_indices, group_indices))

        dataset2 = SupervisedDataset(
            X=DenseContainer(X, X.shape[-1:]),
            Y=DenseContainer(Y, Y.shape[-1:]),
            feature_names=feature_names,
            outcome_names=outcome_names,
            group_indices=group_indices,
        )
        self.assertIsInstance(dataset2.X, Tensor)
        self.assertIsInstance(dataset2._X, DenseContainer)
        self.assertIsInstance(dataset2.Y, Tensor)
        self.assertIsInstance(dataset2._Y, DenseContainer)
        self.assertEqual(dataset, dataset2)

        # Test `_validate`
        with self.assertRaisesRegex(ValueError, "Batch dimensions .* incompatible."):
            SupervisedDataset(
                X=rand(1, 2),
                Y=rand(2, 1),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        with self.assertRaisesRegex(ValueError, "`Y` and `Yvar`"):
            SupervisedDataset(
                X=rand(2, 2),
                Y=rand(2, 1),
                Yvar=rand(2),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        with self.assertRaisesRegex(ValueError, "feature_names"):
            SupervisedDataset(
                X=rand(2, 2),
                Y=rand(2, 1),
                feature_names=[],
                outcome_names=outcome_names,
            )
        with self.assertRaisesRegex(ValueError, "outcome_names"):
            SupervisedDataset(
                X=rand(2, 2),
                Y=rand(2, 1),
                feature_names=feature_names,
                outcome_names=[],
            )
        with self.assertRaisesRegex(ValueError, "group_indices"):
            SupervisedDataset(
                X=rand(2, 2),
                Y=rand(2, 1),
                feature_names=feature_names,
                outcome_names=outcome_names,
                group_indices=tensor(range(n_rows + 1)),
            )

        # Test with Yvar.
        dataset = SupervisedDataset(
            X=X,
            Y=Y,
            Yvar=DenseContainer(Y, Y.shape[-1:]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
        self.assertIsInstance(dataset.X, Tensor)
        self.assertIsInstance(dataset._X, Tensor)
        self.assertIsInstance(dataset.Y, Tensor)
        self.assertIsInstance(dataset._Y, Tensor)
        self.assertIsInstance(dataset.Yvar, Tensor)
        self.assertIsInstance(dataset._Yvar, DenseContainer)

        # More equality checks with & without Yvar.
        self.assertEqual(dataset, dataset)
        self.assertNotEqual(dataset, dataset2)
        self.assertNotEqual(dataset2, dataset)

    def test_clone(self, supervised: bool = True) -> None:
        has_yvar_options = [False]
        if supervised:
            has_yvar_options.append(True)
        for has_yvar in has_yvar_options:
            if supervised:
                dataset = make_dataset(has_yvar=has_yvar)
            else:
                X_val = rand(16, 2)
                X_idx = stack([randperm(len(X_val))[:3] for _ in range(1)])
                X = SliceContainer(
                    X_val, X_idx, event_shape=Size([3 * X_val.shape[-1]])
                )
                dataset = RankingDataset(
                    X=X,
                    Y=tensor([[0, 1, 1]]),
                    feature_names=["x1", "x2"],
                    outcome_names=["ranking indices"],
                )

            for use_deepcopy in [False, True]:
                dataset2 = dataset.clone(deepcopy=use_deepcopy)
                self.assertEqual(dataset, dataset2)
                self.assertTrue(torch.equal(dataset.X, dataset2.X))
                self.assertTrue(torch.equal(dataset.Y, dataset2.Y))
                if has_yvar:
                    self.assertTrue(torch.equal(dataset.Yvar, dataset2.Yvar))
                else:
                    self.assertIsNone(dataset2.Yvar)
                self.assertEqual(dataset.feature_names, dataset2.feature_names)
                self.assertEqual(dataset.outcome_names, dataset2.outcome_names)
                if use_deepcopy:
                    self.assertIsNot(dataset.X, dataset2.X)
                    self.assertIsNot(dataset.Y, dataset2.Y)
                    if has_yvar:
                        self.assertIsNot(dataset.Yvar, dataset2.Yvar)
                    self.assertIsNot(dataset.feature_names, dataset2.feature_names)
                    self.assertIsNot(dataset.outcome_names, dataset2.outcome_names)
                else:
                    self.assertIs(dataset._X, dataset2._X)
                    self.assertIs(dataset._Y, dataset2._Y)
                    self.assertIs(dataset._Yvar, dataset2._Yvar)
                    self.assertEqual(dataset.feature_names, dataset2.feature_names)
                    self.assertEqual(dataset.outcome_names, dataset2.outcome_names)
                # test with mask
                mask = torch.tensor([0, 1, 1], dtype=torch.bool)
                if supervised:
                    dataset2 = dataset.clone(deepcopy=use_deepcopy, mask=mask)
                    self.assertTrue(torch.equal(dataset.X[1:], dataset2.X))
                    self.assertTrue(torch.equal(dataset.Y[1:], dataset2.Y))
                    if has_yvar:
                        self.assertTrue(torch.equal(dataset.Yvar[1:], dataset2.Yvar))
                    else:
                        self.assertIsNone(dataset2.Yvar)
                else:
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        "Masking is not supported for BotorchContainers.",
                    ):
                        dataset.clone(deepcopy=use_deepcopy, mask=mask)

    def test_clone_ranking(self) -> None:
        self.test_clone(supervised=False)

    def test_fixedNoise(self):
        # Generate some data
        X = rand(3, 2)
        Y = rand(3, 1)
        Yvar = rand(3, 1)
        feature_names = ["x1", "x2"]
        outcome_names = ["y"]
        dataset = SupervisedDataset(
            X=X,
            Y=Y,
            Yvar=Yvar,
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
        self.assertTrue(torch.equal(dataset.X, X))
        self.assertTrue(torch.equal(dataset.Y, Y))
        self.assertTrue(torch.equal(dataset.Yvar, Yvar))
        self.assertEqual(dataset.feature_names, feature_names)
        self.assertEqual(dataset.outcome_names, outcome_names)

    def test_ranking(self):
        # Test `_validate`
        X_val = rand(16, 2)
        X_idx = stack([randperm(len(X_val))[:3] for _ in range(1)])
        X = SliceContainer(X_val, X_idx, event_shape=Size([3 * X_val.shape[-1]]))
        feature_names = ["x1", "x2"]
        outcome_names = ["ranking indices"]

        with self.assertRaisesRegex(ValueError, "The `values` field of `X`"):
            RankingDataset(
                X=X,
                Y=tensor([[-1, 0, 1]]),
                feature_names=feature_names[:1],
                outcome_names=outcome_names,
            )

        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            RankingDataset(
                X=X,
                Y=tensor([[-1, 0, 1]]),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        RankingDataset(
            X=X,
            Y=tensor([[2, 0, 1]]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )

        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            RankingDataset(
                X=X,
                Y=tensor([[0, 1, 3]]),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        RankingDataset(
            X=X,
            Y=tensor([[0, 1, 2]]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )

        with self.assertRaisesRegex(ValueError, "missing zero-th rank."):
            RankingDataset(
                X=X,
                Y=tensor([[1, 2, 2]]),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        RankingDataset(
            X=X,
            Y=tensor([[0, 1, 1]]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )

        with self.assertRaisesRegex(ValueError, "ranks not skipped after ties."):
            RankingDataset(
                X=X,
                Y=tensor([[0, 0, 1]]),
                feature_names=feature_names,
                outcome_names=outcome_names,
            )
        RankingDataset(
            X=X,
            Y=tensor([[0, 0, 2]]),
            feature_names=feature_names,
            outcome_names=outcome_names,
        )

    def test_multi_task(self):
        dataset_1 = make_dataset(outcome_names=["y"])
        dataset_2 = make_dataset(outcome_names=["z"])
        dataset_3 = make_dataset(has_yvar=True, outcome_names=["z"])
        dataset_4 = make_dataset(has_yvar=True, outcome_names=["y"])
        # Test validation.
        with self.assertRaisesRegex(
            UnsupportedError, "containing more than one outcome"
        ):
            MultiTaskDataset(datasets=[make_dataset(m=2)], target_outcome_name="y0")
        with self.assertRaisesRegex(
            UnsupportedError, "multiple datasets for the same outcome"
        ):
            MultiTaskDataset(datasets=[dataset_1, dataset_1], target_outcome_name="y")
        with self.assertRaisesRegex(InputDataError, "Target outcome is not present"):
            MultiTaskDataset(datasets=[dataset_1], target_outcome_name="z")
        with self.assertRaisesRegex(UnsupportedError, "modeling batched inputs"):
            MultiTaskDataset(
                datasets=[make_dataset(batch_shape=torch.Size([2]))],
                target_outcome_name="y0",
            )
        with self.assertRaisesRegex(InputDataError, "names of the task features"):
            MultiTaskDataset(
                datasets=[
                    dataset_1,
                    make_dataset(feature_names=["x1", "x3"], outcome_names=["z"]),
                ],
                target_outcome_name="z",
                task_feature_index=1,
            )
        with self.assertRaisesRegex(
            UnsupportedError, "all or none of the datasets to have a Yvar."
        ):
            MultiTaskDataset(datasets=[dataset_1, dataset_3], target_outcome_name="z")

        # Test correct construction.
        mt_dataset = MultiTaskDataset(
            datasets=[dataset_1, dataset_2],
            target_outcome_name="z",
        )
        self.assertEqual(len(mt_dataset.datasets), 2)
        self.assertIsNone(mt_dataset.task_feature_index)
        self.assertIs(mt_dataset.datasets["y"], dataset_1)
        self.assertIs(mt_dataset.datasets["z"], dataset_2)
        self.assertIsNone(mt_dataset.Yvar)
        self.assertFalse(mt_dataset.has_heterogeneous_features)
        expected_X = torch.cat(
            [
                torch.cat([dataset_1.X, torch.ones(3, 1)], dim=-1),
                torch.cat([dataset_2.X, torch.zeros(3, 1)], dim=-1),
            ],
            dim=0,
        )
        expected_Y = torch.cat([ds.Y for ds in [dataset_1, dataset_2]], dim=0)
        self.assertTrue(torch.equal(expected_X, mt_dataset.X))
        self.assertTrue(torch.equal(expected_Y, mt_dataset.Y))
        self.assertIs(
            mt_dataset.get_dataset_without_task_feature(outcome_name="y"), dataset_1
        )

        # Test with Yvar and target_feature_index.
        mt_dataset = MultiTaskDataset(
            datasets=[dataset_3, dataset_4],
            target_outcome_name="z",
            task_feature_index=1,
        )
        self.assertEqual(mt_dataset.task_feature_index, 1)
        expected_X_2 = torch.cat([dataset_3.X, dataset_4.X], dim=0)
        expected_Yvar_2 = torch.cat([dataset_3.Yvar, dataset_4.Yvar], dim=0)
        self.assertTrue(torch.equal(expected_X_2, mt_dataset.X))
        self.assertTrue(torch.equal(expected_Yvar_2, mt_dataset.Yvar))
        # Check that the task feature is removed correctly.
        ds_3_no_task = mt_dataset.get_dataset_without_task_feature(outcome_name="z")
        self.assertTrue(torch.equal(ds_3_no_task.X, dataset_3.X[:, :1]))
        self.assertTrue(torch.equal(ds_3_no_task.Y, dataset_3.Y))
        self.assertTrue(torch.equal(ds_3_no_task.Yvar, dataset_3.Yvar))
        self.assertEqual(ds_3_no_task.feature_names, dataset_3.feature_names[:1])
        self.assertEqual(ds_3_no_task.outcome_names, dataset_3.outcome_names)

        # Test from_joint_dataset.
        sort_idcs = [3, 4, 5, 0, 1, 2]  # X & Y will get sorted based on task feature.
        for outcome_names_per_task in [None, {0: "x", 1: "y"}]:
            joint_dataset = SupervisedDataset(
                X=expected_X,
                Y=expected_Y,
                feature_names=["x0", "x1", "task"],
                outcome_names=["z"],
            )
            mt_dataset = MultiTaskDataset.from_joint_dataset(
                dataset=joint_dataset,
                task_feature_index=-1,
                target_task_value=0,
                outcome_names_per_task=outcome_names_per_task,
            )
            self.assertEqual(len(mt_dataset.datasets), 2)
            if outcome_names_per_task is None:
                self.assertEqual(list(mt_dataset.datasets.keys()), ["z", "task_1"])
                self.assertEqual(mt_dataset.target_outcome_name, "z")
            else:
                self.assertEqual(list(mt_dataset.datasets.keys()), ["x", "y"])
                self.assertEqual(mt_dataset.target_outcome_name, "x")

            self.assertTrue(torch.equal(mt_dataset.X, expected_X[sort_idcs]))
            self.assertTrue(torch.equal(mt_dataset.Y, expected_Y[sort_idcs]))
            self.assertTrue(
                torch.equal(
                    mt_dataset.datasets[mt_dataset.target_outcome_name].Y, dataset_2.Y
                )
            )
            self.assertIsNone(mt_dataset.Yvar)
        with self.assertRaisesRegex(UnsupportedError, "more than one outcome"):
            MultiTaskDataset.from_joint_dataset(
                dataset=make_dataset(m=2),
                task_feature_index=-1,
                target_task_value=0,
            )

        # With heterogeneous feature sets.
        dataset_5 = make_dataset(d=1, outcome_names=["z"])
        mt_dataset = MultiTaskDataset(
            datasets=[dataset_1, dataset_5],
            target_outcome_name="y",
        )
        self.assertTrue(mt_dataset.has_heterogeneous_features)
        with self.assertRaisesRegex(
            UnsupportedError, "datasets with heterogeneous feature sets"
        ):
            mt_dataset.X

        # Test equality.
        self.assertEqual(mt_dataset, mt_dataset)
        self.assertNotEqual(mt_dataset, dataset_5)
        self.assertNotEqual(
            mt_dataset, MultiTaskDataset(datasets=[dataset_1], target_outcome_name="y")
        )
        self.assertNotEqual(
            mt_dataset,
            MultiTaskDataset(datasets=[dataset_1, dataset_5], target_outcome_name="z"),
        )

    def test_clone_multitask(self) -> None:
        for has_yvar in [False, True]:
            dataset_1 = make_dataset(outcome_names=["y"], has_yvar=has_yvar)
            dataset_2 = make_dataset(outcome_names=["z"], has_yvar=has_yvar)
            mt_dataset = MultiTaskDataset(
                datasets=[dataset_1, dataset_2],
                target_outcome_name="z",
            )
            for use_deepcopy in [False, True]:
                mt_dataset2 = mt_dataset.clone(deepcopy=use_deepcopy)
                self.assertEqual(mt_dataset, mt_dataset2)
                self.assertTrue(torch.equal(mt_dataset.X, mt_dataset2.X))
                self.assertTrue(torch.equal(mt_dataset.Y, mt_dataset2.Y))
                if has_yvar:
                    self.assertTrue(torch.equal(mt_dataset.Yvar, mt_dataset2.Yvar))
                else:
                    self.assertIsNone(mt_dataset2.Yvar)
                self.assertEqual(mt_dataset.feature_names, mt_dataset2.feature_names)
                self.assertEqual(mt_dataset.outcome_names, mt_dataset2.outcome_names)
                if use_deepcopy:
                    for ds, ds2 in zip(
                        mt_dataset.datasets.values(), mt_dataset2.datasets.values()
                    ):
                        self.assertIsNot(ds, ds2)
                else:
                    for ds, ds2 in zip(
                        mt_dataset.datasets.values(), mt_dataset2.datasets.values()
                    ):
                        self.assertIs(ds, ds2)
                # test with mask
                mask = torch.tensor([0, 1, 1], dtype=torch.bool)
                mt_dataset2 = mt_dataset.clone(deepcopy=use_deepcopy, mask=mask)
                # mask should only apply to target dataset.
                # All non-target datasets should be included.
                full_mask = torch.tensor([1, 1, 1, 0, 1, 1], dtype=torch.bool)
                self.assertTrue(torch.equal(mt_dataset.X[full_mask], mt_dataset2.X))
                self.assertTrue(torch.equal(mt_dataset.Y[full_mask], mt_dataset2.Y))
                if has_yvar:
                    self.assertTrue(
                        torch.equal(mt_dataset.Yvar[full_mask], mt_dataset2.Yvar)
                    )
                else:
                    self.assertIsNone(mt_dataset2.Yvar)
                self.assertEqual(mt_dataset.feature_names, mt_dataset2.feature_names)
                self.assertEqual(mt_dataset.outcome_names, mt_dataset2.outcome_names)

    def test_contextual_datasets(self):
        num_contexts = 3
        feature_names = [f"x_c{i}" for i in range(num_contexts)]
        parameter_decomposition = {
            "context_2": ["x_c2"],
            "context_1": ["x_c1"],
            "context_0": ["x_c0"],
        }
        context_buckets = list(parameter_decomposition.keys())
        context_outcome_list = [f"y:context_{i}" for i in range(num_contexts)]
        metric_decomposition = {f"{c}": [f"y:{c}"] for c in context_buckets}

        # test construction of agg outcome
        context_dt, dataset_list1 = make_contextual_dataset(
            has_yvar=True, contextual_outcome=False
        )
        self.assertEqual(len(context_dt.datasets), len(dataset_list1))
        self.assertListEqual(context_dt.context_buckets, context_buckets)
        self.assertListEqual(context_dt.outcome_names, ["y"])
        self.assertListEqual(context_dt.feature_names, feature_names)
        self.assertIs(context_dt.datasets["y"], dataset_list1[0])
        self.assertIs(context_dt.X, dataset_list1[0].X)
        self.assertIs(context_dt.Y, dataset_list1[0].Y)
        self.assertIs(context_dt.Yvar, dataset_list1[0].Yvar)

        # test construction of context outcome
        context_dt, dataset_list2 = make_contextual_dataset(
            has_yvar=True, contextual_outcome=True
        )
        self.assertEqual(len(context_dt.datasets), len(dataset_list2))
        # Ordering should match datasets, not parameter_decomposition
        context_order = [f"context_{i}" for i in range(num_contexts)]
        self.assertListEqual(context_dt.context_buckets, context_order)
        self.assertListEqual(context_dt.outcome_names, context_outcome_list)
        self.assertListEqual(context_dt.feature_names, feature_names)
        true_decomp = {f"context_{i}": [i] for i in range(3)}
        self.assertEqual(context_dt.parameter_index_decomp, true_decomp)
        self.assertTrue(torch.equal(context_dt.X, dataset_list2[-1].X))
        self.assertEqual(context_dt.Y.shape[-1], len(context_outcome_list))
        self.assertEqual(context_dt.Yvar.shape[-1], len(context_outcome_list))
        for dt in dataset_list2:
            self.assertIs(context_dt.datasets[dt.outcome_names[0]], dt)

        # Test handling None Yvar
        context_dt, dataset_list3 = make_contextual_dataset(
            has_yvar=False, contextual_outcome=True
        )
        self.assertIsNone(context_dt.Yvar)

        # test dataset validation
        wrong_metric_decomposition1 = {
            f"{c}": [f"y:{c}"] for c in context_buckets if c != "context_0"
        }
        wrong_metric_decomposition1["context_0"] = ["y:context_1"]
        with self.assertRaisesRegex(
            InputDataError,
            "metric_decomposition has same metric for multiple contexts.",
        ):
            ContextualDataset(
                datasets=dataset_list2,
                parameter_decomposition=parameter_decomposition,
                metric_decomposition=wrong_metric_decomposition1,
            )
        wrong_metric_decomposition2 = {
            f"{c}": [f"y:{c}"] for c in context_buckets if c != "context_0"
        }
        wrong_metric_decomposition2["context_0"] = ["y:context_0", "y:context_5"]
        with self.assertRaisesRegex(
            InputDataError,
            "All values in metric_decomposition must have the same length",
        ):
            ContextualDataset(
                datasets=dataset_list2,
                parameter_decomposition=parameter_decomposition,
                metric_decomposition=wrong_metric_decomposition2,
            )

        with self.assertRaisesRegex(
            InputDataError, "Require same X for context buckets"
        ):
            ContextualDataset(
                datasets=[
                    make_dataset(d=num_contexts, outcome_names=[m])
                    for m in context_outcome_list
                ],
                parameter_decomposition=parameter_decomposition,
                metric_decomposition=metric_decomposition,
            )

        with self.assertRaisesRegex(
            InputDataError,
            "metric_decomposition must be provided when there are multiple datasets.",
        ):
            ContextualDataset(
                datasets=dataset_list2,
                parameter_decomposition=parameter_decomposition,
            )

        with self.assertRaisesRegex(
            InputDataError,
            "metric_decomposition is redundant when there is "
            + "one dataset for overall outcome.",
        ):
            ContextualDataset(
                datasets=dataset_list1,
                parameter_decomposition=parameter_decomposition,
                metric_decomposition=metric_decomposition,
            )

        with self.assertRaisesRegex(
            InputDataError,
            "Keys of metric and parameter decompositions do not match.",
        ):
            ContextualDataset(
                datasets=dataset_list2,
                parameter_decomposition=parameter_decomposition,
                metric_decomposition={f"x{c}": [f"y:{c}"] for c in context_buckets},
            )

        wrong_metric_decomposition = {
            f"{c}": [f"y:{c}"] for c in context_buckets if c != "context_0"
        }
        wrong_metric_decomposition["context_0"] = ["wrong_metric"]
        missing_outcome = "y:context_0"
        with self.assertRaisesRegex(
            InputDataError, f"{missing_outcome} is missing in metric_decomposition."
        ):
            ContextualDataset(
                datasets=dataset_list2,
                parameter_decomposition=parameter_decomposition,
                metric_decomposition=wrong_metric_decomposition,
            )

        # Test error for mixed Yvar
        dataset_list3[0]._Yvar = dataset_list2[0]._Yvar
        with self.assertRaisesRegex(
            InputDataError, "Require Yvar to be specified for all buckets, or for none"
        ):
            ContextualDataset(
                datasets=dataset_list3,
                parameter_decomposition=parameter_decomposition,
                metric_decomposition=wrong_metric_decomposition,
            )

    def test_clone_contextual_dataset(self):
        for has_yvar, contextual_outcome in product((False, True), (False, True)):
            context_dt, _ = make_contextual_dataset(
                has_yvar=has_yvar, contextual_outcome=contextual_outcome
            )
            for use_deepcopy in [False, True]:
                context_dt2 = context_dt.clone(deepcopy=use_deepcopy)
                self.assertEqual(context_dt, context_dt2)
                self.assertTrue(torch.equal(context_dt.X, context_dt2.X))
                self.assertTrue(torch.equal(context_dt.Y, context_dt2.Y))
                if has_yvar:
                    self.assertTrue(torch.equal(context_dt.Yvar, context_dt2.Yvar))
                else:
                    self.assertIsNone(context_dt.Yvar)
                self.assertEqual(context_dt.feature_names, context_dt2.feature_names)
                self.assertEqual(context_dt.outcome_names, context_dt2.outcome_names)
                if use_deepcopy:
                    for ds, ds2 in zip(
                        context_dt.datasets.values(), context_dt2.datasets.values()
                    ):
                        self.assertIsNot(ds, ds2)
                else:
                    for ds, ds2 in zip(
                        context_dt.datasets.values(), context_dt2.datasets.values()
                    ):
                        self.assertIs(ds, ds2)
                # test with mask
                mask = torch.tensor([0, 1, 1], dtype=torch.bool)
                context_dt2 = context_dt.clone(deepcopy=use_deepcopy, mask=mask)
                self.assertTrue(torch.equal(context_dt.X[mask], context_dt2.X))
                self.assertTrue(torch.equal(context_dt.Y[mask], context_dt2.Y))
                if has_yvar:
                    self.assertTrue(
                        torch.equal(context_dt.Yvar[mask], context_dt2.Yvar)
                    )
                else:
                    self.assertIsNone(context_dt2.Yvar)
                self.assertEqual(context_dt.feature_names, context_dt2.feature_names)
                self.assertEqual(context_dt.outcome_names, context_dt2.outcome_names)
                self.assertEqual(
                    context_dt.parameter_decomposition,
                    context_dt2.parameter_decomposition,
                )
                if contextual_outcome:
                    self.assertEqual(
                        context_dt.metric_decomposition,
                        context_dt2.metric_decomposition,
                    )
                else:
                    self.assertIsNone(context_dt2.metric_decomposition)

    def test_contextual_dataset_equality(self) -> None:
        context_dt, _ = make_contextual_dataset(has_yvar=True, contextual_outcome=True)
        clone = context_dt.clone()
        self.assertEqual(context_dt, clone)
        for yvar, outcome in (
            (True, False),
            (False, True),
            (False, False),
        ):
            new_dt, _ = make_contextual_dataset(
                has_yvar=yvar, contextual_outcome=outcome
            )
            self.assertNotEqual(context_dt, new_dt)
