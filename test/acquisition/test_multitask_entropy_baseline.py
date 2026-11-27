# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
)
from botorch.models import MultiTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase


class TestMultiTaskEntropyBaselines(BotorchTestCase):
    def test_task_free_baselines_use_selected_output_task(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            for task_feature in (0, 1, -1):
                for acquisition_type in (qMaxValueEntropy, qLowerBoundMaxValueEntropy):
                    with self.subTest(
                        device=device,
                        task_feature=task_feature,
                        acquisition=acquisition_type.__name__,
                    ):
                        continuous = torch.tensor(
                            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.7], [0.8, 0.9]],
                            dtype=torch.double,
                            device=device,
                        )
                        labels = continuous.new_tensor([[2.0], [5.0], [2.0], [5.0]])
                        index = task_feature % 3
                        train_X = torch.cat(
                            (continuous[:, :index], labels, continuous[:, index:]), -1
                        )
                        model = MultiTaskGP(
                            train_X,
                            continuous[:, :1].sin(),
                            task_feature,
                            train_Yvar=torch.full_like(labels, 0.03),
                            output_tasks=[5],
                            outcome_transform=None,
                        )
                        candidate = continuous.new_tensor([[0.2, 0.3], [0.6, 0.8]])
                        kwargs = dict(num_mv_samples=3, use_gumbel=False)
                        if acquisition_type is qMaxValueEntropy:
                            kwargs.update(num_y_samples=8, num_fantasies=2)
                        automatic = acquisition_type(model, candidate, **kwargs)
                        supplied = acquisition_type(
                            model, candidate, train_inputs=train_X, **kwargs
                        )
                        reduced = acquisition_type(
                            model, candidate, train_inputs=continuous, **kwargs
                        )
                        expected = torch.cat((candidate, continuous))
                        for acquisition in (automatic, supplied, reduced):
                            self.assertAllClose(acquisition.candidate_set, expected)
                            acquisition.posterior_max_values = candidate.new_tensor(
                                [[0.8], [1.0], [1.2]]
                            )
                            if acquisition_type is qMaxValueEntropy:
                                acquisition.sampler = SobolQMCNormalSampler(
                                    torch.Size([8]), seed=19
                                )
                        query = candidate[:1].unsqueeze(0).requires_grad_()
                        value = automatic(query)
                        self.assertAllClose(value, reduced(query))
                        self.assertAllClose(value, supplied(query))
                        self.assertTrue(torch.isfinite(value).all())
                        self.assertTrue(
                            torch.isfinite(
                                torch.autograd.grad(value.sum(), query)[0]
                            ).all()
                        )

    def test_explicit_task_columns_are_preserved(self):
        for acquisition_type in (qMaxValueEntropy, qLowerBoundMaxValueEntropy):
            with self.subTest(acquisition=acquisition_type.__name__):
                train_X = torch.tensor(
                    [[0.1, 2.0], [0.3, 5.0], [0.5, 2.0], [0.8, 5.0]], dtype=torch.double
                )
                model = MultiTaskGP(train_X, train_X[:, :1].sin(), -1, output_tasks=[5])
                candidate = train_X.new_tensor([[0.2, 5.0], [0.7, 2.0]])
                acquisition = acquisition_type(
                    model, candidate, num_mv_samples=3, use_gumbel=False
                )
                self.assertAllClose(
                    acquisition.candidate_set, torch.cat((candidate, train_X))
                )
