#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch_community.acquisition.log_expected_improvement_per_cost import (
    LogExpectedImprovementPerCost,
)


class TestLogExpectedImprovementPerCost(BotorchTestCase):
    def test_log_expected_improvement_with_cost(self) -> None:
        for dtype in (torch.float, torch.double):
            self._test_leic(dtype=dtype)

    def _test_leic(self, dtype: torch.dtype) -> None:
        mean = torch.tensor([[-0.5]], dtype=dtype)
        variance = torch.ones(1, 1, dtype=dtype)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))
        X = torch.empty(1, 1, dtype=dtype)  # dummy; posterior is mocked

        # Constant cost c=2: LogEIC should equal LogEI - log(2)
        def cost_fn(X):
            return torch.full(X.shape[:-1], 2.0, dtype=X.dtype)

        lei = LogExpectedImprovement(model=mm, best_f=0.0)
        leic = LogExpectedImprovementPerCost(
            model=mm, best_f=0.0, cost_callable=cost_fn
        )
        # t_batch_mode_transform adds a batch dim (1,1)->(1,1,1), so the cost
        # callable sees (1,d) and returns (1,), while the mock gives log_ei
        # shape () — squeeze to compare scalar values.
        self.assertAllClose(leic(X).squeeze(), lei(X) - math.log(2.0), atol=1e-5)

        # alpha=2: LogEIC = LogEI - 2*log(c)
        leic_alpha2 = LogExpectedImprovementPerCost(
            model=mm, best_f=0.0, cost_callable=cost_fn, alpha=2.0
        )
        self.assertAllClose(
            leic_alpha2(X).squeeze(), lei(X) - 2.0 * math.log(2.0), atol=1e-5
        )

        # maximize=True (explicit)
        lei_max = LogExpectedImprovement(model=mm, best_f=0.0, maximize=True)
        leic_max = LogExpectedImprovementPerCost(
            model=mm, best_f=0.0, cost_callable=cost_fn, maximize=True
        )
        self.assertAllClose(
            leic_max(X).squeeze(), lei_max(X) - math.log(2.0), atol=1e-5
        )

        # maximize=False
        lei_min = LogExpectedImprovement(model=mm, best_f=0.0, maximize=False)
        leic_min = LogExpectedImprovementPerCost(
            model=mm, best_f=0.0, cost_callable=cost_fn, maximize=False
        )
        self.assertAllClose(
            leic_min(X).squeeze(), lei_min(X) - math.log(2.0), atol=1e-5
        )

        # Input-dependent cost: at x=0, c(x)=1.0, so LogEIC = LogEI
        def cost_fn2(X):
            return 1.0 + X[..., 0].abs()

        leic_xdep = LogExpectedImprovementPerCost(
            model=mm, best_f=0.0, cost_callable=cost_fn2
        )
        X_zero = torch.zeros(1, 1, dtype=dtype)
        self.assertAllClose(leic_xdep(X_zero).squeeze(), lei(X_zero), atol=1e-5)

        # Batch mode: X = (b, 1, d)
        X_batch = torch.empty(3, 1, 1, dtype=dtype)
        mean_b = torch.full((3, 1, 1), -0.5, dtype=dtype)
        var_b = torch.ones(3, 1, 1, dtype=dtype)
        mm_b = MockModel(MockPosterior(mean=mean_b, variance=var_b))
        leic_b = LogExpectedImprovementPerCost(
            model=mm_b, best_f=0.0, cost_callable=cost_fn
        )
        self.assertEqual(leic_b(X_batch).shape, torch.Size([3]))
