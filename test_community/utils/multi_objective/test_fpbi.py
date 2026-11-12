# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from types import SimpleNamespace
from unittest import skipUnless
from unittest.mock import patch

import torch
from botorch.models.deterministic import GenericDeterministicModel
from botorch.utils.testing import BotorchTestCase
from botorch_community.utils.multi_objective import fpbi as fpbi_module
from botorch_community.utils.multi_objective.fpbi import (
    fpbi_geometry,
    sample_fpbi,
    sample_model_fpbi,
)
from torch import Tensor


def quadratic(X: Tensor) -> Tensor:
    x = X[..., 0]
    return torch.stack((x.square(), (1 - x).square()), dim=-1)


class TestFPBI(BotorchTestCase):
    def bounds(self, d: int = 1, dtype: torch.dtype = torch.double) -> Tensor:
        return torch.stack((torch.zeros(d), torch.ones(d))).to(
            dtype=dtype, device=self.device
        )

    def test_paper_grid_and_orientation(self) -> None:
        E = self.bounds(2)
        E[0] = E.new_tensor([0, 1])
        E[1] = E.new_tensor([1, 0])
        geometry = fpbi_geometry(E, n_steps=4, maximize=False)
        torch.testing.assert_close(geometry.center, E.new_tensor([0.5, 0.5]))
        torch.testing.assert_close(geometry.direction, E.new_tensor([-0.5, -0.5]))
        expected = E.new_tensor([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]])
        torch.testing.assert_close(geometry.anchors, expected)
        torch.testing.assert_close(fpbi_geometry(-E, n_steps=4).anchors, expected)
        self.assertFalse(geometry.used_fallback_basis)

    def test_batched_flat_geometry(self) -> None:
        E = self.bounds(3).new_tensor([[0, 1, 4], [1, 0, 4], [0.5, 0.5, 4]])
        geometry = fpbi_geometry(torch.stack([E, E]), n_steps=2, maximize=False)
        self.assertEqual(geometry.anchors.shape, torch.Size([2, 4, 3]))
        self.assertEqual(geometry.basis.shape, torch.Size([2, 3, 2]))
        self.assertEqual(geometry.degenerate.shape, torch.Size([2]))
        self.assertFalse(geometry.active_objectives[:, 2].any())
        self.assertFalse(geometry.anchors[..., 2].any())
        self.assertFalse(geometry.basis[..., 1].any())
        with self.assertRaisesRegex(ValueError, "flat"):
            fpbi_geometry(E, focus=E.new_tensor([0, 0, 1]))
        with self.assertRaisesRegex(ValueError, "broadcast"):
            fpbi_geometry(E, focus=E.new_zeros(2, 3))

    def test_generalized_cross_and_rank_deficiency(self) -> None:
        E = torch.ones(4, 4, dtype=torch.double, device=self.device) - torch.eye(
            4, dtype=torch.double, device=self.device
        )
        geometry = fpbi_geometry(E, n_steps=2, maximize=False)
        self.assertEqual(geometry.anchors.shape, torch.Size([8, 4]))
        torch.testing.assert_close(
            geometry.basis.T @ geometry.basis,
            torch.eye(3, dtype=E.dtype, device=E.device),
        )
        torch.testing.assert_close(
            geometry.basis.T @ geometry.direction, E.new_zeros(3)
        )
        deficient = E.new_tensor(
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
        )
        completed = fpbi_geometry(deficient, n_steps=2, maximize=False)
        self.assertTrue(completed.used_fallback_basis)
        torch.testing.assert_close(
            completed.basis.T @ completed.basis,
            torch.eye(3, dtype=E.dtype, device=E.device),
        )

    def test_dominated_and_duplicate_extrema_do_not_distort_geometry(self) -> None:
        E = self.bounds(4).new_tensor(
            [[0, 1, 1, 7], [1, 0, 1, 7], [1, 1, 0, 7], [1, 1, 0, 7]]
        )
        first = fpbi_geometry(E, n_steps=2, maximize=False)
        self.assertFalse(first.used_fallback_basis)
        E[-1] = 100
        second = fpbi_geometry(E, n_steps=2, maximize=False)
        torch.testing.assert_close(second.anchors, first.anchors)
        torch.testing.assert_close(second.nadir, E.new_tensor([1, 1, 1, 7]))

    def test_analytic_quadratic_front(self) -> None:
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                bounds = self.bounds(dtype=dtype)
                tolerance = 2e-5 if dtype == torch.float else 1e-6
                result = sample_fpbi(
                    quadratic,
                    bounds,
                    n_steps=3,
                    num_restarts=2,
                    maximize=False,
                    feasibility_tol=tolerance,
                )
                self.assertEqual(result.X.dtype, dtype)
                self.assertEqual(result.Y.dtype, dtype)
                self.assertEqual(result.X.device, bounds.device)
                self.assertGreaterEqual(len(result.X), 4)
                torch.testing.assert_close(result.Y, quadratic(result.X))
                self.assertTrue(result.ray_succeeded.all())
                self.assertTrue(((result.X[:, 0] > 0.1) & (result.X[:, 0] < 0.9)).any())
                self.assertTrue(
                    all(
                        not d.accepted
                        or (d.solver_success and d.max_violation <= tolerance)
                        for d in result.diagnostics
                    )
                )
                centered = sample_fpbi(
                    quadratic,
                    bounds,
                    n_steps=2,
                    num_restarts=1,
                    maximize=False,
                    feasibility_tol=tolerance,
                )
                kappas = [
                    d.kappa
                    for d in centered.diagnostics
                    if d.kind == "ray" and d.index == 0 and d.accepted
                ]
                self.assertAlmostEqual(max(kappas), 0.5, delta=1e-4)

    def test_constraints_apply_to_extremes_and_rays(self) -> None:
        def feasible(X: Tensor) -> Tensor:
            return torch.stack((X[..., 0] - 0.2, 0.8 - X[..., 0]), dim=-1)

        result = sample_fpbi(
            quadratic,
            self.bounds(),
            n_steps=2,
            num_restarts=2,
            maximize=False,
            inequality_constraints=feasible,
        )
        self.assertTrue((feasible(result.X) >= -1e-6).all())
        self.assertTrue((feasible(result.extreme_X) >= -1e-6).all())
        self.assertTrue(result.ray_succeeded.all())
        kappas = [
            d.kappa
            for d in result.diagnostics
            if d.kind == "ray" and d.index == 0 and d.accepted
        ]
        self.assertAlmostEqual(max(kappas), 0.3, delta=1e-5)

    def test_negative_unrestricted_intersection_parameter(self) -> None:
        def concave(X: Tensor) -> Tensor:
            x = X[..., 0]
            return torch.stack((x.sqrt(), (1 - x).sqrt()), dim=-1)

        result = sample_fpbi(
            concave, self.bounds(), n_steps=2, num_restarts=1, maximize=False
        )
        kappas = [
            d.kappa
            for d in result.diagnostics
            if d.kind == "ray" and d.index == 0 and d.accepted
        ]
        self.assertAlmostEqual(max(kappas), 1 - math.sqrt(2), delta=1e-5)

    def test_actual_model_wrapper_and_equalities(self) -> None:
        model = GenericDeterministicModel(quadratic, num_outputs=2)
        result = sample_model_fpbi(
            model,
            self.bounds(2),
            n_steps=2,
            num_restarts=1,
            maximize=False,
            equality_constraints=lambda X: X[:, 1:] - 0.4,
        )
        torch.testing.assert_close(
            result.X[:, 1], result.X.new_full((len(result.X),), 0.4)
        )
        torch.testing.assert_close(result.Y, quadratic(result.X))

    def test_batched_model_requires_explicit_selection(self) -> None:
        model = SimpleNamespace(
            posterior=lambda X: SimpleNamespace(
                mean=torch.stack((quadratic(X), quadratic(X) + 2))
            )
        )
        with self.assertRaisesRegex(ValueError, "Batched posterior"):
            sample_model_fpbi(model, self.bounds(), n_steps=2, num_restarts=1)
        result = sample_model_fpbi(
            model,
            self.bounds(),
            model_batch_index=(1,),
            n_steps=2,
            num_restarts=1,
            maximize=False,
        )
        torch.testing.assert_close(result.Y, quadratic(result.X) + 2)

    def test_flat_front_and_duplicate_ray_flags(self) -> None:
        result = sample_fpbi(
            lambda X: X.new_full((len(X), 2), 7),
            self.bounds(),
            n_steps=2,
            num_restarts=1,
        )
        self.assertTrue(result.geometry.degenerate)
        self.assertEqual(len(result.X), 1)
        self.assertFalse(result.ray_succeeded.any())
        self.assertTrue(all(d.kind == "extreme" for d in result.diagnostics))
        nonflat = sample_fpbi(
            lambda X: torch.cat((quadratic(X), X.new_full((len(X), 1), 7)), -1),
            self.bounds(),
            n_steps=2,
            num_restarts=1,
            maximize=False,
        )
        self.assertTrue(nonflat.ray_succeeded.all())
        self.assertLess(
            len({d.index for d in nonflat.diagnostics if d.kind == "ray"}),
            nonflat.ray_succeeded.numel(),
        )

    def test_failed_extreme_raises(self) -> None:
        minimize = fpbi_module.minimize

        def fail(*args, **kwargs):
            result = minimize(*args, **kwargs)
            result.success, result.status, result.message = False, 9, "injected failure"
            return result

        with patch.object(fpbi_module, "minimize", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "extreme"):
                sample_fpbi(quadratic, self.bounds(), n_steps=2, num_restarts=1)

    def test_failed_rays_are_diagnosed_and_omitted(self) -> None:
        minimize = fpbi_module.minimize

        def fail_ray(*args, **kwargs):
            result = minimize(*args, **kwargs)
            if len(kwargs["bounds"]) == 2:
                result.success = False
                result.status = 9
                result.message = "injected ray failure"
            return result

        with patch.object(fpbi_module, "minimize", side_effect=fail_ray):
            result = sample_fpbi(
                quadratic, self.bounds(), n_steps=2, num_restarts=1, maximize=False
            )
        self.assertFalse(result.ray_succeeded.any())
        self.assertEqual(len(result.X), 2)
        rays = [d for d in result.diagnostics if d.kind == "ray"]
        self.assertTrue(rays)
        self.assertTrue(all(not d.accepted and d.status == 9 for d in rays))
        self.assertTrue(all(d.message == "injected ray failure" for d in rays))

    def test_success_flag_does_not_override_infeasibility(self) -> None:
        minimize = fpbi_module.minimize

        def infeasible_ray(*args, **kwargs):
            result = minimize(*args, **kwargs)
            if len(kwargs["bounds"]) == 2:
                result.x[-1] += 1
                result.fun = -result.x[-1]
                result.success = True
            return result

        with patch.object(fpbi_module, "minimize", side_effect=infeasible_ray):
            result = sample_fpbi(
                quadratic, self.bounds(), n_steps=2, num_restarts=1, maximize=False
            )
        self.assertFalse(result.ray_succeeded.any())
        self.assertTrue(
            all(
                d.solver_success and not d.accepted and d.max_violation > 1e-6
                for d in result.diagnostics
                if d.kind == "ray"
            )
        )

    def test_objective_and_constraint_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            sample_fpbi(lambda X: X.new_full((len(X), 2), math.nan), self.bounds())
        with self.assertRaisesRegex(ValueError, "shape"):
            sample_fpbi(lambda X: X[:, :1], self.bounds())
        with self.assertRaisesRegex(ValueError, "constraint"):
            sample_fpbi(
                quadratic,
                self.bounds(),
                num_restarts=1,
                inequality_constraints=lambda X: X.new_full((len(X), 1), math.nan),
            )
        calls = 0

        def changed_shape(X: Tensor) -> Tensor:
            nonlocal calls
            calls += 1
            return X.new_zeros((len(X), 2 if calls == 1 else 3))

        with self.assertRaisesRegex(ValueError, "number of outputs"):
            sample_fpbi(changed_shape, self.bounds(), num_restarts=1)

    def test_invalid_options_and_numerical_guards(self) -> None:
        for kwargs in (
            {"n_steps": 0},
            {"n_steps": True},
            {"max_rays": 1},
            {"focus": [math.nan, 0]},
            {"focus": [2, 2]},
            {"seed": -1},
            {"num_restarts": 0},
            {"feasibility_tol": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sample_fpbi(quadratic, self.bounds(), maximize=False, **kwargs)
        with self.assertRaisesRegex(TypeError, "float32 or float64"):
            sample_fpbi(quadratic, self.bounds(dtype=torch.float16))
        with self.assertRaisesRegex(ValueError, "widths"):
            sample_fpbi(quadratic, self.bounds().new_tensor([[-1e308], [1e308]]))
        with self.assertRaisesRegex(ValueError, "rescale"):
            fpbi_geometry(self.bounds().new_tensor([[-1e308, 1e308], [1e308, -1e308]]))
        with self.assertRaisesRegex(ValueError, "at least two"):
            fpbi_geometry(self.bounds().new_zeros(1, 1))


@skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class TestFPBICUDA(BotorchTestCase):
    device = torch.device("cuda")

    def test_callback_boundary_preserves_cuda(self) -> None:
        seen = []

        def objective(X: Tensor) -> Tensor:
            seen.append(X.device)
            return quadratic(X)

        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=self.device)
        result = sample_fpbi(
            objective, bounds, n_steps=2, num_restarts=1, maximize=False
        )
        self.assertTrue(seen and all(d.type == "cuda" for d in seen))
        for output in (
            result.X,
            result.Y,
            result.geometry.anchors,
            result.ray_succeeded,
        ):
            self.assertEqual(output.device.type, "cuda")
        self.assertTrue(result.ray_succeeded.all())
