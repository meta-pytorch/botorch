#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.kernels.orthogonal_additive_kernel import (
    OrthogonalAdditiveKernel,
    SECOND_ORDER_PRIOR_ERROR_MSG,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.constraints import Positive
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.lazy import LazyEvaluatedKernelTensor
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior
from gpytorch.priors.torch_priors import GammaPrior, HalfCauchyPrior, UniformPrior
from torch import nn, Tensor


class TestOrthogonalAdditiveKernel(BotorchTestCase):
    def test_orthogonal_additive_kernel(self):
        self._test_kernel()
        self._test_default_base_kernel()
        self._test_priors()
        self._test_set_coeffs()
        self._test_non_reduced_forward()
        self._test_non_reduced_forward_batch_shape()
        self._test_normalizer_preserves_eval_mode()
        self._test_component_indices()

    def _test_kernel(self):
        n, d = 3, 5
        dtypes = [torch.float, torch.double]
        batch_shapes = [(), (2,), (7, 2)]
        for dtype in dtypes:
            tkwargs = {"dtype": dtype, "device": self.device}

            # test with default args and batch_shape = None in second_order
            oak = OrthogonalAdditiveKernel(
                d,
                RBFKernel(),
                per_dim_lengthscales=False,
                batch_shape=None,
                second_order=True,
            )
            self.assertEqual(oak.batch_shape, torch.Size([]))

            for batch_shape in batch_shapes:
                X = torch.rand(*batch_shape, n, d, **tkwargs)
                base_kernel = MaternKernel().to(device=self.device)
                oak = OrthogonalAdditiveKernel(
                    d,
                    base_kernel,
                    per_dim_lengthscales=False,
                    second_order=False,
                    batch_shape=batch_shape,
                    **tkwargs,
                )
                KL = oak(X)
                self.assertIsInstance(KL, LazyEvaluatedKernelTensor)
                KM = KL.to_dense()
                self.assertIsInstance(KM, Tensor)
                self.assertEqual(KM.shape, (*batch_shape, n, n))
                self.assertEqual(KM.dtype, dtype)
                self.assertEqual(KM.device.type, self.device.type)
                # symmetry
                self.assertTrue(torch.allclose(KM, KM.transpose(-2, -1)))
                # positivity
                self.assertTrue(isposdef(KM))
                # diag=True should match diagonal of full matrix
                K_diag = oak.forward(X, X, diag=True)
                self.assertEqual(K_diag.shape, torch.Size([*batch_shape, n]))
                self.assertAllClose(
                    K_diag, torch.diagonal(KM, dim1=-2, dim2=-1), atol=1e-5
                )

                # testing differentiability
                X.requires_grad = True
                oak(X).to_dense().sum().backward()
                self.assertFalse(X.grad.isnan().any())
                self.assertFalse(X.grad.isinf().any())

                X_out_of_hypercube = torch.rand(n, d, **tkwargs) + 1
                with self.assertRaisesRegex(ValueError, r"x1.*hypercube"):
                    oak(X_out_of_hypercube, X).to_dense()

                with self.assertRaisesRegex(ValueError, r"x2.*hypercube"):
                    oak(X, X_out_of_hypercube).to_dense()

                with self.assertRaisesRegex(UnsupportedError, "does not support"):
                    oak.forward(x1=X, x2=X, last_dim_is_batch=True)

                X2 = torch.rand(*batch_shape, n, d, **tkwargs)
                with self.assertRaisesRegex(UnsupportedError, "diag=True"):
                    oak.forward(x1=X, x2=X2, diag=True)

                oak_2nd = OrthogonalAdditiveKernel(
                    d,
                    base_kernel,
                    per_dim_lengthscales=False,
                    second_order=True,
                    batch_shape=batch_shape,
                    **tkwargs,
                )
                KL2 = oak_2nd(X)
                self.assertIsInstance(KL2, LazyEvaluatedKernelTensor)
                KM2 = KL2.to_dense()
                self.assertIsInstance(KM2, Tensor)
                self.assertEqual(KM2.shape, (*batch_shape, n, n))
                # symmetry
                self.assertTrue(torch.allclose(KM2, KM2.transpose(-2, -1)))
                # positivity
                self.assertTrue(isposdef(KM2))
                self.assertEqual(KM2.dtype, dtype)
                self.assertEqual(KM2.device.type, self.device.type)
                K_diag_2nd = oak_2nd.forward(X, X, diag=True)
                self.assertAllClose(
                    K_diag_2nd,
                    torch.diagonal(KM2, dim1=-2, dim2=-1),
                    atol=1e-5,
                )

                # testing second order coefficient matrices are upper-triangular
                # and contain the transformed values in oak_2nd.raw_coeffs_2
                oak_2nd.raw_coeffs_2 = nn.Parameter(
                    torch.randn_like(oak_2nd.raw_coeffs_2)
                )
                C2 = oak_2nd.coeffs_2
                self.assertTrue(C2.shape == (*batch_shape, d, d))
                self.assertTrue((C2.tril() == 0).all())
                c2 = oak_2nd.coeff_constraint.transform(oak_2nd.raw_coeffs_2)
                i, j = torch.triu_indices(d, d, offset=1)
                self.assertTrue(torch.allclose(C2[..., i, j], c2))

                # second order effects change the correlation structure
                self.assertFalse(torch.allclose(KM, KM2))

                # check normalizer shape: should be (d, 1, 1)
                normalizer = oak.normalizer()
                self.assertEqual(normalizer.shape, (d, 1, 1))
                normalizer_2nd = oak_2nd.normalizer()
                self.assertEqual(normalizer_2nd.shape, (d, 1, 1))

                # check orthogonality of base kernels
                n_test = 7
                # inputs on which to evaluate orthogonality
                X_ortho = torch.rand(n_test, d, **tkwargs)
                # d x quad_deg x quad_deg
                K_ortho = oak._orthogonal_base_kernels(X_ortho, oak.z)

                # NOTE: at each random test input x_i and for each dimension d,
                # sum_j k_d(x_i, z_j) * w_j = 0.
                # Note that this implies the GP mean will be orthogonal as well:
                # mean(x) = sum_j k(x, x_j) alpha_j
                # so
                # sum_i mean(z_i) w_i
                # = sum_j alpha_j (sum_i k(z_i, x_j) w_i) // exchanging summations order
                # = sum_j alpha_j (0) // due to symmetry
                # = 0
                tol = 1e-5
                self.assertTrue(((K_ortho @ oak.w).squeeze(-1) < tol).all())

    def _test_default_base_kernel(self):
        """Test base_kernel=None with per_dim_lengthscales=True/False
        and validation when an explicit base_kernel is provided."""
        d = 5
        tkwargs = {"dtype": torch.double, "device": self.device}
        n = 7

        # --- 1. per_dim_lengthscales=True (default), no batch_shape ---
        oak = OrthogonalAdditiveKernel(dim=d, **tkwargs)
        self.assertTrue(oak.per_dim_lengthscales)
        self.assertIsInstance(oak.base_kernel, RBFKernel)
        self.assertEqual(oak.base_kernel.batch_shape, torch.Size([d]))
        self.assertEqual(oak.base_kernel.lengthscale.shape, torch.Size([d, 1, 1]))

        # Verify the kernel produces correct output shapes
        X = torch.rand(n, d, **tkwargs)
        K = oak(X).to_dense()
        self.assertEqual(K.shape, torch.Size([n, n]))
        self.assertTrue(torch.allclose(K, K.transpose(-2, -1)))
        self.assertTrue(isposdef(K))

        # Verify per-dimension lengthscales are independent: set distinct values
        # and confirm they are stored correctly
        distinct_ls = torch.arange(1.0, d + 1.0, **tkwargs).view(d, 1, 1)
        oak.base_kernel.lengthscale = distinct_ls
        self.assertAllClose(oak.base_kernel.lengthscale, distinct_ls)

        # --- 2. per_dim_lengthscales=True WITH batch_shape ---
        for batch_shape in [(2,), (3, 2)]:
            oak_batch = OrthogonalAdditiveKernel(
                dim=d, batch_shape=batch_shape, **tkwargs
            )
            self.assertIsInstance(oak_batch.base_kernel, RBFKernel)
            expected_base_batch = torch.Size(batch_shape) + torch.Size([d])
            self.assertEqual(oak_batch.base_kernel.batch_shape, expected_base_batch)
            expected_ls_shape = torch.Size(batch_shape) + torch.Size([d, 1, 1])
            self.assertEqual(oak_batch.base_kernel.lengthscale.shape, expected_ls_shape)

            # OAK's reported batch_shape should be the external batch only
            self.assertEqual(oak_batch.batch_shape, torch.Size(batch_shape))

            # Verify normalizer has correct shape: (*batch_shape, d, 1, 1)
            normalizer = oak_batch.normalizer()
            self.assertEqual(normalizer.shape, torch.Size([*batch_shape, d, 1, 1]))

            # Verify the kernel produces correct output shapes with batched inputs
            X_batch = torch.rand(*batch_shape, n, d, **tkwargs)
            K_batch = oak_batch(X_batch).to_dense()
            self.assertEqual(K_batch.shape, torch.Size([*batch_shape, n, n]))
            self.assertTrue(torch.allclose(K_batch, K_batch.transpose(-2, -1)))
            self.assertTrue(isposdef(K_batch))

        # --- 3. per_dim_lengthscales=False, no batch_shape ---
        oak_shared = OrthogonalAdditiveKernel(
            dim=d, per_dim_lengthscales=False, **tkwargs
        )
        self.assertFalse(oak_shared.per_dim_lengthscales)
        self.assertIsInstance(oak_shared.base_kernel, RBFKernel)
        # Shared kernel: batch_shape should be empty
        self.assertEqual(oak_shared.base_kernel.batch_shape, torch.Size([]))
        # Single lengthscale shared across all dimensions
        self.assertEqual(oak_shared.base_kernel.lengthscale.shape, torch.Size([1, 1]))

        # Verify it still produces valid outputs
        K_shared = oak_shared(X).to_dense()
        self.assertEqual(K_shared.shape, torch.Size([n, n]))
        self.assertTrue(isposdef(K_shared))

        # --- 4. per_dim_lengthscales=False WITH batch_shape ---
        oak_shared_batch = OrthogonalAdditiveKernel(
            dim=d, per_dim_lengthscales=False, batch_shape=(2,), **tkwargs
        )
        self.assertEqual(oak_shared_batch.base_kernel.batch_shape, torch.Size([2]))

        # --- 5. Explicit base_kernel, per_dim_lengthscales=False (no warning) ---
        explicit_rbf = RBFKernel()
        oak_explicit = OrthogonalAdditiveKernel(
            d, base_kernel=explicit_rbf, per_dim_lengthscales=False, **tkwargs
        )
        self.assertIs(oak_explicit.base_kernel, explicit_rbf)
        self.assertEqual(oak_explicit.base_kernel.batch_shape, torch.Size([]))

        # --- 6. Explicit base_kernel with mismatched batch_shape (warning) ---
        with self.assertWarnsRegex(UserWarning, "batch_shape"):
            OrthogonalAdditiveKernel(
                d, base_kernel=RBFKernel(), per_dim_lengthscales=True, **tkwargs
            )

        # No warning when base_kernel matches expected batch_shape
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            OrthogonalAdditiveKernel(
                d,
                base_kernel=RBFKernel(batch_shape=torch.Size([d])),
                per_dim_lengthscales=True,
                **tkwargs,
            )

    def _test_priors(self):
        d = 5
        dtypes = [torch.float, torch.double]
        batch_shapes = [(), (2,), (7, 2)]

        # test no prior
        oak = OrthogonalAdditiveKernel(
            d,
            RBFKernel(),
            per_dim_lengthscales=False,
            batch_shape=None,
            second_order=True,
        )
        for dtype, batch_shape in itertools.product(dtypes, batch_shapes):
            # test with default args and batch_shape = None in second_order
            tkwargs = {"dtype": dtype, "device": self.device}
            offset_prior = HalfCauchyPrior(0.1).to(**tkwargs)
            coeffs_1_prior = LogNormalPrior(0, 1).to(**tkwargs)
            coeffs_2_prior = GammaPrior(3, 6).to(**tkwargs)
            oak = OrthogonalAdditiveKernel(
                d,
                RBFKernel(),
                per_dim_lengthscales=False,
                second_order=True,
                offset_prior=offset_prior,
                coeffs_1_prior=coeffs_1_prior,
                coeffs_2_prior=coeffs_2_prior,
                batch_shape=batch_shape,
                **tkwargs,
            )

            self.assertIsInstance(oak.offset_prior, HalfCauchyPrior)
            self.assertIsInstance(oak.coeffs_1_prior, LogNormalPrior)
            self.assertEqual(oak.coeffs_1_prior.scale, 1)
            self.assertEqual(oak.coeffs_2_prior.concentration, 3)

            oak = OrthogonalAdditiveKernel(
                d,
                RBFKernel(),
                per_dim_lengthscales=False,
                second_order=True,
                coeffs_1_prior=None,
                coeffs_2_prior=coeffs_2_prior,
                batch_shape=batch_shape,
                **tkwargs,
            )
            self.assertEqual(oak.coeffs_2_prior.concentration, 3)
            with self.assertRaisesRegex(
                AttributeError,
                "'OrthogonalAdditiveKernel' object has no attribute 'coeffs_1_prior",
            ):
                _ = oak.coeffs_1_prior
                # test with batch_shape = None in second_order
            oak = OrthogonalAdditiveKernel(
                d,
                RBFKernel(),
                per_dim_lengthscales=False,
                second_order=True,
                coeffs_1_prior=coeffs_1_prior,
                batch_shape=batch_shape,
                **tkwargs,
            )
        with self.assertRaisesRegex(AttributeError, SECOND_ORDER_PRIOR_ERROR_MSG):
            OrthogonalAdditiveKernel(
                d,
                RBFKernel(),
                per_dim_lengthscales=False,
                batch_shape=None,
                second_order=False,
                coeffs_2_prior=GammaPrior(1, 1),
            )

        # train the model to ensure that param setters are called
        train_X = torch.rand(5, d, dtype=dtype, device=self.device)
        train_Y = torch.randn(5, 1, dtype=dtype, device=self.device)

        oak = OrthogonalAdditiveKernel(
            d,
            RBFKernel(),
            per_dim_lengthscales=False,
            batch_shape=None,
            second_order=True,
            offset_prior=offset_prior,
            coeffs_1_prior=coeffs_1_prior,
            coeffs_2_prior=coeffs_2_prior,
            **tkwargs,
        )
        model = SingleTaskGP(train_X=train_X, train_Y=train_Y, covar_module=oak)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 2}})

        unif_prior = UniformPrior(10, 11)
        # coeff_constraint is not enforced so that we can check the raw parameter
        # values and not the reshaped (triu transformed) ones
        oak_for_sample = OrthogonalAdditiveKernel(
            d,
            RBFKernel(),
            per_dim_lengthscales=False,
            batch_shape=None,
            second_order=True,
            offset_prior=unif_prior,
            coeffs_1_prior=unif_prior,
            coeffs_2_prior=unif_prior,
            coeff_constraint=Positive(transform=None, inv_transform=None),
            **tkwargs,
        )
        oak_for_sample.sample_from_prior("offset_prior")
        oak_for_sample.sample_from_prior("coeffs_1_prior")
        oak_for_sample.sample_from_prior("coeffs_2_prior")

        # check that all sampled values are within the bounds set by the priors
        self.assertTrue(torch.all(10 <= oak_for_sample.raw_offset <= 11))
        self.assertTrue(
            torch.all(
                (10 <= oak_for_sample.raw_coeffs_1)
                * (oak_for_sample.raw_coeffs_1 <= 11)
            )
        )
        self.assertTrue(
            torch.all(
                (10 <= oak_for_sample.raw_coeffs_2)
                * (oak_for_sample.raw_coeffs_2 <= 11)
            )
        )

    def _test_set_coeffs(self):
        d = 5
        dtype = torch.double
        oak = OrthogonalAdditiveKernel(
            d,
            RBFKernel(),
            per_dim_lengthscales=False,
            batch_shape=None,
            second_order=True,
            dtype=dtype,
        )
        constraint = oak.coeff_constraint
        coeffs_1 = torch.arange(d, dtype=dtype)
        coeffs_2 = torch.ones((d * d), dtype=dtype).reshape(d, d).triu()
        oak.coeffs_1 = coeffs_1
        oak.coeffs_2 = coeffs_2

        self.assertAllClose(
            oak.raw_coeffs_1,
            constraint.inverse_transform(coeffs_1),
        )
        # raw_coeffs_2 has length d * (d-1) / 2
        self.assertAllClose(
            oak.raw_coeffs_2, constraint.inverse_transform(torch.ones(10, dtype=dtype))
        )

        batch_shapes = torch.Size([2]), torch.Size([5, 2])
        for batch_shape in batch_shapes:
            dtype = torch.double
            oak = OrthogonalAdditiveKernel(
                d,
                RBFKernel(),
                per_dim_lengthscales=False,
                batch_shape=batch_shape,
                second_order=True,
                dtype=dtype,
                coeff_constraint=Positive(transform=None, inv_transform=None),
            )
            constraint = oak.coeff_constraint
            coeffs_1 = torch.arange(d, dtype=dtype)
            coeffs_2 = torch.ones((d * d), dtype=dtype).reshape(d, d).triu()
            oak.coeffs_1 = coeffs_1
            oak.coeffs_2 = coeffs_2

            self.assertEqual(oak.raw_coeffs_1.shape, batch_shape + torch.Size([5]))
            # raw_coeffs_2 has length d * (d-1) / 2
            self.assertEqual(oak.raw_coeffs_2.shape, batch_shape + torch.Size([10]))

            # test setting value as float
            oak.offset = 0.5
            self.assertAllClose(oak.offset, 0.5 * torch.ones_like(oak.offset))
            # raw_coeffs_2 has length d * (d-1) / 2
            oak.coeffs_1 = 0.2
            self.assertAllClose(
                oak.raw_coeffs_1, 0.2 * torch.ones_like(oak.raw_coeffs_1)
            )
            oak.coeffs_2 = 0.3
            self.assertAllClose(
                oak.raw_coeffs_2, 0.3 * torch.ones_like(oak.raw_coeffs_2)
            )
            # the lower triangular part is set to 0 automatically
            self.assertAllClose(
                oak.coeffs_2.tril(diagonal=-1), torch.zeros_like(oak.coeffs_2)
            )

    def _test_non_reduced_forward(self):
        """Test _non_reduced_forward returns component-wise kernel matrices."""
        n, d = 5, 4
        tkwargs = {"dtype": torch.double, "device": self.device}
        X1 = torch.rand(n, d, **tkwargs)
        X2 = torch.rand(n + 2, d, **tkwargs)
        base_kernel = RBFKernel().to(**tkwargs)

        # Test first-order only
        oak = OrthogonalAdditiveKernel(
            d,
            base_kernel,
            per_dim_lengthscales=False,
            second_order=False,
            **tkwargs,
        )
        K = oak._non_reduced_forward(X1, X2)
        # Expected shape: (1 + d) x n x (n + 2)
        expected_num_components = 1 + d
        self.assertEqual(K.shape, (expected_num_components, n, n + 2))

        # Test with second-order
        oak_2nd = OrthogonalAdditiveKernel(
            d,
            base_kernel,
            per_dim_lengthscales=False,
            second_order=True,
            **tkwargs,
        )
        K2 = oak_2nd._non_reduced_forward(X1, X2)
        # Expected shape: (1 + d + d*(d-1)/2) x n x (n + 2)
        num_second_order = d * (d - 1) // 2
        expected_num_components = 1 + d + num_second_order
        self.assertEqual(K2.shape, (expected_num_components, n, n + 2))

        # Verify that the sum of components equals the full kernel output
        K_full = oak._non_reduced_forward(X1, X2).sum(dim=0)
        K_standard = oak(X1, X2).to_dense()
        self.assertAllClose(K_full, K_standard, atol=1e-5)

        K2_full = oak_2nd._non_reduced_forward(X1, X2).sum(dim=0)
        K2_standard = oak_2nd(X1, X2).to_dense()
        self.assertAllClose(K2_full, K2_standard, atol=1e-5)

        # Test diag=True path
        X = torch.rand(n, d, **tkwargs)
        for oak_test, nc in [
            (oak, 1 + d),
            (oak_2nd, 1 + d + num_second_order),
        ]:
            K_diag = oak_test._non_reduced_forward(X, X, diag=True)
            K_full_diag = oak_test._non_reduced_forward(X, X)
            self.assertEqual(K_diag.shape, (nc, n))
            self.assertAllClose(
                K_diag, torch.diagonal(K_full_diag, dim1=-2, dim2=-1), atol=1e-5
            )

        # last_dim_is_batch raises UnsupportedError
        with self.assertRaises(UnsupportedError):
            oak._non_reduced_forward(X, X, last_dim_is_batch=True)

    def _test_non_reduced_forward_batch_shape(self):
        """Test _non_reduced_forward with batch-shaped inputs."""
        n, d = 5, 3
        tkwargs = {"dtype": torch.double, "device": self.device}

        for batch_shape in [(2,), (3, 2)]:
            X1 = torch.rand(*batch_shape, n, d, **tkwargs)
            X2 = torch.rand(*batch_shape, n + 1, d, **tkwargs)
            base_kernel = RBFKernel().to(**tkwargs)

            for second_order in [False, True]:
                oak = OrthogonalAdditiveKernel(
                    d,
                    base_kernel,
                    per_dim_lengthscales=False,
                    second_order=second_order,
                    batch_shape=batch_shape,
                    **tkwargs,
                )
                K = oak._non_reduced_forward(X1, X2)
                num_components = oak.num_components
                # Component dimension is inserted at -3, after the kernel batch_shape
                expected_shape = (*batch_shape, num_components, n, n + 1)
                self.assertEqual(K.shape, expected_shape)

                # Sum of components should equal full kernel
                K_sum = K.sum(dim=-3)
                K_standard = oak(X1, X2).to_dense()
                self.assertAllClose(K_sum, K_standard, atol=1e-5)

    def _test_normalizer_preserves_eval_mode(self):
        """Test that normalizer() does not switch the module to training mode."""
        d = 3
        tkwargs = {"dtype": torch.double, "device": self.device}
        oak = OrthogonalAdditiveKernel(
            d,
            RBFKernel(),
            per_dim_lengthscales=False,
            **tkwargs,
        )

        # Switch to eval mode and call normalizer
        oak.eval()
        self.assertFalse(oak.training)

        _ = oak.normalizer()

        # Module should still be in eval mode (not switched by normalizer)
        self.assertFalse(oak.training)

        # Second call should return the same cached normalizer (no recomputation)
        norm1 = oak.normalizer()
        norm2 = oak.normalizer()
        self.assertIs(norm1, norm2)

    def _test_component_indices(self):
        """Test component_indices property returns correct mappings."""
        d = 5
        tkwargs = {"dtype": torch.double, "device": self.device}
        base_kernel = RBFKernel().to(**tkwargs)

        # Test first-order only
        oak = OrthogonalAdditiveKernel(
            d,
            base_kernel,
            per_dim_lengthscales=False,
            second_order=False,
            **tkwargs,
        )
        indices = oak.component_indices
        self.assertIn("bias", indices)
        self.assertIn("first_order", indices)
        self.assertNotIn("second_order", indices)
        self.assertEqual(indices["bias"].tolist(), [0])
        self.assertEqual(indices["first_order"].tolist(), list(range(d)))

        # Test with second-order
        oak_2nd = OrthogonalAdditiveKernel(
            d,
            base_kernel,
            per_dim_lengthscales=False,
            second_order=True,
            **tkwargs,
        )
        indices_2nd = oak_2nd.component_indices
        self.assertIn("bias", indices_2nd)
        self.assertIn("first_order", indices_2nd)
        self.assertIn("second_order", indices_2nd)
        self.assertEqual(indices_2nd["bias"].tolist(), [0])
        self.assertEqual(indices_2nd["first_order"].tolist(), list(range(d)))

        # Verify second_order shape and content
        second_order_indices = indices_2nd["second_order"]
        expected_num_pairs = d * (d - 1) // 2
        self.assertEqual(second_order_indices.shape, (expected_num_pairs, 2))

        # Verify all pairs are upper triangular (i < j)
        for i in range(expected_num_pairs):
            row_idx, col_idx = second_order_indices[i].tolist()
            self.assertLess(row_idx, col_idx)

        # Test get_component_index method
        self.assertEqual(oak.get_component_index("bias"), 0)
        self.assertEqual(oak.get_component_index("first_order", 0), 1)
        self.assertEqual(oak.get_component_index("first_order", d - 1), d)
        self.assertEqual(oak_2nd.get_component_index("second_order", (0, 1)), d + 1)
        with self.assertRaises(ValueError):
            oak.get_component_index("unknown")
        with self.assertRaises(IndexError):
            oak.get_component_index("first_order", 99)
        with self.assertRaises(ValueError):
            oak.get_component_index("second_order", (0, 1))  # second_order disabled
        with self.assertRaises(ValueError):
            oak_2nd.get_component_index("second_order", (1, 0))  # i >= j

        # Additional error paths for get_component_index
        self.assertIsNone(oak.coeffs_2)  # coeffs_2 returns None when disabled
        with self.assertRaises(ValueError):
            oak.get_component_index("first_order", None)  # dim_index=None
        with self.assertRaises(ValueError):
            oak_2nd.get_component_index("second_order", 0)  # non-tuple dim_index
        with self.assertRaises(IndexError):
            oak_2nd.get_component_index("second_order", (0, d))  # j >= d


def isposdef(A: Tensor) -> bool:
    """Determines whether A is positive definite or not, by attempting a Cholesky
    decomposition. Expects batches of square matrices. Throws a RuntimeError otherwise.
    """
    _, info = torch.linalg.cholesky_ex(A)
    return not torch.any(info)
