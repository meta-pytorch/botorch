#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.models.transforms.factory import get_rounding_input_transform
from botorch.models.transforms.input import ChainedInputTransform, Normalize, Round
from botorch.utils.rounding import OneHotArgmaxSTE
from botorch.utils.testing import BotorchTestCase
from botorch.utils.transforms import normalize, unnormalize


class TestGetRoundingInputTransform(BotorchTestCase):
    def test_get_rounding_input_transform(self):
        for dtype in (torch.float, torch.double):
            one_hot_bounds = torch.tensor(
                [
                    [0, 5],
                    [0, 4],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                ],
                dtype=dtype,
                device=self.device,
            ).t()
            with self.assertRaises(ValueError):
                # test no integer or categorical
                get_rounding_input_transform(
                    one_hot_bounds=one_hot_bounds,
                )
            integer_indices = [1]
            categorical_features = {2: 2, 4: 3}
            tf = get_rounding_input_transform(
                one_hot_bounds=one_hot_bounds,
                integer_indices=integer_indices,
                categorical_features=categorical_features,
            )
            self.assertIsInstance(tf, ChainedInputTransform)
            tfs = list(tf.items())
            self.assertEqual(len(tfs), 3)
            # test unnormalize
            tf_name_i, tf_i = tfs[0]
            self.assertEqual(tf_name_i, "unnormalize_tf")
            self.assertIsInstance(tf_i, Normalize)
            self.assertTrue(tf_i.reverse)
            bounds = one_hot_bounds[:, integer_indices]
            offset = bounds[:1, :]
            coefficient = bounds[1:2, :] - offset
            self.assertTrue(torch.equal(tf_i.coefficient, coefficient))
            self.assertTrue(torch.equal(tf_i.offset, offset))
            self.assertEqual(tf_i._d, one_hot_bounds.shape[1])
            self.assertEqual(
                tf_i.indices, torch.tensor(integer_indices, device=self.device)
            )
            # test round
            tf_name_i, tf_i = tfs[1]
            self.assertEqual(tf_name_i, "round")
            self.assertIsInstance(tf_i, Round)
            self.assertEqual(tf_i.integer_indices.tolist(), integer_indices)
            self.assertEqual(tf_i.categorical_features, categorical_features)
            # test normalize
            tf_name_i, tf_i = tfs[2]
            self.assertEqual(tf_name_i, "normalize_tf")
            self.assertIsInstance(tf_i, Normalize)
            self.assertFalse(tf_i.reverse)
            self.assertTrue(torch.equal(tf_i.coefficient, coefficient))
            self.assertTrue(torch.equal(tf_i.offset, offset))
            self.assertEqual(tf_i._d, one_hot_bounds.shape[1])

            # test forward
            X = torch.rand(
                2, 4, one_hot_bounds.shape[1], dtype=dtype, device=self.device
            )
            X_tf = tf(X)
            # assert the continuous param is unaffected
            self.assertTrue(torch.equal(X_tf[..., 0], X[..., 0]))
            # check that integer params are rounded
            X_int = X[..., integer_indices]
            unnormalized_X_int = unnormalize(X_int, bounds)
            rounded_X_int = normalize(unnormalized_X_int.round(), bounds)
            self.assertTrue(torch.equal(rounded_X_int, X_tf[..., integer_indices]))
            # check that categoricals are discretized
            for start, card in categorical_features.items():
                end = start + card
                discretized_feat = OneHotArgmaxSTE.apply(X[..., start:end])
                self.assertTrue(torch.equal(discretized_feat, X_tf[..., start:end]))
            # test transform on train/eval/fantasize
            for tf_i in tf.values():
                self.assertFalse(tf_i.transform_on_train)
                self.assertTrue(tf_i.transform_on_eval)
                self.assertTrue(tf_i.transform_on_fantasize)

            # test no integer
            tf = get_rounding_input_transform(
                one_hot_bounds=one_hot_bounds,
                categorical_features=categorical_features,
            )
            tfs = list(tf.items())
            # round should be the only transform
            self.assertEqual(len(tfs), 1)
            tf_name_i, tf_i = tfs[0]
            self.assertEqual(tf_name_i, "round")
            self.assertIsInstance(tf_i, Round)
            self.assertEqual(tf_i.integer_indices.tolist(), [])
            self.assertEqual(tf_i.categorical_features, categorical_features)
            # test no categoricals
            tf = get_rounding_input_transform(
                one_hot_bounds=one_hot_bounds,
                integer_indices=integer_indices,
            )
            tfs = list(tf.items())
            self.assertEqual(len(tfs), 3)
            _, tf_i = tfs[1]
            self.assertEqual(tf_i.integer_indices.tolist(), integer_indices)
            self.assertEqual(tf_i.categorical_features, {})
            # test initialization
            tf = get_rounding_input_transform(
                one_hot_bounds=one_hot_bounds,
                integer_indices=integer_indices,
                categorical_features=categorical_features,
                initialization=True,
            )
            tfs = list(tf.items())
            self.assertEqual(len(tfs), 3)
            # check that bounds are adjusted for integers on unnormalize
            _, tf_i = tfs[0]
            offset_init = bounds[:1, :] - 0.4999
            coefficient_init = bounds[1:2, :] + 0.4999 - offset_init
            self.assertTrue(torch.equal(tf_i.coefficient, coefficient_init))
            self.assertTrue(torch.equal(tf_i.offset, offset_init))
            # check that bounds are adjusted for integers on normalize
            _, tf_i = tfs[2]
            self.assertTrue(torch.equal(tf_i.coefficient, coefficient))
            self.assertTrue(torch.equal(tf_i.offset, offset))
            # test return numeric
            tf = get_rounding_input_transform(
                one_hot_bounds=one_hot_bounds,
                integer_indices=integer_indices,
                categorical_features=categorical_features,
                return_numeric=True,
            )
            tfs = list(tf.items())
            self.assertEqual(len(tfs), 4)
            tf_name_i, tf_i = tfs[3]
            self.assertEqual(tf_name_i, "one_hot_to_numeric")
            # transform to numeric on train
            # (e.g. for kernels that expect a integer representation)
            self.assertTrue(tf_i.transform_on_train)
            self.assertTrue(tf_i.transform_on_eval)
            self.assertTrue(tf_i.transform_on_fantasize)
            self.assertEqual(tf_i.categorical_features, categorical_features)
            self.assertEqual(tf_i.numeric_dim, 4)
            # test return numeric and no categorical
            tf = get_rounding_input_transform(
                one_hot_bounds=one_hot_bounds,
                integer_indices=integer_indices,
                return_numeric=True,
            )
            tfs = list(tf.items())
            # there should be no one hot to numeric transform
            self.assertEqual(len(tfs), 3)
