#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Log
from botorch.utils.testing import BotorchTestCase


class TestChainedOutcomeHooks(BotorchTestCase):
    def test_child_forward_hooks(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for dtype in (torch.float, torch.double):
            for device in devices:
                for nested in (False, True):
                    with self.subTest(dtype=dtype, device=device, nested=nested):
                        Y = torch.tensor(
                            [[1.1], [1.7], [2.8]], device=device, dtype=dtype
                        )
                        Yvar = torch.full_like(Y, 0.1)
                        X = torch.tensor(
                            [[0.1], [0.4], [0.9]], device=device, dtype=dtype
                        )
                        calls = []
                        child = Log()

                        def before(module, args, kwargs):
                            calls.append("before")
                            return args, {**kwargs, "Y": kwargs["Y"] + 0.2}

                        def after(module, args, kwargs, output):
                            calls.append("after")
                            return output[0] + 0.3, output[1]

                        child.register_forward_pre_hook(before, with_kwargs=True)
                        child.register_forward_hook(after, with_kwargs=True)
                        chain = ChainedOutcomeTransform(log=child)
                        if nested:
                            chain = ChainedOutcomeTransform(inner=chain)
                        actual = chain(Y=Y, Yvar=Yvar, X=X)
                        expected = child(Y=Y, Yvar=Yvar, X=X)
                        self.assertEqual(calls, ["before", "after"] * 2)
                        for a, b in zip(actual, expected):
                            self.assertAllClose(a, b)

    def test_child_backward_hook(self) -> None:
        child = Log()
        calls = []

        def backward(module, grad_input, grad_output):
            calls.append(True)

        child.register_full_backward_hook(backward)
        # Forward uses keyword arguments, so a full backward hook observes its
        # outputs. A hook on a differentiable result must run through a chain.
        Y = torch.tensor([[1.1], [1.7], [2.8]], device=self.device, requires_grad=True)
        direct = deepcopy(child)
        chain = ChainedOutcomeTransform(log=child)
        actual = chain(Y=Y)[0]
        expected = direct(Y=Y)[0]
        self.assertAllClose(actual, expected)
        actual.sum().backward()
        self.assertEqual(len(calls), 1)
