# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from _scratch_bo.em_noise_selection import ledoit_wolf_alpha, oas_alpha, rblw_alpha

torch.manual_seed(0)
g = torch.Generator().manual_seed(3)
B = torch.randn(40, 5, generator=g, dtype=torch.double)
print(f"{'T':>5}{'LW':>10}{'RBLW':>10}{'OAS':>10}")
for T in (6, 17, 50, 200):
    X = torch.randn(T, 5, generator=g, dtype=torch.double) @ B.T
    print(
        f"{T:>5}{ledoit_wolf_alpha(X)[0]:>10.4f}{rblw_alpha(X)[0]:>10.4f}{oas_alpha(X)[0]:>10.4f}"
    )
print()
print("all three must lie in [0,1] and fall with T on this low-rank data")
