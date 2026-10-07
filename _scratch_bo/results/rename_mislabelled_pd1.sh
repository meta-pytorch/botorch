#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Rename the eight mislabelled "PD1" cells (S27.1). run_canon_matrix.sh and
# run_hb_hybrid.sh omitted --benchmark pd1_loo, so these ran LCBench under PD1
# filenames. The new prefix carries the warning and, critically, drops the "pd1"
# substring so a `*pd1*` glob can no longer pick them up.
#
# The runner scripts keep their original output paths on purpose: they are fixed now, so
# re-running them regenerates genuine PD1 results under the correct names, and the
# `[ -f "$out" ] && continue` guard will no longer short-circuit on these stale files.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1  # -> _scratch_bo/results

for pool in matched full; do
  for canon in sumll hyperbo; do
    sl mv "raw/canon_matrix/pd1_${pool}_${canon}.json" \
          "raw/canon_matrix/RETRACTED_ranLCBench_${pool}_${canon}.json"
    sl mv "raw/hb_hybrid/bo_pd1_${pool}_${canon}.json" \
          "raw/hb_hybrid/RETRACTED_ranLCBench_${pool}_${canon}.json"
    sl mv "logs/canon_pd1_${pool}_${canon}.txt" \
          "logs/RETRACTED_ranLCBench_canon_${pool}_${canon}.txt"
    sl mv "logs/hb_bo_pd1_${pool}_${canon}.txt" \
          "logs/RETRACTED_ranLCBench_hb_${pool}_${canon}.txt"
  done
done

echo "renamed 8 result files + 8 logs"
