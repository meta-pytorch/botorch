/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Fused kernel for qLogEHVI / qLogNEHVI inner loop.
//
// Supports both non-batched cell bounds (num_cells, m) for qLogEHVI
// and batched cell bounds (B_cells, num_cells, m) for qLogNEHVI.
//
// Optimizations:
//   - Precomputed 1/tau (division → multiplication)
//   - Fused log_fatplus_fwd_bwd (avoids redundant exp/sigmoid/cauchy)
//   - Fused pareto_val_deriv (single denominator computation)
//   - Local gradient accumulation buffer in backward
//   - Transposed li layout: li[MAX_M][MAX_I] for contiguous fatmin reads

#include <torch/extension.h>
#include <algorithm>
#include <cmath>

namespace {

// Leaky-tail coefficient for the "fat" softplus approximation.
// Must match the Python-side ALPHA_RELU in botorch.utils.safe_math.
constexpr double ALPHA_RELU = 0.1;

// Static upper bounds for stack-allocated scratch arrays.
// MAX_I: maximum subset size (q choose i).
// MAX_M: maximum number of objectives.
constexpr int MAX_I = 16;
constexpr int MAX_M = 8;

// Numerically stable softplus: log(1 + exp(y)).
// Avoids overflow for large y and underflow for very negative y.
template <typename T>
inline T safe_softplus(T y) {
  if (y > T(20)) {
    return y;
  }
  if (y < T(-20)) {
    return std::exp(y);
  }
  return std::log1p(std::exp(y));
}

// Numerically stable sigmoid: 1 / (1 + exp(-y)).
// Uses the exp(y) form for negative y to avoid overflow in exp(-y).
template <typename T>
inline T safe_sigmoid(T y) {
  if (y >= T(0)) {
    T e = std::exp(-y);
    return T(1) / (T(1) + e);
  } else {
    T e = std::exp(y);
    return e / (T(1) + e);
  }
}

// Cauchy-like density kernel: 1 / (1 + y^2).
template <typename T>
inline T cauchy_val(T y) {
  return T(1) / (T(1) + y * y);
}

// Pareto density and its derivative, used in the smooth-min (fatmin).
// f(z) = 2 / (2 + 2z + z^2), f'(z) = -2(2 + 2z) / (2 + 2z + z^2)^2.
template <typename T>
inline void pareto_val_deriv(T z, T& val, T& deriv) {
  T d = T(2) + T(2) * z + z * z;
  val = T(2) / d;
  deriv = T(-2) * (T(2) + T(2) * z) / (d * d);
}

// Forward-only Pareto density (no derivative needed).
template <typename T>
inline T pareto_val_only(T z) {
  T d = T(2) + T(2) * z + z * z;
  return T(2) / d;
}

// log(fatplus(x, tau)): log of the fat-tailed softplus approximation to ReLU.
// fatplus(x, tau) = tau * (softplus(x/tau) + ALPHA_RELU * cauchy(x/tau)).
template <typename T>
inline T log_fatplus_fwd(T x, T tau, T inv_tau) {
  T y = x * inv_tau;
  T f = safe_softplus(y) + T(ALPHA_RELU) * cauchy_val(y);
  T val = tau * f;
  return val > T(0) ? std::log(val) : T(-1e30);
}

// Combined forward + backward for log_fatplus. Computes both the value and
// d(log_fatplus)/dx in a single pass, reusing intermediate quantities.
template <typename T>
inline void log_fatplus_fwd_bwd(T x, T tau, T inv_tau, T& val, T& grad) {
  T y = x * inv_tau;
  T cy = cauchy_val(y);
  T sp = safe_softplus(y);
  T f = sp + T(ALPHA_RELU) * cy;
  T tf = tau * f;
  if (tf <= T(0)) {
    val = T(-1e30);
    grad = T(0);
    return;
  }
  val = std::log(tf);
  T sg = safe_sigmoid(y);
  T fp = sg - T(2) * T(ALPHA_RELU) * y * cy * cy;
  grad = fp / (tau * f);
}

// Smooth minimum ("fatmin") of n values using Pareto-density weighting.
// Returns min_smooth(x[0..n-1]) and optionally the gradient weights gw[].
template <typename T>
inline T compute_fatmin(const T* x, int n, T tau, T inv_tau, T* gw = nullptr) {
  if (n == 1) {
    if (gw) {
      gw[0] = T(1);
    }
    return x[0];
  }

  T mn = x[0];
  int ami = 0;
  for (int j = 1; j < n; j++) {
    if (x[j] < mn) {
      mn = x[j];
      ami = j;
    }
  }

  if (mn < T(-1e29)) {
    if (gw) {
      for (int j = 0; j < n; j++) {
        gw[j] = (j == ami) ? T(1) : T(0);
      }
    }
    return mn;
  }

  T S = T(0);
  T S_pd = T(0);
  T pd[MAX_I];

  if (gw) {
    for (int j = 0; j < n; j++) {
      T zj = (x[j] - mn) * inv_tau;
      T pv, pder;
      pareto_val_deriv(zj, pv, pder);
      S += pv;
      pd[j] = pder;
      S_pd += pder;
    }
  } else {
    for (int j = 0; j < n; j++) {
      T zj = (x[j] - mn) * inv_tau;
      S += pareto_val_only(zj);
    }
  }

  T result = mn - tau * std::log(S);

  if (gw) {
    T pp0 = T(-1);
    for (int j = 0; j < n; j++) {
      if (j == ami) {
        gw[j] = T(1) + (S_pd - pp0) / S;
      } else {
        gw[j] = -pd[j] / S;
      }
    }
  }

  return result;
}

} // namespace

// Forward: obj_subsets (B, n_sub, i, m),
//          cell bounds (B_cells, num_cells, m) or (num_cells, m)
//        → (B, num_cells, n_sub)
torch::Tensor fused_log_areas_forward(
    const torch::Tensor& obj_subsets_,
    const torch::Tensor& cell_lower_,
    const torch::Tensor& cell_upper_,
    double tau_relu,
    double tau_max) {
  auto obj = obj_subsets_.contiguous();
  auto cl = cell_lower_.contiguous();
  auto cu = cell_upper_.contiguous();

  TORCH_CHECK(obj.dim() == 4, "obj_subsets must be 4-D");
  TORCH_CHECK(
      (cl.dim() == 2 || cl.dim() == 3) && cl.dim() == cu.dim(),
      "cell bounds must be 2-D or 3-D");

  const int64_t B = obj.size(0);
  const int64_t n_sub = obj.size(1);
  const int64_t isz = obj.size(2);
  const int64_t m = obj.size(3);

  // Support both (num_cells, m) and (B_cells, num_cells, m)
  const bool batched_cells = cl.dim() == 3;
  const int64_t B_cells = batched_cells ? cl.size(0) : 1;
  const int64_t nc = batched_cells ? cl.size(1) : cl.size(0);

  TORCH_CHECK(isz <= MAX_I && m <= MAX_M, "subset_size or m too large");

  // Reshape cell bounds to 3-D for uniform indexing
  auto cl3 = batched_cells ? cl : cl.unsqueeze(0);
  auto cu3 = batched_cells ? cu : cu.unsqueeze(0);

  auto out = torch::empty({B, nc, n_sub}, obj.options());

  AT_DISPATCH_FLOATING_TYPES(obj.scalar_type(), "fused_fwd", [&] {
    const scalar_t cu_clamp =
        std::is_same<scalar_t, double>::value ? scalar_t(1e10) : scalar_t(1e8);
    const scalar_t inv_tau_relu = scalar_t(1) / scalar_t(tau_relu);
    const scalar_t inv_tau_max = scalar_t(1) / scalar_t(tau_max);

    auto oa = obj.accessor<scalar_t, 4>();
    auto la3 = cl3.accessor<scalar_t, 3>();
    auto ua3 = cu3.accessor<scalar_t, 3>();
    auto ra = out.accessor<scalar_t, 3>();

    // Precompute log cell lengths: (B_cells, nc, m)
    std::vector<scalar_t> lcl_buf(B_cells * nc * m);
    for (int64_t bc = 0; bc < B_cells; bc++) {
      for (int64_t c = 0; c < nc; c++) {
        for (int64_t k = 0; k < m; k++) {
          lcl_buf[(bc * nc + c) * m + k] =
              std::log(std::min(ua3[bc][c][k], cu_clamp) - la3[bc][c][k]);
        }
      }
    }

    at::parallel_for(
        0, B * n_sub, /*grain_size=*/1, [&](int64_t lo, int64_t hi) {
          for (int64_t idx = lo; idx < hi; idx++) {
            const int64_t b = idx / n_sub;
            const int64_t s = idx % n_sub;
            const int64_t bc = B_cells == 1 ? 0 : b;

            for (int64_t c = 0; c < nc; c++) {
              scalar_t li[MAX_M][MAX_I];
              for (int j = 0; j < isz; j++) {
                for (int k = 0; k < m; k++) {
                  li[k][j] = log_fatplus_fwd(
                      oa[b][s][j][k] - la3[bc][c][k],
                      scalar_t(tau_relu),
                      inv_tau_relu);
                }
              }

              scalar_t area = scalar_t(0);
              for (int k = 0; k < m; k++) {
                scalar_t lim_k = compute_fatmin(
                    li[k], (int)isz, scalar_t(tau_max), inv_tau_max);
                scalar_t pair[2] = {lim_k, lcl_buf[(bc * nc + c) * m + k]};
                area += compute_fatmin(pair, 2, scalar_t(tau_max), inv_tau_max);
              }

              ra[b][c][s] = area;
            }
          }
        });
  });

  return out;
}

torch::Tensor fused_log_areas_backward(
    const torch::Tensor& grad_output_,
    const torch::Tensor& obj_subsets_,
    const torch::Tensor& cell_lower_,
    const torch::Tensor& cell_upper_,
    double tau_relu,
    double tau_max) {
  auto go = grad_output_.contiguous();
  auto obj = obj_subsets_.contiguous();
  auto cl = cell_lower_.contiguous();
  auto cu = cell_upper_.contiguous();

  const int64_t B = obj.size(0);
  const int64_t n_sub = obj.size(1);
  const int64_t isz = obj.size(2);
  const int64_t m = obj.size(3);

  const bool batched_cells = cl.dim() == 3;
  const int64_t B_cells = batched_cells ? cl.size(0) : 1;
  const int64_t nc = batched_cells ? cl.size(1) : cl.size(0);

  auto cl3 = batched_cells ? cl : cl.unsqueeze(0);
  auto cu3 = batched_cells ? cu : cu.unsqueeze(0);

  auto g_obj = torch::zeros_like(obj);

  AT_DISPATCH_FLOATING_TYPES(obj.scalar_type(), "fused_bwd", [&] {
    const scalar_t cu_clamp =
        std::is_same<scalar_t, double>::value ? scalar_t(1e10) : scalar_t(1e8);
    const scalar_t inv_tau_relu = scalar_t(1) / scalar_t(tau_relu);
    const scalar_t inv_tau_max = scalar_t(1) / scalar_t(tau_max);

    auto oa = obj.accessor<scalar_t, 4>();
    auto la3 = cl3.accessor<scalar_t, 3>();
    auto ua3 = cu3.accessor<scalar_t, 3>();
    auto ga = go.accessor<scalar_t, 3>();
    auto goa = g_obj.accessor<scalar_t, 4>();

    std::vector<scalar_t> lcl_buf(B_cells * nc * m);
    for (int64_t bc = 0; bc < B_cells; bc++) {
      for (int64_t c = 0; c < nc; c++) {
        for (int64_t k = 0; k < m; k++) {
          lcl_buf[(bc * nc + c) * m + k] =
              std::log(std::min(ua3[bc][c][k], cu_clamp) - la3[bc][c][k]);
        }
      }
    }

    at::parallel_for(
        0, B * n_sub, /*grain_size=*/1, [&](int64_t lo, int64_t hi) {
          for (int64_t idx = lo; idx < hi; idx++) {
            const int64_t b = idx / n_sub;
            const int64_t s = idx % n_sub;
            const int64_t bc = B_cells == 1 ? 0 : b;

            scalar_t g_buf[MAX_I][MAX_M] = {};

            for (int64_t c = 0; c < nc; c++) {
              const scalar_t g_out = ga[b][c][s];

              scalar_t li[MAX_M][MAX_I];
              scalar_t lg[MAX_M][MAX_I];
              for (int j = 0; j < isz; j++) {
                for (int k = 0; k < m; k++) {
                  log_fatplus_fwd_bwd(
                      oa[b][s][j][k] - la3[bc][c][k],
                      scalar_t(tau_relu),
                      inv_tau_relu,
                      li[k][j],
                      lg[k][j]);
                }
              }

              for (int k = 0; k < m; k++) {
                scalar_t fmg[MAX_I];
                scalar_t lim_k = compute_fatmin(
                    li[k], (int)isz, scalar_t(tau_max), inv_tau_max, fmg);

                scalar_t pair[2] = {lim_k, lcl_buf[(bc * nc + c) * m + k]};
                scalar_t fm2g[2];
                compute_fatmin(pair, 2, scalar_t(tau_max), inv_tau_max, fm2g);

                scalar_t g_lim = g_out * fm2g[0];

                for (int j = 0; j < isz; j++) {
                  g_buf[j][k] += g_lim * fmg[j] * lg[k][j];
                }
              }
            }

            for (int j = 0; j < isz; j++) {
              for (int k = 0; k < m; k++) {
                goa[b][s][j][k] = g_buf[j][k];
              }
            }
          }
        });
  });

  return g_obj;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("forward", &fused_log_areas_forward, "Fused log-areas forward");
  mod.def("backward", &fused_log_areas_backward, "Fused log-areas backward");
}
