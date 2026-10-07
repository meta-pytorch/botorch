 ---
  Notation Inconsistency Analysis
  Issue 1: Number of datasets/sample paths ($m$ vs $K$)
  ┌───────────────────────────┬────────┬───────────────────────────────┐
  │          Section          │ Symbol │            Meaning            │
  ├───────────────────────────┼────────┼───────────────────────────────┤
  │ Theory (Sec 3.0)          │ $m$    │ number of i.i.d. sample paths │
  ├───────────────────────────┼────────┼───────────────────────────────┤
  │ Finite-sample theorem     │ $m$    │ number of sample paths        │
  ├───────────────────────────┼────────┼───────────────────────────────┤
  │ Corollary text (line 324) │ $K$    │ number of datasets            │
  ├───────────────────────────┼────────┼───────────────────────────────┤
  │ EM section (Sec 3.1)      │ $K$    │ number of datasets            │
  └───────────────────────────┴────────┴───────────────────────────────┘
  Options:
  - A) Standardize on $m$ everywhere — Change EM section to use $m$. Pros: Consistent with theory. Cons: $K$ is common in meta-learning literature.
  - B) Standardize on $K$ everywhere — Change theory to use $K$. Pros: ML convention. Cons: Requires many edits in theory/appendix.
  - C) Keep both, add bridging sentence — Add explicit statement in EM section that "$K$ datasets correspond to the $m$ sample paths in the preceding theory." Pros: Minimal
  edits. Cons: Two symbols for same thing.
  ---
  Issue 2: $n$ is overloaded
  ┌─────────────────────────────────────┬────────┬──────────────────────────────┐
  │              Location               │ Symbol │           Meaning            │
  ├─────────────────────────────────────┼────────┼──────────────────────────────┤
  │ Corollary after Prop 1              │ $n$    │ test dataset size            │
  ├─────────────────────────────────────┼────────┼──────────────────────────────┤
  │ Theorem 3                           │ $n$    │ observations per sample path │
  ├─────────────────────────────────────┼────────┼──────────────────────────────┤
  │ Corollary 3 (Posterior Contraction) │ $n$    │ test set size                │
  ├─────────────────────────────────────┼────────┼──────────────────────────────┤
  │ EM section                          │ $n_i$  │ observations in dataset $i$  │
  └─────────────────────────────────────┴────────┴──────────────────────────────┘
  Options:
  - A) Rename test set size — Use $n_*$ or $n_{\text{test}}$ for test dataset size, keep $n$ for observations per path
  - B) Keep as-is — The meaning is clear from context in each case
  - C) Use different letter — e.g., $p$ for observations per sample path in the theory
  ---
  Issue 3: $\hat{\boldsymbol{\Sigma}}$ undefined in Corollary 2
  Line 317: $\Vert \hat{\boldsymbol{\Sigma}} - \boldsymbol{\Sigma} \Vert_{\mathrm{op}}$
  This matrix notation appears suddenly in the theory section which otherwise uses function notation ($k_m$, $k$).
  Options:
  - A) Define it — Add: "where $\hat{\boldsymbol{\Sigma}} = [k_m(x_i, x_j)]_{i,j=1}^N$ is the empirical covariance matrix at $N$ fixed locations"
  - B) Use function notation — Replace with $\Vert K_m^{(N)} - K^{(N)} \Vert_{\mathrm{op}}$ (consistent with appendix notation)

  ---
  Issue 4: Bridge between theory and EM notation

  The theory uses:
  - $\mu(x)$, $k(x,x')$ — functions
  - $\mu_m(x)$, $k_m(x,x')$ — empirical functions
  The EM uses:
  - $\boldsymbol{\mu} \in \mathbb{R}^N$, $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$ — vectors/matrices
  Options:
  - A) Add explicit connection — At start of EM section, add: "The vectors $\boldsymbol{\mu}$ and matrix $\boldsymbol{\Sigma}$ represent the evaluations of the mean function
   $\mu$ and covariance function $k$ at the $N$ unique input locations."
  - B) Keep as-is — Readers familiar with GPs will understand
