=== lcbench_highpower: 192 runs x 9 methods (41 trajectory points) ===

| method | @1 | @5 | @10 | @20 | @40 | final | AUC | avg_rank | win% |
|---|---|---|---|---|---|---|---|---|---|
| em_noisefit | 2.490±0.852 | 1.343±0.818 | 0.808±0.524 | 0.276±0.193 | 0.093±0.057 | 0.093±0.057 | 0.669 | 4.22 | 13 |
| em_frozen | 2.829±0.827 | 1.335±0.724 | 0.621±0.318 | 0.181±0.084 | 0.109±0.058 | 0.109±0.058 | 0.639 | 4.32 | 15 |
| em_finetuned | 2.613±0.858 | 1.239±0.647 | 0.906±0.531 | 0.351±0.246 | 0.144±0.078 | 0.144±0.078 | 0.744 | 4.43 | 13 |
| hyperbo_adapt | 2.540±0.762 | 1.237±0.586 | 0.757±0.512 | 0.539±0.410 | 0.312±0.233 | 0.312±0.233 | 0.820 | 4.54 | 13 |
| hyperbo_frozen | 2.618±0.829 | 1.372±0.715 | 0.714±0.420 | 0.410±0.228 | 0.208±0.124 | 0.208±0.124 | 0.810 | 4.78 | 11 |
| vanilla_gp | 4.257±0.922 | 2.370±0.648 | 1.482±0.409 | 0.844±0.323 | 0.278±0.122 | 0.278±0.122 | 1.276 | 4.96 | 11 |
| ablr | 3.236±0.434 | 0.828±0.329 | 0.431±0.190 | 0.279±0.135 | 0.217±0.129 | 0.217±0.129 | 0.619 | 5.04 | 9 |
| pacoh_frozen | 4.587±1.068 | 2.620±0.800 | 1.746±0.769 | 0.937±0.295 | 0.368±0.169 | 0.368±0.169 | 1.397 | 5.64 | 8 |
| random | 5.623±0.893 | 3.134±0.602 | 1.965±0.423 | 1.425±0.330 | 0.971±0.270 | 0.971±0.270 | 1.926 | 7.07 | 7 |

[final] Friedman chi2(8) = 163.2, p = 3.4e-31; Nemenyi CD(0.05) = 0.87
  ranks: em_noisefit 4.22, em_frozen 4.32, em_finetuned 4.43, hyperbo_adapt 4.54, hyperbo_frozen 4.78, vanilla_gp 4.96, ablr 5.04, pacoh_frozen 5.64, random 7.07
  cliques (not significantly different): {em_noisefit, em_frozen, em_finetuned, hyperbo_adapt, hyperbo_frozen, vanilla_gp, ablr}; {hyperbo_frozen, vanilla_gp, ablr, pacoh_frozen}

[auc] Friedman chi2(8) = 360.5, p = 5.16e-73; Nemenyi CD(0.05) = 0.87
  ranks: em_noisefit 3.49, em_frozen 4.00, hyperbo_adapt 4.15, em_finetuned 4.19, ablr 4.58, hyperbo_frozen 4.59, vanilla_gp 6.08, pacoh_frozen 6.60, random 7.32
  cliques (not significantly different): {em_noisefit, em_frozen, hyperbo_adapt, em_finetuned}; {em_frozen, hyperbo_adapt, em_finetuned, ablr, hyperbo_frozen}; {vanilla_gp, pacoh_frozen}; {pacoh_frozen, random}

[@1] Friedman chi2(8) = 216.7, p = 1.88e-42; Nemenyi CD(0.05) = 0.87
  ranks: em_noisefit 3.92, em_finetuned 4.10, hyperbo_adapt 4.23, em_frozen 4.23, hyperbo_frozen 4.48, ablr 5.20, pacoh_frozen 6.10, vanilla_gp 6.19, random 6.55
  cliques (not significantly different): {em_noisefit, em_finetuned, hyperbo_adapt, em_frozen, hyperbo_frozen}; {hyperbo_frozen, ablr}; {pacoh_frozen, vanilla_gp, random}

[@5] Friedman chi2(8) = 317.8, p = 6.69e-64; Nemenyi CD(0.05) = 0.87
  ranks: em_noisefit 3.89, hyperbo_frozen 4.14, hyperbo_adapt 4.19, em_finetuned 4.26, ablr 4.27, em_frozen 4.34, vanilla_gp 6.30, pacoh_frozen 6.79, random 6.83
  cliques (not significantly different): {em_noisefit, hyperbo_frozen, hyperbo_adapt, em_finetuned, ablr, em_frozen}; {vanilla_gp, pacoh_frozen, random}

[@10] Friedman chi2(8) = 291.1, p = 3.15e-58; Nemenyi CD(0.05) = 0.87
  ranks: hyperbo_adapt 4.03, em_noisefit 4.07, hyperbo_frozen 4.09, em_frozen 4.28, em_finetuned 4.37, ablr 4.53, vanilla_gp 6.16, pacoh_frozen 6.38, random 7.09
  cliques (not significantly different): {hyperbo_adapt, em_noisefit, hyperbo_frozen, em_frozen, em_finetuned, ablr}; {vanilla_gp, pacoh_frozen}; {pacoh_frozen, random}

[@20] Friedman chi2(8) = 282.2, p = 2.48e-56; Nemenyi CD(0.05) = 0.87
  ranks: em_noisefit 3.93, em_frozen 4.11, em_finetuned 4.14, hyperbo_adapt 4.22, hyperbo_frozen 4.46, ablr 4.85, vanilla_gp 5.52, pacoh_frozen 6.70, random 7.07
  cliques (not significantly different): {em_noisefit, em_frozen, em_finetuned, hyperbo_adapt, hyperbo_frozen}; {em_frozen, em_finetuned, hyperbo_adapt, hyperbo_frozen, ablr}; {ablr, vanilla_gp}; {pacoh_frozen, random}

[@40] Friedman chi2(8) = 163.2, p = 3.4e-31; Nemenyi CD(0.05) = 0.87
  ranks: em_noisefit 4.22, em_frozen 4.32, em_finetuned 4.43, hyperbo_adapt 4.54, hyperbo_frozen 4.78, vanilla_gp 4.96, ablr 5.04, pacoh_frozen 5.64, random 7.07
  cliques (not significantly different): {em_noisefit, em_frozen, em_finetuned, hyperbo_adapt, hyperbo_frozen, vanilla_gp, ablr}; {hyperbo_frozen, vanilla_gp, ablr, pacoh_frozen}

wrote pytorch/botorch/_scratch_bo/results/figures/lcbench_highpower_summary.json
  wrote pytorch/botorch/_scratch_bo/results/figures/lcbench_highpower_trajectories.png / .pdf
  wrote pytorch/botorch/_scratch_bo/results/figures/lcbench_highpower_perfprofile.png / .pdf
  wrote pytorch/botorch/_scratch_bo/results/figures/lcbench_highpower_dataprofile.png / .pdf
  wrote pytorch/botorch/_scratch_bo/results/figures/lcbench_highpower_cost.png / .pdf
  wrote pytorch/botorch/_scratch_bo/results/figures/lcbench_highpower_cd_final.png / .pdf
  wrote pytorch/botorch/_scratch_bo/results/figures/lcbench_highpower_cd_auc.png / .pdf
  wrote pytorch/botorch/_scratch_bo/results/figures/lcbench_highpower_cd_at1.png / .pdf
  wrote pytorch/botorch/_scratch_bo/results/figures/lcbench_highpower_cd_at5.png / .pdf
  wrote pytorch/botorch/_scratch_bo/results/figures/lcbench_highpower_cd_at10.png / .pdf
  wrote pytorch/botorch/_scratch_bo/results/figures/lcbench_highpower_cd_at20.png / .pdf
  wrote pytorch/botorch/_scratch_bo/results/figures/lcbench_highpower_cd_at40.png / .pdf
