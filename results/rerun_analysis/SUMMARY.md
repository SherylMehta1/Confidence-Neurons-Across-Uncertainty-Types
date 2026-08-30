# Clean bf16 rerun - analysis

## results/ablation_bf16_new
precision=bf16 mean_source=general_baseline rows=23040 neurons=32 (candidates=27) exact-zero shifts=0.001 on-1/256-grid=0.000

- FDR(0.01) survivors among 81 candidate cells: uncertain-shift 3, control-shift 1, uncertain-vs-control interaction 1, activation-slope 3, held-out-only 0
- uncorrected p<.01: uncertain 6, interaction 1, slope 5
- random control NEURONS: mean|shift| 0.01148 vs candidates 0.01027; |interaction| 0.00242 vs 0.00236
- working AND held-out p<.01 same sign: 1 L31_N11541/lack_of_knowledge

Top-5 by interaction_p:
 neuron_id              category  unc_mean  ctrl_mean  interaction  interaction_p    unc_dz     slope      slope_p
L31_N11541     lack_of_knowledge -0.028934   0.001372    -0.030306       0.000100 -0.661145 -0.011618 1.788695e-10
L29_N11791     lack_of_knowledge -0.003504   0.002595    -0.006099       0.018198 -0.210911 -0.022748 3.824535e-01
L31_N14143     lack_of_knowledge -0.005341  -0.000719    -0.004622       0.026497 -0.340797  0.017750 4.136234e-01
 L29_N5866 contradictory_context  0.001531   0.005643    -0.004112       0.045195  0.099196  0.004478 6.736008e-01
  L31_N368 contradictory_context -0.001998   0.000938    -0.002936       0.046495 -0.148180 -0.005258 3.972085e-01

## results/ablation_bf16_old15
precision=bf16 mean_source=general_baseline rows=14400 neurons=20 (candidates=15) exact-zero shifts=0.001 on-1/256-grid=0.000

- FDR(0.01) survivors among 45 candidate cells: uncertain-shift 2, control-shift 2, uncertain-vs-control interaction 0, activation-slope 1, held-out-only 0
- uncorrected p<.01: uncertain 3, interaction 0, slope 5
- random control NEURONS: mean|shift| 0.00889 vs candidates 0.01158; |interaction| 0.00223 vs 0.00181
- working AND held-out p<.01 same sign: 1 L31_N2477/ambiguity

Top-5 by interaction_p:
 neuron_id              category  unc_mean  ctrl_mean  interaction  interaction_p    unc_dz     slope  slope_p
 L24_N7891     lack_of_knowledge -0.001578   0.006713    -0.008292       0.010399 -0.074815  0.047061 0.292979
L29_N11308     lack_of_knowledge -0.002760   0.001896    -0.004656       0.062794 -0.164110  0.010388 0.628792
L29_N10092             ambiguity -0.002002   0.002214    -0.004216       0.135486 -0.083915 -0.001578 0.947185
 L30_N5509             ambiguity -0.001185   0.002698    -0.003883       0.139086 -0.054694 -0.005324 0.679862
L30_N13513 contradictory_context  0.001402  -0.001129     0.002531       0.202680  0.114575  0.023902 0.153378

## results/ablation_bf16_v3set
precision=bf16 mean_source=general_baseline rows=15840 neurons=22 (candidates=17) exact-zero shifts=0.001 on-1/256-grid=0.000

- FDR(0.01) survivors among 51 candidate cells: uncertain-shift 3, control-shift 0, uncertain-vs-control interaction 1, activation-slope 3, held-out-only 0
- uncorrected p<.01: uncertain 6, interaction 2, slope 5
- random control NEURONS: mean|shift| 0.00899 vs candidates 0.01282; |interaction| 0.00209 vs 0.00227
- working AND held-out p<.01 same sign: 2 L20_N5595/lack_of_knowledge, L31_N6772/lack_of_knowledge

Top-5 by interaction_p:
neuron_id          category  unc_mean  ctrl_mean  interaction  interaction_p    unc_dz     slope  slope_p
L31_N6772 lack_of_knowledge -0.007367   0.000212    -0.007579       0.000100 -0.454348  0.093405 0.000012
L20_N5595 lack_of_knowledge -0.013380  -0.000621    -0.012759       0.000600 -0.465595 -0.139398 0.013687
L20_N1399 lack_of_knowledge -0.000787   0.004348    -0.005135       0.020898 -0.042648  0.057329 0.420656
L30_N1457 lack_of_knowledge  0.010612   0.005646     0.004965       0.134687  0.439431  0.104817 0.000002
L30_N1777 lack_of_knowledge  0.003950  -0.002700     0.006650       0.139386  0.122190 -0.001029 0.846415

## Frozen-RMSNorm decomposition
- L31_N2477/ambiguity: full +0.01544 (p=5e-05), frozen-norm +0.00096 (p=0.703) -> norm-mediated fraction 0.94
- L31_N2477/ambiguity (controls): full +0.01419 (p=5e-05), frozen-norm +0.00616 (p=0.0011) -> norm-mediated fraction 0.57
- L31_N2477/contradictory_context: full +0.00707 (p=0.0079), frozen-norm -0.00491 (p=0.03) -> norm-mediated fraction 1.69
- L31_N2477/contradictory_context (controls): full +0.00324 (p=0.258), frozen-norm -0.00718 (p=0.0141) -> norm-mediated fraction 3.22
- L31_N2477/lack_of_knowledge: full +0.01434 (p=5e-05), frozen-norm -0.00129 (p=0.494) -> norm-mediated fraction 1.09
- L31_N2477/lack_of_knowledge (controls): full +0.01016 (p=0.0001), frozen-norm +0.00114 (p=0.638) -> norm-mediated fraction 0.89

## Dose-response (per-prompt Spearman of clamped entropy vs sigma level)
- L31_N2477/ambiguity: mean rho +0.453, sign-consistent 0.81, mean shift by level -2s:-0.0016, -1s:+0.0043, +0s:+0.0154, +1s:+0.0240, +2s:+0.0361
- L31_N2477/ambiguity (controls): mean rho +0.608, sign-consistent 0.88, mean shift by level -2s:-0.0057, -1s:+0.0007, +0s:+0.0142, +1s:+0.0255, +2s:+0.0461
- L31_N2477/contradictory_context: mean rho +0.355, sign-consistent 0.69, mean shift by level -2s:-0.0019, -1s:-0.0001, +0s:+0.0071, +1s:+0.0108, +2s:+0.0185
- L31_N2477/contradictory_context (controls): mean rho +0.357, sign-consistent 0.72, mean shift by level -2s:-0.0035, -1s:-0.0003, +0s:+0.0032, +1s:+0.0108, +2s:+0.0224
- L31_N2477/lack_of_knowledge: mean rho +0.863, sign-consistent 0.99, mean shift by level -2s:-0.0272, -1s:-0.0073, +0s:+0.0143, +1s:+0.0391, +2s:+0.0657
- L31_N2477/lack_of_knowledge (controls): mean rho +0.522, sign-consistent 0.86, mean shift by level -2s:-0.0064, -1s:-0.0005, +0s:+0.0102, +1s:+0.0157, +2s:+0.0277

## Induction check (all prompts)
             category  n_uncertain  n_control  uncertain_mean_entropy  control_mean_entropy  uncertain_median_entropy  control_median_entropy  gap_entropy  welch_p_entropy  mannwhitney_p_entropy  cohens_d_entropy  uncertain_mean_top1_prob  control_mean_top1_prob  uncertain_median_top1_prob  control_median_top1_prob  gap_top1_prob  welch_p_top1_prob  mannwhitney_p_top1_prob  cohens_d_top1_prob  uncertain_mean_top1_working  control_mean_top1_working  uncertain_mean_top1_held_out  control_mean_top1_held_out
            ambiguity          120        120                2.088037              2.342481                  2.159828                2.323434    -0.254444     1.188732e-01           2.787328e-01         -0.202089                  0.498513                0.471291                    0.408801                  0.422933       0.027222       4.251890e-01             4.530642e-01            0.103128                     0.497456                   0.462005                      0.500980                    0.492957
    lack_of_knowledge          120        120                2.845355              1.239193                  2.711335                1.076231     1.606162     5.603582e-38           1.414727e-28          2.014822                  0.290748                0.681073                    0.261835                  0.746530      -0.390325       3.578886e-33             1.095612e-26           -1.951358                     0.292264                   0.678357                      0.287212                    0.687412
contradictory_context          120        120                0.919618              1.247351                  0.864629                1.198662    -0.327734     1.320213e-03           2.763426e-03         -0.419829                  0.741633                0.686352                    0.767542                  0.688298       0.055281       5.188826e-02             9.439853e-02            0.252261                     0.743421                   0.669193                      0.737461                    0.726391

## stolfo_bf16_new
```
Stolfo weight criteria -- 27 candidates vs 1000 random neurons (layers [20, 22, 26, 27, 28, 29, 30, 31], seed 42) and a +/-10% norm-matched null (up to 50 per candidate)

neuron         w_norm    pR |  logit_var    pR    pM |  nullfrac_k64    pR    pM |    b10%    pR    pM | nM   verdict
L29_N11791      1.625   7.7 |    0.00021  98.7  90.0 |       0.02473  91.1  72.0 |  0.4057  91.6  74.0 | 1571 -
L29_N13925      1.752  14.2 |    0.00018  17.8  40.0 |       0.01883  85.0  66.0 |  0.3267  81.4  56.0 | 3458 -
L28_N11622      1.774  16.5 |    0.00018   8.5  10.0 |       0.02626  91.3  94.0 |  0.3654  88.3  90.0 | 12740 -
L30_N13098      1.794  19.5 |    0.00018   8.5  36.0 |       0.02950  92.6  72.0 |  0.4073  91.7  66.0 | 3636 -
L29_N5866       1.773  16.1 |    0.00019  56.2  64.0 |       0.04073  95.2  92.0 |  0.4631  94.3  96.0 | 4996 -
L31_N11541      1.373   3.3 |    0.00016   3.5  26.0 |       0.14009  98.6  82.0 |  0.7124  99.2  90.0 | 1934 -
L29_N12400      1.489   4.9 |    0.00020  98.0  92.0 |       0.03506  94.0  78.0 |  0.3950  91.1  68.0 | 863  -
L27_N45         1.794  19.7 |    0.00019  80.4  84.0 |       0.02835  92.1  98.0 |  0.3851  90.4  96.0 | 13595 -
L28_N5085       1.573   6.1 |    0.00018  12.8  46.0 |       0.02878  92.2  84.0 |  0.4059  91.6  72.0 | 1068 -
L29_N3447       1.809  23.3 |    0.00018  12.4   8.0 |       0.02641  91.4  98.0 |  0.3841  90.4  96.0 | 10819 -
L29_N2646       1.800  21.0 |    0.00017   5.0   4.0 |       0.04306  95.4  98.0 |  0.4855  94.9  98.0 | 9442 -
L20_N4060       1.704  10.9 |    0.00019  88.2  82.0 |       0.02357  90.6  98.0 |  0.3684  89.0  90.0 | 13715 -
L31_N377        1.350   3.0 |    0.00014   0.9  14.0 |       0.03541  94.1  52.0 |  0.6177  98.2  62.0 | 1871 -
L31_N14143      1.630   8.0 |    0.00019  87.9  82.0 |       0.07021  97.1  62.0 |  0.4733  94.6  52.0 | 2201 -
L29_N6625       2.385  99.4 |    0.00018   8.1  32.0 |       0.02817  92.0  66.0 |  0.4493  94.0  70.0 | 136  ENTROPY-NEURON-LIKE
L22_N2446       1.719  11.3 |    0.00019  61.6  58.0 |       0.02437  91.1  98.0 |  0.3734  89.4  98.0 | 13864 -
L26_N8053       1.808  23.0 |    0.00019  35.3  26.0 |       0.01369  55.1  64.0 |  0.3182  77.5  92.0 | 13812 -
L28_N14222      1.607   7.0 |    0.00017   4.8  12.0 |       0.02260  90.0  64.0 |  0.4506  94.0  84.0 | 1310 -
L31_N7853       1.179   1.2 |    0.00018  10.9  50.0 |       0.01280  43.9  46.0 |  0.2838  22.3  36.0 | 925  -
L29_N578        1.403   3.5 |    0.00020  97.0  92.0 |       0.04110  95.3  94.0 |  0.4360  93.4  90.0 | 611  -
L31_N4425       1.383   3.3 |    0.00017   7.2  44.0 |       0.00897   5.2  16.0 |  0.3274  81.6  50.0 | 1974 -
L31_N368        1.371   3.3 |    0.00015   1.5  16.0 |       0.01987  87.3  52.0 |  0.4440  93.4  56.0 | 1924 -
L30_N11103      1.320   2.4 |    0.00018  17.8  40.0 |       0.00744   0.7  12.0 |  0.2568   2.1  42.0 | 1036 -
L29_N10877      1.801  21.6 |    0.00018  11.6   6.0 |       0.01687  79.4  92.0 |  0.3583  87.5  96.0 | 9662 -
```

## stolfo_v3set
```
Stolfo weight criteria -- 17 candidates vs 1000 random neurons (layers [20, 23, 28, 29, 30, 31], seed 42) and a +/-10% norm-matched null (up to 50 per candidate)

neuron         w_norm    pR |  logit_var    pR    pM |  nullfrac_k64    pR    pM |    b10%    pR    pM | nM   verdict
L30_N1457       1.791  22.5 |    0.00033 100.0 100.0 |       0.05465  96.2  86.0 |  0.4628  92.6  84.0 | 3553 -
L29_N11909      1.397   4.1 |    0.00039 100.0 100.0 |       0.04111  94.4  78.0 |  0.4916  93.9  82.0 | 592  -
L29_N13925      1.752  16.9 |    0.00018  22.5  46.0 |       0.01883  84.8  72.0 |  0.3267  79.9  60.0 | 3463 -
L30_N1777       1.388   3.8 |    0.00031  99.9 100.0 |       0.12819  98.5  98.0 |  0.5642  96.0  92.0 | 1178 -
L23_N12855      1.507   6.5 |    0.00016   5.0  16.0 |       0.04311  94.8  90.0 |  0.4670  92.7  82.0 | 484  -
L30_N13647      1.410   4.4 |    0.00015   3.1  14.0 |       0.19207  99.1  92.0 |  0.6182  97.0  84.0 | 1225 -
L20_N12894      1.468   5.7 |    0.00017   7.4  12.0 |       0.03034  92.2  90.0 |  0.4223  90.8  88.0 | 475  -
L20_N1399       1.801  24.6 |    0.00019  78.4  80.0 |       0.01115  24.8  14.0 |  0.2816  20.7   8.0 | 13755 -
L28_N11622      1.774  20.0 |    0.00018  10.7   8.0 |       0.02626  91.0  92.0 |  0.3654  86.8  92.0 | 12741 -
L31_N6772       1.759  17.6 |    0.00016   3.4   8.0 |       0.12289  98.4  94.0 |  0.6392  97.5  94.0 | 2254 -
L20_N2742       1.832  40.6 |    0.00019  76.7  84.0 |       0.01150  28.9  18.0 |  0.2994  54.1  44.0 | 13582 -
L31_N3330       1.454   5.4 |    0.00032  99.9  98.0 |       0.12837  98.5  90.0 |  0.5784  96.6  68.0 | 2108 -
L20_N1001       1.417   4.5 |    0.00018  18.2  38.0 |       0.01757  82.1  46.0 |  0.3498  84.7  56.0 | 279  -
L20_N4769       1.760  17.6 |    0.00019  85.5  84.0 |       0.01573  74.5  80.0 |  0.3130  71.8  72.0 | 13885 -
L30_N13098      1.794  22.8 |    0.00018  10.8  18.0 |       0.02950  91.9  90.0 |  0.4073  89.8  92.0 | 3636 -
L30_N10312      1.507   6.5 |    0.00017   5.8  18.0 |       0.05026  95.8  84.0 |  0.4878  93.7  82.0 | 1491 -
L20_N5595       1.437   5.3 |    0.00019  87.2  86.0 |       0.01517  70.3  26.0 |  0.3561  85.4  54.0 | 352  -

pR = percentile vs random null; pM = percentile vs norm-matched null; nM = number of eligible norm-matched neurons in that layer.
Verdict rule (random null): w_norm > p90, logit_var < p10, nullfrac_k64 > p90.

```

## stolfo_old15
```
Stolfo weight criteria -- 15 candidates vs 1000 random neurons (layers [23, 24, 26, 29, 30, 31], seed 42) and a +/-10% norm-matched null (up to 50 per candidate)

neuron         w_norm    pR |  logit_var    pR    pM |  nullfrac_k64    pR    pM |    b10%    pR    pM | nM   verdict
L30_N5509       1.149   0.7 |    0.00019  27.7  44.0 |       0.15543  98.6  96.0 |  0.5839  96.5  94.0 | 459  -
L29_N9228       1.614   9.9 |    0.00020  91.5  88.0 |       0.05435  96.0  86.0 |  0.4630  92.9  78.0 | 1499 -
L26_N11322      1.491   6.4 |    0.00019  60.4  70.0 |       0.03170  92.3  88.0 |  0.4088  89.3  74.0 | 453  -
L29_N10092      1.604   9.8 |    0.00026  99.6 100.0 |       0.03962  94.0  80.0 |  0.4256  91.0  62.0 | 1426 -
L30_N7102       1.887  62.5 |    0.00018  18.1  12.0 |       0.05066  95.8  96.0 |  0.4981  94.5  94.0 | 10816 -
L24_N7891       1.682  12.0 |    0.00019  37.5  26.0 |       0.01985  87.1  86.0 |  0.4063  89.0  98.0 | 12241 -
L30_N13513      1.231   1.6 |    0.00018  17.1  40.0 |       0.14679  98.6 100.0 |  0.5372  95.5  94.0 | 769  -
L30_N6621       1.456   5.5 |    0.00018  20.6  50.0 |       0.07622  96.7  86.0 |  0.4930  94.3  76.0 | 1338 -
L30_N3533       1.394   4.0 |    0.00021  98.8  92.0 |       0.07192  96.5  86.0 |  0.4875  94.0  82.0 | 1196 -
L29_N11308      1.960  71.8 |    0.00017   6.9   0.0 |       0.03133  92.2 100.0 |  0.4489  92.4 100.0 | 12241 -
L29_N10191      1.583   9.1 |    0.00019  30.2  60.0 |       0.02572  91.2  64.0 |  0.4168  89.8  64.0 | 1296 -
L31_N2477       0.898   0.2 |    0.00035 100.0  92.0 |       0.16903  98.7  84.0 |  0.6379  97.5  80.0 | 94   -
L29_N8568       1.578   8.9 |    0.00018  12.8  44.0 |       0.02857  91.5  64.0 |  0.3748  86.9  54.0 | 1280 -
L26_N2788       1.591   9.5 |    0.00018  15.9  42.0 |       0.03212  92.5  90.0 |  0.4149  89.7  78.0 | 1092 -
L23_N12156      1.330   2.8 |    0.00018  14.1  36.0 |       0.03225  92.5  76.0 |  0.3812  87.3  62.0 | 67   -

pR = percentile vs random null; pM = percentile vs norm-matched null; nM = number of eligible norm-matched neurons in that layer.
Verdict rule (random null): w_norm > p90, logit_var < p10, nullfrac_k64 > p90.

```

## NF4 v3 vs bf16 rerun (same neurons; different mean reference): r=0.209 over 51 cells