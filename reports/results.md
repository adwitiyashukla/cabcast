# CabCast results

Data source: `tlc_shapefile`
Panel: 2,116,584 zone-hours across 123 zones, 2024-01-15 00:00:00 to 2025-12-31 23:00:00, 91 features.

## Point forecast accuracy on the held-out test block

|                        |     mae |    rmse |   mase |   smape |    bias |
|:-----------------------|--------:|--------:|-------:|--------:|--------:|
| seasonal naive         | 15.2285 | 36.8949 | 1      | 64.0194 |  1.7686 |
| drifted seasonal naive | 13.4333 | 31.5622 | 0.8821 | 62.3703 |  0.4151 |
| historical mean        | 14.1398 | 31.4161 | 0.9285 | 66.0664 | -4.8132 |
| lightgbm               |  6.1184 | 12.8603 | 0.4018 | 44.1829 | -0.5858 |

Best model `lightgbm` reduces MAE by 59.8% against `seasonal_naive`.

## Prediction interval calibration

|                        |   coverage |   coverage_active |   target_coverage |   mean_width |   winkler |
|:-----------------------|-----------:|------------------:|------------------:|-------------:|----------:|
| quantile uncalibrated  |   0.894321 |          0.899155 |               0.9 |      29.4915 |   38.8205 |
| conformalized quantile |   0.899402 |          0.900241 |               0.9 |      29.5331 |   38.8138 |
| mondrian cqr           |   0.904665 |          0.908836 |               0.9 |      30.3429 |   38.7817 |
| split conformal        |   0.891647 |          0.88284  |               0.9 |      23.1688 |   57.6338 |

## Fleet rebalancing

At the peak forecast hour (2025-12-11 21:00:00), repositioning at most 25% of a 2,500 vehicle idle fleet by the Sinkhorn plan, counting only arrivals reachable within 30 minutes, cuts unmet demand from 607 to 326 trips, a 46.3% reduction, at a cost of 18.6 vehicle-minutes per additional trip served. 551 vehicles move and 59 are stranded by the arrival horizon.

## Congestion pricing effect

**Parallel trends fails on this panel, so the naive DiD is not identified.** Pre-treatment coefficients trend -0.95% per period with joint F-test p = 5.15e-13 and 46% of pre-periods individually significant. The unadjusted two-way fixed-effects estimate is -14.36% (95% CI -24.51% to -2.85%), reported as a descriptive contrast only.

Adding zone-specific linear time trends, which absorb the differential drift, moves the estimate to +1.77% (95% CI -5.30% to +9.37%), p = 0.6326. Almost all of the apparent effect was the pre-existing divergence continuing. The zone-trend design is rank deficient, so collinear nuisance parameters were absorbed; the reported standard error on the treatment term is still finite and clustered by zone.

## Drift

91 features scanned, 15 at warn level, 17 at alert level. Prediction PSI 0.1927 (ok).
