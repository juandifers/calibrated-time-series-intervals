# Backtest Results

| Station | Coverage (target 0.90) | MAE | RMSE | Mean interval width | Replay windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Station_1 | 0.874 | 0.386 | 0.523 | 1.664 | 10 |
| Station_2 | 0.868 | 1.255 | 1.730 | 5.209 | 10 |
| Station_3 | 0.932 | 4.156 | 6.387 | 22.589 | 10 |
| Station_4 | 0.897 | 1.038 | 1.415 | 4.566 | 10 |
| Station_7 | 0.904 | 2.581 | 3.557 | 15.159 | 9 |
| Station_8 | 0.902 | 1.790 | 2.622 | 7.991 | 10 |
| **Overall** | **0.896** | **1.856** | **3.302** | **9.434** | **59** |

_Empirical coverage vs. a nominal 90% target, aggregated over 59 historical replay windows (base intervals, no runtime overlays), weighted by the number of valid forecast points._
