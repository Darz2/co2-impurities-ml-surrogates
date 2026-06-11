# cDFT map timing — phase equilibria vs cDFT sweep (per feed)

Same physics as `VLE_IFT_V5.ipynb`, default grid (ngrid=500, T_STEP=5, cDFT_NP=5), no critical-region enhancement, hybrid threading on c109.
Phase-equilibria time is computed **once per feed** (critical point + bubble/dew envelope); the per-point column amortizes it over the map points. The cDFT-sweep time is the parallel TP-flash + planar-interface solve over all (T,P) points.

| feed | points | isotherms | phase-eq (s) | cDFT sweep (s) | total (s) | phase-eq/pt (ms, amort.) | cDFT/pt (s) | total/pt (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 95 | 19 | 1.6 | 15.8 | 17.4 | 16.8107 | 0.1666 | 0.1835 |
| 1 | 100 | 20 | 1.8 | 57.9 | 59.7 | 18.1130 | 0.5787 | 0.5969 |
| 2 | 100 | 20 | 2.2 | 51.1 | 53.3 | 22.0180 | 0.5106 | 0.5326 |
| 3 | 105 | 21 | 3.5 | 181.1 | 184.6 | 33.4345 | 1.7248 | 1.7582 |
| 4 | 100 | 20 | 3.1 | 176.1 | 179.1 | 30.5580 | 1.7606 | 1.7911 |

**Totals:** 500 points, phase-eq 12.2 s, cDFT sweep 481.9 s. Phase-eq is 2.5% of the combined stage time.

Node: c109 | outer workers 4 x rayon 4.
