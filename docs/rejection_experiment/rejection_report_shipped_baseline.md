# WasteLens Iteration 4 - rejection experiment report

Model: `models\checkpoints\wastelens_rej_shipped_best.keras`

## 4-bin classification (supported test)
accuracy=0.9813 (baseline 0.9822), macro-F1=0.9721 (baseline 0.9742)

| Bin | P | R | F1 |
|---|---|---|---|
| recyclable | 0.9823 | 0.9928 | 0.9876 |
| organic | 0.9864 | 0.9797 | 0.9831 |
| hazardous | 0.9710 | 0.9437 | 0.9571 |
| general trash | 0.9800 | 0.9423 | 0.9608 |

## Rejection (thresholds frozen on VAL)
AUROC=0.9997 AUPRC=0.9998

| FRR target | thr | val FRR | OOD detect | test FRR |
|---|---|---|---|---|
| 0.01 | 0.0702 | 0.0106 | 0.9964 | 0.0178 |
| 0.05 | 0.0029 | 0.0504 | 0.9985 | 0.0770 |
| 0.10 | 0.0006 | 0.1007 | 1.0000 | 0.1290 |
| 0.15 | 0.0002 | 0.1503 | 1.0000 | 0.1995 |
| 0.20 | 0.0001 | 0.2006 | 1.0000 | 0.2466 |

### Per-source OOD detection @5% FRR target
| source | detect |
|---|---|
| clothes | 1.0000 |
| shoes | 0.9933 |
| collage | 1.0000 |
| nonwaste | 0.9987 |
