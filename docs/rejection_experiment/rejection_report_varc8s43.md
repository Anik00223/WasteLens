# WasteLens Iteration 4 - rejection experiment report

Model: `models\checkpoints\wastelens_rej_varc8s43_10.keras`

## 4-bin classification (supported test)
accuracy=0.9797 (baseline 0.9822), macro-F1=0.9711 (baseline 0.9742)

| Bin | P | R | F1 |
|---|---|---|---|
| recyclable | 0.9880 | 0.9833 | 0.9857 |
| organic | 0.9797 | 0.9797 | 0.9797 |
| hazardous | 0.9645 | 0.9577 | 0.9611 |
| general trash | 0.9358 | 0.9808 | 0.9577 |

## Rejection (thresholds frozen on VAL)
AUROC=0.9996 AUPRC=0.9998

| FRR target | thr | val FRR | OOD detect | test FRR |
|---|---|---|---|---|
| 0.01 | 0.1463 | 0.0106 | 0.9969 | 0.0154 |
| 0.05 | 0.0040 | 0.0504 | 0.9985 | 0.0608 |
| 0.10 | 0.0006 | 0.1007 | 0.9990 | 0.1144 |
| 0.15 | 0.0002 | 0.1503 | 0.9995 | 0.1768 |
| 0.20 | 0.0001 | 0.2006 | 1.0000 | 0.2287 |

### Per-source OOD detection @5% FRR target
| source | detect |
|---|---|
| clothes | 1.0000 |
| shoes | 0.9933 |
| collage | 1.0000 |
| nonwaste | 0.9987 |
