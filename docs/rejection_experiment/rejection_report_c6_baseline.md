# WasteLens Iteration 4 - rejection experiment report

Model: `models\checkpoints\wastelens_rej_frozen_best.keras`

## 4-bin classification (supported test)
accuracy=0.9822 (baseline 0.9822), macro-F1=0.9742 (baseline 0.9742)

| Bin | P | R | F1 |
|---|---|---|---|
| recyclable | 0.9928 | 0.9821 | 0.9874 |
| organic | 0.9735 | 0.9932 | 0.9833 |
| hazardous | 0.9650 | 0.9718 | 0.9684 |
| general trash | 0.9358 | 0.9808 | 0.9577 |

## Rejection (thresholds frozen on VAL)
AUROC=0.9900 AUPRC=0.9937

| FRR target | thr | OOD detect | test FRR |
|---|---|---|---|
| 0.01 | 0.8774 | 0.8426 | 0.0138 |
| 0.05 | 0.5432 | 0.9608 | 0.0665 |
| 0.10 | 0.2827 | 0.9840 | 0.1200 |
| 0.15 | 0.1783 | 0.9923 | 0.1606 |
| 0.20 | 0.1160 | 0.9943 | 0.2101 |

### Per-source OOD detection @5% FRR target
| source | detect |
|---|---|
| clothes | 0.9800 |
| shoes | 0.9430 |
| collage | 0.5778 |
| nonwaste | 0.9933 |
