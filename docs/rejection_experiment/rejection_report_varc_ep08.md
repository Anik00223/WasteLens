# WasteLens Iteration 4 - rejection experiment report

Model: `models\checkpoints\wastelens_rej_varc_08.keras`

## 4-bin classification (supported test)
accuracy=0.9813 (baseline 0.9822), macro-F1=0.9725 (baseline 0.9742)

| Bin | P | R | F1 |
|---|---|---|---|
| recyclable | 0.9835 | 0.9917 | 0.9875 |
| organic | 0.9799 | 0.9865 | 0.9832 |
| hazardous | 0.9778 | 0.9296 | 0.9531 |
| general trash | 0.9709 | 0.9615 | 0.9662 |

## Rejection (thresholds frozen on VAL)
AUROC=0.9997 AUPRC=0.9998

| FRR target | thr | OOD detect | test FRR |
|---|---|---|---|
| 0.01 | 0.4669 | 0.9964 | 0.0146 |
| 0.05 | 0.0446 | 0.9985 | 0.0608 |
| 0.10 | 0.0109 | 0.9990 | 0.1135 |
| 0.15 | 0.0033 | 1.0000 | 0.1784 |
| 0.20 | 0.0016 | 1.0000 | 0.2084 |

### Per-source OOD detection @5% FRR target
| source | detect |
|---|---|
| clothes | 1.0000 |
| shoes | 0.9933 |
| collage | 1.0000 |
| nonwaste | 0.9987 |
