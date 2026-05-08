# Urban Flood Mapping Using Satellite SAR Data

Semantic segmentation of urban flood areas using 8-channel Sentinel-1 SAR imagery.
This is a deep learning course project done across four deliverables, starting from
dataset exploration all the way to a physics-informed final model.

The hard part about this problem is that flooded urban areas do not look like flooded
open areas in SAR. Open floodwater is dark because smooth water deflects radar energy
away from the sensor. In cities, the same water creates a dihedral corner between the
flood surface and a building wall, bouncing energy back at the sensor. The pixel looks
bright, same as a dry building facade. Standard segmentation models just cannot figure
that out without being told the physics. That is what this project is about.

---

## Repository Structure

```
Deliverable 2 — Dataset (UrbanSARFlood preprocessing and exploration)

Deliverable 3 — Baseline (U-Net + ResNet50)

Deliverable 4 — First Improvements
├── mit_transformer.ipynb        (U-Net + MiT-B3 transformer encoder)
└── unet++_efficientnet_attention.ipynb  (U-Net++ + EfficientNet-B3 + decoder attention)

Deliverable 5 — Final System (all phase experiments + final combined model)
├── cnn-mamba.ipynb              (Phase 3a: Hybrid CNN + Mamba SSM bottleneck)
├── upsample.ipynb               (Phase 3b: 2x super-resolution upsampling)
├── physics-approach.ipynb       (Phase 3c: Physics double-bounce loss)
├── gnn-approach.ipynb           (Phase 3d: GNN mass-conservation constraint)
└── final-run.ipynb              (Final combined system — main contribution)
```

---

## Dataset

UrbanSARFlood — 8-channel Sentinel-1 chips per scene.

| Channels | Content |
|----------|---------|
| 1-4 | Pre and post-flood interferometric coherence (VH, VV) |
| 5-8 | Pre and post-flood backscatter intensity (VH, VV) |

Three classes: Non-Flooded (0), Flooded Open (1), Flooded Urban (2).
Dataset is roughly 85-90% Non-Flooded. All training runs apply 4x oversampling
of Flooded Urban chips to partially correct this.

Primary metric is Urban F1. It is the hardest class and the one that matters.

---

## Experiments

| Phase | Architecture | Urban F1 | Stable |
|-------|-------------|----------|--------|
| Phase 1 | U-Net + ResNet50 | 0.1384 | Yes |
| Phase 2a | U-Net++ + EfficientNet-B3 + Attention | 0.1397 | Yes |
| Phase 2b | U-Net + MiT-B3 | 0.0673 | Yes |
| Phase 3a | CNN + Mamba SSM bottleneck | 0.1200 | Partial |
| Phase 3b | 2x Super-Resolution Upsampling | 0.68* | No |
| Phase 3c | Physics Double-Bounce Loss | 0.0534 | No |
| Phase 3d | GNN Mass-Conservation | 0.0076 | No |
| **Final** | **UNet++ + EfficientNet-B4 + Compound Loss** | **0.0864** | **Yes** |

*Inflated by an absent-class metric bug. The model was assigning IoU=1.0 and F1=1.0
to classes absent from both predictions and targets. Since Flooded Urban only appears
in 10-15% of validation chips, this quietly pulled Urban F1 to 0.68. Not real.

---

## Final Model (D4)

This is the main contribution. It combines the components that actually worked across
all prior phases and fixes what caused each one to fail.

### Architecture

**UNet++ with EfficientNet-B4 encoder (ImageNet pretrained), SCSE decoder attention.**

- UNet++ adds dense nested skip connections between encoder and decoder stages.
  Instead of raw skip connections, each node re-processes features through multiple
  convolutions, which helps a lot for fine-grained urban flood boundaries.
- EfficientNet-B4 has ~19M parameters vs B3's ~12M. Better multi-scale feature
  extraction without going over memory at batch size 8.
- SCSE attention applies squeeze-excitation both spatially and channel-wise in the
  decoder. This helps the model focus on post-flood intensity channels and ignore
  background clutter.
- 20.9M total trainable parameters. Input: 8-channel 256x256 SAR chip.
  Output: 3-class segmentation mask.

### What was dropped and why

| Component | Why dropped |
|-----------|-------------|
| Mamba SSM blocks | Sequential pure-PyTorch recurrence, FP32 bugs under AMP, trained from scratch |
| GNN mass-conservation | CPU SLIC overhead (2-3s/batch), needs pretrained backbone as foundation |
| 2x upsampling | Memory bottleneck at batch 4, premature convergence, metric bug made results untrustworthy |

### Loss Function

$$\mathcal{L}_{total} = \mathcal{L}_{WCE} + 0.5 \cdot \mathcal{L}_{Dice} + \lambda_{phys}(e) \cdot \mathcal{L}_{physics}$$

Three terms:

**Weighted Cross-Entropy** — class weights [0.15, 0.35, 0.50] for Non-Flooded,
Flooded Open, Flooded Urban.

**Soft Dice Loss** — directly optimizes boundary overlap. Complements WCE which
is relatively insensitive to boundary precision on small classes.

**Physics Double-Bounce Penalty** — this is the novel part. A pixel is flagged as
a double-bounce candidate when two conditions hold simultaneously:
1. Post-flood minus pre-flood VH or VV intensity exceeds 0.25 (large backscatter
   increase consistent with dihedral scattering)
2. At least one pixel in the 3x3 neighborhood has during-flood VV intensity below
   0.15 (smooth open water surface present nearby as the partner reflector)

On candidate pixels, the model is penalized exponentially for predicting Non-Flooded:

$$\mathcal{L}_{physics} = \frac{1}{|\mathcal{M}_{db}|} \sum_{(i,j) \in \mathcal{M}_{db}} \left( e^{\,\alpha \cdot p_0(i,j)} - 1 \right), \quad \alpha = 4$$

$\lambda_{phys}$ ramps linearly from 0 to 0.8 over epochs 5-20. This is critical.
Phase 3c ran $\lambda = 1.0$ from epoch one and collapsed. Warmup gives the backbone
time to learn basic class structure before the physics penalty starts firing.

### Training Setup

| Config | Value |
|--------|-------|
| Optimizer | AdamW, lr=3e-4, weight decay=1e-4 |
| Schedule | 3-epoch linear warmup then cosine annealing |
| Gradient clipping | norm 1.0 |
| Batch size | 8 |
| Epochs | 50 |
| Mixed precision | AMP |
| FU oversampling | 4x |
| Checkpoint | Best Urban F1 and best val loss saved separately |

### Results
mIoU         : 0.4105
mF1          : 0.4412
Non-Flooded  : IoU=0.9886   F1=0.9941
Flooded Open : IoU=0.1874   F1=0.2430
Flooded Urban: IoU=0.0550   F1=0.0864

Stable across all 50 epochs. No degenerate convergence. Best checkpoint at the epoch
where Urban F1 peaked at 0.0864. The physics penalty at lambda=0.8 still ended up
dominating the gradient on double-bounce-dense batches, which hurt Urban F1 relative
to Phase 2a. A lower lambda_max of 0.3-0.4 is the recommended next step.

---

## Key Takeaways

The best Urban F1 across all phases came from Phase 2a (0.1397), not the final system.
That tells you something honest about how hard this problem is. The physics constraint
targeted the right failure mode but needs more careful tuning to not overpower the
segmentation gradient. The GNN mass-conservation idea is correct in principle but
needs a two-stage training procedure: pretrain a clean segmenter first, then introduce
the graph penalty at a low weight. Introducing it simultaneously with training from
scratch gave it no foundation to work with.

---

## Stack

PyTorch · segmentation-models-pytorch · Albumentations · Rasterio · Kaggle (Tesla T4 16GB)

---

## Authors

Muhammad Sharjeel Nawaz · Syed Moiz Ejaz
