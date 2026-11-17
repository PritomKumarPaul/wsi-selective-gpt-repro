# CONCH-Z patch-coverage MLP: training procedure

This note describes the experiment that produced the patch-ranking MLP used in the CAMELYON16 workflow. The learned head is a **patch-level regressor**: it estimates how much annotated tumor-mask area overlaps a candidate crop. At retrieval time, crops are sorted by this score and the highest-scoring crops can be sent to a vision-language model.

## 1. Assemble the annotated development slides

The source is CAMELYON16: slide-level normal/tumor labels, whole-slide H&E images, tumor XML contours, and pixel-level masks. The split is made at the **slide level before crops are sampled**, so crops from one slide cannot land in both training and validation.

Twenty development slides flagged as having non-exhaustive tumor annotations are excluded. On those slides, a mask pixel that is not marked tumor cannot safely be treated as known normal tissue. The remaining 250 slides are stratified by slide label and center with random seed 42:

| Partition | Slides | Normal | Tumor-positive |
|---|---:|---:|---:|
| Training | 200 | 127 | 73 |
| Held-out validation | 50 | 32 | 18 |

The held-out 50-slide partition is not used for MLP gradient updates or for choosing the number of training epochs. The training code also makes an inner split of the 200 training slides for epoch selection, described below.

## 2. Build candidate crop locations with two grids

Each crop is **1024 × 1024 pixels at level 0** (the slide’s highest-resolution level). The candidate tissue grid is the union of two stride-1024 grids:

1. Grid A starts at slide coordinate `(x=0, y=0)`.
2. Grid B is shifted diagonally by half a crop, `(x=512, y=512)`.

This second grid gives a different crop alignment, so a lesion lying across a boundary in Grid A may appear more centrally in a Grid B crop. The two grids are deduplicated by coordinate. This is two grids only; it does not add the other two half-offset combinations `(512, 0)` and `(0, 512)`.

Before reading every high-resolution crop, the code builds a tissue mask from a slide thumbnail no larger than 2048 × 2048. It converts the thumbnail to HSV, initially marks pixels with saturation `S > 35` and value `V < 250` as tissue, then applies a 3 × 3 morphological opening and a 5 × 5 closing. A grid crop is eligible when at least 50% of its corresponding thumbnail region is marked tissue. This filters many white/background regions; it is an approximate tissue filter, not a tumor detector.

## 3. Sample training crops, using masks only for training labels and positive enrichment

For each tumor-positive training slide, XML tumor contours guide the search for crops likely to overlap a lesion. The sampler centers crops on contour locations and also tries shifted/off-center locations to vary how much lesion and surrounding tissue each crop contains. It measures the overlap against the native-resolution mask and keeps nonzero-overlap crops, aiming for up to 64. It distributes selected positives across four mask-coverage ranges when available; tiny lesions may not provide examples in every range.

The sampler then draws up to 64 crops with zero tumor-mask overlap from the tissue-filtered two-grid pool. A normal training slide contributes up to 64 zero-overlap tissue crops. These are per-slide targets, not a fixed number guaranteed for every slide; available eligible crops and annotations can limit the count.

The final feature cache used for this fit contained **17,420 training crops**, of which **4,672 had nonzero annotated tumor coverage**. The mask is used here to enrich training examples and compute the regression target. The tumor mask is not an input to CONCH-Z or the MLP.

## 4. Encode crops and define the regression target

The image encoder is frozen **CONCH-Z**, run unquantized in FP32. Each 1024 × 1024 RGB source crop goes through CONCH-Z’s evaluation transform to a 448 × 448 model input. The encoder returns a projected, L2-normalized 768-dimensional image feature. For storage efficiency, the cache writes features as FP16; the MLP fitting code converts them back to FP32. **FP16 cache storage is not encoder quantization.**

For crop `i` on slide `s`, the target is its fraction of native level-0 pixels whose mask value is 2:

\[
y_{si}=\frac{\text{number of tumor-labeled mask pixels inside the 1024 × 1024 crop}}{1024^2}.
\]

Thus, a target of 0 means no annotated tumor pixels overlap the crop, and a larger target means a larger annotated tumor fraction. This is neither a binary patch label nor a slide diagnosis. It measures overlap with the available annotation, not clinical severity or every possible abnormality.

For the held-out 50 slides, candidate positions are chosen without consulting the mask, from the tissue-filtered grid by a deterministic spatial shortlist of up to 256 positions per slide. The mask is read only afterward to assign coverage targets and score retrieval. This prevents mask information from choosing validation locations.

## 5. Fit the MLP regression head

The feature vectors are standardized using a `StandardScaler` fit only on the current training fold. The CONCH-Z encoder remains frozen. The trained head is:

```text
768 → Linear(128) → ReLU → Dropout(0.2)
    → Linear(32) → ReLU → Linear(1)
```

It has 102,593 trainable parameters. The scalar output estimates tumor-mask coverage and is used to rank patches; it is **not a calibrated probability**.

### Slide-balanced weighted mean squared error

The loss is mean squared error between the predicted coverage and mask-derived coverage. Each slide receives equal total weight so slides with more cached crops do not dominate. For a tumor-positive slide with both covered and uncovered crops, half of that slide’s weight is assigned across its nonzero-coverage crops and half across its zero-coverage crops. For a normal slide, its zero-coverage crops share that slide’s full weight. The weights are normalized to have mean 1 over the training set.

\[
\mathcal{L}=\frac{1}{N}\sum_{s,i}\widetilde w_{si}
\left(f_\theta(\mathbf z_{si})-y_{si}\right)^2,
\]

where `z` is the standardized CONCH-Z feature, `y` is the crop’s tumor coverage fraction, and `w` is the weight described above. This is supervised regression, not CLIP-style prompt matching and not slide-level classification training.

## 6. Select training duration without using the outer validation set

The 200 training slides are split internally, again stratified by label and center and using seed 42, into 160 inner-training slides and 40 inner-validation slides. The MLP is trained on the 160 slides while inner-validation weighted MSE selects the epoch. Training uses AdamW, learning rate `0.001`, weight decay `0.0001`, batch size 256, a maximum of 100 epochs, and early-stopping patience 10. The selected checkpoint epoch was 14.

The MLP is then reinitialized and trained from scratch on all 200 training slides for exactly 14 epochs. This final refit uses the full training partition while keeping epoch count fixed from the inner split. The separate 50-slide outer validation set remains untouched by fitting and epoch selection; it is used later to evaluate retrieval and, in a separate quantized-encoder phase, calibrate the selective GPT gate.

## 7. Keep training-cache diagnostics distinct from the matched retrieval comparison

The training-cache summary reported pooled patch AUROC 0.9657 and average precision 0.9151, with 17 of 18 tumor-positive outer-validation slides having at least one tumor-overlapping crop in the top 1 through top 32. Those figures describe the cache’s up-to-256-per-slide mask-blind shortlist protocol.

The report’s prompt-similarity-versus-MLP retrieval comparison is a separate, matched experiment: both rankers use the same validation candidate pool and the same mask-overlap scoring rule. Do not compare the cache diagnostic numbers above to that table as if they were produced with an identical candidate pool.

## 8. Reproduce from the repository

The source code is tracked in this repository:

- `code/legacy_repo/patch_fraction_regression.py` — slide split, two-grid tissue candidates, mask-guided training crop sampling, CONCH-Z feature extraction/cache, target generation, and cache diagnostics.
- `code/legacy_repo/train_patch_neural_head.py` — feature standardization, weighted MLP fitting, inner epoch selection, final refit, and validation metrics.
- `code/legacy_repo/train_retrieval_study.py` — shared slide selection, incomplete-annotation exclusion, CONCH-Z preprocessing/encoding helpers, and dataset utilities.

Use data, model weights, cache, and model output directories **outside the Git clone**. Example from the repository root after installing its dependencies and arranging local CAMELYON16 and gated CONCH-Z files:

```bash
DATA_DIR="$HOME/data/camelyon16_full"
MODEL_DIR="$HOME/models/conch-z"
CACHE_DIR="$HOME/wsi_patch_fraction_cache_v1"

python code/legacy_repo/patch_fraction_regression.py plan \
  --data "$DATA_DIR" --model "$MODEL_DIR" --output "$CACHE_DIR"
python code/legacy_repo/patch_fraction_regression.py prepare \
  --data "$DATA_DIR" --model "$MODEL_DIR" --output "$CACHE_DIR" \
  --device cuda:0 --batch-size 8
python code/legacy_repo/train_patch_neural_head.py fit --head mlp \
  --cache "$CACHE_DIR" --output "$CACHE_DIR/neural_mlp_seed42" --device cpu
```

`prepare` is the expensive stage: it reads slides, samples crops, runs CONCH-Z, and writes the feature cache. `fit` reads that cache and trains only the small MLP head; it does not reload slides or rerun CONCH-Z. The cache, slide data, model weights, and trained checkpoint are not included in Git. See the repository README for environment setup, data access/download, and storage cautions.

## What this experiment establishes—and what it does not

It tests whether frozen CONCH-Z features plus a small supervised head can rank mask-overlapping crops above other tissue crops on held-out slides. It does not establish clinical diagnostic performance, exhaustive tumor detection, or edge-device speed. The training sampler is mask-guided while validation candidate selection is mask-blind, so the two distributions differ. Later test-time sampling and GPT classification are separate stages and should be evaluated separately.
