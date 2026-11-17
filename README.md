# WSI Selective GPT: CAMELYON16 source release

A source-only repository for reproducing our CAMELYON16 whole-slide-image (WSI) workflow: dataset download, tissue filtering and patch sampling, frozen CONCH-Z feature extraction, tumor-coverage MLP training, patch retrieval, quantized-encoder gate calibration, GPT classification, random-eight GPT baseline, and an optional CLAM-SB baseline.

This repository contains **code and small download manifests only**. It does not contain WSI slides, masks, embeddings, model weights, GPT outputs, result caches, reports, or figures. Keep all such files outside this clone. The manifests under `code/legacy_repo/examples/` contain dataset object names/checksums, not image data.

There is no project-level `LICENSE` file because the team has not chosen a license for its own code. Decide that before making the repository public; the CAMELYON16 and gated CONCH-Z data/model terms still apply independently.

## Current workflow

1. **Get the data and model separately.** CAMELYON16 images/masks are downloaded to a data directory outside this repository. CONCH-Z is a gated Hugging Face model; each user must obtain access and download it separately.
2. **Prepare local features.** `patch_fraction_regression.py prepare` samples training patches with mask-aware positive-patch enrichment, reads native mask pixels to create coverage labels, and stores CONCH-Z embeddings and metadata in an external cache. No image crops are cached.
3. **Train the MLP head.** `train_patch_neural_head.py fit --head mlp` trains the regression head from cached embeddings. CONCH-Z stays frozen. Its cache/checkpoint are local artifacts, not Git files.
4. **Evaluate retrieval.** `validation_low_tissue_mlp_retrieval.py` and `compare_prompt_mlp_retrieval_validation.py` evaluate ranked crops against masks on validation. Masks are used for training labels and post-ranking retrieval evaluation; they are not an inference-time signal.
5. **Calibrate the edge-oriented gate.** `recalibrate_quantized_mlp_gate.py` recalculates validation features with W8A8 quantized CONCH-Z visual linear layers, applies the already-trained FP32 MLP, and saves a validation gate. The MLP is not quantized or retrained.
6. **Run end to end.** `final_test_quantized_gate_top8_gpt.py` samples test candidates, encodes/ranks them, applies the validation gate, sends top-eight images to GPT when called, asks for the API key interactively, and saves resumable results outside the repository.
7. **Compare a random baseline.** `random8_gpt_baseline.py` samples eight crops uniformly from all eligible crops (no 500-candidate cap) and calls GPT on every slide. Its default 3 seeds mean 387 billable requests on the 129-slide mirror test split. Use only when that cost is intended; it has a $6 preflight budget guard.

The included CLAM-SB code is an optional, separate baseline. Its feature extraction/training can be long; prior run outputs are deliberately not included.

## Repository layout

```text
code/legacy_repo/       Shared WSI utilities, downloaders, feature preparation,
                        MLP training, validation, gate calibration, GPT runners
code/legacy_repo/examples/  Small JSON download manifests (metadata only)
code/experiments/      Prompt-similarity vs MLP retrieval comparison
code/baselines/        Random-eight GPT and CLAM-SB baselines
scripts/               Safe helper for downloading gated CONCH-Z outside Git
requirements.txt       Python dependencies; install PyTorch separately
```

## Environment setup

The reference experiments used Python 3.11, PyTorch 2.5.1+cu121, Transformers 4.57.6, and the pinned packages in `requirements.txt`. Install a PyTorch/torchvision build matching your system first. For the reference CUDA 12.1 environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

OpenSlide's native runtime is supplied by `openslide-bin` on supported Linux wheels. GPU quantization uses CUDA integer matrix multiplication and requires a compatible NVIDIA GPU/PyTorch build. CPU-only users can run data preparation/training with the appropriate script flags, but the final W8A8 gate evaluation requires CUDA.

## Download CAMELYON16 outside this repository

The full pinned public-mirror manifest describes roughly 762 GB of files. Do not run the full download unless you have storage and time for it. These scripts support plan/status/stop/resume/verify operations; check their `--help` before starting a large transfer. Set an external destination explicitly:

```bash
DATA_DIR="${HOME}/data/camelyon16_full"
python code/legacy_repo/download_camelyon16_full.py --start --dest "$DATA_DIR"
python code/legacy_repo/download_camelyon16_full.py --status --dest "$DATA_DIR"
python code/legacy_repo/download_camelyon16_full.py --stop --dest "$DATA_DIR"
python code/legacy_repo/download_camelyon16_full.py --verify --dest "$DATA_DIR"
```

For a smaller development subset, use `download_camelyon16_subset.py --help` and its plan/download/status/stop/verify modes. Do not place the dataset under this repository. The full-data script also includes the earlier subset/expansion download modules because they provide shared checksum and resume helpers.

## Get gated CONCH-Z outside this repository

Request/accept access to [MahmoodLab/CONCH-Z on Hugging Face](https://huggingface.co/MahmoodLab/CONCH-Z) with your own account, then authenticate the Hugging Face CLI. The helper pins the model revision and refuses to save weights inside the clone:

```bash
hf auth login
CONCHZ_MODEL_DIR="${HOME}/models/conch-z" ./scripts/download_conchz.sh
```

The model is gated and carries its own academic-use terms. Do not redistribute its weights. They are not included here.

## Reproduce the feature-cache and MLP stages

For the full rationale and step-by-step description of the two-grid sampler, mask-derived coverage targets, loss weighting, and MLP fitting procedure, see [`docs/patch_coverage_mlp_training.md`](docs/patch_coverage_mlp_training.md).

Set paths to your external dataset, downloaded model, and local cache. The cache defaults to mask-aware patch-fraction regression: it samples positive patches using annotations during training, uses native mask pixels as coverage targets, and uses the selected split to train/evaluate. Validation/test sampling is mask-blind.

```bash
DATA_DIR="${HOME}/data/camelyon16_full"
MODEL_DIR="${HOME}/models/conch-z"
CACHE_DIR="${HOME}/wsi_patch_fraction_cache_v1"

python code/legacy_repo/patch_fraction_regression.py plan \
  --data "$DATA_DIR" --model "$MODEL_DIR" --output "$CACHE_DIR"
python code/legacy_repo/patch_fraction_regression.py prepare \
  --data "$DATA_DIR" --model "$MODEL_DIR" --output "$CACHE_DIR" \
  --device cuda:0 --batch-size 8
python code/legacy_repo/train_patch_neural_head.py fit --head mlp \
  --cache "$CACHE_DIR" --output "$CACHE_DIR/neural_mlp_seed42" --device cpu
```

`prepare` performs the expensive CONCH-Z image encoding. `fit` uses those cached features and does not run CONCH-Z again. Do not commit `$CACHE_DIR`; it contains embeddings, trained weights, and annotation-derived metadata. If rebuilding a different split/configuration, use a new cache path rather than mixing runs.

## Validation retrieval and gate calibration

```bash
python code/legacy_repo/validation_low_tissue_mlp_retrieval.py plan \
  --training "$CACHE_DIR" --output "${HOME}/wsi_validation_low_tissue_retrieval_v1"
python code/legacy_repo/validation_low_tissue_mlp_retrieval.py run \
  --training "$CACHE_DIR" --output "${HOME}/wsi_validation_low_tissue_retrieval_v1" \
  --device cuda:0 --batch-size 8
python code/legacy_repo/recalibrate_quantized_mlp_gate.py \
  --training "$CACHE_DIR" --output "${HOME}/mlp_validation_gate_quantized_v1" \
  --device cuda:0 --batch-size 8
```

The prompt-vs-MLP comparison can be run after its required validation feature cache exists:

```bash
python code/experiments/compare_prompt_mlp_retrieval_validation.py \
  --training-cache "$CACHE_DIR" \
  --feature-cache "${HOME}/wsi_validation_low_tissue_retrieval_v1" \
  --prompts code/legacy_repo/prompts_camelyon16.json \
  --output "${HOME}/retrieval_prompt_vs_mlp_validation" --device cpu
```

The prompt comparison is a retrieval experiment; it does not call GPT. The all-GPT validation runners are `validation_fp32_mlp_always_gpt.py` and `validation_fp32_zeroshot_always_gpt.py`; inspect `--help` and the cost guard before using them.

## End-to-end GPT test and random-eight baseline

The following test runner processes the full mirror test split. It uses the validation gate, quantized CONCH-Z visual linears, the unquantized FP32 MLP head, top eight crops, and an interactive hidden API-key prompt. It makes billable requests for slides that pass the gate. Outputs and cached GPT responses are written to the specified external run directory. This runner does not currently have a hard dollar-budget cap; review the estimated call count and account budget before starting it. The random-eight runner separately includes a $6 preflight guard.

```bash
python code/legacy_repo/final_test_quantized_gate_top8_gpt.py \
  --training "$CACHE_DIR" --data "$DATA_DIR" \
  --gate-file "${HOME}/mlp_validation_gate_quantized_v1/gate.json" \
  --output "${HOME}/wsi_test_quantized_runtime_gpt_v1" \
  --device cuda:0 --batch-size 8
```

The random-eight baseline makes fresh GPT calls for every slide, for each seed. It is intentionally not a selective-call run and does not use CONCH-Z/MLP or a candidate cap. Its API budget guard defaults to $6, but a single request can exceed the per-call reserve if actual usage differs.

```bash
python code/baselines/random8_gpt_baseline.py \
  --data "$DATA_DIR" --output "${HOME}/random8_gpt" --seeds 1 2 3
```

Both GPT scripts prompt for the key without echoing it; keys are not stored in run records. Review the prompts, intended slide split, and expected request count before running. Never commit run output, logs, API keys, or `.env` files.

## Optional CLAM-SB baseline

`code/baselines/clam_sb_baseline.py` implements the slide-level CLAM-SB-style MIL baseline over frozen CONCH-Z patch features. `clam_parallel_run.py` schedules extraction workers across chosen GPU IDs. The run requires the split manifest from the training cache, external WSI/model paths, and substantial disk/time. Read `--help` and the script docstrings before launching; prior cache/results are not bundled.

## Important scope notes

- Project task: binary CAMELYON16 slide classification and retrieval of patches overlapping annotated metastasis.
- The feature encoder is frozen CONCH-Z; the learned component is an MLP regression head trained to predict annotated tumor-pixel fraction in a crop.
- Masks are used as training targets and for offline retrieval evaluation, not as inputs to test-time sampling or GPT.
- CONCH-Z gets RGB 1024×1024 level-0 crops through its own evaluation transform to 448×448 before encoding. GPT gets resized 512×512 crops.
- The random-eight baseline and the gated MLP pipeline have different sampling and call protocols; their accuracies are not a perfectly controlled head-to-head comparison.
- This is an exploratory academic project, not a clinically validated diagnostic system. Follow CAMELYON16 and CONCH-Z terms.
