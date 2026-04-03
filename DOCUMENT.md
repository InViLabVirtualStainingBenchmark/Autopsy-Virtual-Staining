# DOCUMENT.md -- Benchmark Smoke Test Log

<!--
  This file lives in the root of every forked repo.
  Fill it in as you go. Do not reconstruct it after the fact.
  Keep entries factual and brief. The audience is a future person
  reproducing your setup on a different machine or the HPC cluster.
-->

---

## Model Info

- **Model name:** Autopsy Virtual Staining (RegiStain)
- **Upstream repo URL:** https://github.com/liyuzhu1998/Autopsy-Virtual-Staining
- **Fork URL:** (fill in your fork URL)
- **Upstream last commit date:** (check GitHub)
- **Paper / citation:** Li et al., "Virtual histological staining of unlabeled autopsy tissue", Nature Communications 2024
- **Paired or unpaired assumption:** paired
- **Intended staining task (if domain-specific):** autofluorescence (DAPI + TxRed) → H&E; adapted here for H&E → IHC (BCI/MIST-HER2)

---

## Environment Claimed by Authors

- **Python version:** 3.8.15
- **PyTorch version:** N/A — this model uses TensorFlow, not PyTorch
- **CUDA version:** 11.3
- **Installation method:** conda
- **Requirements file present:** tf2_env.yaml
- **Pretrained weights available:** yes — Generator checkpoint (model_G_iter=87700.h5)
- **Pretrained weights notes:**
  Hosted on Zenodo at https://doi.org/10.5281/zenodo.10203424 (stable host).
  Also includes 10 example FOV images (2048x2048 px, autofluorescence .mat files).

---

## Environment Actually Used

- **Python version:** 3.8.15
- **TensorFlow version:** 2.5.0
- **CUDA version:** 11.3.1
- **Conda environment name:** tf2_env
- **Date tested:** 2026-04-03
- **Hardware:** RTX 4090, WSL2 on Windows 11

### GPU Confirmation

<!--
  TensorFlow GPU check (not PyTorch):
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  Expected output: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
-->

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# RTX 4090 detected, 21306 MB VRAM allocated by TensorFlow
```

---

## Installation

### Commands Run

```bash
cd ~/internship-models/Autopsy-Virtual-Staining
# tf2_env.yaml has Windows build strings — use the Linux-compatible version instead
conda env create -f tf2_env_linux.yaml
conda activate tf2_env
# fix numpy/scipy ABI mismatch (see Issues below)
pip install --force-reinstall scipy==1.7.3
# verify imports
python -c "import ops; import batch_utils; import watcher; print('imports ok')"
# verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Issues and Fixes

| Issue | Fix Applied |
|-------|-------------|
| `watcher.py` not in repo, `from watcher import Watcher` crashes on import | Created `watcher.py` stub (see Changes table below) |
| `tf2_env.yaml` uses Windows-only packages (`vc`, `vs2015_runtime`, `wincertstore`) and Windows build strings (e.g. `h2bbff1b_7`) — fails on Linux | Created `tf2_env_linux.yaml`: removed Windows packages and all build hashes, added `conda-forge` channel, added `tensorflow-addons==0.13.0` (missing from original), moved `numpy` to pip section pinned at `1.19.5` |
| `RuntimeError: module compiled against API version 0xe but this version of numpy is 0xd` — numpy/scipy ABI mismatch at import | `scipy=1.9.3` (conda) was built against numpy 1.23.x; TF 2.5.0 requires numpy 1.19.5. Fix: `pip install --force-reinstall scipy==1.7.3` |
| `tensorflow.python.framework.errors_impl.NotFoundError` when `tf.io.gfile.mkdir()` is called with non-existent parent directory | Replaced `tf.io.gfile.mkdir()` with `os.makedirs(path, exist_ok=True)` in `test_G.py` and `train_stage2_seperate_train_by_iters.py` |
| `Could not load library libcudnn_cnn_infer.so.8 ... libcuda.so: cannot open shared object file` on first forward pass (WSL2) | `libcuda.so` (CUDA driver stub) lives at `/usr/lib/wsl/lib/` in WSL2 and is not in the default library path. Fix: `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`; made permanent via `tf2_env` conda activation script at `~/miniconda3/envs/tf2_env/etc/conda/activate.d/wsl_cuda.sh` |

---

## Dataset Preparation

<!--
  BCI images are stored as side-by-side JPEG pairs: left half = H&E, right half = IHC.
  They must be split into separate A (H&E) and B (IHC) folders before use.
-->

- **Dataset used:** BCI
- **Format expected by model (after refactor):** Separate A/B folders — `trainA/`, `trainB/`, `testA/`, `testB/`
- **Conversion applied:**
  ```bash
  # BCI images are side-by-side JPEG pairs (left=H&E, right=IHC).
  # For smoke test, reused the already-split dataset from the CUT repo:
  #   ../contrastive-unpaired-translation/datasets/BCI_dataset/BCI_dataset/
  # No additional conversion needed — trainA/trainB/testA/testB already exist.

  # If starting from raw BCI download, split with:
  python - << 'EOF'
  import os, glob
  from PIL import Image
  for split in ['train', 'test']:
      src = f'~/BCI/{split}/'
      os.makedirs(f'{src}A', exist_ok=True)
      os.makedirs(f'{src}B', exist_ok=True)
      for p in glob.glob(f'{src}*.jpg'):
          img = Image.open(p)
          w, h = img.size
          img.crop((0, 0, w//2, h)).save(f'{src}A/' + os.path.basename(p))
          img.crop((w//2, 0, w, h)).save(f'{src}B/' + os.path.basename(p))
  EOF
  ```
- **Final folder layout used:**
  ```
  contrastive-unpaired-translation/datasets/BCI_dataset/BCI_dataset/
    trainA/   <-- H&E source images (1024x1024)
    trainB/   <-- IHC target images (1024x1024)
    testA/    <-- 977 images
    testB/    <-- 977 images

  contrastive-unpaired-translation/datasets/MIST/HER2-004/TrainValAB/
    trainA/   <-- H&E source images (1024x1024)
    trainB/   <-- IHC target images (1024x1024)
    testA/    <-- 1000 images
    testB/    <-- 1000 images
    valA/
    valB/
  ```
- **Number of images used for smoke test (train / test):** BCI: 977 test / full train; MIST: 1000 test / full train

---

## Pretrained Weights

- **Download source URL:** https://doi.org/10.5281/zenodo.10203424
- **Host stability:** stable (Zenodo)
- **Weights placed at (relative path):** `pretrained/model_G_iter=87700.h5`
- **Size on disk:** (fill in after download)

---

## Inference Smoke Test

- **Script / command run:**
  ```bash
  conda activate tf2_env
  cd ~/internship-models/Autopsy-Virtual-Staining
  python test_G.py \
    --data_dir ../contrastive-unpaired-translation/datasets/BCI_dataset/BCI_dataset/testB/ \
    --checkpoint pretrained/model_G_iter=87700.h5 \
    --output_dir ../autopsy-VS-outputs/BCI-pretrained/ \
    --image_size 512 \
    --gpu 0
  ```
- **Output folder:** `outputs/BCI-pretrained-smoke/`
- **Number of output images produced:** 50
- **Output image dimensions:** 1024×1024
- **Visual check result:** outputs show colorful pink/purple staining — pipeline confirmed working. Note: pretrained weights were trained on autofluorescence→H&E (not H&E→IHC), so outputs are H&E-like rather than IHC brown DAB — domain mismatch is expected with these weights
- **Time to run (approx):** ~10 min for 50 images (includes CUDA JIT warm-up for first ~10 images; TF 2.5 has no precompiled kernels for RTX 4090 sm_89 — driver JIT fallback is used)
- **Errors or warnings during inference:** `ptxas not found` / `Relying on driver to perform ptx compilation` — harmless warning, driver JIT used as fallback. `LD_LIBRARY_PATH=/usr/lib/wsl/lib` required (WSL2 libcuda.so fix)

---

## Training Smoke Test

- **Script / command run:**
  ```bash
  conda activate tf2_env
  cd ~/internship-models/Autopsy-Virtual-Staining
  python train_stage2_seperate_train_by_iters.py \
    --model_dir ../autopsy-VS-train/ \
    --train_data "../contrastive-unpaired-translation/datasets/BCI_dataset/BCI_dataset/trainB/*.jpg" \
    --val_data "../contrastive-unpaired-translation/datasets/BCI_dataset/BCI_dataset/testB/*.jpg" \
    --gpu 0
  ```
  Note: For a 2-epoch smoke test, temporarily set `tc.N_epoch = 2` in `init_parameters()`
  before running (or wait for `--epochs` arg to be added).
- **Dataset used:**
- **Epochs run:**
- **Batch size:** 4 (default)
- **Input resolution:** 256x256 patches (extracted from full images)
- **Time per epoch (approx):**
- **Peak GPU memory (approx, from nvidia-smi):**
- **Checkpoint saved:** yes / no
- **Checkpoint path:**
- **Crash or error during training:**

---

## Output Verification

- **Output folder:**
- **Example output filenames:**
- **Dimensions match input:** yes / no
- **Visual sanity check:**
- **Any obvious artifacts or failure modes:**

---

## Changes Made to Original Code

| File | Change Description | Reason |
|------|--------------------|--------|
| `watcher.py` | Created new file — minimal no-op stub for `Watcher` class | File was not committed to the original repo; `from watcher import Watcher` crashes training script on import |
| `tf2_env_linux.yaml` | Created new file — Linux-compatible environment derived from `tf2_env.yaml` | Original yaml contains Windows-only conda packages (`vc`, `vs2015_runtime`, `wincertstore`) and Windows build hashes that are unavailable on Linux; also adds missing `tensorflow-addons==0.13.0` |
| `ops.py` | Added `import os`; replaced `model_path + 'code'` (×4) with `os.path.join(model_path, 'code')` | String concatenation without separator gives wrong paths when model_path lacks trailing slash |
| `train_stage2_seperate_train_by_iters.py` | Added `import argparse` and `parse_args()` function | No CLI args existed; all paths were hardcoded Windows drive letters |
| `train_stage2_seperate_train_by_iters.py` | Replaced hardcoded `tc.model_path = 'L:/...'` with `args.model_dir` | Hardcoded Windows path |
| `train_stage2_seperate_train_by_iters.py` | Replaced hardcoded `tc.image_path = 'L:/.../*.mat'` with `args.train_data` | Hardcoded Windows path |
| `train_stage2_seperate_train_by_iters.py` | Replaced hardcoded `vc.image_path = 'J:/.../*.mat'` with `args.val_data` | Hardcoded Windows path on different drive |
| `train_stage2_seperate_train_by_iters.py` | Replaced hardcoded `CUDA_VISIBLE_DEVICES = "1"` with `args.gpu`; moved after `parse_args()` | Hardcoded GPU index; env var must be set before TF import |
| `train_stage2_seperate_train_by_iters.py` | `is_mat` now set from `args.is_mat` (default False); added conditional `convert_inp_path_from_target` for A/B folder layout | `.mat` was hardcoded; BCI/MIST-HER2 use PNG in A/B folders |
| `train_stage2_seperate_train_by_iters.py` | `tf.io.gfile.mkdir(tc.model_path + '/output')` → `os.makedirs(os.path.join(tc.model_path, 'output'), exist_ok=True)` | `tf.io.gfile.mkdir` raises `NotFoundError` if parent directory does not exist; also fixed string concatenation to use `os.path.join` |
| `test_G.py` | Added `import argparse` and `parse_args()` function | No CLI args existed |
| `test_G.py` | Replaced hardcoded `L:\\...\\*.mat` image path with `os.path.join(args.data_dir, '*.{ext}')` | Hardcoded Windows backslash path |
| `test_G.py` | Replaced hardcoded `model_path`, `checkpoint_path`, `output_path` Windows strings with `args.checkpoint`, `args.output_dir` | Hardcoded Windows paths |
| `test_G.py` | Replaced `CUDA_VISIBLE_DEVICES = "0"` with `args.gpu`; moved after `parse_args()` | Hardcoded GPU index |
| `test_G.py` | `tc.image_size = 2048` → `args.image_size` (default 256) | Hardcoded 2048 (autopsy slides); BCI images are smaller |
| `test_G.py` | `is_mat` now from `args.is_mat`; added conditional `convert_inp_path_from_target` | Same as train script |
| `test_G.py` | `valid_image_path.split('\\')[-1]` → `os.path.splitext(os.path.basename(valid_image_path))[0] + '.png'` | Backslash split breaks on Linux |
| `test_G.py` | `valid_image_path.split('\\')[-3]` → `os.path.basename(os.path.dirname(os.path.dirname(...)))` | Backslash split breaks on Linux |
| `test_G.py` | `tf.concat([valid_x[j,:,:,0:2], valid_x[j,:,:,3:4]], ...)` → `valid_x[j]` | Channel index 3 does not exist for 2-channel input (visualization-only code) |
| `test_G.py` | `tf.io.gfile.mkdir(output_path)` → `os.makedirs(output_path, exist_ok=True)` | `tf.io.gfile.mkdir` raises `NotFoundError` if parent directory does not exist |
| `test_G.py` | Added `--ext` argument (default `png`); `ext = 'mat' if args.is_mat else args.ext` | Glob pattern was hardcoded to `*.png`; MIST-HER2 images are `.jpg` so the file list was empty |
| `batch_utils.py` | Added `import cv2` | Needed for RGB image loading |
| `batch_utils.py` | `ImageTransformationBatchLoader`: replaced `np.load()` else-branch with `cv2.imread()` + `/255.0` | BCI/MIST-HER2 are RGB images, not `.npy` arrays |
| `batch_utils.py` | `ImageTransformationBatchLoader_Testing`: same replacement | Same reason |

### Known Limitation Documented Here

The Generator (Attention U-Net) was designed for 2-channel input (`num_slices=2`). H&E source
images have 3 RGB channels. With `channel_end_index=2` (default), only the R and G channels
are passed as input — the B channel is discarded. This is the minimal approach that avoids
any architecture change. When training from scratch on BCI/MIST-HER2, this means some
blue-channel structural information is not used. The pretrained weights are not compatible
with 3-channel input.

---

## Frozen Environment

- **Environment file:** `environment_autopsy-vs.yml`
- **Committed to fork:** yes / no
- **Notes on unusual or heavy dependencies:**
  TensorFlow 2.5.0 is pinned; this is an older version incompatible with newer CUDA.
  Requires CUDA 11.3 + cuDNN 8.2.1 specifically. Cannot share environment with PyTorch models.

---

## HPC Readiness Notes

- **Display/GUI dependencies to remove or neutralize:** `matplotlib` uses `plt.imsave()` only (no `plt.show()`); safe for headless use as-is
- **System-level dependencies (non-pip/conda):** none identified
- **Estimated GPU memory requirement:** (fill in from nvidia-smi during smoke test)
- **Estimated storage requirement (weights + data):** weights ~200 MB (estimate); BCI dataset ~10 GB
- **Other notes for cluster adaptation:**
  CUDA 11.3 module must be loaded on cluster. Check VSC module availability before porting.

---

## Summary

**Overall result:** (fill in after smoke test)

<!-- Example:
"autopsy-VS smoke test completed on [date]. [Result here].
Refactoring required 5 files and 1 new file (watcher.py stub).
Frozen environment committed. Ready for full benchmark run."
-->
