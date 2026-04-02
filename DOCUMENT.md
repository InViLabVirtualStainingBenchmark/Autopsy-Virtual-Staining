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

- **Python version:**
- **TensorFlow version:** 2.5.0
- **CUDA version:**
- **Conda environment name:** tf2_env
- **Date tested:**
- **Hardware:** RTX 4090, WSL2 on Windows 11

### GPU Confirmation

<!--
  TensorFlow GPU check (not PyTorch):
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  Expected output: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
-->

```
# paste output here
```

---

## Installation

### Commands Run

```bash
cd ~/thomas/internship-models/autopsy-VS
conda env create -f tf2_env.yaml
conda activate tf2_env
# verify imports
python -c "import ops; import batch_utils; import watcher; print('imports ok')"
```

### Issues and Fixes

| Issue | Fix Applied |
|-------|-------------|
| `watcher.py` not in repo, `from watcher import Watcher` crashes on import | Created `watcher.py` stub (see Changes table below) |
|       |             |

---

## Dataset Preparation

<!--
  BCI images are stored as side-by-side JPEG pairs: left half = H&E, right half = IHC.
  They must be split into separate A (H&E) and B (IHC) folders before use.
-->

- **Dataset used:** BCI / MIST-HER2 / both
- **Format expected by model (after refactor):** Separate A/B folders — `trainA/`, `trainB/`, `testA/`, `testB/`
- **Conversion applied:**
  ```bash
  # BCI images are 1024x512 side-by-side (left=H&E, right=IHC).
  # Split them with:
  python - << 'EOF'
  import os, glob
  from PIL import Image

  for split in ['train', 'test']:
      src = f'~/thomas/internship-models/dataset/BCI/{split}/'
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
  dataset/BCI/
    train/
      A/   <-- H&E source images (1024x512 each half → 512x512)
      B/   <-- IHC target images
    test/
      A/
      B/
  ```
- **Number of images used for smoke test (train / test):**

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
  cd ~/thomas/internship-models/autopsy-VS
  python test_G.py \
    --data_dir ~/thomas/internship-models/dataset/BCI/test/B/ \
    --checkpoint pretrained/model_G_iter=87700.h5 \
    --output_dir ~/thomas/internship-models/autopsy-VS-outputs/ \
    --image_size 256 \
    --gpu 0
  ```
- **Output folder:**
- **Number of output images produced:**
- **Output image dimensions:**
- **Visual check result:**
- **Time to run (approx):**
- **Errors or warnings during inference:**

---

## Training Smoke Test

- **Script / command run:**
  ```bash
  conda activate tf2_env
  cd ~/thomas/internship-models/autopsy-VS
  python train_stage2_seperate_train_by_iters.py \
    --model_dir ~/thomas/internship-models/autopsy-VS-train/ \
    --train_data ~/thomas/internship-models/dataset/BCI/train/B/*.png \
    --val_data ~/thomas/internship-models/dataset/BCI/test/B/*.png \
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
| `ops.py` | Added `import os`; replaced `model_path + 'code'` (×4) with `os.path.join(model_path, 'code')` | String concatenation without separator gives wrong paths when model_path lacks trailing slash |
| `train_stage2_seperate_train_by_iters.py` | Added `import argparse` and `parse_args()` function | No CLI args existed; all paths were hardcoded Windows drive letters |
| `train_stage2_seperate_train_by_iters.py` | Replaced hardcoded `tc.model_path = 'L:/...'` with `args.model_dir` | Hardcoded Windows path |
| `train_stage2_seperate_train_by_iters.py` | Replaced hardcoded `tc.image_path = 'L:/.../*.mat'` with `args.train_data` | Hardcoded Windows path |
| `train_stage2_seperate_train_by_iters.py` | Replaced hardcoded `vc.image_path = 'J:/.../*.mat'` with `args.val_data` | Hardcoded Windows path on different drive |
| `train_stage2_seperate_train_by_iters.py` | Replaced hardcoded `CUDA_VISIBLE_DEVICES = "1"` with `args.gpu`; moved after `parse_args()` | Hardcoded GPU index; env var must be set before TF import |
| `train_stage2_seperate_train_by_iters.py` | `is_mat` now set from `args.is_mat` (default False); added conditional `convert_inp_path_from_target` for A/B folder layout | `.mat` was hardcoded; BCI/MIST-HER2 use PNG in A/B folders |
| `train_stage2_seperate_train_by_iters.py` | `tf.io.gfile.mkdir(tc.model_path + '/output')` → `os.path.join(tc.model_path, 'output')` | Cross-platform path safety |
| `test_G.py` | Added `import argparse` and `parse_args()` function | No CLI args existed |
| `test_G.py` | Replaced hardcoded `L:\\...\\*.mat` image path with `os.path.join(args.data_dir, '*.{ext}')` | Hardcoded Windows backslash path |
| `test_G.py` | Replaced hardcoded `model_path`, `checkpoint_path`, `output_path` Windows strings with `args.checkpoint`, `args.output_dir` | Hardcoded Windows paths |
| `test_G.py` | Replaced `CUDA_VISIBLE_DEVICES = "0"` with `args.gpu`; moved after `parse_args()` | Hardcoded GPU index |
| `test_G.py` | `tc.image_size = 2048` → `args.image_size` (default 256) | Hardcoded 2048 (autopsy slides); BCI images are smaller |
| `test_G.py` | `is_mat` now from `args.is_mat`; added conditional `convert_inp_path_from_target` | Same as train script |
| `test_G.py` | `valid_image_path.split('\\')[-1]` → `os.path.splitext(os.path.basename(valid_image_path))[0] + '.png'` | Backslash split breaks on Linux |
| `test_G.py` | `valid_image_path.split('\\')[-3]` → `os.path.basename(os.path.dirname(os.path.dirname(...)))` | Backslash split breaks on Linux |
| `test_G.py` | `tf.concat([valid_x[j,:,:,0:2], valid_x[j,:,:,3:4]], ...)` → `valid_x[j]` | Channel index 3 does not exist for 2-channel input (visualization-only code) |
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
