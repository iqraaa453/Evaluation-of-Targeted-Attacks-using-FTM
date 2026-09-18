# Colab NIPS Run

Run these commands from a Colab runtime with GPU enabled.

## 1. Verify GPU

```bash
nvidia-smi
```

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

If `timm` model loading fails because of an old package version, upgrade it:

```bash
pip install -U timm
```

## 3. Verify Dataset

The folder should contain:

```text
data/images/      1000 NIPS PNG images
data/images.csv   1000-row NIPS metadata CSV
```

Quick check:

```bash
python - <<'PY'
import csv, glob
print("images:", len(glob.glob("data/images/*.png")))
with open("data/images.csv", newline="") as f:
    print("csv rows:", sum(1 for _ in csv.DictReader(f)))
PY
```

Both counts should be `1000`.

## 4. Run FTM on NIPS

```bash
python main.py \
  --device cuda:0 \
  --batch_size 20 \
  --model_name ResNet50 \
  --save_dir ./exp/ResNet50/ftm_nips \
  --eval
```

For a smaller test run before the full job:

```bash
python main.py \
  --device cuda:0 \
  --batch_size 4 \
  --model_name ResNet50 \
  --save_dir ./exp/ResNet50/debug_nips \
  --eval \
  --debug
```

## 5. Outputs To Save

Main adversarial images:

```text
exp/ResNet50/ftm_nips/adv_imgs/
```

Model-wise success results:

```text
exp/ResNet50/ftm_nips/eval_results.txt
exp/ResNet50/ftm_nips/results/model_wise_summary.csv
```

Per-image, per-model detailed results:

```text
exp/ResNet50/ftm_nips/results/results_summary.csv
```

Human-readable summary:

```text
exp/ResNet50/ftm_nips/results/summary.txt
```

Optional zip for download:

```bash
zip -r ftm_nips_results.zip exp/ResNet50/ftm_nips
```
