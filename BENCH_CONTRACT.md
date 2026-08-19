# Comparison-method reproduction — shared contract

Every per-method agent MUST follow this so all results are comparable and the
context stays clean. Work happens on the remote GPU box over SSH.

## 0. Proxy (REQUIRED before any git/pip/download)
```
source /ytech_m2v4_hdd/lizhongyin/proxy_env.sh
```
GitHub/HuggingFace/PyTorch/Google-Drive are only reachable through this proxy.
The internal PyPI mirror (default in pip) works without proxy and is fast.

## 1. Standardized benchmark inputs (READ-ONLY)
For each task the test pairs are pre-materialized as 8-bit grayscale:
```
/ytech_m2v4_hdd/lizhongyin/fusion_bench/inputs/<task>/A/<stem>.png   # source A (gray = Y of color src)
/ytech_m2v4_hdd/lizhongyin/fusion_bench/inputs/<task>/B/<stem>.png   # source B (gray)
/ytech_m2v4_hdd/lizhongyin/fusion_bench/inputs/<task>/cbcr/<stem>.npy   # (2,H,W) Cb,Cr of color src (optional, for RGB viz)
/ytech_m2v4_hdd/lizhongyin/fusion_bench/inputs/<task>/colorA/<stem>.png # original RGB color source (optional)
```
tasks (and #pairs):  `gfp_pc` (30) · `irvis` (50, MSRS subset) · `medical` (48, Harvard)

Semantics: A = (color/functional) source, B = (gray/structure) source.
- gfp_pc: A=GFP fluorescence(Y), B=phase-contrast
- irvis : A=visible(Y), B=infrared
- medical: A=PET/SPECT(Y), B=MRI
If a method has a fixed IR/VIS role, map **B→IR slot, A→VIS slot** consistently.

## 2. Output (what you must produce)
One fused PNG per stem (grayscale OR RGB — metrics convert to gray) into:
```
/ytech_m2v4_hdd/lizhongyin/fusion_bench/fused/<Method>/<task>/<stem>.png
```
File stem must equal the input stem. Cover ALL 3 tasks.

## 3. Score it (shared evaluator — do NOT write your own metrics)
```
/ytech_m2v4_hdd/lizhongyin/venv/bin/python \
  /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper/bench/eval_method.py \
  --task <task> --name <Method> --fused-dir /ytech_m2v4_hdd/lizhongyin/fusion_bench/fused/<Method>/<task>
```
Run for each of the 3 tasks. It writes per_image.csv, means.csv and appends to
`fusion_bench/reports/<task>/leaderboard.csv`, and prints the metric means.
The evaluator uses the shared `metrics/` package (EN/MI/SD/SF/AG/SSIM/MS_SSIM/
Qabf/VIF + SCD/Nabf/CC + FuncCorr/FuncSal). Always use the base venv python
`/ytech_m2v4_hdd/lizhongyin/venv/bin/python` for the evaluator (it has the deps).

## 4. Environment isolation (REQUIRED)
- Create a dedicated venv at `/ytech_m2v4_hdd/lizhongyin/venv/<method>` (lowercase).
- Never modify system python or the shared base venv.
- `python -m venv` using `/opt/conda/bin/python3.11` (or python3.8 if the method
  needs old torch). pip installs default to the internal mirror (fast); use the
  proxy env for github/HF/pytorch wheels.
- Record the exact create+install commands in your EXP doc.

## 5. GPU discipline
- Use ONLY your assigned GPU: `export CUDA_VISIBLE_DEVICES=<n>` (single digit).
- Inference is light; if you must TRAIN, keep it on your one GPU.
- Check `nvidia-smi` before heavy work; never use a GPU with >1GB used by others.

## 6. Repos & weights
- Clone original repo into `/ytech_m2v4_hdd/lizhongyin/code/ref/<Method>` (skip if
  already present). Prefer the authors' pretrained weights (download via proxy:
  GitHub releases / HF / Google-Drive with gdown). If no weights and training is
  quick (<~1h on H800), train from the LOCAL data; otherwise note it and skip
  training, using whatever pretrained exists.
- Training data lives at: MSRS `/ytech_m2v4_hdd/lizhongyin/data/MSRS/train/{vi,ir}`,
  Harvard `/ytech_m2v4_hdd/lizhongyin/data/Harvard-Medical/train/{func,mri}`,
  GFP-PC `/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper/source images/GFP-PC`.

## 7. Deliverable doc (REQUIRED)
Write `/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md/EXP-CMP-<NN>-<method>.md`
following the EXP template (motivation, repo+weights provenance, env setup cmds,
how fusion was run per task, the 3 metric tables, problems hit, conclusion).
Do NOT push — the orchestrator batches commits/pushes.

## 8. Report back (your final message = structured summary)
Return: method name, repo+commit, weights source, venv path, per-task status
(done/failed + #images), the metric means per task (copy from evaluator output),
and any blockers. Keep it concise — raw logs stay on the remote.
```
