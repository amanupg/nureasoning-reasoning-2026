# Shared dependencies: files in this folder that other agents rely on

**Do not delete, move or rename anything listed here without first checking with the dependent agent and
getting Aman's approval.** This applies on the Insomnia cluster
(`/path/to/cluster/scratch/nureasoning`) and on the Mac.

Set up 2026-10-18 at Aman's request (cluster dedupe between `nureasoning/` and `jepa-nureasoning/`).
Rule: a shared file stays in the folder of the agent that produced it; the other agent reads it from here.

## Depended on by the jepa planning thread (`jepa-nureasoning`, session "jepa-nureasoning-planning")

Confirmed by the jepa thread on 2026-10-18.

### On the Insomnia cluster
| path (relative to this folder) | used by (jepa-nureasoning/...) | what for |
|---|---|---|
| `.cache/hf/token` (via `HF_HOME=nureasoning/.cache/hf`) | `cluster/ingest_train.sbatch` | HF token for frame downloads; needed whenever they ingest again (mode 600; never copy it into argv) |

Nothing else here is read by the jepa thread on the cluster (their RL jobs use their own packed files).

### On the Mac (`<local checkout>`), read-only
| path | used by (jepa-nureasoning/...) | what for |
|---|---|---|
| `cache/train`, `cache/validation`, `cache/test` | `wajepa_run.clip_inputs`, `build_submission.py`, `rl_wajepa.py grade`, all scoring | clip metadata, ego states, annotations, test inputs |
| `src/` | `hint_check.py`, `decision_oracle.py` (import our loaders), `v71_val.py` (`src/make_submission.py`) | loaders and v7.1 helpers |
| `recon/devkit/` | `src/pipeline_*.sh` | devkit (scoring, submission format) |
| `submissions/v7.1_chooser-a15ft-valcv-ens5_a15ft-fallback_reasoning-cot.json` | `build_submission.py` (default `--answers`) | fallback answers |
| `artifacts/a15ft_index_train.json` | `src/djft.py` | training index |
| `runs_handoff/` | `src/ensemble_eval2.py` | planning handoff files (2026-10-01) |
| `vqa_train/` | `src/decision_oracle.py` | train questions (decision labels) |
| `artifacts/lon_decision_probs_{train,validation,test}.json` (also on the cluster) | planning decision-conditioned distance model (2026-10-21) | per-clip longitudinal decision probabilities (src/export_lon_decisions.py, from src/dec_head.py outputs) |

## Depended on by the AlpaSim PAI thread (`alpasim/`, session "alpasim-alpamayo-pai")

Confirmed by that thread on 2026-10-19 (also listed in alpasim/BORROWED.md). All read-only, on the cluster.

| path | what for |
|---|---|
| `ext/alpamayo1.5/` | Alpamayo 1.5 model code (on PYTHONPATH of their per-job env) |
| `ext/setup_a15.sh` | package list for their per-job env |
| `.cache/hf/hub/models--nvidia--Alpamayo-1.5-10B` | policy weights |
| `.cache/hf/hub/models--Qwen--Qwen3-VL-8B-Instruct`, `models--Qwen--Qwen3-VL-2B-Instruct` | VLM / processor |
| `src/` (`a15ft_data.build_sample`, `a15_infer.load_model`; keep signatures stable) | policy checks |
| `recon/devkit/`, `cache/validation/`, `img_a15/validation/` | policy checks (`img_a15/validation` is all that remains of img_a15 since 2026-10-20; keep it) |
| `env/bin/python` | one older script |

Incident 2026-10-19: `ext/a15env` (deleted that day as "unused", with Aman's approval) was the Python env of
their simulator jobs; job 13541844 failed during the deletion. They now build the env per job on node-local
disk. Lesson: before deleting anything here, grep ALL sibling folders under /path/to/cluster/scratch/
(alpasim, alpasim-jepa, jepa-nureasoning, wearable) for the path, not only this folder's code.

## Dedupe result (2026-10-18)

Scan `cluster/dedupe_scan.py` (files >= 1 MB, same size then same sha256; output
`logs/dedupe_2026-10-18.tsv` on the cluster): **no shared data files**. The only byte-identical files
(75 groups, 3.6 GB on the jepa side) are pip-installed libraries (CUDA, OpenCV, Python) inside each
agent's own Python environment. Not deduplicated: sharing environments would let one agent's package
upgrade break the other's jobs, for 3.6 GB.

The jepa thread's cluster folder holds nothing this folder depends on.
