# sMRI on NVIDIA Holoscan

A minimal port of the [sMRI structural-MRI preprocessing pipeline](https://github.com/mhabbasiit/sMRI)
to the [NVIDIA Holoscan SDK](https://docs.nvidia.com/holoscan/sdk-user-guide/).

**Author:** Mohammad H. Abbasi — Stanford Translational AI Lab (STAI)

Each preprocessing stage of the sMRI pipeline is expressed as a Holoscan
**Operator**, and the stages are wired into an application **graph**:

```
SkullStrip → Registration → PostProcess → QC → Metadata
```

Operators pass a lightweight context dict (subject id, modality, file paths)
along the graph edges; the NIfTI volumes themselves stay on disk. This is the
same operator-graph model Holoscan uses for real-time medical-device
pipelines, applied here to an offline structural-MRI workflow.

The operators call the **real** sMRI functions:

| Stage | sMRI function |
|-------|---------------|
| SkullStrip   | `skullstrip.extract_brain_with_synthstrip` (SynthStrip container) + `perform_quality_check` QC montage |
| Registration | `reg.register_image` — two-stage rigid + affine (SimpleITK) + warped-over-MNI overlay |
| PostProcess  | `warp_mask_to_mni` → `n4_bias_correction` → `zscore_normalization_masked` → `crop_with_mask` + `create_visualization` montage |
| QC           | `structural_qc.calculate_dice_coefficient` (registration Dice vs MNI) |
| Metadata     | per-subject artifact + visualization manifest CSV |

Each stage emits a PNG QC visualization (skull-strip montage, registration
overlay, and final multi-slice montage).

> `reg.py` runs a batch pipeline at import time and is not importable as a
> library, so its `register_image()` is extracted verbatim into `reg_lib.py`.

## Why Holoscan

- **Composable graph** — add/remove/reorder stages without touching the others.
- **Scales to streaming** — the same operators can be driven by a scheduler for
  multi-subject or near-real-time throughput.
- **GPU-native** — operators can exchange `holoscan.Tensor` data and run CUDA
  kernels (e.g. for registration / normalization) without host round-trips.

## Prerequisites

- Python 3.10+
- [Holoscan SDK](https://pypi.org/project/holoscan/) (`pip install holoscan`)
- Singularity/Apptainer **or** Docker (for the SynthStrip container)
- The sMRI repo cloned next to this app (see Setup)

## Setup

```bash
# 1. clone this repo, then clone the sMRI pipeline beside the app
git clone https://github.com/mhabbasiit/sMRI smri_src

# 2. create an environment and install deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. pull the SynthStrip container once and point config at it
singularity pull synthstrip.sif docker://freesurfer/synthstrip:latest
#   then edit smri_src/config.py:
#     SYNTHSTRIP_SIF_PATH = "/abs/path/to/synthstrip.sif"
#     OUTPUT_DIR          = "/abs/path/to/out"
```

## Run

```bash
./run.sh \
  --input /path/to/sub_t1w.nii.gz \
  --subject SUB01 \
  --modality T1w \
  --outdir ./out
```

`run.sh` sets the recommended Holoscan stack size and a local TemplateFlow
cache, then launches the graph. To run directly:

```bash
ulimit -s 32768
.venv/bin/python smri_holoscan_app.py --input ... --subject SUB01 --outdir ./out
```

Without the sMRI toolchain installed, the app runs in **dry-run mode** and just
prints the graph execution order — useful for inspecting the pipeline shape.

## Example output

Real run on one subject (T1w, CPU, ~80 s end-to-end), registration Dice
**0.94** against the MNI brain mask (good alignment to template space).

## Outputs

The layout follows the sMRI `STAGE_ROOTS` + `STRUCTURE` convention
(`<stage>/<subject>/anat/...`):

```
<outdir>/
├── skullstrip/<subject>/anat/
│   ├── <stem>_brain.nii.gz, <stem>_brain_mask.nii.gz   # brain + binary mask
│   └── <stem>_desc-qc.png                              # skull-strip QC montage
├── skullstrip/qc_summary.csv                           # per-subject QC table
├── registration/<subject>/anat/
│   ├── <stem>_mni_warped.nii.gz                        # brain in MNI space
│   ├── other/<stem>_rigid.mat, <stem>_affine.mat       # transforms
│   ├── <stem>_reg_overlay.png                          # registration QC overlay
│   ├── T1w_mni_mask.nii.gz                             # mask warped to MNI
│   ├── T1w_mni_warped_n4.nii.gz                        # N4 bias-corrected
│   ├── T1w_mni_zscore_fixed.nii.gz                     # masked z-scored
│   ├── T1w_mni_zscore_fixed_cropped.nii.gz             # FINAL preprocessed volume
│   └── multislice_visualization_<subject>.png          # final multi-slice montage
├── metadata/<subject>_<modality>_metadata.csv          # artifact + QC manifest
└── logs/
```

Each stage drops its QC visualization into its own stage folder (skull-strip
montage, registration overlay, final multi-slice montage).

## Troubleshooting

- **`undefined symbol: cudaGetDriverEntryPointByVersion`** — the system CUDA
  runtime is older than 12.5. Point at a newer one:
  `export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH`
- **`stack size below recommended minimum`** — `ulimit -s 32768` (run.sh does this).
- **No GPU present** — Holoscan falls back to CPU-only execution automatically.

## License

MIT. Built on the sMRI pipeline (Abbasi & Adeli, 2025, Stanford STAI Lab).
