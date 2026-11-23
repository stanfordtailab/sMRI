"""
sMRI → Holoscan
================
Author : Mohammad H. Abbasi  (mabbasi [at] stanford.edu)
Lab    : Stanford Translational AI Lab (STAI), Stanford University
License: MIT

Wraps the structural-MRI preprocessing pipeline from
https://github.com/mhabbasiit/sMRI as an NVIDIA Holoscan application graph.

Each preprocessing stage of the sMRI repo becomes a Holoscan Operator. The
operators pass a small "context" dict (subject id, modality, file paths)
along the edges of the graph, so the heavy NIfTI volumes stay on disk while
only references move between nodes — exactly the operator-graph model
Holoscan uses for medical-device pipelines.

    SkullStrip → Registration → PostProcess → QC → Metadata

The stages call the *real* sMRI functions:
    skullstrip.extract_brain_with_synthstrip(...)        # SynthStrip container
    reg_lib.register_image(...)                          # verbatim from reg.py
    postprocess_*.{warp_mask_to_mni,n4_bias_correction,
                   zscore_normalization_masked,crop_with_mask}
    structural_qc.calculate_dice_coefficient(...)        # registration QC

(reg.py runs a batch pipeline at import time and is not importable as a
library, so its register_image() is extracted verbatim into reg_lib.py.)

Run:
    python smri_holoscan_app.py \
        --input /path/to/sub_t1w.nii.gz --subject SUB --outdir ./out

Holoscan SDK (Python):  pip install holoscan
Docs: https://docs.nvidia.com/holoscan/sdk-user-guide/
"""

import argparse
import os
import sys

# The sMRI modules live in the cloned repo next to this file.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "smri_src"))

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

# ---------------------------------------------------------------------------
# Import the real sMRI functions. Wrapped in try/except so the graph stays
# inspectable on a machine that only has the Holoscan SDK (e.g. an interview
# laptop) without the full neuroimaging toolchain.
# ---------------------------------------------------------------------------
try:
    from skullstrip import extract_brain_with_synthstrip
    from reg_lib import register_image
    from postprocess_mni_mask_zscore_crop import (
        warp_mask_to_mni,
        n4_bias_correction,
        zscore_normalization_masked,
        crop_with_mask,
        create_visualization,
    )
    from structural_qc import calculate_dice_coefficient
    from structure_resolver import make_path
    import config

    _SMRI_AVAILABLE = True
except ImportError as e:
    print(f"[warn] sMRI modules not importable ({e}); graph will run in dry-run mode.")
    _SMRI_AVAILABLE = False


def canon_anat(ctx, stage):
    """Per-subject 'anat' directory under a stage root, following the sMRI
    STAGE_ROOTS + STRUCTURE convention (e.g. <root>/<subject>/anat)."""
    root = ctx["stage_roots"][stage]
    if _SMRI_AVAILABLE:
        return make_path(config.STRUCTURE, root, ctx["subject"], ctx["session"], mkdirs=True)
    path = os.path.join(root, ctx["subject"], "anat")
    os.makedirs(path, exist_ok=True)
    return path


def save_registration_overlay(warped_path, template_path, out_png, subject=""):
    """Registration QC visualization: warped brain (red contour) over the MNI
    template at 3 mid-slices. reg.py has no built-in PNG, so this is added here.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    tpl = nib.load(template_path).get_fdata(dtype=np.float32)
    war = nib.load(warped_path).get_fdata(dtype=np.float32)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(f"Registration QC — {subject}: warped (red) over MNI template", fontsize=13)
    for ax, axis in zip(axes, range(3)):
        mid = tpl.shape[axis] // 2
        tsl = np.rot90(np.take(tpl, mid, axis=axis))
        wsl = np.rot90(np.take(war, mid, axis=axis))
        ax.imshow(tsl, cmap="gray")
        ax.contour(wsl > wsl.mean(), colors="red", linewidths=0.6)
        ax.set_title(["Sagittal", "Coronal", "Axial"][axis])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Registration] QC overlay -> {out_png}")


class SkullStripOp(Operator):
    """Step 1 — brain extraction (SynthStrip, via Singularity/Docker container)."""

    def setup(self, spec: OperatorSpec):
        spec.output("ctx")

    def compute(self, op_input, op_output, context):
        ctx = self.app_ctx
        # Canonical layout (matches sMRI STAGE_ROOTS + STRUCTURE):
        #   {skullstrip}/{subject}/anat/{stem}_brain.nii.gz  (+ _brain_mask, _desc-qc.png)
        ss_anat = canon_anat(ctx, "skullstrip")
        brain = os.path.join(ss_anat, f"{ctx['stem']}_brain.nii.gz")
        ctx["brain"] = brain
        ctx["brain_mask"] = brain.replace(".nii.gz", "") + "_mask.nii.gz"
        # qc_dir == the skullstrip anat dir, so the QC montage lands beside the brain
        # (the repo does the same: qc_path = out_path in process_subject()).
        ctx["qc_dir"] = ss_anat
        if _SMRI_AVAILABLE:
            # perform_quality_check appends a row to STAGE_ROOTS["skullstrip"]/qc_summary.csv
            os.makedirs(ctx["stage_roots"]["skullstrip"], exist_ok=True)
            os.makedirs(config.STAGE_ROOTS["skullstrip"], exist_ok=True)
            extract_brain_with_synthstrip(
                input_file=ctx["input"],
                output_file=brain,
                modality=ctx["modality"],
                qc_dir=ss_anat,
                qc_root=ctx["stage_roots"]["qc"],
                subject=ctx["subject"],
                session=ctx["session"],
            )
        print(f"[SkullStrip] {ctx['input']} -> {brain}")
        op_output.emit(ctx, "ctx")


class RegistrationOp(Operator):
    """Step 2 — two-stage (rigid + affine) registration to MNI space."""

    def setup(self, spec: OperatorSpec):
        spec.input("ctx")
        spec.output("ctx")

    def compute(self, op_input, op_output, context):
        ctx = op_input.receive("ctx")
        # Canonical layout: {registration}/{subject}/anat/{stem}_mni_warped.nii.gz
        reg_anat = canon_anat(ctx, "registration")
        ctx["reg_anat"] = reg_anat
        prefix = os.path.join(reg_anat, ctx["stem"])
        if _SMRI_AVAILABLE:
            res = register_image(
                fixed=ctx["mni_template"],
                moving=ctx["brain"],
                output_prefix=prefix,
                subject_id=ctx["subject"],
            )
            ctx["warped"] = res.outputs.warped_image
            ctx["rigid_mat"] = res.outputs.rigid_transform
            ctx["affine_mat"] = res.outputs.affine_transform
            if ctx.get("mni_template"):
                # Registration QC overlay, stored alongside the warped image.
                ctx["reg_overlay"] = os.path.join(reg_anat, f"{ctx['stem']}_reg_overlay.png")
                save_registration_overlay(
                    ctx["warped"], ctx["mni_template"], ctx["reg_overlay"], subject=ctx["subject"],
                )
        else:
            ctx["warped"] = f"{prefix}_mni_warped.nii.gz"
            ctx["rigid_mat"] = os.path.join(reg_anat, "other", f"{ctx['stem']}_rigid.mat")
            ctx["affine_mat"] = os.path.join(reg_anat, "other", f"{ctx['stem']}_affine.mat")
        print(f"[Registration] {ctx['brain']} -> {ctx['warped']}")
        op_output.emit(ctx, "ctx")


class PostProcessOp(Operator):
    """Step 3 — warp mask → N4 bias correction → masked z-score → tight crop.

    Mirrors the call order in postprocess_mni_mask_zscore_crop.main().
    """

    def setup(self, spec: OperatorSpec):
        spec.input("ctx")
        spec.output("ctx")

    def compute(self, op_input, op_output, context):
        ctx = op_input.receive("ctx")
        sub, mod = ctx["subject"], ctx["modality"]
        # Like postprocess_*.main(): outputs go INTO the registration anat dir,
        # with the repo's exact filenames so create_visualization() finds them.
        anat_dir = ctx["reg_anat"]
        mni_mask = os.path.join(anat_dir, f"{mod}_mni_mask.nii.gz")
        n4_out = os.path.join(anat_dir, f"{mod}_mni_warped_n4.nii.gz")
        z_fixed = os.path.join(anat_dir, f"{mod}_mni_zscore_fixed.nii.gz")
        cropped = os.path.join(anat_dir, f"{mod}_mni_zscore_fixed_cropped.nii.gz")

        if _SMRI_AVAILABLE:
            warp_mask_to_mni(ctx["brain_mask"], ctx["rigid_mat"], ctx["affine_mat"], ctx["warped"], mni_mask)
            n4_bias_correction(ctx["warped"], mni_mask, n4_out)
            zscore_normalization_masked(n4_out, mni_mask, z_fixed, keep_background_zero=True)
            crop_with_mask(z_fixed, mni_mask, cropped, margin=config.POSTPROCESS_CROP_MARGIN)
            # Multi-slice montage PNG (axial/sagittal/coronal) of the final volume.
            create_visualization(sub, anat_dir)

        ctx["mni_mask"] = mni_mask
        ctx["final"] = cropped
        ctx["montage"] = os.path.join(anat_dir, f"multislice_visualization_{sub}.png")
        print(f"[PostProcess] {ctx['warped']} -> {cropped}")
        op_output.emit(ctx, "ctx")


class QCOp(Operator):
    """Step 4 — registration QC: Dice between the warped brain mask and the
    MNI template brain mask (uses structural_qc.calculate_dice_coefficient)."""

    def setup(self, spec: OperatorSpec):
        spec.input("ctx")
        spec.output("ctx")

    def compute(self, op_input, op_output, context):
        ctx = op_input.receive("ctx")
        dice = None
        if _SMRI_AVAILABLE and ctx.get("mni_template_mask"):
            # image1/image2 only define the reference grid; the two binary masks
            # being compared are passed as mask1_path / mask2_path.
            dice = calculate_dice_coefficient(
                image1_path=ctx["mni_template"],
                image2_path=ctx["mni_template"],
                mask1_path=ctx["mni_mask"],
                mask2_path=ctx["mni_template_mask"],
            )
        ctx["dice"] = dice
        print(f"[QC] Dice(warped mask vs MNI template mask) = {dice}")
        op_output.emit(ctx, "ctx")


class MetadataOp(Operator):
    """Step 5 — write a per-subject manifest CSV of the produced artifacts.

    (The repo's full CSVMetadataGenerator is dataset-level and joins a
    participants.tsv; here we emit a single-subject artifact manifest.)
    """

    def setup(self, spec: OperatorSpec):
        spec.input("ctx")

    def compute(self, op_input, op_output, context):
        ctx = op_input.receive("ctx")
        sub, mod = ctx["subject"], ctx["modality"]
        meta_dir = os.path.join(ctx["outdir"], "metadata")
        os.makedirs(meta_dir, exist_ok=True)
        csv_path = os.path.join(meta_dir, f"{sub}_{mod}_metadata.csv")
        qc_png = os.path.join(ctx.get("qc_dir", ""),
                              os.path.basename(ctx["input"]).replace(".nii.gz", "_desc-qc.png"))
        rows = [
            ("subject", sub),
            ("modality", mod),
            ("input", ctx["input"]),
            ("brain", ctx.get("brain", "")),
            ("warped", ctx.get("warped", "")),
            ("final", ctx.get("final", "")),
            ("registration_dice", ctx.get("dice", "")),
            ("qc_skullstrip_png", qc_png),
            ("qc_registration_png", ctx.get("reg_overlay", "")),
            ("qc_montage_png", ctx.get("montage", "")),
        ]
        with open(csv_path, "w") as f:
            f.write("key,value\n")
            for k, v in rows:
                f.write(f"{k},{v}\n")
        print(f"[Metadata] subject={ctx['subject']} -> {csv_path}")


class SMRIApp(Application):
    """The full sMRI preprocessing graph."""

    def __init__(self, app_ctx, *args, **kwargs):
        self._ctx = app_ctx
        super().__init__(*args, **kwargs)

    def compose(self):
        # CountCondition(1): the source has no input port, so without it the
        # scheduler would call compute() forever. One subject => run once.
        src = SkullStripOp(self, CountCondition(self, 1), name="skullstrip")
        reg = RegistrationOp(self, name="registration")
        post = PostProcessOp(self, name="postprocess")
        qc = QCOp(self, name="qc")
        meta = MetadataOp(self, name="metadata")

        src.app_ctx = self._ctx
        for upstream, downstream in [(src, reg), (reg, post), (post, qc), (qc, meta)]:
            self.add_flow(upstream, downstream, {("ctx", "ctx")})


def fetch_mni(resolution=1):
    """Resolve the MNI152NLin2009cAsym T1w brain template + its brain mask."""
    from templateflow import api as tflow

    try:
        tmpl = str(tflow.get("MNI152NLin2009cAsym", resolution=resolution, suffix="T1w", desc="brain"))
    except Exception:
        tmpl = str(tflow.get("MNI152NLin2009cAsym", resolution=resolution, suffix="T1w"))
    try:
        mask = str(tflow.get("MNI152NLin2009cAsym", resolution=resolution, suffix="mask", desc="brain"))
    except Exception:
        mask = None
    return tmpl, mask


def main():
    parser = argparse.ArgumentParser(description="sMRI preprocessing on Holoscan")
    parser.add_argument("--input", required=True, help="native-space T1w/T2w NIfTI")
    parser.add_argument("--subject", default=None, help="subject id (default: from filename)")
    parser.add_argument("--modality", default="T1w", choices=["T1w", "T2w"])
    parser.add_argument("--outdir", default=None,
                        help="output root (default: config.OUTPUT_DIR, else ./out)")
    args = parser.parse_args()

    outdir = args.outdir or (config.OUTPUT_DIR if _SMRI_AVAILABLE else "./out")
    os.makedirs(outdir, exist_ok=True)
    subject = args.subject or os.path.basename(args.input).split("_")[0]
    base = os.path.basename(args.input)
    stem = base[:-7] if base.endswith(".nii.gz") else os.path.splitext(base)[0]

    # Stage roots mirror the sMRI STAGE_ROOTS layout, rooted at outdir.
    stage_roots = {
        "skullstrip": os.path.join(outdir, "skullstrip"),
        "registration": os.path.join(outdir, "registration"),
        "qc": os.path.join(outdir, "QC"),
    }
    ctx = {
        "subject": subject,
        "session": "",
        "stem": stem,
        "input": args.input,
        "modality": args.modality,
        "outdir": outdir,
        "stage_roots": stage_roots,
        "mni_template": None,
        "mni_template_mask": None,
    }
    if _SMRI_AVAILABLE:
        ctx["mni_template"], ctx["mni_template_mask"] = fetch_mni()
        print(f"[setup] outdir: {outdir}")
        print(f"[setup] MNI template: {ctx['mni_template']}")

    SMRIApp(ctx).run()
    print("[done] graph completed.")


if __name__ == "__main__":
    main()
