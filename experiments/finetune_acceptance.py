r"""Apply the fine-tuning acceptance rule ("Fine-tuning plan and acceptance rule",
committed 87b3319 before training). Val only.

Adopt the fine-tuned model only if, on Val (E2-FS, raw scans, native space), all hold:
  1. a checkpoint passed the min-class rule (every class Val Dice >= 0.5 at the saved epoch);
  2. mean raw-scan Dice >= 0.8622 - 0.01 = 0.8522 (0.8622 = v2, results/e2_fs/v2_val);
  3. fewer than 23 structures with |volume bias| > 5 % (23 = v2);
  4. |bias| decreases for each of L/R accumbens, pallidum, putamen.
Otherwise keep v2 and report the bias as a limitation. The v2 numbers in 2-4 are the pre-registered ones (v2_val).
Diagnostics (not part of the rule): the same comparison against v2 re-segmented from the cached uint8 inputs
(--v2-reseg), which shows how much of any change comes from the cache quantisation rather than fine-tuning; and the
in-domain Val bias of the fine-tuned model (--e1-ft).

Writes results/phase3/finetune_acceptance.json.
Usage: python experiments/finetune_acceptance.py --ft results/e2_fs/ft_val --v2 results/e2_fs/v2_val
       [--v2-reseg results/e2_fs/v2_reseg_val] [--e1-ft results/e1_unet_in_domain/unet3d_ft_best_val] [--ckpt unet3d_ft_best.pth]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.finetune_decision import in_domain_bias  # noqa: E402
from neurosphere.provenance import provenance  # noqa: E402

V2_DICE, DICE_MARGIN, V2_N_OVER, BIAS_LIMIT = 0.8622, 0.01, 23, 5.0
KEY = ["left_accumbens", "right_accumbens", "left_pallidum", "right_pallidum", "left_putamen", "right_putamen"]


def summary(m):
    b = {k: v["mean_bias_pct"] for k, v in m["per_structure"].items()}
    return {"dice": m["mean_fg_dice_over_subjects"], "dice_ci95": m["mean_fg_dice_ci95"], "bias": b,
            "n_over": sum(abs(x) > BIAS_LIMIT for x in b.values()), "mean_abs_bias": float(np.mean([abs(x) for x in b.values()]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True); ap.add_argument("--v2", required=True)
    ap.add_argument("--v2-reseg"); ap.add_argument("--e1-ft"); ap.add_argument("--ckpt", default="unet3d_ft_best.pth")
    a = ap.parse_args()
    load = lambda d: json.loads((ROOT / d / "metrics.json").read_text())  # noqa: E731
    ft_m, v2_m = load(a.ft), load(a.v2)
    assert ft_m["split"] == v2_m["split"] == "Val"
    ft, v2 = summary(ft_m), summary(v2_m)
    assert abs(v2["dice"] - V2_DICE) < 5e-5 and v2["n_over"] == V2_N_OVER, "v2 baseline differs from the pre-registered numbers"
    ck = torch.load(ROOT / "checkpoints" / a.ckpt, map_location="cpu", weights_only=False)
    hist = json.loads((ROOT / "checkpoints" / (Path(a.ckpt).stem.replace("_best", "") + "_history.json")).read_text())
    saved = [h for h in hist if h["epoch"] == ck["epoch"] + 1][0]
    c1 = saved["val_min_class_dice"] >= ck["recipe"]["min_class_dice"]
    c2 = ft["dice"] >= V2_DICE - DICE_MARGIN
    c3 = ft["n_over"] < V2_N_OVER
    key = {k: {"v2": round(v2["bias"][k], 2), "ft": round(ft["bias"][k], 2), "reduced": abs(ft["bias"][k]) < abs(v2["bias"][k])} for k in KEY}
    c4 = all(v["reduced"] for v in key.values())
    out = {"rule": __doc__.split("Diagnostics")[0].strip(),
           "checkpoint": {"file": a.ckpt, "epoch_index": ck["epoch"], "recipe": ck["recipe"], "ft_e2fs_checkpoint": ft_m["checkpoint"]},
           "c1_min_class": {"met": c1, "saved_epoch_min_class_dice": saved["val_min_class_dice"], "worst_label": saved["worst_label"]},
           "c2_dice": {"met": c2, "ft": ft["dice"], "ft_ci95": ft["dice_ci95"], "threshold": V2_DICE - DICE_MARGIN, "v2": v2["dice"]},
           "c3_n_structures_over_5pct": {"met": c3, "ft": ft["n_over"], "v2": v2["n_over"]},
           "c4_key_structures": {"met": c4, "per_structure": key},
           "decision": "adopt_finetuned" if (c1 and c2 and c3 and c4) else "keep_v2 (report the bias as a limitation)",
           "mean_abs_bias_pct": {"ft": ft["mean_abs_bias"], "v2": v2["mean_abs_bias"]},
           "bias_pct_ft": {k: round(v, 2) for k, v in ft["bias"].items()}, "bias_pct_v2": {k: round(v, 2) for k, v in v2["bias"].items()}}
    if a.v2_reseg:
        r = summary(load(a.v2_reseg))
        out["diagnostic_v2_reseg_uint8"] = {"dice": r["dice"], "n_over": r["n_over"], "mean_abs_bias": r["mean_abs_bias"],
                                            "key_bias": {k: round(r["bias"][k], 2) for k in KEY}}
    if a.e1_ft:
        e1 = load(a.e1_ft); ib = in_domain_bias(ROOT / a.e1_ft)
        out["diagnostic_in_domain_val_ft"] = {"dice": e1["mean_fg_dice_over_subjects"], "bias_pct": ib,
                                              "n_over": sum(abs(v) > BIAS_LIMIT for v in ib.values()),
                                              "mean_abs_bias": float(np.mean([abs(v) for v in ib.values()]))}
    out["provenance"] = provenance()
    d = ROOT / "results" / "phase3"; d.mkdir(parents=True, exist_ok=True)
    (d / "finetune_acceptance.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("c1_min_class", "c2_dice", "c3_n_structures_over_5pct", "c4_key_structures", "decision",
                                          "mean_abs_bias_pct")}, indent=1))
    for k in ("diagnostic_v2_reseg_uint8", "diagnostic_in_domain_val_ft"):
        if k in out:
            print(k, json.dumps({x: y for x, y in out[k].items() if x != "bias_pct"}))


if __name__ == "__main__":
    main()
