# NeuroSphere: brain MRI to structured JSON

Code for the paper *MRI to Structured JSON: A Quality-Gated Deep Learning Pipeline for Compact Brain Morphometry Records* (OCIT 2026).

NeuroSphere converts a raw T1-weighted MRI file (NIfTI) into a small JSON file that is checked against a versioned schema (2.0.0). Each file contains:

- volumes for 29 brain structures (FreeSurfer label definitions), measured in the scan's own space;
- estimated intracranial volume (eTIV), eTIV-normalised volumes and left–right asymmetry;
- tissue volumes and within-tissue intensity summaries;
- image-quality metrics (MRIQC definitions) with a pass/warn/fail gate;
- normative z-scores (age, sex, site and eTIV model fitted on training subjects only);
- provenance: software versions, code revision, model checksum and per-stage run time.

Missing values are written as `null` with an explanation. Five output formats are available: full JSON, compact JSON, CSV, plain-text sentences grouped by feature, and JSON with meaningless keys.

## Pipeline

```
raw T1 NIfTI -> RAS reorientation -> N4 bias correction -> HD-BET brain mask -> white-matter intensity scaling
  -> ANTs affine registration to the training template -> 3D U-Net (29 structures) -> inverse affine to native space
  -> volumes, eTIV, tissue, intensity, quality metrics -> quality gate and normative model -> JSON record
```

Segmentation uses a 3D U-Net (about 5.6 M parameters). It was trained with FreeSurfer labels from the TransMorph IXI release, then adapted to images produced by this pipeline's preprocessing.

## Results (IXI, raw scans of 115 held-out test subjects)

| | Value |
|---|---|
| Mean Dice vs. FreeSurfer labels of the same subject | 0.878 (95% CI 0.875–0.880) |
| Mean absolute volume bias against FreeSurfer | 2.8% |
| ICC(2,1) with FastSurfer volumes | ≥ 0.94 for 23 of 29 structures |
| Processing time per scan (RTX 4060 laptop GPU) | median 25.6 s |
| Full JSON record | median 15 KB (raw scan: 19.7 MB) |

Per-structure numbers, the known limitations and all other results are in the paper and in `results/`.

## Installation

Developed and tested on Windows with Python 3.12 and an NVIDIA GPU (CUDA 12.4 build of PyTorch).

```
python -m venv .venv
.venv\Scripts\python -m pip install -c constraints.txt -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

On Linux use `.venv/bin/python`. `requirements-lock.txt` lists the exact environment used for the paper.

## Model files

The model weights and the registration template are attached to this repository's release. Download them and place them as follows:

| File | Place it in | SHA-256 |
|---|---|---|
| `unet3d_ft_best.pth` (adopted model) | `checkpoints/` | `578a1d68328d5231fa0551b7fc385ea4232af335391a42327a29a06be9a65b05` |
| `train_mean_T1.nii.gz` (template: average of the 403 training scans) | `data/atlas/` | `464be8d17c28daaffdb81f2dd8d14157d06324dc704301209d8aebbd69b816d1` |

## Data

- **IXI T1 scans** (CC BY-SA 3.0): https://brain-development.org/ixi-dataset/. Put the raw T1 files in `ixi-t1/` and the demographics spreadsheet `IXI.xls` in `data/raw/`.
- **TransMorph IXI release** (FreeSurfer labels, training data): https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration. Put the `Train/`, `Val/` and `Test/` folders in `IXI_data/`.
- The subject splits (403 training, 58 validation, 120 test) and the link between TransMorph indices and IXI identifiers are in `data/`.

## How the paper's numbers were produced

Every number in the paper is read from the files in `results/`, which these scripts write.

| Step | Script | Output |
|---|---|---|
| Preprocess and segment all 581 raw scans | `scripts/run_pipeline_all.py` | `work/pipeline_v2/` |
| In-domain Dice of a checkpoint | `python train_unet3d.py --eval-only --ckpt-name <file> --split Test --out-dir <dir>` | `results/e1_unet_in_domain/` |
| Sensitivity to head position | `experiments/e1b_position_robustness.py` | `results/e1b_position_robustness/` |
| Fine-tuning pairs (label transfer) | `scripts/make_finetune_data.py` | `work/finetune_data/` |
| Re-segment the cached inputs with a checkpoint | `python scripts/resegment_cached.py --ckpt unet3d_ft_best.pth --split all --out work/pipeline_ft` | `work/pipeline_ft/` |
| Agreement with transferred FreeSurfer labels | `python experiments/e2fs_transferred_fs.py --split Test --labels-dir work/pipeline_ft --out-name ft_test` | `results/e2_fs/` |
| FastSurfer on the raw test scans (Docker) | `scripts/run_fastsurfer.py --test-split` | `work/fastsurfer/` |
| Agreement with FastSurfer | `python experiments/e2_fastsurfer.py --labels-dir work/pipeline_ft --cache-dir work/pipeline_v2 --out-name ft_test` | `results/e2_fastsurfer/` |
| Fine-tuning acceptance rule (validation only) | `python experiments/finetune_acceptance.py --ft results/e2_fs/ft_val --v2 results/e2_fs/v2_val` | `results/phase3/finetune_acceptance.json` |
| JSON records, quality gate, normative model, record sizes | `python scripts/build_json_all.py --pipeline-dir work/pipeline_ft` | `work/json/`, `results/phase3/` |

The serialisation profiles (`full`, `compact`, `csv`, `text`, `obfuscated`) are implemented in `neurosphere/serialize.py`, and `build_json_all.py` reports their sizes. Per-epoch validation scores for R2 and the fine-tuned model are in `results/training/`. A one-command interface for new scans is not part of this release; the scripts above process the IXI cohort.

## Limitations

- Trained and evaluated on healthy adults from one dataset (IXI, three scanners); behaviour on other scanners, children or pathology is untested.
- The left accumbens remains over-estimated relative to FreeSurfer (+18% on test), and the 3rd and 4th ventricles by about 6%.
- Cerebral cortex and white matter volumes depend on the reference: they match FreeSurfer within 1% but differ from FastSurfer by +20% and −6%.
- The quality gate compares each scan with a single reference built from all three IXI sites, so most of its flags reflect scanner differences rather than poor images; per-site references are planned.
- For research use only; not for diagnosis.

## Licence

The code is released under the MIT licence (see `LICENSE`). The model weights and the registration template attached to the release are derived from the IXI dataset and are shared under CC BY-SA 3.0, the licence of IXI.

## Citation

Rishit V N S, P Manas Siddhardha, Rohith Karning, Kushal G and Mahitha G, "MRI to Structured JSON: A Quality-Gated Deep Learning Pipeline for Compact Brain Morphometry Records," in *Proc. OITS International Conference on Information Technology (OCIT)*, 2026.
