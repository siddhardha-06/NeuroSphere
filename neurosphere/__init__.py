"""NeuroSphere v2: compact, verifiable JSON records of T1-weighted brain MRI.

Modules: io (RAS loading), preprocess (N4, HD-BET mask, intensity scaling, affine registration),
segment (3D U-Net and native-space volumes), transfer (FreeSurfer label transfer), features (volumes,
eTIV, tissue and intensity), qc (image-quality metrics and gate), normative (cross-fitted model),
jsonio (schema 2.0.0 records), serialize (serialisation profiles), splits, metrics, provenance.
"""

__version__ = "2.0.0.dev1"
