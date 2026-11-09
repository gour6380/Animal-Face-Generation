# Data and pretrained-model notices

The project Git ignore rules exclude full datasets, downloaded pretrained-weight
files and trained checkpoints. The compact `reports/` bundle contains measured
results and selected generated-image figures from `run_001`; it does not contain
the AFHQ dataset or model weights. The public notebook is output-free. The
original executed notebook is retained locally and excluded from Git because it
may embed dataset examples and reconstruction comparisons in its outputs.

AFHQ v2 is provided by the StarGAN v2 authors for noncommercial use under
CC BY-NC 4.0. Download provenance and the upstream license are linked in the
README. Keep the authors' attribution and license terms with any permitted
redistribution of downloaded data; this project's source does not grant rights
to dataset assets.

The reconstruction objective uses pretrained torchvision VGG-16 features and
LPIPS v0.1 calibration weights. Distribution metrics use the torch-fidelity
Inception-v3-compatible extractor. Their source projects, documentation and
applicable licenses are linked in the README and installed distributions.
Explicit acquisition records the exact files and SHA-256 identities used.

The launcher follows the isolated-worker design used in the owner's earlier
project. This archive includes its own implementation and has no dependency on
that project's files, environment or artifacts. Third-party dependency licenses
remain those of their respective distributions.
