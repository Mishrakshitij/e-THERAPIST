#!/usr/bin/env bash
set -euo pipefail

python -m e_therapist inspect
python -m e_therapist train-classifiers --config configs/classifiers.yaml
python -m e_therapist train-sft --config configs/sft.yaml
python -m e_therapist train-nlpo --config configs/nlpo.yaml
python -m e_therapist evaluate --config configs/nlpo.yaml \
    --checkpoint runs/nlpo/last --classifiers runs/classifiers --bertscore
