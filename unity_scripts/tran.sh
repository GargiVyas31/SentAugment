#!/bin/bash

#SBATCH --job-name=master_translation
#SBATCH --output=./mylogs/master_translation.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=3:00:00

set -e

cd /work/gvyas_umass_edu/gvyas/SentAugment
export PYTHONPATH=/work/gvyas_umass_edu/gvyas/Sentaugment:${PYTHONPATH}
source activate sent_augment


python src/translation.py
