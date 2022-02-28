#!/bin/bash

#SBATCH --job-name=embed_sentences
#SBATCH --output=./mylogs/embed_sentences_stdout.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --exclude=node92

set -e

cd /home/ahattimare_umass_edu/scratch/amit/SentAugment
source activate sent_augment

input=data/keys_small.txt  # input text file
output=data/keys_small.pt  # output pytorch file
python3 src/sase.py --input $input --model data/sase.pth --spm_model data/sase.spm --batch_size 64 --cuda "True" --output $output

echo "end of embedding sentences of common crawl!"
