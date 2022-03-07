#!/bin/bash

#SBATCH --job-name=faiss_query
#SBATCH --output=./mylogs/knn_faiss_cpu_stdout.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=node92

set -e

cd /home/ahattimare_umass_edu/scratch/amit/SentAugment

source activate sent_augment

## encode input sentences as sase embedding
input=data/sentence.txt  # input file containing a few (query) sentences
python src/sase.py --input $input --model data/sase.pth --spm_model data/sase.spm --batch_size 64 --cuda "True" --output $input.pt

index=data/100M_1GPU_16GB.faiss.idx  # FAISS index path
bank=data/keys.txt  # text file with all the data (the compressed file keys.ref.bin64 should also be present in the same folder)
K=3  # number of sentences to retrieve per query
NPROBE=1024 # number of probes for querying the index

python src/faiss_retrieve.py --input $input.pt --bank $bank --index $index --K $K --nprobe $NPROBE --gpu "False" > knn_faiss.txt
