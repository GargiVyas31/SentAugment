#!/bin/bash

#SBATCH --job-name=faiss_query
#SBATCH --output=./mylogs/faiss_query_stdout.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=node92

set -e

cd /home/ahattimare_umass_edu/scratch/amit/SentAugment
source activate sent_augment

index=data/100M_1GPU_16GB.faiss.idx  # FAISS index path
input=data/sentence.txt.pt  # embeddings of input sentences
bank=data/keys.txt  # text file with all the data (the compressed file keys.ref.bin64 should also be present in the same folder)
K=5  # number of sentences to retrieve per query
NPROBE=1024 # number of probes for querying the index

python src/faiss_retrieve.py --input $input --bank $bank --index $index --K $K --nprobe $NPROBE --gpu "True" > nn.txt &
