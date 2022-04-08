#!/bin/bash

#SBATCH --job-name=mc4_sase_knn
#SBATCH --output=./mylogs/mc4_sase_knn_stdout.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00

set -e

cd /work/ahattimare_umass_edu/SentAugment

source activate sent_augment

echo "Embed Fr bank sentences using sase."
input=data/mc4_fr100k_para.txt
output=data/mc4_fr100k_para_sase.pt
python src/sase.py --input $input --model data/sase.pth --spm_model data/sase.spm --batch_size 64 --cuda "True" --output $output

echo "Embed De bank sentences using sase."
input=data/mc4_de100k_para.txt
output=data/mc4_de100k_para_sase.pt
python src/sase.py --input $input --model data/sase.pth --spm_model data/sase.spm --batch_size 64 --cuda "True" --output $output

echo "Embed query sentences using sase."
input=data/nc_body_1k.txt
output=data/nc_body_1k_sase.pt
python src/sase.py --input $input --model data/sase.pth --spm_model data/sase.spm --batch_size 64 --cuda "True" --output $output

echo "Perform KNN search for Fr."
input=data/nc_body_1k.txt
input_emb=data/nc_body_1k_sase.pt
bank=data/mc4_fr100k_para.txt
emb=data/mc4_fr100k_para_sase.pt
output=data/nc_body_1k_fr100k_para_sase.txt
K=3
python src/flat_retrieve.py --input $input --input_emb $input_emb --bank $bank --emb $emb --K $K --pretty_print True --output $output

echo "Perform KNN search for De."
input=data/nc_body_1k.txt
input_emb=data/nc_body_1k_sase.pt
bank=data/mc4_de100k_para.txt
emb=data/mc4_de100k_para_sase.pt
output=data/nc_body_1k_de100k_para_sase.txt
K=3
python src/flat_retrieve.py --input $input --input_emb $input_emb --bank $bank --emb $emb --K $K --pretty_print True --output $output
