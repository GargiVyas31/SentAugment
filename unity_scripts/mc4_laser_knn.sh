#!/bin/bash

#SBATCH --job-name=mc4_laser_knn
#SBATCH --output=./mylogs/mc4_laser_knn_stdout.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00

set -e

cd /work/ahattimare_umass_edu/SentAugment

source activate sent_augment

echo "Embed Fr bank sentences using laser."
input=data/mc4_fr100k_para.txt
output=data/mc4_fr100k_para_laser.pt
python src/laser.py --input $input --output $output --input_lang "fr" --cuda "True"

echo "Embed De bank sentences using laser."
input=data/mc4_de100k_para.txt
output=data/mc4_de100k_para_laser.pt
python src/laser.py --input $input --output $output --input_lang "de" --cuda "True"

echo "Embed query sentences using laser."
input=data/nc_body_1k.txt
output=data/nc_body_1k_laser.pt
python src/laser.py --input $input --output $output --input_lang "en" --cuda "True"

echo "Perform KNN search for Fr."
input=data/nc_body_1k.txt
input_emb=data/nc_body_1k_laser.pt
bank=data/mc4_fr100k_para.txt
emb=data/mc4_fr100k_para_laser.pt
output=data/nc_body_1k_fr100k_para_laser.txt
K=3
python src/flat_retrieve.py --input $input --input_emb $input_emb --bank $bank --emb $emb --K $K --pretty_print True --output $output

echo "Perform KNN search for De."
input=data/nc_body_1k.txt
input_emb=data/nc_body_1k_laser.pt
bank=data/mc4_de100k_para.txt
emb=data/mc4_de100k_para_laser.pt
output=data/nc_body_1k_de100k_para_laser.txt
K=3
python src/flat_retrieve.py --input $input --input_emb $input_emb --bank $bank --emb $emb --K $K --pretty_print True --output $output
