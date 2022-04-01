#!/bin/bash

#SBATCH --job-name=mc4_mdpr_knn
#SBATCH --output=./mylogs/mc4_mdpr_knn_stdout.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --exclude=node92,node41,node42

set -e

cd /home/ahattimare_umass_edu/scratch/amit/SentAugment

source activate sent_augment

export HF_DATASETS_CACHE="/home/ahattimare_umass_edu/scratch"

echo "Download MC4 data for German."
file_name=data/mc4_de100k.txt
python src/generate_data.py --num_rows=10 --output $file_name --language=de --split_by=sentence
python src/compress_text.py --input $file_name

#echo "Embed bank sentences."
#input=data/mc4_fr10k.txt
#output=data/mc4_fr10k.pt
#python src/mdpr.py --input $input --output $output --batch_size=256 --cuda "True" --load_save "True"

#echo "Embed query sentences."
#input=data/titles.txt
#python src/mdpr.py --input $input --output $input.pt --batch_size=256 --cuda "True" --load_save "True"
#
#echo "Perform KNN search."
#input=data/titles.txt
#bank=data/mc4_fr10000.txt
#emb=data/mc4_fr10000.pt
#output=data/titles_nn.txt
#K=3
#python src/flat_retrieve.py --input $input --bank $bank --emb $emb --K $K --pretty_print True --output $output

exit 0
