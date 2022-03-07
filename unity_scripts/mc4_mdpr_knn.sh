#!/bin/bash

set -e

cd /home/ahattimare_umass_edu/scratch/amit/SentAugment

echo "Download MC4 data."
file_name=/scratch/hananthakris_umass_edu/haritha/mc4_fr1K.txt
python src/generate_data.py --num_rows=1000 --output $file_name --language=fr
python src/compress_text.py --input $file_name

echo "Embed bank sentences"
#input=data/keys_small.txt
#output=data/keys_small.pt
input=data/mc4_fr10.txt
output=data/mc4_fr10.pt
python src/mdpr.py --input $input --output $output --batch_size=4 --cuda "True"

echo "Embed query sentences"
input=data/sentence.txt
python src/mdpr.py --input $input --output $input.pt --batch_size=4 --cuda "True"

echo "Perform KNN search"
#input=data/sentence.txt
#bank=data/keys_small.txt
emb=data/keys_small.pt
bank=data/mc4_fr10.txt
emb=data/mc4_fr10.pt
K=2
python src/flat_retrieve.py --input $input.pt --bank $bank --emb $emb --K $K
