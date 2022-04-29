#!/bin/bash

#SBATCH --job-name=test_script1
#SBATCH --output=./mylogs/test_script1_stdout.txt
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --time=1:00:00

set -e

cd /work/ahattimare_umass_edu/SentAugment
export PYTHONPATH=/work/ahattimare_umass_edu/SentAugment:${PYTHONPATH}

source activate sent_augment


#echo "Embed De bank sentences using sase. (~15-25min per file)"
#count=$1
#for (( i = count; i <= count; i++ ))
#do
#  input="data/de10M/mc4_de10M_para_split_${i}.txt"
#  output="data/de10M/mc4_de10M_para_split_${i}_sase.pt"
#  echo "Creating sase embedding for ${input} in ${output}"
#  python src/sase.py --input $input --model data/sase.pth --spm_model data/sase.spm --batch_size 64 --cuda "True" --output $output
#done


#echo "Create Faiss index for all the bank files. (~20min per file)"
#count=$1
#for (( i = count; i <= count; i++ ))
#do
#  emb="data/de10M/mc4_de10M_para_split_${i}_sase.pt"
#  index_path="data/de10M/mc4_de10M_para_split_${i}_sase_faiss_index.idx"
#  python src/index_creation/faiss_index.py --create_index --emb $emb --M 32 --index_path $index_path
#done


echo "Search across all indices for a given query file. (~20min per file)"
input=data/nc_body_10k.txt
input_emb=data/nc_body_10k_sase.pt
K=3
pretty_print=False
count=$1

for (( i = count; i <= count; i++ ))
do
  bank="data/de10M/mc4_de10M_para_split_${i}.txt"
  index_path="data/de10M/mc4_de10M_para_split_${i}_sase_faiss_index.idx"

  output="data/nc_body_100k/nc_body_10k_de_split_${i}_sase_knn_faiss.csv"
  offset=$(((i-1) * 1000000))

  echo "input arguments: ${bank} ${index_path} ${output} ${offset}"

  python src/index_creation/faiss_index.py --search_index --input $input --input_emb $input_emb --bank $bank \
  --index_path $index_path --K $K --output $output --pretty_print $pretty_print --offset $offset --multiple_banks_output
done
