#!/bin/bash

#SBATCH --job-name=multiple_banks_pipeline
#SBATCH --output=./mylogs/multiple_banks_pipeline_stdout.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=6:00:00

set -e

cd /work/ahattimare_umass_edu/SentAugment
export PYTHONPATH=/work/ahattimare_umass_edu/SentAugment:${PYTHONPATH}

source activate sent_augment

echo "Download MC4 para data for German. (~2h 20min)"
file_name="data/de10M/mc4_de10M_para.txt"
python src/generate_data.py --num_rows=10_000_000 --output $file_name --language=de --split_by=paragraph


echo "Split MC4 data file into multiple files. (~7min)"
file_name="data/de10M/mc4_de10M_para.txt"
target_template="data/de10M/mc4_de10M_para_split_XXX.txt"
count=10
python src/generate_data.py --split --source $file_name --target_template $target_template --count $count --rows_per_file 1000_000


echo "Compress all the individual files."
for (( i = 1; i <= count; i++ ))
do
  file_name="data/de10M/mc4_de10M_para_split_${i}.txt"
  echo "Compressing ${file_name} file."
  python src/compress_text.py --input $file_name
done


echo "Embed De bank sentences using sase. (~15-20min per file)"
for (( i = 1; i <= count; i++ ))
do
  input="data/de10M/mc4_de10M_para_split_${i}.txt"
  output="data/de10M/mc4_de10M_para_split_${i}_sase.pt"
  echo "Creating sase embedding for ${input} in ${output}"
  python src/sase.py --input $input --model data/sase.pth --spm_model data/sase.spm --batch_size 64 --cuda "True" --output $output
done


echo "Create Faiss index for all the bank files. (~20 min per file)"
for (( i = 1; i <= count; i++ ))
do
  emb="data/de10M/mc4_de10M_para_split_${i}_sase.pt"
  index_path="data/de10M/mc4_de10M_para_split_${i}_sase_faiss_index.idx"
  python src/index_creation/faiss_index.py --create_index --emb $emb --M 32 --index_path $index_path
done


echo "Search across all indices for a given query file. (~20min per file)"
input=data/nc_body_100k.txt
input_emb=data/nc_body_100k_sase.pt
K=3
pretty_print=False

for (( i = 1; i <= count; i++ ))
do
  bank="data/de10M/mc4_de10M_para_split_${i}.txt"
  index_path="data/de10M/mc4_de10M_para_split_${i}_sase_faiss_index.idx"

  output="data/nc_body_100k/nc_body_100k_de_split_${i}_sase_knn_faiss.csv"
  offset=$(((i-1) * 1000000))

  echo "input arguments: ${bank} ${index_path} ${output} ${offset}"

  python src/index_creation/faiss_index.py --search_index --input $input --input_emb $input_emb --bank $bank \
  --index_path $index_path --K $K --output $output --pretty_print $pretty_print --offset $offset --multiple_banks_output
done


echo "Combine results obtained from multiple index searches."
file_name_template=data/nc_body_100k/nc_body_100k_de_split_XXX_sase_knn_faiss.csv
num_files=10

echo "K=3 (~17 min)"
K=3
output_file=data/nc_body_100k/nc_body_100k_de_combined_K3_sase_knn_faiss.csv
python src/combine_bank_retrieval.py --file_name_template $file_name_template --num_files $num_files \
--output_file $output_file --K $K

echo "K=10 (~17 min)"
K=10
output_file=data/nc_body_100k/nc_body_100k_de_combined_K10_sase_knn_faiss.csv
python src/combine_bank_retrieval.py --file_name_template $file_name_template --num_files $num_files \
--output_file $output_file --K $K


echo "end of script!"
