#!/bin/bash

#SBATCH --job-name=mc4_mdpr_knn
#SBATCH --output=./mylogs/mc4_mdpr_knn_stdout.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --exclude=node92,node41,node42

set -e

cd /work/ahattimare_umass_edu/SentAugment

source activate sent_augment

# To change huggingface default cache location.
#export HF_DATASETS_CACHE="/home/ahattimare_umass_edu/scratch"

echo "Download MC4 data for German."
#file_name=data/mc4_de100k.txt
#python src/generate_data.py --num_rows=100000 --output $file_name --language=de --split_by=sentence
#python src/compress_text.py --input $file_name

echo "Embed bank sentences."
#input=data/mc4_de100k.txt
#output=data/mc4_de100k_mdpr_ques.pt
#python src/mdpr.py --input $input --output $output --batch_size=256 --cuda "True" --load_save "True" --model_type="question"

echo "Embed query sentences."
#input=data/10k_examples_retriever.txt
#output=data/titles_10k_mdpr_pass.pt
#python src/mdpr.py --input $input --output $output --batch_size=256 --cuda "True" --load_save "True" --model_type="passage"

echo "Perform KNN search for De."

#input_arr=("data/titles.txt" "data/titles.txt" "data/titles.txt" "data/titles.txt")
#input_emb_arr=("data/titles_1k_mdpr_pass.pt" "data/titles_1k_mdpr_pass.pt" "data/titles_1k_mdpr_ques.pt" "data/titles_1k_mdpr_ques.pt")
#K=3
#
##bank_arr=("data/mc4_fr100k.txt" "data/mc4_fr100k.txt" "data/mc4_fr100k.txt" "data/mc4_fr100k.txt")
##emb_arr=("data/mc4_fr100k_mdpr_pass.pt" "data/mc4_fr100k_mdpr_ques.pt" "data/mc4_fr100k_mdpr_pass.pt" "data/mc4_fr100k_mdpr_ques.pt")
##output_arr=("data/titles_1k_pass_fr100k_pass_mdpr.txt" "data/titles_1k_pass_fr100k_ques_mdpr.txt" "data/titles_1k_ques_fr100k_pass_mdpr.txt" "data/titles_1k_ques_fr100k_ques_mdpr.txt")
#
#bank_arr=("data/mc4_de100k.txt" "data/mc4_de100k.txt" "data/mc4_de100k.txt" "data/mc4_de100k.txt")
#emb_arr=("data/mc4_de100k_mdpr_pass.pt" "data/mc4_de100k_mdpr_ques.pt" "data/mc4_de100k_mdpr_pass.pt" "data/mc4_de100k_mdpr_ques.pt")
#output_arr=("data/titles_1k_pass_de100k_pass_mdpr.txt" "data/titles_1k_pass_de100k_ques_mdpr.txt" "data/titles_1k_ques_de100k_pass_mdpr.txt" "data/titles_1k_ques_de100k_ques_mdpr.txt")
#
#for index in ${!input_arr[*]}; do
#  input=${input_arr[$index]}
#  input_emb=${input_emb_arr[$index]}
#  bank=${bank_arr[$index]}
#  emb=${emb_arr[$index]}
#  output=${output_arr[$index]}
#
#  echo "Running for ${input} ${input_emb} ${bank} ${emb} ${output}"
#  python src/flat_retrieve.py --input $input --input_emb $input_emb --bank $bank --emb $emb --K $K --pretty_print True --output $output
#done

input=data/titles.txt
input_emb=data/titles_1k_mdpr_pass.pt
bank=data/mc4_fr100k.txt
emb=data/mc4_fr100k_mdpr_pass.pt
output=data/titles_1k_pass_fr100k_pass_mdpr_test.txt
K=3
python src/flat_retrieve.py --input $input --input_emb $input_emb --bank $bank --emb $emb --K $K --pretty_print True --output $output

exit 0
