#!/bin/bash

#SBATCH --job-name=faiss_indexing
#SBATCH --output=./mylogs/faiss_indexing_stdout.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00

set -e

cd /work/ahattimare_umass_edu/SentAugment
export PYTHONPATH=/work/ahattimare_umass_edu/SentAugment:${PYTHONPATH}

source activate sent_augment

# nc titles with mdpr passage.

#input=data/titles.txt
#input_emb=data/titles_mdpr_pass.pt
#bank=data/mc4_fr100k.txt
#emb=data/mc4_fr100k_mdpr_pass.pt
#output=data/titles_knn_faiss.txt
#output_flat=data/titles_knn_flat.txt
#pretty_print=True
#index_path=data/mc4_fr100k_mdpr_pass_faiss_index.idx

# nc_body with mdpr passage.

#input=data/nc_body_10k.txt
#input_emb=data/nc_body_10k_mdpr_pass.pt
#bank=data/mc4_fr1M_para.txt
#emb=data/mc4_fr1M_para_mdpr_pass.pt
#output=data/nc_body_10k_mdpr_pass_knn_faiss.csv
#output_flat=data/nc_body_10k_mdpr_pass_knn_flat.csv
#pretty_print=True
#index_path=data/mc4_fr1M_para_mdpr_pass_faiss_index.idx

# nc_body with sase.

#input=data/nc_body_10k.txt
#input_emb=data/nc_body_10k_sase.pt
#bank=data/mc4_fr1M_para.txt
#emb=data/mc4_fr1M_para_sase.pt
#output=data/nc_body_10k_sase_knn_faiss.csv
#output_flat=data/nc_body_10k_sase_knn_flat.csv
#pretty_print=True
#index_path=data/mc4_fr1M_para_sase_faiss_index.idx

# nc_body with laser.

input=data/nc_body_10k.txt
input_emb=data/nc_body_10k_laser.pt
bank=data/mc4_fr1M_para.txt
emb=data/mc4_fr1M_para_laser.pt
output=data/nc_body_10k_laser_knn_faiss.csv
output_flat=data/nc_body_10k_laser_knn_flat.csv
pretty_print=True
index_path=data/mc4_fr1M_para_laser_faiss_index.idx

# test flat retrieve.

#input=data/titles.txt
#input_emb=data/titles_mdpr_pass.pt
#input_emb=data/titles_laser.pt
#bank=data/mc4_fr100k.txt
#emb=data/mc4_fr100k_mdpr_pass.pt
#emb=data/mc4_fr100k_laser.pt
#output_flat=data/titles_mdpr_pass_fr100k_mdpr_pass_knn_flat.csv
#output_flat=data/titles_laser_fr100k_laser_knn_flat.csv

# create and save index.
#python src/index_creation/faiss_index.py --create_index --emb $emb --M 32 --index_path $index_path

# load and use index.
python src/index_creation/faiss_index.py --search_index --input $input --input_emb $input_emb --bank $bank --index_path $index_path --K 3 --output $output --pretty_print $pretty_print

# To compare time with flat retrieve method.
#python src/flat_retrieve.py --input $input --input_emb $input_emb --bank $bank --emb $emb --K 3 --pretty_print True --output $output_flat

# Create NC body data.
#python src/generate_nc_data.py

exit 0
