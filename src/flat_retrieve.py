#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
"""
Script that retrieve nearest neighbors of sentences from the bank
Example: python src/flat_retrieve.py --input $input --bank $bank --emb data/keys.pt --K $K
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mdpr import get_torch_device

DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR + '/../src/lib')
from indexing import IndexTextOpen, IndexTextQuery

device = get_torch_device()

parser = argparse.ArgumentParser(description="retrieve nearest neighbors of sentences")
parser.add_argument("--input", type=str, required=True, help="input text file.")
parser.add_argument("--input_emb", type=str, required=True, help="input text file embeddings.")
parser.add_argument("--bank", type=str, required=True, help="compressed text file")
parser.add_argument("--emb", type=str, required=True, help="pytorch embeddings of text bank")
parser.add_argument("--K", type=int, default=100, help="number of nearest neighbors per sentence")
parser.add_argument("--pretty_print", type=str, choices=["True", "False"], default="False")
parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()

pretty_print = args.pretty_print == "True"
pretty_print_file = ".".join(args.output.split(".")[:-1]) + "_prettry_print.txt" if pretty_print else ""
if pretty_print:
    assert Path(args.input).is_file(), "--pretty_print is True but --input file does not exist so can't print input."
ppf = open(pretty_print_file, 'w') if pretty_print_file != "" else None

assert args.output.split(".")[-1] == "csv", "--output file should be .csv"

# load query embedding and bank embedding
print('loading query and bank embeddings.')
t0 = time.time()
query_emb = torch.load(args.input_emb, map_location=torch.device(device))
print(f"time to load query embeddings={(time.time() - t0):.3f}s")
t0 = time.time()
bank_emb = torch.load(args.emb, map_location=torch.device(device))
print(f"time to load bank embeddings={(time.time() - t0):.3f}s")

# normalize embeddings
query_emb.div_(query_emb.norm(2, 1, keepdim=True).expand_as(query_emb))
bank_emb.div_(bank_emb.norm(2, 1, keepdim=True).expand_as(bank_emb))

# score and rank
start = time.time()
scores = bank_emb.mm(query_emb.t())  # B x Q
_, indices = torch.topk(scores, args.K, dim=0)  # K x Q
print(f"time to find knn={(time.time() - start):.3f}s")

# fetch and print retrieved text
txt_mmap, ref_mmap = IndexTextOpen(args.bank)

bank_index_used = []

with open(args.input, "r") as input_file:
    with open(args.output, "w") as output_file:
        csv_writer = csv.writer(output_file)
        for i, (query_idx, line) in enumerate(tqdm(zip(range(indices.size(1)), input_file),
                                                   total=indices.size(1), desc="Processing Input file.")):
            if ppf:
                ppf.write(f"{i + 1} En: {line}")
            for k in range(args.K):
                bank_index = (indices[k][query_idx]).cpu().item()
                bank_index_used.append(bank_index)
                sentence = IndexTextQuery(txt_mmap, ref_mmap, bank_index)
                if ppf:
                    ppf.write(f"{k + 1} Fr: {sentence}\n")
                csv_writer.writerow([bank_index, sentence])
            if ppf:
                ppf.write(f"\n")
if ppf:
    ppf.close()

# This shows how the KNN retrieved data is spread out in the bank.
for perc in range(0, 101, 10):
    print(f"{perc} percentile index: {np.percentile(bank_index_used, perc):.2f}")
