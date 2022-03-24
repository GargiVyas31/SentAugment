#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
"""
Script that retrieve nearest neighbors of sentences from the bank
Example: python src/flat_retrieve.py --input $input --bank $bank --emb data/keys.pt --K $K
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from mdpr import get_torch_device

DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR + '/../src/lib')
from indexing import IndexTextOpen, IndexTextQuery

device = get_torch_device()

parser = argparse.ArgumentParser(description="retrieve nearest neighbors of sentences")
parser.add_argument("--input", type=str, required=True , help="input text file. We assume embeddings are present in"
                                                              "input.pt file.")
parser.add_argument("--bank", type=str, required=True, help="compressed text file")
parser.add_argument("--emb", type=str, required=True, help="pytorch embeddings of text bank")
parser.add_argument("--K", type=int, default=100, help="number of nearest neighbors per sentence")
parser.add_argument("--pretty_print", type=str, choices=["True", "False"], default="False")
parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()

pretty_print = args.pretty_print == "True"
pretty_print_file = ".".join(args.output.split(".")[:-1]) + "_prettry_print.txt" if pretty_print else ""
assert pretty_print and Path(args.input).is_file(), "--pretty_print is True but --input file does not exist."
ppf = open(pretty_print_file, 'w') if pretty_print_file != "" else None

# load query embedding and bank embedding
query_emb = torch.load(args.input + ".pt", map_location=torch.device(device))
bank_emb = torch.load(args.emb, map_location=torch.device(device))

# normalize embeddings
query_emb.div_(query_emb.norm(2, 1, keepdim=True).expand_as(query_emb))
bank_emb.div_(bank_emb.norm(2, 1, keepdim=True).expand_as(bank_emb))

# score and rank
scores = bank_emb.mm(query_emb.t())  # B x Q
_, indices = torch.topk(scores, args.K, dim=0)  # K x Q

# fetch and print retrieved text
txt_mmap, ref_mmap = IndexTextOpen(args.bank)

with open(args.input, "r") as input_file:
    with open(args.output, "w") as output_file:
        for i, (query_idx, line) in enumerate(zip(range(indices.size(1)), input_file)):
            if ppf:
                ppf.write(f"{i + 1} En: {line}")
            for k in range(args.K):
                sentence = IndexTextQuery(txt_mmap, ref_mmap, indices[k][query_idx])
                if ppf:
                    ppf.write(f"{k + 1} Fr: {sentence}\n")
                output_file.write(sentence + "\n")
            if ppf:
                ppf.write(f"\n")

ppf.close()
