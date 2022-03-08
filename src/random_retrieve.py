#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
"""
Script that retrieve nearest neighbors of sentences from the bank
Example: python src/flat_retrieve.py --input $input --bank $bank --emb data/keys.pt --K $K
"""

import os
import sys
import torch
import argparse
import time

from mdpr import get_torch_device

DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR + '/../src/lib')
from indexing import IndexTextOpen, IndexTextQuery

device = get_torch_device()

parser = argparse.ArgumentParser(description="retrieve random sentences")
parser.add_argument("--input", type=str, required=True , help="input pytorch embeddings")
parser.add_argument("--bank", type=str, required=True, help="compressed text file")
parser.add_argument("--emb", type=str, required=True, help="pytorch embeddings of text bank")
parser.add_argument("--K", type=int, default=100, help="number of random per sentence")

args = parser.parse_args()

# load query embedding and bank embedding
query_emb = torch.load(args.input, map_location=torch.device(device))
bank_emb = torch.load(args.emb, map_location=torch.device(device))

# normalize embeddings
query_emb.div_(query_emb.norm(2, 1, keepdim=True).expand_as(query_emb))
bank_emb.div_(bank_emb.norm(2, 1, keepdim=True).expand_as(bank_emb))

# score and rank
scores = bank_emb.mm(query_emb.t())  # B x Q
indices = torch.randint(len(scores), (args.K,len(scores[0]))) # K x Q
txt_mmap, ref_mmap = IndexTextOpen(args.bank)
for qeury_idx in range(indices.size(1)):
    for k in range(args.K):
        print(IndexTextQuery(txt_mmap, ref_mmap, indices[k][qeury_idx]))
    
