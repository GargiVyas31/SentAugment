#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
#

"""
Script that takes text as input and output SASE sentence embeddings
Example: python src/sase.py --input $input --model $modelpath --spm_model $spmmodel --batch_size 64 --cuda "True" --output $output
"""

import argparse

import torch
from laserembeddings import Laser

parser = argparse.ArgumentParser(description="LASER encoding")


def main():
    parser.add_argument("--input", type=str, default="", help="input file")
    parser.add_argument("--input_lang", type=str, default="", help="input language")
    # parser.add_argument("--model", type=str, default="", help="model path")
    # parser.add_argument("--laser_model", type=str, default="", help="laser model path")
    # parser.add_argument("--laser_bpe_codes", type=str, default="", help="laser bpe codes path")
    # parser.add_argument("--laser_bpe_vocab", type=str, default="", help="laser bpe vocab path")
    # parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    # parser.add_argument("--max_words", type=int, default=100, help="max words")
    parser.add_argument("--cuda", type=str, default="True", help="use cuda")
    parser.add_argument("--output", type=str, default="", help="output file")
    args = parser.parse_args()

    laser_bpe_codes = "data/93langs.fcodes"
    laser_bpe_vocab = "data/93langs.fvocab"

    # Load the model
    laser_model = "data/bilstm.93langs.2018-12-26.pt"

    # cuda
    assert args.cuda in ["True", "False"]
    args.cuda = eval(args.cuda)

    # build model / reload weights
    laser_model = Laser(laser_bpe_codes, laser_bpe_vocab, laser_model)

    # load sentences
    sentences = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            sentences.append(line.rstrip())
            if i % 10_000 == 0:
                print(f"loading sentences line {i + 1}...")

    # encode sentences
    embs = []

    embeddings_input = laser_model.embed_sentences(sentences, lang=args.input_lang)
    # print(type(embeddings_input))
    embeddings_input = torch.tensor(embeddings_input).double().cpu()
    embs.append(embeddings_input)

    # save embeddings
    torch.save(torch.cat(embs, dim=0).squeeze(0), args.output)


if __name__ == "__main__":
    main()
