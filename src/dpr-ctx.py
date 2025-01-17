"""
Script that takes text as input and output dpr-ctx sentence embeddings.
"""

import argparse
import io
import os
import pathlib
import pandas as pd

import torch
from tqdm import tqdm

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

def get_torch_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        n_gpu = torch.cuda.device_count()
        print(f"Found device: {device_name}, n_gpu: {n_gpu}")
        device_ = torch.device("cuda")
    else:
        device_ = torch.device('cpu')
    return device_


device = get_torch_device()

parser = argparse.ArgumentParser(description="dpr-ctx embeddings.")


def main():
    parser.add_argument("--input", type=str, default="", help="input file")
    parser.add_argument("--cuda", type=str, default="True", help="use cuda")
    parser.add_argument("--output", type=str, default="", help="output file")
    parser.add_argument("--model_type", type=str, default="passage", help="Select if you want to use the question or"
                                                                           "passage encoder of mDPR. Acceptable value "
                                                                           "are 'question' or 'passage'.")
    parser.add_argument("--batch_size", type=int, default=256, help="number of sentences to embed at a time.")
    parser.add_argument("--load_saved", type=str, default="False", choices=["True", "False"], help="load locally saved "
                                                            "pre-trained model if possible. Also, to save pre-trained"
                                                            "model locally that is downloaded from huggingface.")
    args = parser.parse_args()

    # cuda
    assert args.cuda in ["True", "False"]
    args.cuda = eval(args.cuda)

    assert args.model_type in ["question", "passage"], "--model_type only supports 'question' or 'passage'."
    model_name = "voidful/dpr-question_encoder-bert-base-multilingual" if args.model_type == "question" \
        else "voidful/dpr-ctx_encoder-bert-base-multilingual"
    print(f"Using model {model_name}")

    current_path = str(pathlib.Path(__file__).parent.absolute())
    model_path = current_path + "/../models/" + model_name
    load_local_copy = True if args.load_saved == "True" and os.path.isdir(model_path) else False

    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_path if load_local_copy else model_name)
    if args.load_saved:
        tokenizer.save_pretrained(model_path)
    print("MDPR tokenizer loaded.")

    model = DPRQuestionEncoder.from_pretrained(model_path if load_local_copy else model_name)
    if args.load_saved:
        model.save_pretrained(model_path)
    print("MDPR-CTX model loaded.")

    model.eval()
    if args.cuda:
        model.cuda()

    print(f"Max position embedding for model: {model.config.max_position_embeddings}")
    max_length = 512

    # load sentences
    sentences = []

    if args.input.endswith(".txt"):
        with io.open(args.input, encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                sentences.append(line.rstrip())
                if i % 10000 == 0:
                    print(f"loading sentences line {i + 1}...")
    elif args.input.endswith(".csv"):
        input_df = pd.read_csv(args.input, header=None)
        for idx, line in enumerate(input_df[1]):
            sentences.append(line.rstrip())
            if idx % 10000 == 0:
                print(f"loading sentences line {idx + 1}...")

    # encode sentences
    embs = None
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), args.batch_size), desc="Embedding sentences"):
            batch = sentences[i:i + args.batch_size]
            batch_sent_tok = tokenizer(batch, padding="max_length",
                                       max_length=max_length, truncation=True,
                                       return_tensors="pt")
            if args.cuda:
                batch_sent_tok = batch_sent_tok.to(device)
            batch_embeddings = model(**batch_sent_tok)
            embeddings_cpu = batch_embeddings.pooler_output.cpu()

            if embs is None:
                # preallocate the tensor with the correct expected size.
                embs = torch.zeros(len(sentences), embeddings_cpu.size()[1])
                embs[i:i + embeddings_cpu.size()[0]] = embeddings_cpu
            else:
                embs[i:i + embeddings_cpu.size()[0]] = embeddings_cpu

    # save embeddings
    torch.save(embs, args.output)


if __name__ == "__main__":
    main()
