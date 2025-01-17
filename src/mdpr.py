"""
Script that takes text as input and output mDPR sentence embeddings.
"""

import argparse
import io
import os
import pathlib

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


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

parser = argparse.ArgumentParser(description="mDPR embeddings.")


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
    load_saved = args.load_saved == "True"

    assert args.model_type in ["question", "passage"], "--model_type only supports 'question' or 'passage'."
    model_name = "castorini/mdpr-question-nq" if args.model_type == "question" else "castorini/mdpr-passage-nq"
    print(f"Using model {model_name}")

    current_path = str(pathlib.Path(__file__).parent.absolute())
    model_path = current_path + "/../models/" + model_name
    load_local_copy = True if load_saved and os.path.isdir(model_path) else False

    tokenizer = AutoTokenizer.from_pretrained(model_path if load_local_copy else model_name)
    if load_saved:
        tokenizer.save_pretrained(model_path)
    print("MDPR tokenizer loaded.")

    model = AutoModel.from_pretrained(model_path if load_local_copy else model_name)
    if load_saved:
        model.save_pretrained(model_path)
    print("MDPR model loaded.")

    model.eval()
    if args.cuda:
        model.cuda()

    max_length = 512
    print(f"Position embedding for model: {max_length}. Max possible: {model.config.max_position_embeddings}")

    # load sentences
    sentences = []
    with io.open(args.input, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            sentences.append(line.rstrip())
            if i % 10_000 == 0:
                print(f"loading sentences line {i + 1}...")

    # encode sentences and save in a tensor.
    embeds = None
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

            if embeds is None:
                # preallocate the tensor with the correct expected size.
                embeds = torch.zeros(len(sentences), embeddings_cpu.size()[1])
                embeds[i:i + embeddings_cpu.size()[0]] = embeddings_cpu
            else:
                embeds[i:i + embeddings_cpu.size()[0]] = embeddings_cpu

    # save embeddings
    torch.save(embeds, args.output)
    print("mdpr embedding done")


if __name__ == "__main__":
    main()
