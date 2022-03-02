"""
Script that takes text as input and output mDPR sentence embeddings.
"""

import argparse
import torch
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
    parser.add_argument("--model_type", type=str, default="question", help="Select if you want to use the question or"
                                                                           "passage encoder of mDPR. Acceptable value "
                                                                           "are 'question' or 'passage'.")
    parser.add_argument("--batch_size", type=int, default=256, help="number of sentences to embed at a time.")

    args = parser.parse_args()

    # cuda
    assert args.cuda in ["True", "False"]
    args.cuda = eval(args.cuda)

    assert args.model_type in ["question", "passage"], "--model_type only supports 'question' or 'passage'."
    model_name = "castorini/mdpr-question-nq" if args.model_type == "question" else "castorini/mdpr-passage-nq"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if args.cuda:
        model.cuda()

    max_length = model.config.max_length
    print(f"model max length = {max_length}")

    # load sentences
    sentences = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            sentences.append(line.rstrip())
            if i % 10_000 == 0:
                print(f"loading sentences line {i + 1}...")

    # encode sentences
    embs = []
    with torch.no_grad():
        for i in range(0, len(sentences), args.batch_size):
            batch = sentences[i:i + args.batch_size]
            batch_sent_tok = tokenizer(batch, padding="max_length",
                                       max_length=max_length, truncation=True,
                                       return_tensors="pt")
            if args.cuda:
                batch_sent_tok = batch_sent_tok.to(device)
            batch_embeddings = model(**batch_sent_tok)
            embs.append(batch_embeddings.pooler_output)

    # save embeddings
    torch.save(torch.cat(embs, dim=0).squeeze(0), args.output)


if __name__ == "__main__":
    main()
