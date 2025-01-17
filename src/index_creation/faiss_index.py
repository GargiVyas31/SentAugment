import csv
import time

import faiss
import numpy as np
import torch
import argparse
import pandas as pd

# if you see import error while running the below line from terminal, you need to set the python path.
# run the below command in terminal before running python command.
# export PYTHONPATH=/work/ahattimare_umass_edu/SentAugment:${PYTHONPATH}
from tqdm import tqdm

from src.indexing import IndexTextOpen, IndexTextQuery, IndexLoad
from src.mdpr import get_torch_device

device = get_torch_device()

parser = argparse.ArgumentParser(description="faiss index creation.")


def check_waters():
    # Create input.

    d = 64  # dimension
    nb = 100_000  # database size
    nq = 10_000  # nb of queries
    np.random.seed(1234)  # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    # Build index.

    index = faiss.IndexFlatL2(d)  # build the index
    print(index.is_trained)
    index.add(xb)  # add vectors to the index
    print(index.ntotal)

    # Search index.

    k = 4  # we want to see 4 nearest neighbors
    D, I = index.search(xb[:5], k)  # sanity check
    print(I)
    print(D)
    D, I = index.search(xq, k)  # actual search
    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries


def create_and_save_index(emb: str, M: int, index_path: str):
    # Validate data.
    assert emb is not None, "pass --emb for bank embeddings"
    assert M is not None, "pass --M for HNSW faiss index creation"
    assert index_path is not None, "pass --index_path for saving the index"

    bank_emb = torch.load(emb, map_location=torch.device(device))  # (nb x d)
    bank_emb.div_(bank_emb.norm(2, 1, keepdim=True).expand_as(bank_emb))
    bank_emb_np = bank_emb.cpu().numpy().astype('float32')
    d = bank_emb.shape[1]

    t0 = time.time()
    index = faiss.IndexHNSWFlat(d, M)
    index.verbose = True
    index.add(bank_emb_np)
    t1 = time.time()
    print(f"Index created. time={(t1 - t0):.3f}s. ntotal={index.ntotal}")

    faiss.write_index(index, index_path)
    print(f"Faiss index has been saved in {index_path}")


def load_and_search_index(input, input_emb, bank, index_path, K, output, pretty_print,
                          multiple_banks_output=False, offset=0):
    """
    offset (int): for multiple banks, this tells the offset to be used for sentence ids.
    """

    # Validate
    assert all(var is not None for var in [input, input_emb, bank, index_path, K, output, pretty_print])
    print(f"passed offset is {offset}")

    pretty_print = pretty_print == "True"
    pretty_print_file = ".".join(output.split(".")[:-1]) + "_prettry_print.txt" if pretty_print else ""
    ppf = open(pretty_print_file, 'w') if pretty_print_file != "" else None
    assert args.output.split(".")[-1] == "csv", "--output file should be .csv"

    # Load the index.

    t0 = time.time()
    index = IndexLoad(index_path)
    t1 = time.time()
    print(f"faiss index has been loaded. time taken={(t1 - t0):.3f}s.")

    # Load queries.

    query_emb = torch.load(input_emb, map_location=torch.device(device))  # (nq x d)
    query_emb.div_(query_emb.norm(2, 1, keepdim=True).expand_as(query_emb))
    query_emb_np = query_emb.cpu().numpy().astype('float32')

    # Search on the index.

    t0 = time.time()
    D, I = index.search(query_emb_np, K)
    t1 = time.time()
    print(f"time to search index for all queries={(t1 - t0):.3f}s.")

    # Print/save the result.

    txt_mmap, ref_mmap = IndexTextOpen(bank)
    indices = I.T  # shape (k x nq) supported in flat_retrieve code.
    scores = D.T  # shape (k x nq)
    input_df = pd.read_csv(input, header=None)

    with open(output, "w") as output_file:
        csv_writer = csv.writer(output_file)
        for query_idx, query_label, line in tqdm(zip(range(indices.shape[1]), input_df[0], input_df[1]),
                                                 total=indices.shape[1],
                                                 desc="Processing Input."):
            toprint = f"qid: {query_idx}, En: {line}\n"
            if ppf:
                ppf.write(toprint)
            for k in range(K):
                sent_idx = indices[k][query_idx]
                sentence = IndexTextQuery(txt_mmap, ref_mmap, sent_idx)
                toprint = f"k={k + 1}: label={query_label}, score={scores[k][query_idx]}, sent={sentence}\n\n\n"
                if ppf:
                    ppf.write(toprint)
                if multiple_banks_output:
                    csv_writer.writerow([query_idx, sent_idx + offset, scores[k][query_idx], query_label, sentence])
                else:
                    csv_writer.writerow([sent_idx, query_label, sentence])

            if ppf:
                ppf.write("\n")
    if ppf:
        ppf.close()


if __name__ == '__main__':
    parser.add_argument("--create_index", action="store_true")
    parser.add_argument("--emb", type=str, help="pytorch embeddings of text bank")
    parser.add_argument("--M", type=int, default=32, help="M argument for HNSW faiss index")
    parser.add_argument("--index_path", type=str, help="path to save the index in or load it from")

    parser.add_argument("--search_index", action="store_true")
    parser.add_argument("--input", type=str, help="input text file.")
    parser.add_argument("--input_emb", type=str, help="input text file embeddings.")
    parser.add_argument("--bank", type=str, help="compressed text file")
    parser.add_argument("--K", type=int, default=100, help="number of nearest neighbors per sentence")
    parser.add_argument("--output", type=str, help="file path to save KNN search output")
    parser.add_argument("--pretty_print", type=str, choices=["True", "False"], default="False")
    parser.add_argument("--multiple_banks_output", action="store_true", help="pass this flag if the output needs to be"
                                                                             "later filtered for multiple banks.")
    parser.add_argument("--offset", type=int, default=0, help="the sentence id offset used to save the retrieval result"
                                                              " so that it can later be separated for multiple banks.")
    args = parser.parse_args()

    if args.create_index:
        t0 = time.time()
        create_and_save_index(**{"emb": args.emb, "M": args.M, "index_path": args.index_path})
        print(f"total time for create_and_save_index(): {(time.time() - t0):.3f}s.")
    elif args.search_index:
        t0 = time.time()
        load_and_search_index(**{"input": args.input, "input_emb": args.input_emb, "bank": args.bank,
                                 "index_path": args.index_path, "K": args.K, "output": args.output,
                                 "pretty_print": args.pretty_print, "multiple_banks_output": args.multiple_banks_output,
                                 "offset": args.offset})
        print(f"total time for load and search: {(time.time() - t0):.3f}s.")
    exit(0)
    # check_waters()
