import argparse

from rank_bm25 import BM25Plus, BM25Okapi, BM25L

parser = argparse.ArgumentParser(description="bm25 ranking.")


def main():
    """
    This method takes in a bank/corpus file, a sentence file having queries, number of neighbours K and
    prints K closest matching sentences for each input query.

    A sample usage of this method
    python src/bm25.py --input=data/sentence.txt --bank=data/keys_small.txt --K=2 --lowercase=True
    """
    parser.add_argument("--bank", type=str, default="", help="corpus bank file")
    parser.add_argument("--input", type=str, default="", help="input file")
    parser.add_argument("--lowercase", type=str, default="true", help="convert text to lowercase")
    parser.add_argument("--K", type=int, default=100, help="number of ranked sentences")
    parser.add_argument("--model", type=str, default="bm25plus", help="One of bm25plus, bm25okapi, bm25l")

    args = parser.parse_args()

    to_lowercase: bool = eval(args.lowercase)

    assert args.model in ["bm25plus", "bm25okapi", "bm25l"], "--model only supports bm25plus, bm25okapi and bm25l."
    if args.model == "bm25plus":
        bm25_model = BM25Plus
    elif args.model == "bm25okapi":
        bm25_model = BM25Okapi
    else:
        bm25_model = BM25L

    # load the corpus sentences
    corpus = []
    with open(args.bank) as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            if to_lowercase:
                line = line.lower()
            corpus.append(line)
            if i % 10_000 == 0:
                print(f"loading corpus sentence line {i + 1}...")

    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = bm25_model(tokenized_corpus)

    # load the query sentences
    queries = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            if to_lowercase:
                line = line.lower()
            queries.append(line)
            if i % 10_00 == 0:
                print(f"loading query sentence line {i + 1}...")

    K = args.K

    for query in queries:
        tokenized_query = query.split(" ")
        top_docs = bm25.get_top_n(tokenized_query, corpus, n=K)

        print(f"Query: {query}")
        for k, doc in enumerate(top_docs):
            print(f"{k+1}: {doc}")
        print()


if __name__ == '__main__':
    main()
