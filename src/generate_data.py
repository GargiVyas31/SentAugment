import argparse
from typing import List
from os.path import exists
from datasets import load_dataset


def process(sentence: str) -> List[str]:
    """
    Takes in a sentence obtained from MC4 sampling and converts it into smaller sentences. The strategy to split
    sentence is to divide the sentence into paragraphs, then combine smaller adjacent paragraphs into one.
    Each resulting sentence is truncated to 2000-2500 characters long (roughly 250-400 words).
    Returns a list of sentences.
    """
    sent_min_char, sent_max_char = 2000, 2500
    paragraphs = sentence.split("\n")
    sentences = []
    curr = ""
    for para in paragraphs:
        curr += (para + " ")
        if len(curr) >= sent_min_char:
            # truncate if max limit has been crossed.
            if len(curr) > sent_max_char:
                curr = curr[:sent_max_char]
            sentences.append(curr)
            curr = ""
    return sentences


def sample_mc4_data(num_rows=100, batch_size=100, language_code="fr", save_path=None):
    mc4random = load_dataset(
        "bertin-project/mc4-sampling", language_code,
        split="train",
        streaming=True,
        sampling_method="random",
        factor=0.5,
    )

    # Create the file if it does not exist.
    if not exists(save_path):
        fp = open(save_path, "w")
        fp.close()

    sentences = []
    for i, sample in enumerate(mc4random):
        processed_sentences = process(sample["text"])
        sentences += processed_sentences
        i += len(processed_sentences)

        if i % 1000 == 0:
            print(f"Processed {i} sentences so far...")

        if len(sentences) >= batch_size:
            with open(save_path, "a") as f:
                # writelines() does not add newline after each sentence. So add it explicitly.
                sentences = [sent + "\n" for sent in sentences]
                f.writelines(sentences)
            sentences = []

        if i >= num_rows:
            break
    if len(sentences) > 0:
        with open(save_path, "a") as f:
            sentences = [sent + "\n" for sent in sentences]
            f.writelines(sentences)
        sentences = []


parser = argparse.ArgumentParser(description="Create MC4 data file.")


if __name__ == '__main__':
    parser.add_argument("--num_rows", type=int, default=10, help="number of sentences to write.")
    parser.add_argument("--output", type=str, help="path of the output text file.")
    parser.add_argument("--language", type=str, default="fr", help="language code for MC4 from "
                                                                   "https://huggingface.co/datasets/mc4")
    args = parser.parse_args()

    assert args.num_rows >= 1, "--num_rows need to be positive integer."

    sample_mc4_data(num_rows=args.num_rows, language_code=args.language, save_path=args.output)
    print("done")
