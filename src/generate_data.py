import argparse
import time
from typing import List

import spacy
from datasets import load_dataset


def split_mixed(document: str) -> List[str]:
    """
    Takes in a document obtained from MC4 sampling and converts it into smaller sentences. The strategy to split
    sentence is to divide the sentence into paragraphs, then combine smaller adjacent paragraphs into one.
    Each resulting sentence is truncated to 2000-2500 characters long (roughly 250-400 words).
    Returns a list of sentences.
    """
    sent_min_char, sent_max_char = 2000, 2500
    paragraphs = document.split("\n")
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


def split_into_paragraphs(document: str) -> list:
    paragraphs = document.split("\n")

    # remove very small paragraphs.
    def keep_para(para: str) -> bool:
        words = para.split(' ')
        return len(words) >= 100 and all(len(word) < 50 for word in words)

    paragraphs = list(filter(keep_para, paragraphs))
    return paragraphs


def split_into_sentences(document: str, spacy_module) -> list:
    doc = spacy_module(document)
    if not doc.has_annotation("SENT_START"):
        print("Spacy document doesn't have SENT_START.")
        return []
    # remove newline character from sentences.
    sentences = [str(sent).replace('\n', ' ') for sent in doc.sents]

    # remove very short sentences or when words are not actual words.
    def keep_sent(sent: str) -> bool:
        words = sent.split(' ')
        return len(words) >= 8 and all(len(word) < 50 for word in words)

    sentences = list(filter(keep_sent, sentences))
    return sentences


def _sample_into_file(num_rows, batch_size, file_path, split_by, print_sno, mc4random, nlp_spacy):
    # Clear out file content.
    with open(file_path, 'w') as f:
        pass

    sentences = []
    curr_size = 0
    print_ctr = 1
    for i, sample in enumerate(mc4random):
        if split_by == "mixed":
            processed_sentences = split_mixed(sample["text"])
        elif split_by == "paragraph":
            processed_sentences = split_into_paragraphs(sample["text"])
        else:
            processed_sentences = split_into_sentences(sample["text"], nlp_spacy)

        sentences += processed_sentences
        curr_size += len(processed_sentences)

        if i % 10_000 == 0:
            print(f"Iteration {i}, processed {curr_size} sentences so far...")

        if len(sentences) >= batch_size:
            _append_to_file(file_path, sentences, print_ctr if print_sno else -1)
            print_ctr += len(sentences)
            sentences = []

        if curr_size >= num_rows:
            break
    if len(sentences) > 0:
        _append_to_file(file_path, sentences, print_ctr if print_sno else -1)
        print_ctr += len(sentences)
        sentences = []


def sample_mc4_data(num_rows=100, batch_size=100, language_code="fr", save_path=None, split_by=None,
                    print_sno=False):
    assert save_path is not None, "provide a save_path to save the output."
    assert split_by in ["sentence", "paragraph", "mixed"], "provide a valid splitting technique."
    assert language_code in ["fr", "de"], "only fr and de are supported."

    mc4random = load_dataset(
        "bertin-project/mc4-sampling", language_code,
        split="train",
        streaming=True,
        sampling_method="random",
        factor=0.6,
    )

    spacy_modules = {"fr": "fr_core_news_md", "de": "de_core_news_md"}
    nlp_spacy = spacy.load(spacy_modules[language_code])

    _sample_into_file(num_rows, batch_size, save_path, split_by, print_sno, mc4random, nlp_spacy)


def _append_to_file(save_path: str, sentences: list, counter_start: int):
    with open(save_path, "a") as f:
        # writelines() does not add newline after each sentence. So add it explicitly.
        if counter_start == -1:
            # don't print sentence count in the beginning.
            f.writelines([sent + "\n" for sent in sentences])
        else:
            new_sentences = []
            for sent in sentences:
                new_sentences.append(str(counter_start) + ": " + sent + "\n")
                counter_start += 1
            f.writelines(new_sentences)


parser = argparse.ArgumentParser(description="Create MC4 data file.")


def split_file(source: str, target_template: str, count: int, rows_per_file: int):
    assert "XXX" in target_template

    batch_size = 128
    with open(source) as fin:
        sentences = []
        file_counter = 1
        file_name = target_template.replace("XXX", str(file_counter))
        file_size = 0
        with open(file_name, 'w') as f:
            print(f"created an empty file {file_name} for writing.")

        for line in fin:
            sentences.append(line[:-1])
            if (len(sentences) >= batch_size) or (file_size + len(sentences) >= rows_per_file):
                _append_to_file(file_name, sentences, -1)
                file_size += len(sentences)
                sentences = []

            # check if we should create new file.
            if file_size >= rows_per_file:
                # All files have been created.
                if file_counter == count:
                    break

                # We have new files to create.
                file_counter += 1
                file_name = target_template.replace("XXX", str(file_counter))
                file_size = 0
                with open(file_name, 'w') as f:
                    print(f"created an empty file {file_name} for writing.")


if __name__ == '__main__':
    parser.add_argument("--num_rows", type=int, default=10, help="number of rows (sentences or paragraphs) to write.")
    parser.add_argument("--output", type=str, help="path of the output text file.")
    parser.add_argument("--language", type=str, default="fr", help="language code for MC4 from "
                                                                   "https://huggingface.co/datasets/mc4")
    parser.add_argument("--split_by", type=str, choices=["sentence", "paragraph", "mixed"],
                        help="How to split the MC4 sentences.")

    parser.add_argument("--multiple", action="store_true")
    parser.add_argument("--count", type=int, default=0, help="how many multiple files do you want to create.")

    parser.add_argument("--split", action="store_true")
    parser.add_argument("--source", type=str, help="the source file to split")
    parser.add_argument("--target_template", type=str, help="target file name template")
    parser.add_argument("--rows_per_file", type=int, help="how many rows to add per file")

    args = parser.parse_args()

    assert args.num_rows >= 1, "--num_rows need to be positive integer."

    if args.split:
        assert args.source is not None
        assert args.count > 0
        assert args.target_template is not None
        assert args.rows_per_file > 0

    t0 = time.time()
    if args.split:
        split_file(args.source, args.target_template, args.count, args.rows_per_file)
    else:
        sample_mc4_data(num_rows=args.num_rows, language_code=args.language, save_path=args.output,
                        split_by=args.split_by, print_sno=False)
    print(f"Done. Time taken={(time.time() - t0):.3f} sec")

