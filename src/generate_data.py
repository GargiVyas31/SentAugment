import argparse
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
    paragraphs = [para for para in paragraphs if len(para) > 5]
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
        factor=0.5,
    )

    spacy_modules = {"fr": "fr_core_news_md", "de": "de_core_news_md"}
    nlp_spacy = spacy.load(spacy_modules[language_code])

    # Clear out file content.
    with open(save_path, 'w') as f:
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

        if i % 1000 == 0:
            print(f"Iteration {i}, processed {curr_size} sentences so far...")

        if len(sentences) >= batch_size:
            _append_to_file(save_path, sentences, print_ctr if print_sno else -1)
            print_ctr += len(sentences)
            sentences = []

        if curr_size >= num_rows:
            break
    if len(sentences) > 0:
        _append_to_file(save_path, sentences, print_ctr if print_sno else -1)
        print_ctr += len(sentences)
        sentences = []


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


if __name__ == '__main__':
    parser.add_argument("--num_rows", type=int, default=10, help="number of rows (sentences or paragraphs) to write.")
    parser.add_argument("--output", type=str, help="path of the output text file.")
    parser.add_argument("--language", type=str, default="fr", help="language code for MC4 from "
                                                                   "https://huggingface.co/datasets/mc4")
    parser.add_argument("--split_by", type=str, choices=["sentence", "paragraph", "mixed"], required=True,
                        help="How to split the MC4 sentences.")
    args = parser.parse_args()

    assert args.num_rows >= 1, "--num_rows need to be positive integer."

    sample_mc4_data(num_rows=args.num_rows, language_code=args.language, save_path=args.output,
                    split_by=args.split_by, print_sno=False)
    print("done")
