import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def download_nc():
    nc_data = load_dataset("xglue", "nc")
    data = pd.DataFrame(nc_data["train"])
    sample_articles = data["news_body"].sample(n=10_000, random_state=20)

    with open("data/nc_body_10k.txt", "w") as f:
        for article in tqdm(sample_articles, desc="Creating NC body file"):
            f.write(article)
            f.write("\n")


if __name__ == '__main__':
    download_nc()
