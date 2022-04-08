import pandas as pd
from datasets import load_dataset


def download_nc():
    nc_data = load_dataset("xglue", "nc")
    data = pd.DataFrame(nc_data["train"])
    sample_articles = data["news_body"].sample(n=1000, random_state=20)

    with open("data/nc_body_1k.txt", "w") as f:
        for article in sample_articles:
            f.write(article)
            f.write("\n")


if __name__ == '__main__':
    download_nc()
