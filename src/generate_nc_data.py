import csv

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def download_nc():
    nc_data = load_dataset("xglue", "nc")
    data = pd.DataFrame(nc_data["train"])
    data = data[["news_body", "news_category"]].sample(n=100_000, random_state=10)

    # Find the data distribution.
    data_perc = data["news_category"].value_counts().apply(lambda x: x*100/data.shape[0])
    print(data_perc)

    def stratified_sample_df(df, col, n_samples_per_class):
        n = min(n_samples_per_class, df[col].value_counts().min())
        df_ = df.groupby(col).apply(lambda x: x.sample(n))
        df_.index = df_.index.droplevel(0)
        return df_

    data_balanced = stratified_sample_df(data, "news_category", 1000)
    # data_balanced = data

    # Now again check the data distribution.
    data_balanced_perc = data_balanced["news_category"].value_counts().apply(lambda x: x*100/data_balanced.shape[0])
    print(data_balanced_perc)

    # shuffle the data.
    data_balanced.sample(frac=1).reset_index(drop=True)

    output_file = "data/nc_body_10k.csv"

    with open(output_file, "w") as output_file:
        csv_writer = csv.writer(output_file)
        for article, category in tqdm(zip(data_balanced["news_body"], data_balanced["news_category"]),
                            desc="Creating NC body file",
                            total=data_balanced.shape[0]):
            csv_writer.writerow([category, article])


if __name__ == '__main__':
    download_nc()
