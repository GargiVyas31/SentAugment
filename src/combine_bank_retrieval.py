import argparse
import time

import pandas as pd

parser = argparse.ArgumentParser(description="faiss index creation.")


def combine_files_test():
    # The columns are [query_id, sent_id, sent_score, query_label, sent]
    data1 = [
        [1, 1, 1.23, 0, "sent1"],
        [2, 4, 1.20, 8, "sent4"],
        [2, 5, 1.54, 8, "sent5"],
        [2, 3, 1.43, 8, "sent3"],
        [1, 2, 1.34, 0, "sent2"],
        [1, 3, 1.43, 0, "sent3"],
    ]

    data2 = [
        [1, 10, 1.28, 0, "sent10"],
        [2, 40, 1.20, 8, "sent40"],
        [3, 50, 1.54, 5, "sent50"],
        [2, 30, 1.57, 8, "sent30"],
        [1, 20, 1.34, 0, "sent20"],
        [1, 30, 1.55, 0, "sent30"],
    ]

    df1 = pd.DataFrame(data1)
    print(df1)

    df2 = pd.DataFrame(data2)
    print(df2)

    df = pd.concat([df1, df2], ignore_index=True)

    k = 2
    df_new = df.groupby([0]).apply(lambda x: x.sort_values(by=[2], ascending=False).head(k))
    print(df_new)

    print('done')


def combine_files(file_name_template: str, num_files: int, output_file: str, k: int):
    assert num_files > 1, "At least 2 files are required to combine."
    assert "XXX" in file_name_template

    # Load the first file.
    file_name = file_name_template.replace("XXX", str(1))
    df = pd.read_csv(file_name, header=None)

    for i in range(2, num_files + 1):
        file_name2 = file_name_template.replace("XXX", str(i))
        print(f"Combining file {file_name2}")

        df2 = pd.read_csv(file_name2, header=None)
        df = pd.concat([df, df2], ignore_index=True)
        # group by query_id, sort each by decreasing sent_score, then keep top k.
        df = df.groupby([0]).apply(lambda x: x.sort_values(by=[2], ascending=False).head(k))

    # Save combined result in disk. Only save [sent_id, query_label, sentence] with no header.
    df.to_csv(output_file, columns=[1, 3, 4], header=False, index=False)


if __name__ == '__main__':
    parser.add_argument("--file_name_template", type=str, required=True, help="source file name template.")
    parser.add_argument("--num_files", type=int, required=True, help="the number of source files. We assume 1 to n files.")
    parser.add_argument("--output_file", type=str, required=True, help="file where to store the combined result.")
    parser.add_argument("--K", type=int, required=True, help="number of examples to keep for each query sentence.")
    args = parser.parse_args()

    t0 = time.time()
    combine_files(args.file_name_template, args.num_files, args.output_file, args.K)
    print(f"total time combining files: {(time.time() - t0):.3f}s.")

    # combine_files_test()

    exit(0)
