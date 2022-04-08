import numpy as np

words = []
with open("../data/nc_body_1k.txt", "r") as f:
    for line in f:
        words.append(len(line.split(' ')))

print(f"sentences parsed: {len(words)}")
for perc in range(0, 101, 10):
    print(f"{perc} percentile index: {np.percentile(words, perc):.2f}")
