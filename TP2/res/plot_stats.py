import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('results.txt', delimiter='|')
print(df.columns.values)
length_sentences = df[" words "].values
tag_accuracy = df[" tag_accracy "].values
hist = {}
for idx_sent in range(np.shape(length_sentences)[0]):
    length = length_sentences[idx_sent]
    if length in hist:
       hist[length].append(tag_accuracy[idx_sent])   
    else:
       hist[length] = [tag_accuracy[idx_sent]]
for length in hist.keys():
    hist[length] = np.mean(hist[length])
plt.bar(list(hist.keys()),list(hist.values()))
plt.show()
