import gensim.downloader as api
import pandas as pd
import pickle

import torch
from nltk.tokenize import wordpunct_tokenize

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# glove
glove_embs = api.load("glove-wiki-gigaword-50")
# print(glove_embs["<UNK>"])

# f = open("demofile2.txt", "a")
# for word in glove_embs.index_to_key:
#     f.write(word)
#     f.write("\n")
# f.close()

glove_embeddings = {}
for df in [train, test]:
    for idx, row in df.iterrows():
        poem = row.Poem
        tokens = wordpunct_tokenize(poem.lower())
        glove_embeddings[poem] = torch.stack([torch.as_tensor(glove_embs[token].copy()) if token in glove_embs else torch.zeros(50) for token in tokens]).unsqueeze(0)

with open("glove_embeddings.pkl", "wb") as f:
    pickle.dump(glove_embeddings, f)