import pandas as pd
import pickle

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# bert
from transformers import BertTokenizer, BertModel
import torch
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")

bert_embeddings = {}
for df in [train, test]:
    for idx, row in df.iterrows():
        poem = row.Poem
        tokens = tokenizer(poem, return_tensors='pt')
        with torch.no_grad():
            bert_embeddings[poem] = model(**tokens)

with open("bert_embeddings.pkl", "wb") as f:
    pickle.dump(bert_embeddings, f)