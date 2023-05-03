import pandas as pd
import pickle

pd.set_option("display.max_columns", None)

# read in Kaggle dataset
data = pd.read_csv("PoetryFoundationData.csv")

data["Poet"] = data["Poet"].apply(lambda x: x.strip("\t\r\n ").replace("\r", "").replace("\t", ""))
data["Title"] = data["Title"].apply(lambda x: x.strip("\t\r\n ").replace("\r", "").replace("\t", ""))
data["Poem"] = data["Poem"].apply(lambda x: x.strip("\t\r\n ").replace("\r", "").replace("\t", ""))

# filter out poems with more than 512 tokens
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

data["Tokens"] = data["Poem"].apply(lambda x: len(tokenizer.tokenize(x)))
data = data[data["Tokens"] <= 512] 

# unpickle Rupi Kaur poems
poem_pkl = open("rupi_kaur_poems.pkl", "rb")
poems = pickle.load(poem_pkl)

# choose top 50 longest poems
poems = sorted(poems, key=lambda x: -len(x))[:50]

# count the number of poems by each poet
poet_counts = data.groupby("Poet")["Poem"].count()

# sort the poets by the number of poems in descending order
sorted_poets = poet_counts.sort_values(ascending=False)

print(sorted_poets.head(10))

# select the top 10 poets with the highest number of poems
top_10_poets = sorted_poets.head(10).index.tolist()

print(top_10_poets)

# filter the dataset to include only the poems by the selected top 10 poets
df = data[data["Poet"].isin(top_10_poets)][["Poet", "Poem", "Title"]]

# add Rupi Kaur poems to dataset
for poem in poems:
    df.loc[len(df)] = {"Poet":"Rupi Kaur", "Poem":poem}

poet_info = {
    "John Ashbery":["American", "20th-century", "Male"],
    "William Butler Yeats":["Irish", "20th-century", "Male"],
    "William Shakespeare":["English", "16th-century", "Male"],
    # "Robert Browning":["English", "19th-century", "Male"],
    "Kay Ryan":["American", "20th-century", "Female"],
    "Emily Dickinson":["American", "19th-century", "Female"],
    "Alfred, Lord Tennyson":["English", "19th-century", "Male"],
    "Rae Armantrout":["American", "21st-century", "Female"],
    "Yusef Komunyakaa":["American", "20th-century", "Male"],
    # "John Donne":["English", "17th-century", "Male"],
    "Samuel Menashe":["American", "20th-century", "Male"],
    "William Wordsworth":["English", "19th-century", "Male"],
    "Rupi Kaur":["Canadian", "21st-century", "Female"],
}

nationality = list()
century = list()
gender = list()

for idx, row in df.iterrows():
    n, c, g = poet_info[row.Poet]
    nationality.append(n)
    century.append(c)
    gender.append(g)

df["Nationalty"] = nationality
df["Century"] = century
df["Gender"] = gender

df.to_csv("poems.csv", index=False)