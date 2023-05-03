from torch.utils.data import Dataset
import pandas as pd
import torch
import nltk

class PoetryDataset(Dataset):
    def __init__(self, csv_dir, embeddings, is_glove=False, root_path="./data/"):
        self.embeddings = embeddings
        self.is_glove = is_glove

        data = root_path + csv_dir + ".csv"
        gold = root_path + csv_dir + "_labels.csv"

        data_df = pd.read_csv(data)
        gold_df = pd.read_csv(gold)

        self.df = pd.concat([data_df, gold_df], axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        poem = self.df.iloc[idx, 0]
        title = self.df.iloc[idx, 1]
        nationality = self.df.iloc[idx, 2]
        century = self.df.iloc[idx, 3]
        gender = self.df.iloc[idx, 4]
        poet = poet2label(self.df.iloc[idx, 5])

        if self.is_glove:
            inputs = self.embeddings[poem][0]
        else:
            inputs = self.embeddings[poem][0][0]

        sample = {"inputs":inputs, "poem":poem, "title":title, "nationality":nationality, "century":century, "gender":gender, "poet":poet}
        return sample

def poet2label(poet):
    labels = {
        "John Ashbery":0,
        "William Butler Yeats":1,
        "William Shakespeare":2,
        "Kay Ryan":3,
        "Emily Dickinson":4,
        "Alfred, Lord Tennyson":5,
        "Rae Armantrout":6,
        "Yusef Komunyakaa":7,
        "Samuel Menashe":8,
        "William Wordsworth":9,
        "Rupi Kaur":10
    }

    return labels[poet]

def label2poet(label):
    poets = {
        0: "John Ashbery",
        1: "William Butler Yeats",
        2: "William Shakespeare",
        3: "Kay Ryan",
        4: "Emily Dickinson",
        5: "Alfred, Lord Tennyson",
        6: "Rae Armantrout",
        7: "Yusef Komunyakaa",
        8: "Samuel Menashe",
        9: "William Wordsworth",
        10: "Rupi Kaur"
    }

    return poets[label]

def train(model, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for i, batch in enumerate(train_loader):
        inputs = batch["inputs"]
        labels = batch["poet"]

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        outputs = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (preds == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    return avg_train_loss, avg_train_acc

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch["inputs"]
            labels = batch["poet"]

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            val_loss += loss.item()
            val_acc += (preds == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    return avg_val_loss, avg_val_acc

def test(model, test_loader):
    predictions = list()
    labels = list()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch["inputs"]
            label = batch["poet"]
            labels.append(label)

            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
            predictions.append(prediction)
        
    return torch.as_tensor(predictions), torch.as_tensor(labels)