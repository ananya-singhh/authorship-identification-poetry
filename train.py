import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)

import torch
torch.set_default_tensor_type(torch.FloatTensor)
torch.use_deterministic_algorithms(True)
torch.manual_seed(577)

import nltk
import matplotlib.pyplot as plt

import pickle

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
'''
NOTE: Do not change any of the statements above regarding random/numpy/pytorch.
You can import other built-in libraries (e.g. collections) or pre-specified external libraries
such as pandas, nltk and gensim below. 
Also, if you'd like to declare some helper functions, please do so in utils.py and
change the last import statement below.
'''

import gensim.downloader as api

from models import DAN, FCNN, LSTM
from utils import PoetryDataset, train, validate, test

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--neural_arch', choices=['dan', 'lstm', 'lr', 'fcnn'], default='dan', type=str)
    parser.add_argument('--word_embs', choices=['bert', 'glove'], default='bert', type=str)
    parser.add_argument('--gen_plot', choices=[True, False], default=False, type=bool())

    args = parser.parse_args()
    
    embeddings = None
    input_size = 0
    is_glove = None
    if args.word_embs == "glove":
        pkl = open("./data/glove_embeddings.pkl", "rb")
        embeddings = pickle.load(pkl)
        input_size = 50
        is_glove = True
    else:
        pkl = open("./data/bert_embeddings.pkl", "rb")
        embeddings = pickle.load(pkl)
        input_size = 768
        is_glove = False
    
    train_data = PoetryDataset("train", embeddings, is_glove=is_glove)
    test_data = PoetryDataset("test", embeddings, is_glove=is_glove)

    model = None
    hidden_size = 0
    learning_rate = 0
    num_epochs = 0
    if args.neural_arch == "dan":
        if is_glove:
            hidden_size = 60
            learning_rate = 1e-4
            num_epochs = 500
        else:
            hidden_size = 35
            learning_rate = 5e-5
            num_epochs = 250
        model = DAN(input_size, hidden_size, is_glove=is_glove)
    elif args.neural_arch == "lstm":
        if is_glove:
            hidden_size = 35
            learning_rate = 1e-5
            num_epochs = 1000
        else:
            hidden_size = 10
            learning_rate = 5e-5
            num_epochs = 210
        model = LSTM(input_size, hidden_size, is_glove=is_glove)
    elif args.neural_arch == "fcnn":
        hidden_size = 50
        learning_rate = 5e-5
        num_epochs = 150
        model = FCNN(input_size, hidden_size)

    batch_size = 1
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    beta1 = 0.9
    beta2 = 0.999
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
    criterion = torch.nn.CrossEntropyLoss()

    generate_plot = args.gen_plot

    if generate_plot:
        writer = SummaryWriter("runs/"+args.neural_arch+"-lr="+str(learning_rate)+"-hd"+str(hidden_size)+"-"+args.word_embs)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, test_loader, criterion)

        print(f'Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f} - Training Acc: {train_acc:.4f} - Validation Acc: {val_acc:.4f}')
        
        if generate_plot:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/test", val_acc, epoch)
    
    predictions, labels = test(model, test_loader)
    print("preds", predictions)
    print("labels", labels)

    f1score = MulticlassF1Score(num_classes=11)
    score = f1score(predictions, labels)
    print("score", score)

    confusion_matrix = MulticlassConfusionMatrix(num_classes=11)
    matrix = confusion_matrix(predictions, labels)
    print("matrix", matrix)

    if generate_plot:
        writer.close()
