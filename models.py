import torch

class DAN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, is_glove=False, output_size=11):
        super(DAN, self).__init__()
        self.is_glove = is_glove
        self.dan = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        if not self.is_glove:
            x = x[:, 1:, :]
        x_avg = torch.mean(x, dim=1)
        output = self.dan(x_avg)
        return output

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, is_glove=False, output_size=11, num_layers=1, bidirectional=False):
        super(LSTM, self).__init__()
        self.is_glove = is_glove
        self.lstm = torch.nn.Sequential(
            torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True),
        )
        self.dropout = torch.nn.Dropout(p=0)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if not self.is_glove:
            x = x[:, 1:, :]
        _, (hidden, cell) = self.lstm(x)
        final_hidden = hidden[-1]
        output = self.dropout(final_hidden)
        output = self.linear(output)
        return output


class FCNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=11):
        super(FCNN, self).__init__()
        self.fcnn = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = x[:, 0, :]
        output = self.fcnn(x)
        return output