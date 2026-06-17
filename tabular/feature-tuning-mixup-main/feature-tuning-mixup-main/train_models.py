from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim

def train_mlp(model, X, y):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.long)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return model


def train_sklearn_models(X, y):
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier()

    lr.fit(X, y)
    rf.fit(X, y)

    return lr, rf