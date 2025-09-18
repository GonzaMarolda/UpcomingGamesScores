import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

CSV_PATH = "../data/processed.csv"

def get_dataloaders(batch_size=32, test_size=0.2):
    df = pd.read_csv(CSV_PATH)

    X = df.drop(columns=["pct_pos_total_norm"]).values.astype("float32") # Pytorch works better with float32
    y = df["pct_pos_total_norm"].values.astype("float32").reshape(-1, 1) # Reshape for as many rows as needed and 1 column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X.shape[1] # number of features