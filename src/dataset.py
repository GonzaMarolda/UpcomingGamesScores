import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

CSV_PATH = "./data/processed.csv"

def get_dataloaders(batch_size=32, test_size=0.2):
    df = pd.read_csv(CSV_PATH)

    X = df.drop(columns=["pct_pos_total_norm"]).values.astype("float32") # Pytorch works better with float32
    y = df["pct_pos_total_norm"].values.astype("float32").reshape(-1, 1) # Reshape for as many rows as needed and 1 column
    relevance = df["num_reviews_total_norm"].values.astype("float32")

    X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(X, y, relevance, test_size=test_size)
    test_filter = r_test >= 0.3
    X_test = X_test[test_filter]
    y_test = y_test[test_filter]

    weights_train = torch.tensor(r_train) ** 0.2
    sampler_train = WeightedRandomSampler(weights=weights_train, num_samples=len(weights_train), replacement=True)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get input column names
    column_names = df.drop(columns=["pct_pos_total_norm"]).columns.tolist()

    return train_loader, test_loader, column_names