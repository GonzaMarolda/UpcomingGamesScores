# Steam Game Score Predictor

## Description
This project aims to predict the user score calculated by averaging good and bad reviews.
It uses features like price, required age, tags and supported languages.

## Dataset
- Source: https://www.kaggle.com/datasets/artermiloff/steam-games-dataset/data
- Rows: 6,285 games
- Columns: 157 features
- Preprocessing:
- - Normalization.
- - One-hot encoding.
- - Added features: supports_english, for_mature_audiences.

## Model
- Architecture: Feed-forward neural network.
- Layers: 128 -> 64 -> 32 -> 1
- Activation: ReLU
- Loss: MSE
- Optimizer: SGD

## Results
Results can be found at trained_models/results, contains:
- Train loss, Val loss
- MSE, MAE
