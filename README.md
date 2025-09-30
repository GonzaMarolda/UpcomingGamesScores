# Steam Game Score Predictor

## Description
This project aims to predict the user score calculated by averaging good and bad reviews.
It uses features like price, required age, tags and supported languages.

## Dataset
- Source: https://www.kaggle.com/datasets/artermiloff/steam-games-dataset/data
- Rows: 89,618 games
- Columns: 186 features
- Preprocessing:
- - Normalization
- - Filtering
- - Embedding
- - Additional features

## Model
- Architecture: Feed-forward neural network.
- Layers: 256 -> 128 -> 64 -> 32 -> 1
- Activation: ReLU
- Loss: MSE
- Optimizer: SGD

## Results
Results can be found at trained_models/results, contains:
- Train loss, Val loss
- MSE, MAE

## Usage
The model should be used from the web app: https://github.com/GonzaMarolda/GameScorePredictorApp
- User input: price, required age, is indie?, supports english?, supported languages amount, tags, and publishers
