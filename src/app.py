import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import GameScorePredictor
from preprocessing_utils import get_top_tags, get_top_publishers, preprocess_input
import torch
import torch.nn as nn
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

with open('../data/metadata.json', 'r') as f:
    metadata = json.load(f)
    input_columns = metadata.get("input_columns", [])
model = GameScorePredictor(input_columns)
model.load_state_dict(torch.load('../trained_models/models/gamescore_model_v11.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = camel_to_snake(request.get_json())
        if not is_data_valid(data):
            return jsonify({"error": "Invalid input data"}), 400
        
        print("Data is valid:")
        print(data)
        
        features = preprocess_input(data)
        print("Features obtained for prediction")

        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction = model(input_tensor).item()

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

@app.route('/tags', methods=['GET'])
def get_tags():
    top_tags = get_top_tags()
    return jsonify({"tags": top_tags})

@app.route('/publishers', methods=['GET'])
def get_publishers():
    top_publishers = get_top_publishers()
    return jsonify({"publishers": top_publishers + ["Other"]})

def is_data_valid(data): 
    tags = data.get("tags", None)
    if tags is None or not isinstance(tags, list) or tags == []:
        return False
    top_tags = get_top_tags()
    for tag in tags:
        if tag not in top_tags:
            return False
        
    publishers = data.get("publishers", None)
    if publishers is None or not isinstance(publishers, list) or publishers == []:
        return False
    top_publishers = get_top_publishers()
    for pub in publishers:
        if pub not in top_publishers and pub != "Other":
            return False

    release_year = data.get("release_year", None)
    if release_year is None or not release_year.is_integer() or release_year < 2015 or release_year > 2035:
        return False

    price = data.get("price", None)
    if price is None or not (isinstance(price, int) or isinstance(price, float)):
        return False

    required_age = data.get("required_age", None)
    if required_age is None or not required_age.is_integer() or required_age < 0 or required_age > 21:
        return False

    is_indie = data.get("is_indie", None)
    if is_indie is None or not isinstance(is_indie, bool):
        return False

    supports_english = data.get("supports_english", None)
    if supports_english is None or not isinstance(supports_english, bool):
        return False

    supported_languages_amount = data.get("supported_languages_amount", None)
    if supported_languages_amount is None or not supported_languages_amount.is_integer() or supported_languages_amount < 0:
        return False

    return True

def camel_to_snake(data):
    snake_data = {}
    for key, value in data.items():
        snake_key = ''.join(['_'+c.lower() if c.isupper() else c for c in key]).lstrip('_')
        snake_data[snake_key] = value
    return snake_data

# Run the app
if __name__ == '__main__':
    app.run(debug=True)