import json
import pandas as pd
import ast
from preprocessing_utils import filter_games, normalize_data, add_additional_features, process_tags, process_publishers
import warnings
import globals
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

RAW_PATH = "../data/raw.csv"
PROCESSED_PATH = "../data/processed.csv"
EXPLORATION_PATH = "../data/exploration.csv"
METADATA_PATH = "../data/metadata.json"

raw_df = pd.read_csv(RAW_PATH)
print(f"[INFO] Games in raw data: {len(raw_df)}")

processed_output_cols = [
    "price_norm", 
    "is_free",
    "is_indie",
    "required_age_norm", 
    "year_norm", 
    "price_year",
    "pct_pos_total_norm", 
    "num_reviews_total_norm",
    "supports_english", 
    "supports_few_languages", 
    "supports_several_languages",
    "supports_many_languages",
    "for_mature_audiences"
]

raw_df = filter_games(raw_df)
raw_df = normalize_data(raw_df)

raw_df = process_tags(raw_df, globals.top_n_tags, globals.tags_columns_amount)
raw_df = process_publishers(raw_df, globals.top_n_publishers, globals.publishers_columns_amount)

raw_df = add_additional_features(raw_df)

# Build processed dataframe

tag_cols = [f"tag_{i+1}" for i in range(globals.tags_columns_amount)]
publisher_cols = [f"publisher_{i+1}" for i in range(globals.publishers_columns_amount)]
processed_df = raw_df[processed_output_cols + tag_cols + publisher_cols]

processed_df.to_csv(PROCESSED_PATH, index=False)
print(f"[INFO] Processed dataframe saved in {PROCESSED_PATH} with {processed_df.shape[0]} rows and {processed_df.shape[1]} columns")

# Update input columns metadata
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
with open(METADATA_PATH, "w") as f:
    column_names = processed_df.drop(columns=["pct_pos_total_norm"]).columns.tolist()
    metadata["input_columns"] = column_names
    json.dump(metadata, f, indent=4)

# Build exploration dataframe

exploration_output_cols = [
    "name", 
    "price",
    "publishers",
    "is_indie",
    "required_age", 
    "release_date", 
    "pct_pos_total", 
    "tags",
    "supports_english", 
    "supports_few_languages",
    "supports_several_languages",
    "supports_many_languages",
]
exploration_df = raw_df[exploration_output_cols]

exploration_df.to_csv(EXPLORATION_PATH, index=False)
print(f"[INFO] Exploration dataframe saved in {EXPLORATION_PATH} with {exploration_df.shape[0]} rows and {exploration_df.shape[1]} columns")
