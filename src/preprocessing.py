import pandas as pd
import ast
from preprocessing_utils import filter_games, normalize_data, add_additional_features
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

RAW_PATH = "../data/raw.csv"
PROCESSED_PATH = "../data/processed.csv"
EXPLORATION_PATH = "../data/exploration.csv"

raw_df = pd.read_csv(RAW_PATH)
print(f"[INFO] Games in raw data: {len(raw_df)}")

output_cols = [
    "price_norm", 
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
raw_df = add_additional_features(raw_df)

#Top 300 tags and one-hot encoding
top_n_tags = 300
all_tags = raw_df["tags"].apply(lambda tags: ast.literal_eval(tags)).explode()
top_tags = all_tags.value_counts().head(top_n_tags).index
print(f"[INFO] Total tags found: {all_tags.nunique()}")
print(f"[INFO] Using top {top_n_tags} most common")
for tag in top_tags:
    col_name = f"tag_{tag}".replace(" ", "_").replace("-", "_")
    raw_df[col_name] = raw_df["tags"].apply(lambda game_tags: 1 if tag in game_tags else 0)

# Build processed dataframe
tag_cols = [f"tag_{tag}".replace(" ", "_").replace("-", "_") for tag in top_tags]
processed_df = raw_df[output_cols + tag_cols]

processed_df.to_csv(PROCESSED_PATH, index=False)
print(f"[INFO] Processed dataframe saved in {PROCESSED_PATH} with {processed_df.shape[0]} rows and {processed_df.shape[1]} columns")

# Build exploration dataframe
output_cols = [
    "name", 
    "price",
    "required_age", 
    "release_date", 
    "pct_pos_total", 
    "tags",
    "supports_english", 
    "supports_few_languages",
    "supports_several_languages",
    "supports_many_languages",
]
exploration_df = raw_df[output_cols]

exploration_df.to_csv(EXPLORATION_PATH, index=False)
print(f"[INFO] Exploration dataframe saved in {EXPLORATION_PATH} with {exploration_df.shape[0]} rows and {exploration_df.shape[1]} columns")