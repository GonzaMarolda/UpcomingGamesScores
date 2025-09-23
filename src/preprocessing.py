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
    "is_free",
    "required_age_norm", 
    "year_norm", 
    "price_year",
    "pct_pos_total_norm", 
    "is_really_good",
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

# Create tags index columns for embedding
top_n_tags = 300
tags_columns_amount = 20

raw_df["tags_list"] = raw_df["tags"].apply(lambda tags: list(ast.literal_eval(tags).keys()))

all_tags = raw_df["tags_list"].explode()
top_tags = all_tags.value_counts().head(top_n_tags).index.tolist()

tag_to_index = {tag: idx+1 for idx, tag in enumerate(top_tags)}  # 1 is most popular tag. 0 is reserved for padding

def assign_tag_indices(tags):
    filtered = [t for t in tags if t in tag_to_index]
    filtered.sort(key=lambda t: tag_to_index[t])

    top_tags_for_game = filtered[:tags_columns_amount]
    indices = [tag_to_index[t] for t in top_tags_for_game]

    # Add padding if less tags than tags_columns_amount
    while len(indices) < tags_columns_amount:
        indices.append(0)
    return indices

tags_index_rows = raw_df["tags_list"].apply(assign_tag_indices)
tags_df = pd.DataFrame(
    tags_index_rows.tolist(),
    columns=[f"tag_{i+1}" for i in range(tags_columns_amount)]
)

raw_df = raw_df.reset_index(drop=True) # Reset index to align with tags_df
raw_df = pd.concat([raw_df, tags_df.astype("int64")], axis=1)

#Top 300 tags and multi-hot encoding
# top_n_tags = 300
# all_tags = raw_df["tags"].apply(lambda tags: ast.literal_eval(tags)).explode()
# top_tags = all_tags.value_counts().head(top_n_tags).index
# print(f"[INFO] Total tags found: {all_tags.nunique()}")
# print(f"[INFO] Using top {top_n_tags} most common")
# for tag in top_tags:
#     col_name = f"tag_{tag}".replace(" ", "_").replace("-", "_")
#     raw_df[col_name] = raw_df["tags"].apply(lambda game_tags: 1 if tag in game_tags else 0)

# Build processed dataframe
tag_cols = [f"tag_{i+1}" for i in range(tags_columns_amount)]
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