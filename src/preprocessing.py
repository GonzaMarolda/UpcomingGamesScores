import pandas as pd
import ast
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

RAW_PATH = "../data/raw.csv"
PROCESSED_PATH = "../data/processed.csv"
EXPLORATION_PATH = "../data/exploration.csv"

raw_df = pd.read_csv(RAW_PATH)
print(f"[INFO] Games in raw data: {len(raw_df)}")

# Filter by date (2015–2025) 
raw_df["release_date"] = pd.to_datetime(raw_df["release_date"], errors="coerce")
raw_df = raw_df.dropna(subset=["release_date"])
raw_df = raw_df[(raw_df["release_date"].dt.year >= 2015) & (raw_df["release_date"].dt.year <= 2025)]
print(f"[INFO] Games after filtering by date (2015–2025): {len(raw_df)}")

# Filter out lesser known games (less than 10k reviews)
raw_df = raw_df[raw_df["num_reviews_total"] >= 2000]
print(f"[INFO] Games after filtering by reviews count (x >= 2,000): {len(raw_df)}")

# Filter out games without tags
raw_df = raw_df[raw_df["tags"].notna() & (raw_df["tags"] != "[]")]
print(f"[INFO] Games after filtering out games without tags: {len(raw_df)}")

# Normalize year [2015–2030] to [0–1]
raw_df["year_norm"] = (raw_df["release_date"].dt.year - 2015) / (2030 - 2015)
# Normalize price [0–100] to [0–1]
raw_df["price_norm"] = raw_df["price"].clip(0, 80) / 80
# Normalize required_age [0–21] to [0–1]
raw_df["required_age_norm"] = raw_df["required_age"].clip(0, 21) / 21
# Normalize pct_pos_total [0–100] to [0–1]
raw_df["pct_pos_total_norm"] = raw_df["pct_pos_total"].clip(0, 100) / 100

# Top 150 tags and one-hot encoding
top_n_tags = 300
all_tags = raw_df["tags"].apply(lambda tags: ast.literal_eval(tags)).explode()
top_tags = all_tags.value_counts().head(top_n_tags).index
print(f"[INFO] Total tags found: {all_tags.nunique()}")
print(f"[INFO] Using top {top_n_tags} most common")

for tag in top_tags:
    col_name = f"tag_{tag}".replace(" ", "_").replace("-", "_")
    raw_df[col_name] = raw_df["tags"].apply(lambda game_tags: 1 if tag in game_tags else 0)

# Count supported languages and normalize [0–15] to [0–1]
raw_df["lang_count"] = raw_df["supported_languages"].apply(lambda langs: len(ast.literal_eval(langs)))
raw_df["lang_norm"] = raw_df["lang_count"].clip(0, 15) / 15

# Add supports_english,and is_for_adults columns
raw_df["supports_english"] = raw_df["supported_languages"].apply(
    lambda langs: "English" in ast.literal_eval(langs)
)
raw_df["for_mature_audiences"] = raw_df["required_age"].apply(lambda age: int(age) >= 17)

# Build processed dataframe
output_cols = [
    "price_norm", 
    "required_age_norm", 
    "year_norm", 
    "lang_norm", 
    "pct_pos_total_norm", 
    "supports_english", 
    "for_mature_audiences"
]
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
    "lang_count"
]
exploration_df = raw_df[output_cols]

exploration_df.to_csv(EXPLORATION_PATH, index=False)
print(f"[INFO] Exploration dataframe saved in {EXPLORATION_PATH} with {exploration_df.shape[0]} rows and {exploration_df.shape[1]} columns")