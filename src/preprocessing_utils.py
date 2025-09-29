import pandas as pd
import ast

def filter_games(df):
    # Filter by date (2015–2025) 
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])
    df = df[(df["release_date"].dt.year >= 2015) & (df["release_date"].dt.year <= 2025)]
    print(f"[INFO] Games after filtering by date (2015–2025): {len(df)}")

    # Filter out lesser known games (less than 1000 reviews)
    df = df[df["num_reviews_total"] >= 1000]
    print(f"[INFO] Games after filtering by reviews count (x >= 1,000): {len(df)}")

    # Filter out games without tags
    df = df[df["tags"].notna() & (df["tags"] != "[]")]
    print(f"[INFO] Games after filtering out games without tags: {len(df)}")

    # Filter by goodness (pct_pos_total >= 70)
    df = df[df["pct_pos_total"] >= 70]
    print(f"[INFO] Games after filtering by goodness (pct_pos_total >= 70): {len(df)}")

    return df

def normalize_value(value, column):
    match column:
        case "year":
            return (value - 2015) / (2030 - 2015)
        case "price":
            return value.clip(0, 80) / 80
        case "required_age":
            return value.clip(0, 21) / 21
        case "pct_pos_total":
            return value.clip(70, 100) / 100
        case _:
            raise ValueError(f"Unknown column: {column}")

def normalize_data(df):
    # [2015–2030] to [0–1]
    df["year_norm"] = (df["release_date"].dt.year - 2015) / (2030 - 2015)
    # [0–100] to [0–1]
    df["price_norm"] = df["price"].clip(0, 80) / 80
    # [0–21] to [0–1]
    df["required_age_norm"] = df["required_age"].clip(0, 21) / 21
    # [70–100] to [0–1]
    df["pct_pos_total_norm"] = df["pct_pos_total"].clip(70, 100) / 100
    # [0-n] to [1000-25k]
    df["num_reviews_total_norm"] = df["num_reviews_total"].clip(1000, 25000) / 25000

    return df

def add_additional_features(df):
    df["for_mature_audiences"] = df["required_age"].apply(lambda age: int(age) >= 17)
    df["price_year"] = df["price_norm"] * df["year_norm"]
    df["is_free"] = df["price"] == 0

    df["supports_english"] = df["supported_languages"].apply(
        lambda langs: "English" in ast.literal_eval(langs)
    )
    df["supports_few_languages"] = df["supported_languages"].apply(
        lambda langs: len(ast.literal_eval(langs)) <= 3
    )
    df["supports_several_languages"] = df["supported_languages"].apply(
        lambda langs: 4 <= len(ast.literal_eval(langs)) <= 8
    )
    df["supports_many_languages"] = df["supported_languages"].apply(
        lambda langs: len(ast.literal_eval(langs)) >= 9
    )

    # Considering that a game is indie when its publisher published less or equal than 2 games
    publisher_counts = df["publishers"].explode().value_counts().to_dict()
    df["is_indie"] = df["publishers"].apply(
        lambda pubs: all(publisher_counts.get(pub, 0) <= 2 for pub in pubs) if len(pubs) > 0 else True
    )
    return df

# Tag columns for multi-hot => embedding
def process_tags(df, top_n_tags, tags_columns_amount):
    df["tags_list"] = df["tags"].apply(lambda tags: list(ast.literal_eval(tags).keys()))

    all_tags = df["tags_list"].explode()
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

    tags_index_rows = df["tags_list"].apply(assign_tag_indices)
    tags_df = pd.DataFrame(
        tags_index_rows.tolist(),
        columns=[f"tag_{i+1}" for i in range(tags_columns_amount)]
    )

    df = df.reset_index(drop=True) # Reset index to align with tags_df
    df = pd.concat([df, tags_df.astype("int64")], axis=1)
    return df

# Publisher columns for multi-hot => embedding
def process_publishers(df, top_n_publishers, publishers_columns_amount):
    # Top publishers and grouping
    df["publishers"] = df["publishers"].apply(lambda pubs: ast.literal_eval(pubs))
    top_publishers = df["publishers"].explode().value_counts().head(top_n_publishers).index.tolist()
    df["filtered_publishers"] = df["publishers"].apply(
        lambda pubs: [pub if pub in top_publishers else "Other" for pub in pubs] if (len(pubs) > 0) else ["Other"]
    )

    publisher_to_index = {"Other": 1}
    for idx, pub in enumerate(top_publishers, start=2):
        publisher_to_index[pub] = idx

    def assign_publisher_indices(publishers):
        filtered = [p for p in publishers if p in publisher_to_index]
        filtered.sort(key=lambda p: publisher_to_index[p])
        top_publishers_for_game = filtered[:publishers_columns_amount]
        indices = [publisher_to_index[p] for p in top_publishers_for_game]

        # Padding if less than publishers_columns_amount
        while len(indices) < publishers_columns_amount:
            indices.append(0)

        return indices

    publishers_index_rows = df["filtered_publishers"].apply(assign_publisher_indices)
    publishers_df = pd.DataFrame(
        publishers_index_rows.tolist(),
        columns=[f"publisher_{i+1}" for i in range(publishers_columns_amount)]
    )

    df = df.reset_index(drop=True)
    df = pd.concat([df, publishers_df.astype("int64")], axis=1)
    return df