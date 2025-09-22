import pandas as pd
import ast

def filter_games(df):
    # Filter by date (2015–2025) 
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])
    df = df[(df["release_date"].dt.year >= 2015) & (df["release_date"].dt.year <= 2025)]
    print(f"[INFO] Games after filtering by date (2015–2025): {len(df)}")

    # Filter out lesser known games (less than 10k reviews)
    df = df[df["num_reviews_total"] >= 1000]
    print(f"[INFO] Games after filtering by reviews count (x >= 2,000): {len(df)}")

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
    df["is_really_good"] = df["pct_pos_total"] >= 95

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

    return df