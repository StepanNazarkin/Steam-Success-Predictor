import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "games.json"
PROCESSED_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "steam_games_processed_v2.parquet"
)

MAX_RELEASE_YEAR = 2024

POST_RELEASE_AND_UNUSED_COLUMNS = [
    "appid",
    "name",
    "metacritic_url",
    "reviews",
    "metacritic_score",
    "recommendations",
    "user_score",
    "score_rank",
    "positive",
    "negative",
    "average_playtime_forever",
    "average_playtime_2weeks",
    "median_playtime_forever",
    "median_playtime_2weeks",
    "discount",
    "peak_ccu",
    "dlc_count",
    "notes",
    "packages",
    "detailed_description",
    "about_the_game",
    "movies",
    "tags",
]

LEAKY_CATEGORIES = {
    # Post-release Valve-gated features
    "Steam Trading Cards",
    "Profile Features Limited",
    "SteamVR Collectibles",
    # Late-2024 feature
    "Steam Timeline",
    # May-2025 Steam Accessibility categories (retroactive survivorship leakage on 2021-2024)
    "Adjustable Difficulty",
    "Adjustable Text Size",
    "Camera Comfort",
    "Chat Speech-to-text",
    "Chat Text-to-speech",
    "Color Alternatives",
    "Custom Volume Controls",
    "Keyboard Only Option",
    "Mouse Only Option",
    "Narrated Game Menus",
    "Playable without Timed Input",
    "Save Anytime",
    "Stereo Sound",
    "Subtitle Options",
    "Surround Sound",
    "Touch Only Option",
}

TIER_MAPPING = {"Low": 0, "Medium": 1, "High": 2}


def get_clean_tier(owners_str: str) -> str:
    if owners_str == "0 - 20000":
        return "Low"
    if owners_str in ("20000 - 50000", "50000 - 100000"):
        return "Medium"
    return "High"


def clean_languages_list(raw_value: object) -> list[str]:
    if isinstance(raw_value, list):
        text = ", ".join(map(str, raw_value))
    elif isinstance(raw_value, str):
        text = raw_value
    else:
        return []

    text = (
        text.replace("&amp;", "")
        .replace("amp;", "")
        .replace("lt;", "<")
        .replace("gt;", ">")
    )
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("\r", ",").replace("\n", ",")

    cleaned_list = [lang.strip() for lang in text.split(",") if lang.strip()]

    final_list = []
    for lang in cleaned_list:
        if "all with full audio support" in lang:
            final_list.append("Japanese")
        elif lang.startswith("category_"):
            continue
        else:
            final_list.append(lang)

    return sorted(set(final_list))


def prepare_data(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    print(f"Loading raw dataset from {raw_path}...")
    df = pd.read_json(raw_path, orient="index")
    df.index.name = "appid"
    df = df.reset_index()
    print(f"Raw dataset rows: {len(df)}")

    drop_cols = [c for c in POST_RELEASE_AND_UNUSED_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols).copy()

    df = df[df["estimated_owners"] != "0 - 0"].copy()

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"]).copy()
    df["release_year"] = df["release_date"].dt.year.astype(int)
    df["release_month"] = df["release_date"].dt.month.astype(int)
    df = df[df["release_year"] <= MAX_RELEASE_YEAR].copy()

    df["target_num"] = (
        df["estimated_owners"].apply(get_clean_tier).map(TIER_MAPPING).astype(int)
    )
    df = df.drop(columns=["release_date", "estimated_owners"])

    df["screenshots_count"] = df["screenshots"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    df = df.drop(columns=["screenshots"])

    for col in ("header_image", "website", "support_url", "support_email"):
        df[f"has_{col}"] = (df[col].fillna("").astype(str).str.strip() != "").astype(
            int
        )
    df = df.drop(columns=["header_image", "website", "support_url", "support_email"])

    for col in ("windows", "mac", "linux"):
        df[col] = df[col].astype(int)

    df["categories"] = df["categories"].apply(
        lambda cats: (
            [c for c in cats if c not in LEAKY_CATEGORIES]
            if isinstance(cats, list)
            else []
        )
    )
    df["genres"] = df["genres"].apply(lambda x: x if isinstance(x, list) else [])
    df["developers"] = df["developers"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    df["publishers"] = df["publishers"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    df["supported_languages"] = df["supported_languages"].apply(clean_languages_list)
    df["full_audio_languages"] = df["full_audio_languages"].apply(clean_languages_list)
    df["short_description"] = df["short_description"].fillna("").astype(str)

    df = df.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(
        f"Saved {len(df)} processed rows ({df['release_year'].min()}-{df['release_year'].max()}) "
        f"to {output_path}"
    )
    print(
        "Rows in 2021-2024 training window:",
        int(df["release_year"].between(2021, 2024).sum()),
    )
    return df


if __name__ == "__main__":
    prepare_data()
