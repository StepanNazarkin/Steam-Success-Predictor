from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from app.schema import GamePredictionRequest, GamePredictionResponse

TIER_MAPPING = {
    0: "Low",
    1: "Medium",
    2: "High",
}

MLB_CONFIGS = [
    ("genres", "genre", "mlb_genres"),
    ("categories", "cat", "mlb_categories"),
    ("full_audio_languages", "audio_lang", "mlb_full_audio_languages"),
    ("supported_languages", "lang", "mlb_supported_languages"),
]


def get_metadata(artifacts: dict[str, Any]) -> dict[str, list[str]]:
    metadata = {
        col: sorted(artifacts[encoder_key].classes_.tolist())
        for col, _, encoder_key in MLB_CONFIGS
    }
    return metadata


def _dedup_clean_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        s = str(item).strip()
        if s and s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def _lookup_entity_history(
    names: list[str],
    lookup: dict[str, dict[str, int]],
    prefix: str,
    prior_20k: float,
    prior_high: float,
    smoothing_m: float,
) -> dict[str, int | float]:
    # Deduplicate normalized entity keys so repeated names aren't double-counted
    unique_keys = list(
        dict.fromkeys(key for name in names if (key := str(name).strip().lower()))
    )
    matched = [lookup[key] for key in unique_keys if key in lookup]

    prev_games = sum(int(m["prev_games"]) for m in matched)
    prev_20k_count = sum(int(m["prev_20k_count"]) for m in matched)
    prev_high_count = sum(int(m["prev_high_count"]) for m in matched)

    if prev_games > 0:
        prev_20k_rate = prev_20k_count / prev_games
        prev_high_rate = prev_high_count / prev_games
    else:
        prev_20k_rate = np.nan
        prev_high_rate = np.nan

    smoothed_20k_rate = (prev_20k_count + smoothing_m * prior_20k) / (
        prev_games + smoothing_m
    )
    smoothed_high_rate = (prev_high_count + smoothing_m * prior_high) / (
        prev_games + smoothing_m
    )

    return {
        f"{prefix}_prev_games": prev_games,
        f"{prefix}_prev_20k_count": prev_20k_count,
        f"{prefix}_prev_high_count": prev_high_count,
        f"{prefix}_prev_20k_rate": prev_20k_rate,
        f"{prefix}_prev_high_rate": prev_high_rate,
        f"{prefix}_smoothed_20k_rate": smoothed_20k_rate,
        f"{prefix}_smoothed_high_rate": smoothed_high_rate,
    }


def _compute_engineered_features(
    raw_data: dict[str, Any], devs: list[str], pubs: list[str]
) -> dict[str, int | float]:
    price = float(raw_data["price"])
    short_desc = raw_data["short_description"]
    release_month = int(raw_data["release_month"])
    angle = 2.0 * np.pi * (release_month - 1) / 12.0

    dev_keys = {d.strip().lower() for d in devs if d.strip()}
    pub_keys = {p.strip().lower() for p in pubs if p.strip()}

    return {
        "is_free": int(price == 0.0),
        "log_price": float(np.log1p(max(price, 0.0))),
        "supported_languages_count": len(raw_data["supported_languages"]),
        "full_audio_languages_count": len(raw_data["full_audio_languages"]),
        "genres_count": len(raw_data["genres"]),
        "categories_count": len(raw_data["categories"]),
        "platforms_count": (
            int(raw_data["windows"]) + int(raw_data["mac"]) + int(raw_data["linux"])
        ),
        "store_completeness_score": (
            int(raw_data["has_header_image"])
            + int(raw_data["has_website"])
            + int(raw_data["has_support_url"])
            + int(raw_data["has_support_email"])
        ),
        "is_self_published": int(bool(dev_keys and pub_keys and (dev_keys & pub_keys))),
        "description_char_len": len(short_desc),
        "description_word_count": len(short_desc.split()),
        "release_month_sin": float(np.sin(angle)),
        "release_month_cos": float(np.cos(angle)),
    }


def _transform_multilabel(df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    encoded_dfs = []

    for col, prefix, encoder_key in MLB_CONFIGS:
        mlb = artifacts[encoder_key]
        known_classes = set(mlb.classes_)

        cleaned_series = df[col].apply(
            lambda items: (
                [x for x in items if x in known_classes]
                if isinstance(items, list)
                else []
            )
        )

        transformed = mlb.transform(cleaned_series).astype(np.int8)
        cols = [f"{prefix}_{c}" for c in mlb.classes_]
        encoded_dfs.append(pd.DataFrame(transformed, columns=cols))

    return pd.concat(encoded_dfs, axis=1)


def _transform_tfidf(df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    tfidf = artifacts["tfidf"]
    tfidf_cols = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]

    transformed = tfidf.transform(df["short_description"]).toarray()
    return pd.DataFrame(transformed, columns=tfidf_cols)


def _calibrated_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(float(temperature), 1e-6)
    scaled = scaled - np.max(scaled)
    exp_scores = np.exp(scaled)
    return exp_scores / np.sum(exp_scores)


def predict_game_success(
    request: GamePredictionRequest,
    model: xgb.XGBClassifier,
    artifacts: dict[str, Any],
) -> GamePredictionResponse:
    raw_data = request.model_dump()

    devs = _dedup_clean_strings(raw_data["developers"])
    pubs = _dedup_clean_strings(raw_data["publishers"])
    leaky_categories = set(artifacts.get("leaky_categories", ()))

    raw_data["developers"] = devs
    raw_data["publishers"] = pubs
    raw_data["genres"] = _dedup_clean_strings(raw_data["genres"])
    raw_data["categories"] = [
        c
        for c in _dedup_clean_strings(raw_data["categories"])
        if c not in leaky_categories
    ]
    raw_data["supported_languages"] = _dedup_clean_strings(
        raw_data["supported_languages"]
    )
    raw_data["full_audio_languages"] = _dedup_clean_strings(
        raw_data["full_audio_languages"]
    )
    raw_data["short_description"] = raw_data.get("short_description") or ""

    prior_20k = float(artifacts["prior_20k"])
    prior_high = float(artifacts["prior_high"])
    smoothing_m = float(artifacts["smoothing_m"])

    raw_data.update(
        _lookup_entity_history(
            devs, artifacts["dev_history"], "dev", prior_20k, prior_high, smoothing_m
        )
    )
    raw_data.update(
        _lookup_entity_history(
            pubs, artifacts["pub_history"], "pub", prior_20k, prior_high, smoothing_m
        )
    )
    raw_data.update(_compute_engineered_features(raw_data, devs, pubs))

    df = pd.DataFrame([raw_data])
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    base_features = artifacts["base_features"]
    mlb_df = _transform_multilabel(df, artifacts)
    tfidf_df = _transform_tfidf(df, artifacts)

    X_final = pd.concat(
        [df[base_features].astype(np.float32), mlb_df, tfidf_df], axis=1
    )
    X_final = X_final.reindex(columns=artifacts["feature_columns"], fill_value=0)

    logits = model.predict(X_final, output_margin=True)[0]
    temperature = float(artifacts.get("temperature", 1.0))
    probs = _calibrated_softmax(logits, temperature)
    predicted_class = int(np.argmax(probs))

    probabilities = {
        TIER_MAPPING[idx]: round(float(prob), 4) for idx, prob in enumerate(probs)
    }
    prob_20k_plus = round(probabilities["Medium"] + probabilities["High"], 4)
    predicted_tier = TIER_MAPPING[predicted_class]

    return GamePredictionResponse(
        success_tier=predicted_tier,
        prob_20k_plus=prob_20k_plus,
        probabilities=probabilities,
        summary=(
            f"Most likely tier: {predicted_tier} ({probabilities[predicted_tier]:.1%}). "
            f"Estimated chance of reaching 20k+ owners: {prob_20k_plus:.1%}."
        ),
    )
