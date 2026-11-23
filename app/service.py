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


def _lookup_entity_history(
    names: list[str],
    lookup: dict[str, dict[str, int | float]],
    prefix: str,
) -> dict[str, int | float]:
    matched = [
        lookup[key]
        for name in names
        if (key := str(name).strip().lower()) and key in lookup
    ]

    if not matched:
        return {
            f"{prefix}_prev_games": 0,
            f"{prefix}_prev_20k_count": 0,
            f"{prefix}_prev_high_count": 0,
            f"{prefix}_prev_20k_rate": np.nan,
            f"{prefix}_prev_high_rate": np.nan,
        }

    return {
        f"{prefix}_prev_games": max(int(m["prev_games"]) for m in matched),
        f"{prefix}_prev_20k_count": max(int(m["prev_20k_count"]) for m in matched),
        f"{prefix}_prev_high_count": max(int(m["prev_high_count"]) for m in matched),
        f"{prefix}_prev_20k_rate": max(float(m["prev_20k_rate"]) for m in matched),
        f"{prefix}_prev_high_rate": max(float(m["prev_high_rate"]) for m in matched),
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

        transformed = mlb.transform(cleaned_series)
        cols = [f"{prefix}_{c}" for c in mlb.classes_]
        encoded_dfs.append(pd.DataFrame(transformed, columns=cols))

    return pd.concat(encoded_dfs, axis=1)


def _transform_target_encoding(
    df: pd.DataFrame, artifacts: dict[str, Any]
) -> pd.DataFrame:
    te_encoder = artifacts["target_encoder"]
    te_cols = ["primary_developer", "primary_publisher"]

    transformed = te_encoder.transform(df[te_cols])
    feature_names = te_encoder.get_feature_names_out(te_cols)

    return pd.DataFrame(transformed, columns=feature_names)


def _transform_tfidf(df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    tfidf = artifacts["tfidf"]
    tfidf_cols = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]

    transformed = tfidf.transform(df["short_description"]).toarray()
    return pd.DataFrame(transformed, columns=tfidf_cols)


def predict_game_success(
    request: GamePredictionRequest,
    model: xgb.XGBClassifier,
    artifacts: dict[str, Any],
) -> GamePredictionResponse:
    raw_data = request.model_dump()

    devs = [d.strip() for d in raw_data["developers"] if d.strip()]
    pubs = [p.strip() for p in raw_data["publishers"] if p.strip()]

    raw_data["primary_developer"] = devs[0] if devs else "Unknown"
    raw_data["primary_publisher"] = pubs[0] if pubs else "Unknown"
    raw_data["short_description"] = raw_data.get("short_description") or ""

    raw_data.update(_lookup_entity_history(devs, artifacts["dev_history"], "dev"))
    raw_data.update(_lookup_entity_history(pubs, artifacts["pub_history"], "pub"))

    df = pd.DataFrame([raw_data])
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    base_features = artifacts["base_features"]
    mlb_df = _transform_multilabel(df, artifacts)
    te_df = _transform_target_encoding(df, artifacts)
    tfidf_df = _transform_tfidf(df, artifacts)

    X_final = pd.concat([df[base_features], mlb_df, te_df, tfidf_df], axis=1)
    X_final = X_final.reindex(columns=artifacts["feature_columns"], fill_value=0)

    probs = model.predict_proba(X_final)[0]
    predicted_class = int(np.argmax(probs))

    probabilities = {
        TIER_MAPPING[idx]: round(float(prob), 4) for idx, prob in enumerate(probs)
    }
    prob_20k_plus = round(float(probs[1] + probs[2]), 4)
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