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
    ("tags", "tag", "mlb_tags"),
    ("full_audio_languages", "audio_lang", "mlb_full_audio_languages"),
    ("supported_languages", "lang", "mlb_supported_languages"),
]


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
    raw_data["primary_developer"] = raw_data["primary_developer"].strip() or "Unknown"
    raw_data["primary_publisher"] = raw_data["primary_publisher"].strip() or "Unknown"

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
    predicted_tier = TIER_MAPPING[predicted_class]
    confidence = probabilities[predicted_tier]

    return GamePredictionResponse(
        success_tier=predicted_tier,
        probabilities=probabilities,
        summary=f"Predicted {predicted_tier} success tier with {confidence:.1%} confidence.",
    )
