import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_JOBS = -1

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "steam_games_processed.json"
MODELS_DIR = BASE_DIR / "models"

BASE_FEATURES = [
    "required_age",
    "price",
    "windows",
    "mac",
    "linux",
    "achievements",
    "release_month",
    "market_competition",
    "screenshots_count",
    "has_header_image",
    "has_website",
    "has_support_url",
    "has_support_email",
]

BEST_XGB_PARAMS = {
    "n_estimators": 224,
    "learning_rate": 0.0651641427774416,
    "max_depth": 7,
    "min_child_weight": 14,
    "subsample": 0.865423045790244,
    "colsample_bytree": 0.5074318643962501,
    "reg_alpha": 0.18684985740759733,
    "reg_lambda": 0.4659707668127148,
    "objective": "multi:softprob",
    "num_class": 3,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}


def prepare_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["primary_developer"] = df["developers"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown"
    )
    df["primary_publisher"] = df["publishers"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown"
    )
    df["short_description"] = df["short_description"].fillna("").astype(str)
    return df


def _encode_single_multilabel(
    train_series: pd.Series,
    test_series: pd.Series,
    prefix: str,
    top_k: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, MultiLabelBinarizer]:
    train, test = train_series.copy(), test_series.copy()

    if top_k:
        counts = train.explode().value_counts()
        top_classes = set(counts.head(top_k).index)
        train = train.apply(
            lambda items: (
                [x for x in items if x in top_classes]
                if isinstance(items, list)
                else []
            )
        )
        test = test.apply(
            lambda items: (
                [x for x in items if x in top_classes]
                if isinstance(items, list)
                else []
            )
        )

    mlb = MultiLabelBinarizer()
    train_raw = mlb.fit_transform(train)
    cols = [f"{prefix}_{c}" for c in mlb.classes_]

    return (
        pd.DataFrame(train_raw, columns=cols),
        pd.DataFrame(mlb.transform(test), columns=cols),
        mlb,
    )


def apply_multilabel_encoding(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    configs = [
        ("genres", "genre", None),
        ("categories", "cat", None),
        ("tags", "tag", 100),
        ("full_audio_languages", "audio_lang", 30),
        ("supported_languages", "lang", 30),
    ]

    train_dfs, test_dfs, encoders = [], [], {}

    for col, prefix, top_k in configs:
        tr_enc, te_enc, mlb = _encode_single_multilabel(
            df_train[col], df_test[col], prefix, top_k
        )
        train_dfs.append(tr_enc)
        test_dfs.append(te_enc)
        encoders[f"mlb_{col}"] = mlb

    return pd.concat(train_dfs, axis=1), pd.concat(test_dfs, axis=1), encoders


def apply_target_encoding(
    df_train: pd.DataFrame, y_train: pd.Series, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, TargetEncoder]:
    te_safe = TargetEncoder(smooth="auto", cv=5, random_state=RANDOM_STATE)
    te_cols = ["primary_developer", "primary_publisher"]

    dev_pub_tr_raw = te_safe.fit_transform(df_train[te_cols], y_train)
    feature_names = te_safe.get_feature_names_out(te_cols)

    dev_pub_train = pd.DataFrame(dev_pub_tr_raw, columns=feature_names)
    dev_pub_test = pd.DataFrame(
        te_safe.transform(df_test[te_cols]), columns=feature_names
    )

    return dev_pub_train, dev_pub_test, te_safe


def apply_nlp_tfidf(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, TfidfVectorizer]:
    tfidf = TfidfVectorizer(max_features=300, stop_words="english")
    tfidf.fit(df_train["short_description"])
    tfidf_cols = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]

    tfidf_train = pd.DataFrame(
        tfidf.transform(df_train["short_description"]).toarray(), columns=tfidf_cols
    )
    tfidf_test = pd.DataFrame(
        tfidf.transform(df_test["short_description"]).toarray(), columns=tfidf_cols
    )

    return tfidf_train, tfidf_test, tfidf


def evaluate_and_log(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    print("\n=== FINAL MODEL TEST METRICS ===")
    print(f"Accuracy:         {acc:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    print(f"ROC AUC (Macro):  {roc_auc:.4f}")


def train() -> None:
    print(f"Loading processed dataset from {DATA_PATH}...")
    df = pd.read_json(DATA_PATH)

    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["target_num"]
    )
    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_full_train["target_num"]
    y_test = df_test["target_num"]

    df_full_train = prepare_raw_columns(df_full_train.drop(columns=["target_num"]))
    df_test = prepare_raw_columns(df_test.drop(columns=["target_num"]))

    print("Running feature engineering pipelines...")
    mlb_tr, mlb_te, mlb_encoders = apply_multilabel_encoding(df_full_train, df_test)
    te_tr, te_te, te_encoder = apply_target_encoding(df_full_train, y_train, df_test)
    tfidf_tr, tfidf_te, tfidf_vectorizer = apply_nlp_tfidf(df_full_train, df_test)

    X_train_final = pd.concat(
        [df_full_train[BASE_FEATURES], mlb_tr, te_tr, tfidf_tr], axis=1
    )
    X_test_final = pd.concat([df_test[BASE_FEATURES], mlb_te, te_te, tfidf_te], axis=1)
    print(f"Final training matrix shape: {X_train_final.shape}")

    print("Training final XGBoost Classifier...")
    sample_weights = np.sqrt(compute_sample_weight("balanced", y_train))
    model = XGBClassifier(**BEST_XGB_PARAMS)
    model.fit(X_train_final, y_train, sample_weight=sample_weights)

    evaluate_and_log(model, X_test_final, y_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = MODELS_DIR / "best_xgboost.json"
    artifacts_path = MODELS_DIR / "preprocessing_artifacts.joblib"

    model.save_model(model_path)

    artifacts = {
        "tfidf": tfidf_vectorizer,
        "target_encoder": te_encoder,
        **mlb_encoders,
        "base_features": BASE_FEATURES,
        "feature_columns": X_train_final.columns.tolist(),
    }
    joblib.dump(artifacts, artifacts_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Artifacts saved to: {artifacts_path}")


if __name__ == "__main__":
    train()
