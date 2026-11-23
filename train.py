import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, TargetEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_JOBS = -1

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "steam_games_processed_v2.json"
MODELS_DIR = BASE_DIR / "models"

TRAIN_YEARS_RANGE = (2021, 2024)
TEST_YEAR = 2024

TIME_WEIGHT_MAP = {
    2021: 0.4,
    2022: 0.6,
    2023: 0.8,
    2024: 1.0,
}

LEAKY_CATEGORIES = {"Steam Trading Cards"}

HISTORY_FEATURES = [
    "dev_prev_games",
    "dev_prev_20k_count",
    "dev_prev_high_count",
    "dev_prev_20k_rate",
    "dev_prev_high_rate",
    "pub_prev_games",
    "pub_prev_20k_count",
    "pub_prev_high_count",
    "pub_prev_20k_rate",
    "pub_prev_high_rate",
]

BASE_FEATURES = [
    "required_age",
    "price",
    "windows",
    "mac",
    "linux",
    "achievements",
    "release_month",
    "screenshots_count",
    "has_header_image",
    "has_website",
    "has_support_url",
    "has_support_email",
    *HISTORY_FEATURES,
]

BEST_XGB_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.018493277707481513,
    "max_depth": 7,
    "min_child_weight": 8,
    "subsample": 0.8061303046919053,
    "colsample_bytree": 0.5437713481633702,
    "reg_alpha": 3.238199209465676,
    "reg_lambda": 0.7609796758482871,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}


def compute_entity_history(
    df_all: pd.DataFrame, col: str, prefix: str
) -> pd.DataFrame:
    ym = (
        df_all["release_year"].fillna(0).astype(int) * 100
        + df_all["release_month"].fillna(1).astype(int)
    )
    work = pd.DataFrame(
        {
            "_row_id": np.arange(len(df_all)),
            "ym": ym.to_numpy(),
            col: df_all[col].to_numpy(),
            "is_20k": (df_all["target_num"].to_numpy() >= 1).astype(int),
            "is_high": (df_all["target_num"].to_numpy() == 2).astype(int),
        }
    )

    exploded = work.explode(col)
    exploded[col] = exploded[col].fillna("").astype(str).str.strip().str.lower()
    exploded = exploded[exploded[col] != ""]

    monthly = (
        exploded.groupby([col, "ym"], as_index=False)
        .agg(
            m_games=("_row_id", "count"),
            m_20k=("is_20k", "sum"),
            m_high=("is_high", "sum"),
        )
        .sort_values([col, "ym"])
    )

    g = monthly.groupby(col)
    monthly["prev_games"] = g["m_games"].cumsum() - monthly["m_games"]
    monthly["prev_20k"] = g["m_20k"].cumsum() - monthly["m_20k"]
    monthly["prev_high"] = g["m_high"].cumsum() - monthly["m_high"]

    monthly["prev_20k_rate"] = np.where(
        monthly["prev_games"] > 0, monthly["prev_20k"] / monthly["prev_games"], np.nan
    )
    monthly["prev_high_rate"] = np.where(
        monthly["prev_games"] > 0, monthly["prev_high"] / monthly["prev_games"], np.nan
    )

    merged = exploded[["_row_id", col, "ym"]].merge(
        monthly[
            [
                col,
                "ym",
                "prev_games",
                "prev_20k",
                "prev_high",
                "prev_20k_rate",
                "prev_high_rate",
            ]
        ],
        on=[col, "ym"],
        how="left",
    )

    per_game = (
        merged.groupby("_row_id")
        .agg(
            prev_games=("prev_games", "max"),
            prev_20k_count=("prev_20k", "max"),
            prev_high_count=("prev_high", "max"),
            prev_20k_rate=("prev_20k_rate", "max"),
            prev_high_rate=("prev_high_rate", "max"),
        )
        .reindex(np.arange(len(df_all)))
    )

    out = pd.DataFrame(index=df_all.index)
    out[f"{prefix}_prev_games"] = (
        per_game["prev_games"].fillna(0).astype(int).to_numpy()
    )
    out[f"{prefix}_prev_20k_count"] = (
        per_game["prev_20k_count"].fillna(0).astype(int).to_numpy()
    )
    out[f"{prefix}_prev_high_count"] = (
        per_game["prev_high_count"].fillna(0).astype(int).to_numpy()
    )
    out[f"{prefix}_prev_20k_rate"] = per_game["prev_20k_rate"].to_numpy()
    out[f"{prefix}_prev_high_rate"] = per_game["prev_high_rate"].to_numpy()
    return out


def build_entity_lookup(df_all: pd.DataFrame, col: str) -> dict[str, dict]:
    work = pd.DataFrame(
        {
            col: df_all[col].to_numpy(),
            "is_20k": (df_all["target_num"].to_numpy() >= 1).astype(int),
            "is_high": (df_all["target_num"].to_numpy() == 2).astype(int),
        }
    )
    exploded = work.explode(col)
    exploded[col] = exploded[col].fillna("").astype(str).str.strip().str.lower()
    exploded = exploded[exploded[col] != ""]

    agg = exploded.groupby(col).agg(
        prev_games=("is_20k", "count"),
        prev_20k_count=("is_20k", "sum"),
        prev_high_count=("is_high", "sum"),
    )
    agg["prev_20k_rate"] = agg["prev_20k_count"] / agg["prev_games"]
    agg["prev_high_rate"] = agg["prev_high_count"] / agg["prev_games"]

    return {
        name: {
            "prev_games": int(row.prev_games),
            "prev_20k_count": int(row.prev_20k_count),
            "prev_high_count": int(row.prev_high_count),
            "prev_20k_rate": float(row.prev_20k_rate),
            "prev_high_rate": float(row.prev_high_rate),
        }
        for name, row in agg.iterrows()
    }


def prepare_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["primary_developer"] = df["developers"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown"
    )
    df["primary_publisher"] = df["publishers"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown"
    )
    df["categories"] = df["categories"].apply(
        lambda items: (
            [c for c in items if c not in LEAKY_CATEGORIES]
            if isinstance(items, list)
            else []
        )
    )
    df["short_description"] = df["short_description"].fillna("").astype(str)
    return df


def compute_time_weights(years: pd.Series) -> np.ndarray:
    raw = years.map(TIME_WEIGHT_MAP).fillna(1.0).to_numpy(dtype=float)
    return raw / raw.mean()


def _encode_single_multilabel(
    train_series: pd.Series,
    test_series: pd.Series | None,
    prefix: str,
    top_k: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, MultiLabelBinarizer]:
    train = train_series.copy()
    test = test_series.copy() if test_series is not None else None

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
        if test is not None:
            test = test.apply(
                lambda items: (
                    [x for x in items if x in top_classes]
                    if isinstance(items, list)
                    else []
                )
            )

    mlb = MultiLabelBinarizer()
    train_df = pd.DataFrame(mlb.fit_transform(train), columns=[f"{prefix}_{c}" for c in mlb.classes_])
    test_df = (
        pd.DataFrame(mlb.transform(test), columns=train_df.columns)
        if test is not None
        else None
    )
    return train_df, test_df, mlb


def apply_multilabel_encoding(
    df_train: pd.DataFrame, df_test: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    configs = [
        ("genres", "genre", None),
        ("categories", "cat", None),
        ("full_audio_languages", "audio_lang", 30),
        ("supported_languages", "lang", 30),
    ]

    train_dfs, test_dfs, encoders = [], [], {}

    for col, prefix, top_k in configs:
        tr_enc, te_enc, mlb = _encode_single_multilabel(
            df_train[col],
            df_test[col] if df_test is not None else None,
            prefix,
            top_k,
        )
        train_dfs.append(tr_enc)
        if te_enc is not None:
            test_dfs.append(te_enc)
        encoders[f"mlb_{col}"] = mlb

    return (
        pd.concat(train_dfs, axis=1),
        pd.concat(test_dfs, axis=1) if df_test is not None else None,
        encoders,
    )


def apply_target_encoding(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    df_test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, TargetEncoder]:
    te_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    te_safe = TargetEncoder(smooth="auto", cv=te_cv)
    te_cols = ["primary_developer", "primary_publisher"]

    tr_raw = te_safe.fit_transform(df_train[te_cols], y_train)
    feature_names = te_safe.get_feature_names_out(te_cols)

    dev_pub_train = pd.DataFrame(tr_raw, columns=feature_names)
    dev_pub_test = (
        pd.DataFrame(te_safe.transform(df_test[te_cols]), columns=feature_names)
        if df_test is not None
        else None
    )
    return dev_pub_train, dev_pub_test, te_safe


def apply_nlp_tfidf(
    df_train: pd.DataFrame, df_test: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None, TfidfVectorizer]:
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=10,
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf.fit(df_train["short_description"])
    tfidf_cols = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]

    tfidf_train = pd.DataFrame(
        tfidf.transform(df_train["short_description"]).toarray(), columns=tfidf_cols
    )
    tfidf_test = (
        pd.DataFrame(
            tfidf.transform(df_test["short_description"]).toarray(), columns=tfidf_cols
        )
        if df_test is not None
        else None
    )
    return tfidf_train, tfidf_test, tfidf


def build_feature_matrices(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    df_test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    df_train = prepare_raw_columns(df_train.reset_index(drop=True))
    df_test = prepare_raw_columns(df_test.reset_index(drop=True)) if df_test is not None else None

    mlb_tr, mlb_te, mlb_encoders = apply_multilabel_encoding(df_train, df_test)
    te_tr, te_te, te_encoder = apply_target_encoding(df_train, y_train, df_test)
    tfidf_tr, tfidf_te, tfidf_vectorizer = apply_nlp_tfidf(df_train, df_test)

    X_train = pd.concat([df_train[BASE_FEATURES], mlb_tr, te_tr, tfidf_tr], axis=1)
    X_test = (
        pd.concat([df_test[BASE_FEATURES], mlb_te, te_te, tfidf_te], axis=1)
        if df_test is not None
        else None
    )

    encoders = {
        "tfidf": tfidf_vectorizer,
        "target_encoder": te_encoder,
        **mlb_encoders,
    }
    return X_train, X_test, encoders


def evaluate_and_log(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    y_prob = model.predict_proba(X_test)
    ll = log_loss(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    auc_per_class = [
        roc_auc_score((y_test == i).astype(int), y_prob[:, i]) for i in range(3)
    ]

    actual_dist = y_test.value_counts(normalize=True).sort_index().to_numpy()
    pred_dist = y_prob.mean(axis=0)

    print(f"\n=== HONEST TEMPORAL TEST ({TEST_YEAR}) ===")
    print(f"LogLoss:               {ll:.4f}")
    print(f"ROC AUC (Macro):       {roc_auc:.4f}")
    print(
        f"ROC AUC Low/Med/High:  {[round(x, 4) for x in auc_per_class]}"
    )
    print(f"Actual class share:    {np.round(actual_dist, 4)}")
    print(f"Mean predicted probs:  {np.round(pred_dist, 4)}")


def train() -> None:
    print(f"Loading processed dataset from {DATA_PATH}...")
    df_all = pd.read_json(DATA_PATH).reset_index(drop=True)
    print(
        f"Loaded {len(df_all)} rows | Years: "
        f"{df_all['release_year'].min()} - {df_all['release_year'].max()}"
    )

    print("Computing strictly prior developer/publisher history...")
    dev_hist = compute_entity_history(df_all, "developers", "dev")
    pub_hist = compute_entity_history(df_all, "publishers", "pub")
    df_all = pd.concat([df_all, dev_hist, pub_hist], axis=1)

    dev_lookup = build_entity_lookup(df_all, "developers")
    pub_lookup = build_entity_lookup(df_all, "publishers")

    min_year, max_year = TRAIN_YEARS_RANGE
    df = df_all[df_all["release_year"].between(min_year, max_year)].reset_index(drop=True)

    df_tr = df[df["release_year"] < TEST_YEAR].reset_index(drop=True)
    df_te = df[df["release_year"] == TEST_YEAR].reset_index(drop=True)

    y_tr = df_tr["target_num"].copy()
    y_te = df_te["target_num"].copy()
    w_tr = compute_time_weights(df_tr["release_year"])

    print(
        f"\n[Stage 1] Validation split: train={len(df_tr)} ({min_year}-{TEST_YEAR - 1}), "
        f"test={len(df_te)} ({TEST_YEAR})"
    )
    X_tr, X_te, _ = build_feature_matrices(df_tr, y_tr, df_te)

    val_model = XGBClassifier(**BEST_XGB_PARAMS)
    val_model.fit(X_tr, y_tr, sample_weight=w_tr)
    evaluate_and_log(val_model, X_te, y_te)

    print(f"\n[Stage 2] Refitting production model on full {min_year}-{max_year} ({len(df)} rows)...")
    y_full = df["target_num"].copy()
    w_full = compute_time_weights(df["release_year"])

    X_full, _, prod_encoders = build_feature_matrices(df, y_full, df_test=None)
    print(f"Final production matrix shape: {X_full.shape}")

    prod_model = XGBClassifier(**BEST_XGB_PARAMS)
    prod_model.fit(X_full, y_full, sample_weight=w_full)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = MODELS_DIR / "best_xgboost.json"
    artifacts_path = MODELS_DIR / "preprocessing_artifacts.joblib"

    prod_model.save_model(model_path)

    artifacts = {
        **prod_encoders,
        "base_features": BASE_FEATURES,
        "feature_columns": X_full.columns.tolist(),
        "dev_history": dev_lookup,
        "pub_history": pub_lookup,
    }
    joblib.dump(artifacts, artifacts_path)

    print(f"\nProduction model saved to: {model_path}")
    print(f"Artifacts saved to:        {artifacts_path}")


if __name__ == "__main__":
    train()