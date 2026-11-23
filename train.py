import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier

from prepare_data import LEAKY_CATEGORIES

RANDOM_STATE = 42
N_JOBS = -1
SMOOTHING_M = 5.0

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "steam_games_processed_v2.parquet"
MODELS_DIR = BASE_DIR / "models"

TRAIN_YEARS_RANGE = (2021, 2024)
CAL_VAL_YEAR = 2023
TEST_YEAR = 2024

TIME_WEIGHT_MAP = {
    2021: 0.4,
    2022: 0.6,
    2023: 0.8,
    2024: 1.0,
}

HISTORY_FEATURES = [
    "dev_prev_games",
    "dev_prev_20k_count",
    "dev_prev_high_count",
    "dev_prev_20k_rate",
    "dev_prev_high_rate",
    "dev_smoothed_20k_rate",
    "dev_smoothed_high_rate",
    "pub_prev_games",
    "pub_prev_20k_count",
    "pub_prev_high_count",
    "pub_prev_20k_rate",
    "pub_prev_high_rate",
    "pub_smoothed_20k_rate",
    "pub_smoothed_high_rate",
]

ENGINEERED_FEATURES = [
    "is_free",
    "log_price",
    "supported_languages_count",
    "full_audio_languages_count",
    "genres_count",
    "categories_count",
    "platforms_count",
    "store_completeness_score",
    "is_self_published",
    "description_char_len",
    "description_word_count",
    "release_month_sin",
    "release_month_cos",
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
    *ENGINEERED_FEATURES,
]

BEST_XGB_PARAMS = {
    "n_estimators": 373,
    "learning_rate": 0.03126580463454852,
    "max_depth": 6,
    "min_child_weight": 13,
    "subsample": 0.7520153931537937,
    "colsample_bytree": 0.5151308485893303,
    "reg_alpha": 0.5293240431651446,
    "reg_lambda": 0.12682166037503817,
    "gamma": 0.6659740669084924,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}


def _to_list(value: object) -> list[str]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value if isinstance(value, list) else []


def compute_entity_history(df_all: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    ym = df_all["release_year"].fillna(0).astype(int) * 100 + df_all[
        "release_month"
    ].fillna(1).astype(int)
    work = pd.DataFrame(
        {
            "_row_id": np.arange(len(df_all)),
            "ym": ym.to_numpy(),
            col: df_all[col].apply(_to_list).to_numpy(),
            "is_20k": (df_all["target_num"].to_numpy() >= 1).astype(int),
            "is_high": (df_all["target_num"].to_numpy() == 2).astype(int),
        }
    )

    exploded = work.explode(col)
    exploded[col] = exploded[col].fillna("").astype(str).str.strip().str.lower()
    exploded = exploded[exploded[col] != ""]
    exploded = exploded.drop_duplicates(subset=["_row_id", col])

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

    merged = exploded[["_row_id", col, "ym"]].merge(
        monthly[[col, "ym", "prev_games", "prev_20k", "prev_high"]],
        on=[col, "ym"],
        how="left",
    )

    per_game = (
        merged.groupby("_row_id")
        .agg(
            prev_games=("prev_games", "sum"),
            prev_20k_count=("prev_20k", "sum"),
            prev_high_count=("prev_high", "sum"),
        )
        .reindex(np.arange(len(df_all)))
        .fillna(0)
    )

    pg = per_game["prev_games"].astype(int).to_numpy()
    p20 = per_game["prev_20k_count"].astype(int).to_numpy()
    ph = per_game["prev_high_count"].astype(int).to_numpy()

    out = pd.DataFrame(index=df_all.index)
    out[f"{prefix}_prev_games"] = pg
    out[f"{prefix}_prev_20k_count"] = p20
    out[f"{prefix}_prev_high_count"] = ph
    out[f"{prefix}_prev_20k_rate"] = np.divide(
        p20, pg, out=np.full(len(pg), np.nan, dtype=float), where=pg > 0
    )
    out[f"{prefix}_prev_high_rate"] = np.divide(
        ph, pg, out=np.full(len(pg), np.nan, dtype=float), where=pg > 0
    )
    return out


def build_entity_lookup(df_all: pd.DataFrame, col: str) -> dict[str, dict[str, int]]:
    work = pd.DataFrame(
        {
            "_row_id": np.arange(len(df_all)),
            col: df_all[col].apply(_to_list).to_numpy(),
            "is_20k": (df_all["target_num"].to_numpy() >= 1).astype(int),
            "is_high": (df_all["target_num"].to_numpy() == 2).astype(int),
        }
    )
    exploded = work.explode(col)
    exploded[col] = exploded[col].fillna("").astype(str).str.strip().str.lower()
    exploded = exploded[exploded[col] != ""]
    exploded = exploded.drop_duplicates(subset=["_row_id", col])

    agg = exploded.groupby(col).agg(
        prev_games=("_row_id", "count"),
        prev_20k_count=("is_20k", "sum"),
        prev_high_count=("is_high", "sum"),
    )

    return {
        name: {
            "prev_games": int(row.prev_games),
            "prev_20k_count": int(row.prev_20k_count),
            "prev_high_count": int(row.prev_high_count),
        }
        for name, row in agg.iterrows()
    }


def add_smoothed_history_rates(
    df: pd.DataFrame, prior_20k: float, prior_high: float, m: float = SMOOTHING_M
) -> pd.DataFrame:
    df = df.copy()
    for prefix in ("dev", "pub"):
        pg = df[f"{prefix}_prev_games"].to_numpy(dtype=float)
        p20 = df[f"{prefix}_prev_20k_count"].to_numpy(dtype=float)
        ph = df[f"{prefix}_prev_high_count"].to_numpy(dtype=float)
        df[f"{prefix}_smoothed_20k_rate"] = (p20 + m * prior_20k) / (pg + m)
        df[f"{prefix}_smoothed_high_rate"] = (ph + m * prior_high) / (pg + m)
    return df


def prepare_raw_and_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in (
        "genres",
        "categories",
        "full_audio_languages",
        "supported_languages",
        "developers",
        "publishers",
    ):
        df[col] = df[col].apply(_to_list)

    df["categories"] = df["categories"].apply(
        lambda items: [c for c in items if c not in LEAKY_CATEGORIES]
    )
    df["short_description"] = df["short_description"].fillna("").astype(str)

    df["is_free"] = (df["price"] == 0.0).astype(int)
    df["log_price"] = np.log1p(df["price"].clip(lower=0.0))
    df["supported_languages_count"] = df["supported_languages"].apply(len)
    df["full_audio_languages_count"] = df["full_audio_languages"].apply(len)
    df["genres_count"] = df["genres"].apply(len)
    df["categories_count"] = df["categories"].apply(len)
    df["platforms_count"] = df["windows"] + df["mac"] + df["linux"]
    df["store_completeness_score"] = (
        df["has_header_image"]
        + df["has_website"]
        + df["has_support_url"]
        + df["has_support_email"]
    )

    def _is_self_published(devs: list[str], pubs: list[str]) -> int:
        dev_set = {str(d).strip().lower() for d in devs if str(d).strip()}
        pub_set = {str(p).strip().lower() for p in pubs if str(p).strip()}
        return int(bool(dev_set and pub_set and (dev_set & pub_set)))

    df["is_self_published"] = [
        _is_self_published(d, p) for d, p in zip(df["developers"], df["publishers"])
    ]
    df["description_char_len"] = df["short_description"].str.len()
    df["description_word_count"] = df["short_description"].str.split().str.len()
    angle = 2.0 * np.pi * (df["release_month"] - 1) / 12.0
    df["release_month_sin"] = np.sin(angle)
    df["release_month_cos"] = np.cos(angle)
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
        train = train.apply(lambda items: [x for x in items if x in top_classes])
        if test is not None:
            test = test.apply(lambda items: [x for x in items if x in top_classes])

    mlb = MultiLabelBinarizer()
    train_raw = mlb.fit_transform(train).astype(np.int8)
    cols = [f"{prefix}_{c}" for c in mlb.classes_]
    train_df = pd.DataFrame(train_raw, columns=cols)

    if test is not None:
        known_classes = set(mlb.classes_)
        test_clean = test.apply(lambda items: [x for x in items if x in known_classes])
        test_df = pd.DataFrame(mlb.transform(test_clean).astype(np.int8), columns=cols)
    else:
        test_df = None
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


def apply_nlp_tfidf(
    df_train: pd.DataFrame, df_test: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None, TfidfVectorizer]:
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=10,
        sublinear_tf=True,
        stop_words="english",
        dtype=np.float32,
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
    prior_20k = float((y_train >= 1).mean())
    prior_high = float((y_train == 2).mean())

    df_train = add_smoothed_history_rates(
        df_train.reset_index(drop=True), prior_20k, prior_high
    )
    df_train = prepare_raw_and_engineered_columns(df_train)

    if df_test is not None:
        df_test = add_smoothed_history_rates(
            df_test.reset_index(drop=True), prior_20k, prior_high
        )
        df_test = prepare_raw_and_engineered_columns(df_test)

    mlb_tr, mlb_te, mlb_encoders = apply_multilabel_encoding(df_train, df_test)
    tfidf_tr, tfidf_te, tfidf_vectorizer = apply_nlp_tfidf(df_train, df_test)

    X_train = pd.concat(
        [df_train[BASE_FEATURES].astype(np.float32), mlb_tr, tfidf_tr], axis=1
    )
    X_test = (
        pd.concat([df_test[BASE_FEATURES].astype(np.float32), mlb_te, tfidf_te], axis=1)
        if df_test is not None
        else None
    )

    encoders = {
        "tfidf": tfidf_vectorizer,
        "prior_20k": prior_20k,
        "prior_high": prior_high,
        "smoothing_m": SMOOTHING_M,
        **mlb_encoders,
    }
    return X_train, X_test, encoders


def fit_temperature(logits: np.ndarray, y_true: pd.Series) -> float:
    """Fit single-scalar temperature scaling to calibrate multiclass probabilities."""
    y_arr = y_true.to_numpy()

    def _nll(temp: float) -> float:
        probs = softmax(logits / temp, axis=1)
        return float(log_loss(y_arr, probs))

    result = minimize_scalar(_nll, bounds=(0.5, 2.5), method="bounded")
    return float(result.x)


def evaluate_and_log(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    temperature: float = 1.0,
    label: str = "HONEST TEMPORAL TEST",
) -> np.ndarray:
    logits = model.predict(X_test, output_margin=True)
    y_prob = softmax(logits / temperature, axis=1)

    ll = log_loss(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    auc_per_class = [
        roc_auc_score((y_test == i).astype(int), y_prob[:, i]) for i in range(3)
    ]
    y_oh = np.eye(3)[y_test.to_numpy()]
    brier = float(np.mean(np.sum((y_prob - y_oh) ** 2, axis=1)))

    actual_dist = y_test.value_counts(normalize=True).sort_index().to_numpy()
    pred_dist = y_prob.mean(axis=0)

    print(f"\n=== {label} ({TEST_YEAR}, T={temperature:.4f}) ===")
    print(f"LogLoss:               {ll:.4f}")
    print(f"Brier Score:           {brier:.4f}")
    print(f"ROC AUC (Macro):       {roc_auc:.4f}")
    print(f"ROC AUC Low/Med/High:  {[round(x, 4) for x in auc_per_class]}")
    print(f"Actual class share:    {np.round(actual_dist, 4)}")
    print(f"Mean predicted probs:  {np.round(pred_dist, 4)}")
    return logits


def train() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {DATA_PATH}. Run `uv run python prepare_data.py` first."
        )

    print(f"Loading processed dataset from {DATA_PATH}...")
    df_all = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    print(
        f"Loaded {len(df_all)} rows | Years: "
        f"{df_all['release_year'].min()} - {df_all['release_year'].max()}"
    )

    print(
        "Computing strictly prior developer/publisher history across full timeline..."
    )
    dev_hist = compute_entity_history(df_all, "developers", "dev")
    pub_hist = compute_entity_history(df_all, "publishers", "pub")
    df_all = pd.concat([df_all, dev_hist, pub_hist], axis=1)

    dev_lookup = build_entity_lookup(df_all, "developers")
    pub_lookup = build_entity_lookup(df_all, "publishers")

    min_year, max_year = TRAIN_YEARS_RANGE
    df = df_all[df_all["release_year"].between(min_year, max_year)].reset_index(
        drop=True
    )

    df_cal_tr = df[df["release_year"] < CAL_VAL_YEAR].reset_index(drop=True)
    df_cal_va = df[df["release_year"] == CAL_VAL_YEAR].reset_index(drop=True)
    y_cal_tr = df_cal_tr["target_num"].copy()
    y_cal_va = df_cal_va["target_num"].copy()
    w_cal_tr = compute_time_weights(df_cal_tr["release_year"])

    print(
        f"\n[Stage 0] Fitting validation temperature on {min_year}-{CAL_VAL_YEAR - 1} "
        f"({len(df_cal_tr)}) -> {CAL_VAL_YEAR} ({len(df_cal_va)})..."
    )
    X_cal_tr, X_cal_va, _ = build_feature_matrices(df_cal_tr, y_cal_tr, df_cal_va)
    cal_params = {**BEST_XGB_PARAMS, "n_estimators": 339}
    cal_model = XGBClassifier(**cal_params)
    cal_model.fit(X_cal_tr, y_cal_tr, sample_weight=w_cal_tr)
    val_temperature = fit_temperature(
        cal_model.predict(X_cal_va, output_margin=True), y_cal_va
    )
    print(f"Fitted validation temperature T_val = {val_temperature:.4f}")

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
    test_logits = evaluate_and_log(
        val_model, X_te, y_te, temperature=1.0, label="HONEST TEMPORAL TEST (RAW)"
    )
    evaluate_and_log(
        val_model,
        X_te,
        y_te,
        temperature=val_temperature,
        label="HONEST TEMPORAL TEST (CALIBRATED)",
    )

    prod_temperature = fit_temperature(test_logits, y_te)
    print(
        f"\nProduction temperature fitted on 2024 OOT logits: T_prod = {prod_temperature:.4f}"
    )

    print(
        f"\n[Stage 2] Refitting production model on full {min_year}-{max_year} "
        f"({len(df)} rows)..."
    )
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
        "leaky_categories": sorted(LEAKY_CATEGORIES),
        "temperature": prod_temperature,
    }
    joblib.dump(artifacts, artifacts_path)

    print(f"\nProduction model saved to: {model_path}")
    print(f"Artifacts saved to:        {artifacts_path}")


if __name__ == "__main__":
    train()
