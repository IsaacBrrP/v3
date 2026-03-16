"""
House Prices: Advanced Regression — V3 Pipeline
Target: RMSLE ≤ 0.05 on Kaggle public leaderboard

Architecture:
  - 6 base models: Lasso, Ridge, ElasticNet, XGBoost, LightGBM, CatBoost
  - Bayesian hyperparameter optimization via Optuna
  - 2-level stacking with OOF predictions
  - Optimized blending via scipy.optimize

Estimated runtime: 2–4 hours (dominated by Optuna trials)
"""

# =============================================================================
# SECTION 0 — IMPORTS AND GLOBAL CONSTANTS
# =============================================================================

import os
import sys
import json
import time
import warnings
import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import chi2

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("catboost").setLevel(logging.ERROR)

# Third-party models with graceful fallback
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not found. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not found. Install with: pip install lightgbm")

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("WARNING: catboost not found. Install with: pip install catboost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not found. Install with: pip install optuna")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not found. Install with: pip install shap")

# Global constants
N_FOLDS: int = 10
RANDOM_STATE: int = 42
OPTUNA_TRIALS_LINEAR: int = 100
OPTUNA_TRIALS_XGB: int = 200
OPTUNA_TRIALS_LGB: int = 200
OPTUNA_TRIALS_CB: int = 150
SMOOTHING_M: int = 10  # Bayesian target encoding smoothing
MIN_PRICE: float = 10_000.0
MAX_PRICE: float = 1_500_000.0

np.random.seed(RANDOM_STATE)

# =============================================================================
# SECTION 1 — UTILITY FUNCTIONS
# =============================================================================


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Log Error computed in log-space.
    Because we model log1p(SalePrice) directly, this is equivalent to RMSE
    when both arrays are already in log-space (i.e., log1p-transformed).
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_oof_predictions(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_folds: int = N_FOLDS,
    use_early_stopping: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Out-Of-Fold predictions to build stacking meta-features.

    Guarantees no data leakage: the meta-learner in stage 10 sees OOF
    predictions that were never produced by a model trained on the same
    observation.  Test predictions are averaged across all folds.

    Parameters
    ----------
    model : sklearn-compatible estimator
    X_train : training feature matrix (numpy)
    y_train : log1p-transformed target
    X_test  : test feature matrix (numpy)
    n_folds : number of CV folds
    use_early_stopping : if True, pass eval_set for XGB/LGB-style early stop

    Returns
    -------
    oof_preds   : shape (n_train,)
    test_preds  : shape (n_test,)  — averaged across folds
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        if use_early_stopping and HAS_XGB and isinstance(
            model, xgb.XGBRegressor
        ):
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        elif use_early_stopping and HAS_LGB and isinstance(
            model, lgb.LGBMRegressor
        ):
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
            )
        elif use_early_stopping and HAS_CB and isinstance(
            model, cb.CatBoostRegressor
        ):
            model.fit(
                X_tr,
                y_tr,
                eval_set=(X_val, y_val),
                verbose=False,
            )
        else:
            model.fit(X_tr, y_tr)

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / n_folds

    oof_score = rmsle(y_train, oof_preds)
    print(f"  OOF RMSLE: {oof_score:.5f}")
    return oof_preds, test_preds


def target_encode_kfold(
    train_series: pd.Series,
    y: np.ndarray,
    test_series: pd.Series,
    n_folds: int = 5,
    smoothing: int = SMOOTHING_M,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-Fold Bayesian target encoding with smoothing.

    enc_i = (n_i * mean_i + m * global_mean) / (n_i + m)

    The smoothing parameter m controls regularization: larger m pushes
    group estimates toward the global mean when n_i is small.
    This prevents target leakage while retaining neighborhood-level signal.
    """
    global_mean = float(y.mean())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    # Reset index so positional indexing is safe
    train_series = train_series.reset_index(drop=True)
    train_enc = np.zeros(len(train_series))

    for tr_idx, val_idx in kf.split(train_series):
        tr_s = train_series.iloc[tr_idx]
        tr_y = y[tr_idx]  # positional — tr_idx comes from kf.split
        val_s = train_series.iloc[val_idx]

        # Build a category → smooth mean mapping using only the train fold
        # Use a simple aggregation (no lambda with g.index to avoid index issues)
        df_fold = pd.DataFrame({"cat": tr_s.values, "target": tr_y})
        agg = df_fold.groupby("cat")["target"].agg(["mean", "count"])
        agg["smooth_enc"] = (
            agg["count"] * agg["mean"] + smoothing * global_mean
        ) / (agg["count"] + smoothing)
        stats_map = agg["smooth_enc"].to_dict()
        train_enc[val_idx] = (
            val_s.map(stats_map).fillna(global_mean).values
        )

    # For test: use global train statistics (no leakage since test has no target)
    df_all = pd.DataFrame({"cat": train_series.values, "target": y})
    agg_global = df_all.groupby("cat")["target"].agg(["mean", "count"])
    agg_global["smooth_enc"] = (
        agg_global["count"] * agg_global["mean"] + smoothing * global_mean
    ) / (agg_global["count"] + smoothing)
    stats_global = agg_global["smooth_enc"].to_dict()
    test_enc = (
        test_series.reset_index(drop=True).map(stats_global).fillna(global_mean).values
    )

    return train_enc, test_enc


def add_ordinal_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map quality/condition ordinal strings to integers (Section 4.7).
    Must run BEFORE computing interaction features in Section 4.3 so that
    those interactions are numeric from the start.
    """
    qual_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

    ordinal_cols_qual = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
        "HeatingQC", "KitchenQual", "FireplaceQu",
        "GarageQual", "GarageCond", "PoolQC",
    ]
    for col in ordinal_cols_qual:
        if col in df.columns:
            df[col + "_enc"] = df[col].map(qual_map).fillna(0).astype(int)

    bsmt_exp_map = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
    if "BsmtExposure" in df.columns:
        df["BsmtExposure_enc"] = (
            df["BsmtExposure"].map(bsmt_exp_map).fillna(0).astype(int)
        )

    bsmt_fin_map = {
        "None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6,
    }
    for col in ["BsmtFinType1", "BsmtFinType2"]:
        if col in df.columns:
            df[col + "_enc"] = (
                df[col].map(bsmt_fin_map).fillna(0).astype(int)
            )

    garage_fin_map = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
    if "GarageFinish" in df.columns:
        df["GarageFinish_enc"] = (
            df["GarageFinish"].map(garage_fin_map).fillna(0).astype(int)
        )

    fence_map = {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}
    if "Fence" in df.columns:
        df["Fence_enc"] = (
            df["Fence"].map(fence_map).fillna(0).astype(int)
        )

    return df


# =============================================================================
# SECTION 2 — MAIN PIPELINE
# =============================================================================


def main() -> None:
    start_time = time.time()
    print("=" * 60)
    print("HOUSE PRICES V3 — KAGGLE GRANDMASTER PIPELINE")
    print("=" * 60)

    # =========================================================================
    # ETAPA 1: DATA LOADING AND VALIDATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("ETAPA 1: CARGA Y VALIDACIÓN DE DATOS")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")

    for p in [train_path, test_path]:
        if not os.path.exists(p):
            sys.exit(f"ERROR: File not found: {p}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_id = train["Id"].values
    test_id = test["Id"].values
    y_raw = train["SalePrice"].copy()

    # Transform target to log-space — RMSLE in original space becomes RMSE
    # in log-space, which is numerically better-conditioned and matches the
    # metric's emphasis on relative (not absolute) errors.
    y = np.log1p(y_raw.values)

    print(f"Train shape : {train.shape}")
    print(f"Test  shape : {test.shape}")
    print(f"y_raw stats : mean={y_raw.mean():.0f}, median={y_raw.median():.0f}")
    print(f"y (log1p)   : mean={y.mean():.4f}, std={y.std():.4f}")

    # Missing value report
    miss_train = train.isnull().sum()
    miss_train = miss_train[miss_train > 0]
    print(f"\nFeatures with missing in train ({len(miss_train)}):")
    print(miss_train.to_string())

    # Concatenate for joint preprocessing (prevents train/test distribution drift)
    all_data = pd.concat(
        [train.drop(["SalePrice", "Id"], axis=1), test.drop("Id", axis=1)],
        axis=0,
        ignore_index=True,
    )
    n_train = len(train)
    print(f"\nall_data shape: {all_data.shape}")

    # =========================================================================
    # ETAPA 2: ANÁLISIS EXPLORATORIO ESTADÍSTICO
    # =========================================================================
    print("\n" + "=" * 60)
    print("ETAPA 2: ANÁLISIS EXPLORATORIO ESTADÍSTICO")
    print("=" * 60)

    # 2.1 Target skewness
    skew_raw = float(y_raw.skew())
    skew_log = float(pd.Series(y).skew())
    print(f"Skewness SalePrice raw : {skew_raw:.4f}")
    print(f"Skewness log1p(SalePrice): {skew_log:.4f}")
    if abs(skew_log) > 0.5:
        print("WARNING: log1p skewness > 0.5; consider Box-Cox.")

    # 2.2 Multivariate outlier detection
    num_feats_corr = [
        "OverallQual", "GrLivArea", "TotalBsmtSF", "1stFlrSF",
        "GarageArea", "YearBuilt", "FullBath", "TotRmsAbvGrd",
        "GarageCars", "YearRemodAdd",
    ]
    _X_out = train[num_feats_corr].fillna(train[num_feats_corr].median())
    _cov = np.cov(_X_out.values.T)
    _cov_inv = np.linalg.pinv(_cov)
    _diff = _X_out.values - _X_out.values.mean(axis=0)
    mahal_sq = np.array(
        [float(_diff[i] @ _cov_inv @ _diff[i]) for i in range(len(_diff))]
    )
    chi2_thresh = chi2.ppf(0.999, df=len(num_feats_corr))
    mahal_outliers = set(np.where(mahal_sq > chi2_thresh)[0])

    # IQR 3× on SalePrice
    q1, q3 = y_raw.quantile(0.25), y_raw.quantile(0.75)
    iqr = q3 - q1
    iqr_outliers = set(
        y_raw[(y_raw < q1 - 3 * iqr) | (y_raw > q3 + 3 * iqr)].index
    )

    # IQR 3× on Price/SF (GrLivArea as SF proxy)
    price_sf = y_raw / train["GrLivArea"]
    q1p, q3p = price_sf.quantile(0.25), price_sf.quantile(0.75)
    iqrp = q3p - q1p
    iqr_sf_outliers = set(
        price_sf[(price_sf < q1p - 3 * iqrp) | (price_sf > q3p + 3 * iqrp)].index
    )

    # Remove observations that are outliers in ≥2 of 3 criteria
    outlier_counts = {}
    for idx in range(n_train):
        cnt = (
            int(idx in mahal_outliers)
            + int(idx in iqr_outliers)
            + int(idx in iqr_sf_outliers)
        )
        if cnt >= 2:
            outlier_counts[idx] = cnt

    rows_to_drop = sorted(outlier_counts.keys())
    print(f"\nMultivariate outliers removed: {len(rows_to_drop)}")
    if rows_to_drop:
        print(f"  Row indices: {rows_to_drop}")

    # Drop from train portion (all_data rows 0..n_train-1)
    keep_mask = np.ones(n_train, dtype=bool)
    keep_mask[rows_to_drop] = False

    y = y[keep_mask]
    y_raw_clean = y_raw.values[keep_mask]
    train_id = train_id[keep_mask]

    # Rebuild all_data with cleaned train
    train_clean = train.iloc[keep_mask].reset_index(drop=True)
    all_data = pd.concat(
        [
            train_clean.drop(["SalePrice", "Id"], axis=1),
            test.drop("Id", axis=1),
        ],
        axis=0,
        ignore_index=True,
    )
    n_train = len(train_clean)
    print(f"n_train after outlier removal: {n_train}")

    # =========================================================================
    # ETAPA 3: ADVANCED IMPUTATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("ETAPA 3: IMPUTACIÓN AVANZADA")
    print("=" * 60)

    # 3.1 MNAR: structural absence → 'None' or 0
    mnar_cat = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType",
    ]
    mnar_num = [
        "GarageYrBlt", "GarageArea", "GarageCars",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
    ]
    for col in mnar_cat:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna("None")
    for col in mnar_num:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna(0)

    # 3.2 MAR: LotFrontage → median by Neighborhood
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")[
        "LotFrontage"
    ].transform(lambda x: x.fillna(x.median()))
    # Fallback for neighborhoods with all NaN
    all_data["LotFrontage"] = all_data["LotFrontage"].fillna(
        all_data["LotFrontage"].median()
    )

    # 3.3 MCAR: mode / median for remaining
    mcar_cat = [
        "MSZoning", "Utilities", "Exterior1st", "Exterior2nd",
        "Electrical", "KitchenQual", "Functional", "SaleType",
    ]
    for col in mcar_cat:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    # Remaining numerics
    num_cols = all_data.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col] = all_data[col].fillna(all_data[col].median())

    # Remaining categoricals
    cat_cols = all_data.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    # 3.4 Validation
    assert all_data.isnull().sum().sum() == 0, "FATAL: missing values remain!"
    print("Imputation complete. No missing values.")

    # =========================================================================
    # ETAPA 4: FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "=" * 60)
    print("ETAPA 4: FEATURE ENGINEERING")
    print("=" * 60)

    # ---- 4.7 Ordinal encodings FIRST (needed for interaction features) -------
    all_data = add_ordinal_encodings(all_data)

    # ---- 4.1 Area / volume features -----------------------------------------
    all_data["TotalSF"] = (
        all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
    )
    all_data["TotalLivingSF"] = all_data["GrLivArea"] + all_data["TotalBsmtSF"]
    all_data["TotalPorchSF"] = (
        all_data["OpenPorchSF"]
        + all_data["3SsnPorch"]
        + all_data["EnclosedPorch"]
        + all_data["ScreenPorch"]
        + all_data["WoodDeckSF"]
    )
    all_data["AllBath"] = (
        all_data["FullBath"]
        + 0.5 * all_data["HalfBath"]
        + all_data["BsmtFullBath"]
        + 0.5 * all_data["BsmtHalfBath"]
    )
    all_data["BsmtFinRatio"] = all_data["BsmtFinSF1"] / (
        all_data["TotalBsmtSF"] + 1
    )
    all_data["FloorRatio"] = all_data["2ndFlrSF"] / (all_data["1stFlrSF"] + 1)
    all_data["LivLotRatio"] = all_data["GrLivArea"] / (all_data["LotArea"] + 1)
    all_data["GarageRatio"] = all_data["GarageArea"] / (all_data["TotalSF"] + 1)
    all_data["StorageRatio"] = all_data["TotalBsmtSF"] / (all_data["TotalSF"] + 1)
    all_data["OverallScore"] = all_data["OverallQual"] * all_data["OverallCond"]

    # ---- 4.2 Temporal features -----------------------------------------------
    all_data["HouseAge"] = all_data["YrSold"] - all_data["YearBuilt"]
    all_data["RemodAge"] = all_data["YrSold"] - all_data["YearRemodAdd"]
    all_data["GarageAge"] = all_data["YrSold"] - all_data["GarageYrBlt"]
    all_data["IsRemodeled"] = (
        all_data["YearRemodAdd"] != all_data["YearBuilt"]
    ).astype(int)
    all_data["IsNewHouse"] = (
        all_data["YrSold"] == all_data["YearBuilt"]
    ).astype(int)
    all_data["YearBuilt_Sq"] = all_data["YearBuilt"] ** 2
    all_data["RemodDelta"] = all_data["YearRemodAdd"] - all_data["YearBuilt"]

    def season_map(month: int) -> int:
        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        return 3

    all_data["SeasonSold"] = all_data["MoSold"].apply(season_map)

    # ---- 4.3 Quality interaction features ------------------------------------
    all_data["Qual_TotalSF"] = all_data["OverallQual"] * all_data["TotalSF"]
    all_data["Qual_LivArea"] = all_data["OverallQual"] * all_data["GrLivArea"]
    all_data["Qual_BsmtSF"] = all_data["OverallQual"] * all_data["TotalBsmtSF"]
    all_data["Qual_Age"] = all_data["OverallQual"] * all_data["HouseAge"]
    all_data["Qual_GarageCars"] = (
        all_data["OverallQual"] * all_data["GarageCars"]
    )
    all_data["Cond_TotalSF"] = all_data["OverallCond"] * all_data["TotalSF"]
    all_data["ExterQual_SF"] = all_data["ExterQual_enc"] * all_data["GrLivArea"]
    all_data["KitchenQual_SF"] = (
        all_data["KitchenQual_enc"] * all_data["GrLivArea"]
    )
    all_data["Bath_Qual"] = all_data["AllBath"] * all_data["OverallQual"]
    all_data["Garage_Qual"] = (
        all_data["GarageCars"] * all_data["GarageQual_enc"]
    )
    all_data["Fireplace_Qual"] = (
        all_data["Fireplaces"] * all_data["FireplaceQu_enc"]
    )
    all_data["Bsmt_Qual_SF"] = (
        all_data["BsmtQual_enc"] * all_data["TotalBsmtSF"]
    )

    # ---- 4.4 Polynomial features --------------------------------------------
    # log1p applied to sq/cube features to tame their scale
    all_data["OverallQual_sq"] = all_data["OverallQual"] ** 2
    all_data["GrLivArea_sq"] = np.log1p(all_data["GrLivArea"] ** 2)
    all_data["TotalSF_sq"] = np.log1p(all_data["TotalSF"] ** 2)
    all_data["GarageArea_sq"] = np.log1p(all_data["GarageArea"] ** 2)
    all_data["YearBuilt_cube"] = all_data["YearBuilt"] ** 3
    all_data["Qual_sq"] = all_data["OverallQual"] ** 3
    all_data["AllBath_sq"] = all_data["AllBath"] ** 2
    all_data["RemodAge_sq"] = all_data["RemodAge"] ** 2

    # ---- 4.5 Location features (target encoding) ----------------------------
    # Split to avoid leakage
    all_data_train = all_data.iloc[:n_train].copy()
    all_data_test = all_data.iloc[n_train:].copy()

    # Neighborhood target encoding
    train_nb_enc, test_nb_enc = target_encode_kfold(
        all_data_train["Neighborhood"], y, all_data_test["Neighborhood"]
    )
    all_data_train["Neighborhood_TargetEnc"] = train_nb_enc
    all_data_test["Neighborhood_TargetEnc"] = test_nb_enc

    # Neighborhood price tier (quintiles by median sale price)
    nb_median = (
        train_clean.groupby("Neighborhood")["SalePrice"].median()
    )
    nb_tiers = pd.qcut(nb_median, q=5, labels=[1, 2, 3, 4, 5])
    nb_tier_map = nb_tiers.to_dict()
    all_data_train["Neighborhood_PriceTier"] = (
        all_data_train["Neighborhood"].map(nb_tier_map).fillna(3).astype(int)
    )
    all_data_test["Neighborhood_PriceTier"] = (
        all_data_test["Neighborhood"].map(nb_tier_map).fillna(3).astype(int)
    )

    # Neighborhood quality tier (quintiles by median OverallQual)
    nb_qual_median = (
        all_data_train.groupby("Neighborhood")["OverallQual"].median()
    )
    nb_qual_tiers = pd.qcut(
        # rank(method="first") converts median values to unique ranks so that
        # qcut can divide them into equal-sized quintiles without raising a
        # "Bin edges must be unique" error when two neighborhoods share the same
        # median OverallQual (duplicate bin-edge problem).
        nb_qual_median.rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    )
    nb_qual_tier_map = nb_qual_tiers.to_dict()
    all_data_train["Neighborhood_QualTier"] = (
        all_data_train["Neighborhood"].map(nb_qual_tier_map).fillna(3).astype(int)
    )
    all_data_test["Neighborhood_QualTier"] = (
        all_data_test["Neighborhood"].map(nb_qual_tier_map).fillna(3).astype(int)
    )

    # MSSubClass target encoding
    train_sc_enc, test_sc_enc = target_encode_kfold(
        all_data_train["MSSubClass"].astype(str),
        y,
        all_data_test["MSSubClass"].astype(str),
    )
    all_data_train["SubClass_TargetEnc"] = train_sc_enc
    all_data_test["SubClass_TargetEnc"] = test_sc_enc

    # Merge back
    all_data = pd.concat(
        [all_data_train, all_data_test], axis=0, ignore_index=True
    )

    # ---- 4.6 Binary amenity features ----------------------------------------
    all_data["HasPool"] = (all_data["PoolArea"] > 0).astype(int)
    all_data["HasFireplace"] = (all_data["Fireplaces"] > 0).astype(int)
    all_data["HasGarage"] = (all_data["GarageArea"] > 0).astype(int)
    all_data["HasBasement"] = (all_data["TotalBsmtSF"] > 0).astype(int)
    all_data["Has2ndFloor"] = (all_data["2ndFlrSF"] > 0).astype(int)
    all_data["HasMasVnr"] = (all_data["MasVnrArea"] > 0).astype(int)

    # ---- 4.8 One-hot encoding for nominal categoricals ----------------------
    # Drop original ordinal columns (already encoded) and Neighborhood
    # (replaced by target encoding)
    ordinal_orig = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual",
        "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond",
        "PoolQC", "Fence", "Functional",
    ]
    drop_before_ohe = [c for c in ordinal_orig if c in all_data.columns]
    drop_before_ohe += ["Neighborhood"]  # replaced by encodings

    all_data = all_data.drop(columns=drop_before_ohe, errors="ignore")

    # OHE remaining object columns
    cat_remain = all_data.select_dtypes(include=["object"]).columns.tolist()
    print(f"Columns to OHE: {cat_remain}")
    all_data = pd.get_dummies(all_data, columns=cat_remain, drop_first=True)

    # ---- 4.9 Skewness transformation ----------------------------------------
    num_feats = all_data.select_dtypes(include=[np.number]).columns.tolist()
    skewness = all_data[num_feats].skew().sort_values(ascending=False)
    high_skew = skewness[abs(skewness) > 0.5].index.tolist()
    print(f"Features with |skew| > 0.5 before transform: {len(high_skew)}")

    for col in high_skew:
        col_min = all_data[col].min()
        if col_min >= 0:
            all_data[col] = np.log1p(all_data[col])
        else:
            # Yeo-Johnson handles negative values; fit on train, apply to all
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            vals = all_data[col].values.reshape(-1, 1)
            all_data[col] = pt.fit_transform(vals).ravel()

    print(f"\nall_data shape after feature engineering: {all_data.shape}")

    # =========================================================================
    # ETAPA 5: FEATURE SELECTION
    # =========================================================================
    print("\n" + "=" * 60)
    print("ETAPA 5: SELECCIÓN DE FEATURES")
    print("=" * 60)

    # 5.1 Near-zero variance removal
    X_train_tmp = all_data.iloc[:n_train]
    variances = X_train_tmp.var()
    low_var = variances[variances < 0.01].index.tolist()
    print(f"Low-variance features removed: {len(low_var)}")
    all_data = all_data.drop(columns=low_var, errors="ignore")

    # Also remove features where > 99% rows have same value
    def pct_dominant(s: pd.Series) -> float:
        return s.value_counts(normalize=True).iloc[0] if len(s) > 0 else 1.0

    train_portion = all_data.iloc[:n_train]
    dominant_feats = [
        col
        for col in all_data.columns
        if pct_dominant(train_portion[col]) > 0.99
    ]
    print(f"Dominant-value features removed: {len(dominant_feats)}")
    all_data = all_data.drop(columns=dominant_feats, errors="ignore")

    # 5.2 Multicollinearity removal (|r| > 0.92 → keep feature with higher
    #     correlation with target)
    X_tr_corr = all_data.iloc[:n_train].copy()
    corr_matrix = X_tr_corr.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    y_series = pd.Series(y, name="target")
    target_corr = X_tr_corr.corrwith(y_series).abs()

    to_drop_collinear = []
    for col in upper.columns:
        high_corr_feats = upper[col][upper[col] > 0.92].index.tolist()
        for hc in high_corr_feats:
            # Drop the one with lower correlation to target
            if target_corr.get(col, 0) < target_corr.get(hc, 0):
                if col not in to_drop_collinear:
                    to_drop_collinear.append(col)
            else:
                if hc not in to_drop_collinear:
                    to_drop_collinear.append(hc)

    print(f"Collinear features removed (|r|>0.92): {len(to_drop_collinear)}")
    all_data = all_data.drop(columns=to_drop_collinear, errors="ignore")

    # 5.3 Integrity check
    all_data = all_data.replace([np.inf, -np.inf], np.nan)
    for col in all_data.columns:
        if all_data[col].isnull().any():
            all_data[col] = all_data[col].fillna(all_data[col].median())

    assert all_data.isnull().sum().sum() == 0
    assert not np.isinf(all_data.values.astype(np.float64)).any()

    # 5.4 Report
    feature_names = all_data.columns.tolist()
    print(f"\nFinal feature space: {len(feature_names)} features")
    X_tr_final = all_data.iloc[:n_train].copy()
    y_series_full = pd.Series(y, index=X_tr_final.index, name="target")
    top20 = X_tr_final.corrwith(y_series_full).abs().sort_values(ascending=False).head(20)
    print("\nTop 20 features by |correlation| with log(SalePrice):")
    print(top20.to_string())

    # =========================================================================
    # ETAPA 6: DIFFERENTIAL SCALING
    # =========================================================================
    print("\n" + "=" * 60)
    print("ETAPA 6: ESCALADO DIFERENCIADO")
    print("=" * 60)

    X_train_all = all_data.iloc[:n_train].values.astype(np.float64)
    X_test_all = all_data.iloc[n_train:].values.astype(np.float64)

    # Linear models: RobustScaler (fitted on train only)
    scaler = RobustScaler()
    X_train_linear = scaler.fit_transform(X_train_all)
    X_test_linear = scaler.transform(X_test_all)

    # Tree models: no scaling needed
    X_train_tree = X_train_all.copy()
    X_test_tree = X_test_all.copy()

    print(f"X_train_linear shape: {X_train_linear.shape}")
    print(f"X_train_tree   shape: {X_train_tree.shape}")

    # =========================================================================
    # ETAPA 7 + 8: BASE MODELS WITH OPTUNA HYPERPARAMETER OPTIMIZATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("ETAPA 7+8: MODELOS BASE + OPTIMIZACIÓN BAYESIANA")
    print("=" * 60)

    kfold_cv = KFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )  # 5-fold for Optuna speed; 10-fold for final OOF

    best_params: dict = {}

    # ---- Lasso ---------------------------------------------------------------
    print("\n--- Lasso ---")
    if HAS_OPTUNA:
        def objective_lasso(trial: "optuna.Trial") -> float:
            alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
            m = Lasso(alpha=alpha, max_iter=100_000, tol=1e-4, random_state=RANDOM_STATE)
            scores = []
            for tr_i, val_i in kfold_cv.split(X_train_linear):
                m.fit(X_train_linear[tr_i], y[tr_i])
                scores.append(rmsle(y[val_i], m.predict(X_train_linear[val_i])))
            return float(np.mean(scores))

        study_lasso = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(),
        )
        study_lasso.optimize(objective_lasso, n_trials=OPTUNA_TRIALS_LINEAR, show_progress_bar=False)
        best_params["lasso"] = study_lasso.best_params
        print(f"  Best Lasso params: {best_params['lasso']}")
    else:
        best_params["lasso"] = {"alpha": 0.0003}

    best_lasso = Lasso(
        alpha=best_params["lasso"]["alpha"],
        max_iter=100_000,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )

    # ---- Ridge ---------------------------------------------------------------
    print("\n--- Ridge ---")
    if HAS_OPTUNA:
        def objective_ridge(trial: "optuna.Trial") -> float:
            alpha = trial.suggest_float("alpha", 0.01, 100.0, log=True)
            m = Ridge(alpha=alpha, max_iter=10_000, random_state=RANDOM_STATE)
            scores = []
            for tr_i, val_i in kfold_cv.split(X_train_linear):
                m.fit(X_train_linear[tr_i], y[tr_i])
                scores.append(rmsle(y[val_i], m.predict(X_train_linear[val_i])))
            return float(np.mean(scores))

        study_ridge = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(),
        )
        study_ridge.optimize(objective_ridge, n_trials=OPTUNA_TRIALS_LINEAR, show_progress_bar=False)
        best_params["ridge"] = study_ridge.best_params
        print(f"  Best Ridge params: {best_params['ridge']}")
    else:
        best_params["ridge"] = {"alpha": 8.0}

    best_ridge = Ridge(
        alpha=best_params["ridge"]["alpha"],
        max_iter=10_000,
        random_state=RANDOM_STATE,
    )

    # ---- ElasticNet ----------------------------------------------------------
    print("\n--- ElasticNet ---")
    if HAS_OPTUNA:
        def objective_elastic(trial: "optuna.Trial") -> float:
            alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.99)
            m = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=100_000,
                tol=1e-4,
                random_state=RANDOM_STATE,
            )
            scores = []
            for tr_i, val_i in kfold_cv.split(X_train_linear):
                m.fit(X_train_linear[tr_i], y[tr_i])
                scores.append(rmsle(y[val_i], m.predict(X_train_linear[val_i])))
            return float(np.mean(scores))

        study_elastic = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(),
        )
        study_elastic.optimize(objective_elastic, n_trials=OPTUNA_TRIALS_LINEAR, show_progress_bar=False)
        best_params["elasticnet"] = study_elastic.best_params
        print(f"  Best ElasticNet params: {best_params['elasticnet']}")
    else:
        best_params["elasticnet"] = {"alpha": 0.0003, "l1_ratio": 0.85}

    best_elastic = ElasticNet(
        alpha=best_params["elasticnet"]["alpha"],
        l1_ratio=best_params["elasticnet"]["l1_ratio"],
        max_iter=100_000,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )

    # ---- XGBoost -------------------------------------------------------------
    if HAS_XGB:
        print("\n--- XGBoost ---")
        if HAS_OPTUNA:
            def objective_xgb(trial: "optuna.Trial") -> float:
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 6),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                    "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                    "n_estimators": 2000,
                    "objective": "reg:squarederror",
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1,
                    "tree_method": "hist",
                    "early_stopping_rounds": 50,
                }
                scores = []
                for tr_i, val_i in kfold_cv.split(X_train_tree):
                    m = xgb.XGBRegressor(**params)
                    m.fit(
                        X_train_tree[tr_i],
                        y[tr_i],
                        eval_set=[(X_train_tree[val_i], y[val_i])],
                        verbose=False,
                    )
                    scores.append(rmsle(y[val_i], m.predict(X_train_tree[val_i])))
                return float(np.mean(scores))

            study_xgb = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                pruner=optuna.pruners.MedianPruner(),
            )
            study_xgb.optimize(objective_xgb, n_trials=OPTUNA_TRIALS_XGB, show_progress_bar=False)
            best_params["xgb"] = study_xgb.best_params
            print(f"  Best XGB params: {best_params['xgb']}")
        else:
            best_params["xgb"] = {}

        best_xgb = xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=best_params["xgb"].get("learning_rate", 0.01),
            max_depth=best_params["xgb"].get("max_depth", 4),
            min_child_weight=best_params["xgb"].get("min_child_weight", 1),
            subsample=best_params["xgb"].get("subsample", 0.8),
            colsample_bytree=best_params["xgb"].get("colsample_bytree", 0.8),
            reg_alpha=best_params["xgb"].get("reg_alpha", 0.001),
            reg_lambda=best_params["xgb"].get("reg_lambda", 1.0),
            gamma=best_params["xgb"].get("gamma", 0.0),
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=100,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        print("WARNING: Falling back to GradientBoostingRegressor for XGB slot.")
        best_xgb = GradientBoostingRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=4,
            random_state=RANDOM_STATE
        )

    # ---- LightGBM ------------------------------------------------------------
    if HAS_LGB:
        print("\n--- LightGBM ---")
        if HAS_OPTUNA:
            def objective_lgb(trial: "optuna.Trial") -> float:
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                    "n_estimators": 2000,
                    "objective": "regression",
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1,
                    "verbose": -1,
                    "subsample_freq": 5,
                    "early_stopping_rounds": 50,
                }
                scores = []
                for tr_i, val_i in kfold_cv.split(X_train_tree):
                    m = lgb.LGBMRegressor(**params)
                    m.fit(
                        X_train_tree[tr_i],
                        y[tr_i],
                        eval_set=[(X_train_tree[val_i], y[val_i])],
                    )
                    scores.append(rmsle(y[val_i], m.predict(X_train_tree[val_i])))
                return float(np.mean(scores))

            study_lgb = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                pruner=optuna.pruners.MedianPruner(),
            )
            study_lgb.optimize(objective_lgb, n_trials=OPTUNA_TRIALS_LGB, show_progress_bar=False)
            best_params["lgb"] = study_lgb.best_params
            print(f"  Best LGB params: {best_params['lgb']}")
        else:
            best_params["lgb"] = {}

        best_lgbm = lgb.LGBMRegressor(
            n_estimators=3000,
            learning_rate=best_params["lgb"].get("learning_rate", 0.01),
            num_leaves=best_params["lgb"].get("num_leaves", 31),
            min_child_samples=best_params["lgb"].get("min_child_samples", 20),
            subsample=best_params["lgb"].get("subsample", 0.8),
            subsample_freq=5,
            colsample_bytree=best_params["lgb"].get("colsample_bytree", 0.8),
            reg_alpha=best_params["lgb"].get("reg_alpha", 0.001),
            reg_lambda=best_params["lgb"].get("reg_lambda", 0.001),
            objective="regression",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            early_stopping_rounds=100,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        print("WARNING: Falling back to GradientBoostingRegressor for LGBM slot.")
        best_lgbm = GradientBoostingRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=4,
            random_state=RANDOM_STATE
        )

    # ---- CatBoost ------------------------------------------------------------
    if HAS_CB:
        print("\n--- CatBoost ---")
        if HAS_OPTUNA:
            def objective_cb(trial: "optuna.Trial") -> float:
                params = {
                    "depth": trial.suggest_int("depth", 4, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                    "random_strength": trial.suggest_float("random_strength", 0.1, 1.0),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.1, 1.0),
                    "iterations": 1000,
                    "loss_function": "RMSE",
                    "random_seed": RANDOM_STATE,
                    "verbose": 0,
                    "od_type": "Iter",
                    "od_wait": 50,
                }
                scores = []
                for tr_i, val_i in kfold_cv.split(X_train_tree):
                    m = cb.CatBoostRegressor(**params)
                    m.fit(
                        X_train_tree[tr_i],
                        y[tr_i],
                        eval_set=(X_train_tree[val_i], y[val_i]),
                        verbose=False,
                    )
                    scores.append(rmsle(y[val_i], m.predict(X_train_tree[val_i])))
                return float(np.mean(scores))

            study_cb = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                pruner=optuna.pruners.MedianPruner(),
            )
            study_cb.optimize(objective_cb, n_trials=OPTUNA_TRIALS_CB, show_progress_bar=False)
            best_params["catboost"] = study_cb.best_params
            print(f"  Best CatBoost params: {best_params['catboost']}")
        else:
            best_params["catboost"] = {}

        best_catboost = cb.CatBoostRegressor(
            iterations=3000,
            learning_rate=best_params["catboost"].get("learning_rate", 0.01),
            depth=best_params["catboost"].get("depth", 6),
            l2_leaf_reg=best_params["catboost"].get("l2_leaf_reg", 3.0),
            min_data_in_leaf=best_params["catboost"].get("min_data_in_leaf", 10),
            random_strength=best_params["catboost"].get("random_strength", 0.5),
            bagging_temperature=best_params["catboost"].get("bagging_temperature", 0.5),
            od_type="Iter",
            od_wait=100,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=0,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        print("WARNING: Falling back to GradientBoostingRegressor for CatBoost slot.")
        best_catboost = GradientBoostingRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=4,
            random_state=RANDOM_STATE
        )

    # Save best hyperparameters
    params_path = os.path.join(base_dir, "best_params_v3.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {params_path}")

    # =========================================================================
    # ETAPA 9: OOF PREDICTIONS FOR STACKING
    # =========================================================================
    print("\n" + "=" * 60)
    print("ETAPA 9: OOF PREDICTIONS PARA STACKING")
    print("=" * 60)

    print("\nLasso OOF:")
    oof_lasso, test_lasso = get_oof_predictions(
        best_lasso, X_train_linear, y, X_test_linear
    )

    print("\nRidge OOF:")
    oof_ridge, test_ridge = get_oof_predictions(
        best_ridge, X_train_linear, y, X_test_linear
    )

    print("\nElasticNet OOF:")
    oof_elastic, test_elastic = get_oof_predictions(
        best_elastic, X_train_linear, y, X_test_linear
    )

    print("\nXGBoost OOF:")
    oof_xgb, test_xgb = get_oof_predictions(
        best_xgb, X_train_tree, y, X_test_tree,
        use_early_stopping=HAS_XGB,
    )

    print("\nLightGBM OOF:")
    oof_lgbm, test_lgbm = get_oof_predictions(
        best_lgbm, X_train_tree, y, X_test_tree,
        use_early_stopping=HAS_LGB,
    )

    print("\nCatBoost OOF:")
    oof_catboost, test_catboost = get_oof_predictions(
        best_catboost, X_train_tree, y, X_test_tree,
        use_early_stopping=HAS_CB,
    )

    print("\n--- Summary of OOF RMSLE ---")
    oof_map = {
        "Lasso": oof_lasso,
        "Ridge": oof_ridge,
        "ElasticNet": oof_elastic,
        "XGBoost": oof_xgb,
        "LightGBM": oof_lgbm,
        "CatBoost": oof_catboost,
    }
    for name, oof in oof_map.items():
        print(f"  {name:12s}: {rmsle(y, oof):.5f}")

    # =========================================================================
    # ETAPA 10: META-LEARNER (STACKING LEVEL 2)
    # =========================================================================
    print("\n" + "=" * 60)
    print("ETAPA 10: META-LEARNER (NIVEL 2)")
    print("=" * 60)

    meta_train = np.column_stack(
        [oof_lasso, oof_ridge, oof_elastic, oof_xgb, oof_lgbm, oof_catboost]
    )
    meta_test = np.column_stack(
        [test_lasso, test_ridge, test_elastic, test_xgb, test_lgbm, test_catboost]
    )

    # Ridge as meta-learner: simple, regularized, avoids level-2 overfitting
    # since the meta-feature space is only 6-dimensional.
    meta_model = Ridge(alpha=0.5, random_state=RANDOM_STATE)
    meta_model.fit(meta_train, y)
    stacking_oof_pred = meta_model.predict(meta_train)
    stacking_test_pred = meta_model.predict(meta_test)

    stacking_oof_rmsle = rmsle(y, stacking_oof_pred)
    print(f"Stacking OOF RMSLE: {stacking_oof_rmsle:.5f}")

    # Optionally try BayesianRidge
    from sklearn.linear_model import BayesianRidge
    br = BayesianRidge()
    br.fit(meta_train, y)
    br_oof = br.predict(meta_train)
    br_oof_rmsle = rmsle(y, br_oof)
    print(f"BayesianRidge meta OOF RMSLE: {br_oof_rmsle:.5f}")

    if br_oof_rmsle < stacking_oof_rmsle:
        print("  → Using BayesianRidge as meta-learner (better OOF)")
        meta_model = br
        stacking_oof_pred = br_oof
        stacking_test_pred = br.predict(meta_test)
        stacking_oof_rmsle = br_oof_rmsle

    # =========================================================================
    # ETAPA 11: ENSEMBLE FINAL (OPTIMIZED BLENDING)
    # =========================================================================
    print("\n" + "=" * 60)
    print("ETAPA 11: ENSEMBLE FINAL")
    print("=" * 60)

    def blend_loss(weights: np.ndarray) -> float:
        w = np.abs(weights) / (np.sum(np.abs(weights)) + 1e-10)
        blend = (
            w[0] * stacking_oof_pred
            + w[1] * oof_xgb
            + w[2] * oof_lgbm
            + w[3] * oof_catboost
        )
        return rmsle(y, blend)

    result = minimize(
        blend_loss,
        x0=np.array([0.4, 0.2, 0.2, 0.2]),
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6},
    )
    optimal_weights = np.abs(result.x) / (np.sum(np.abs(result.x)) + 1e-10)
    print(f"Optimal blend weights: stacking={optimal_weights[0]:.3f}, "
          f"xgb={optimal_weights[1]:.3f}, lgbm={optimal_weights[2]:.3f}, "
          f"catboost={optimal_weights[3]:.3f}")

    blend_oof = (
        optimal_weights[0] * stacking_oof_pred
        + optimal_weights[1] * oof_xgb
        + optimal_weights[2] * oof_lgbm
        + optimal_weights[3] * oof_catboost
    )
    final_blend_rmsle = rmsle(y, blend_oof)
    print(f"Final Blend OOF RMSLE: {final_blend_rmsle:.5f}")

    final_test_pred = (
        optimal_weights[0] * stacking_test_pred
        + optimal_weights[1] * test_xgb
        + optimal_weights[2] * test_lgbm
        + optimal_weights[3] * test_catboost
    )

    # Invert log1p transformation → original price scale
    submission_prices = np.expm1(final_test_pred)
    assert (submission_prices > 0).all(), "Negative prices detected!"

    # =========================================================================
    # ETAPA 12: SHAP INTERPRETABILITY
    # =========================================================================
    print("\n" + "=" * 60)
    print("ETAPA 12: INTERPRETABILIDAD SHAP")
    print("=" * 60)

    if HAS_SHAP and HAS_XGB:
        try:
            # Re-train XGBoost on full train for SHAP (no val set needed).
            # n_estimators is fixed at 3000 to match the OOF model; it is not
            # included in best_params because Optuna tunes only the tree structure
            # and regularization hypers, while n_estimators is controlled via
            # early stopping.
            xgb_shap = xgb.XGBRegressor(
                n_estimators=3000,
                learning_rate=best_params["xgb"].get("learning_rate", 0.01),
                max_depth=best_params["xgb"].get("max_depth", 4),
                min_child_weight=best_params["xgb"].get("min_child_weight", 1),
                subsample=best_params["xgb"].get("subsample", 0.8),
                colsample_bytree=best_params["xgb"].get("colsample_bytree", 0.8),
                reg_alpha=best_params["xgb"].get("reg_alpha", 0.001),
                reg_lambda=best_params["xgb"].get("reg_lambda", 1.0),
                gamma=best_params["xgb"].get("gamma", 0.0),
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            )
            xgb_shap.fit(X_train_tree, y)
            explainer = shap.TreeExplainer(xgb_shap)
            shap_values = explainer.shap_values(X_train_tree)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X_train_tree,
                feature_names=feature_names,
                plot_type="bar",
                max_display=20,
                show=False,
            )
            plt.tight_layout()
            shap_summary_path = os.path.join(base_dir, "shap_summary_v3.png")
            plt.savefig(shap_summary_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"SHAP summary saved to {shap_summary_path}")

            # Top 5 dependence plots
            mean_shap = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame(
                {"feature": feature_names, "shap": mean_shap}
            ).sort_values("shap", ascending=False)
            print("\nTop 20 features by mean |SHAP|:")
            print(shap_df.head(20).to_string(index=False))

            low_shap = shap_df[shap_df["shap"] < 0.001]["feature"].tolist()
            print(f"\nFeatures with mean |SHAP| < 0.001: {len(low_shap)}")
            print("  (documented but not auto-removed in V3)")

            # Save top-5 dependence plots in a grid
            top5_feats = shap_df["feature"].iloc[:5].tolist()
            fig2, axes = plt.subplots(1, 5, figsize=(25, 5))
            for i, feat in enumerate(top5_feats):
                feat_idx = feature_names.index(feat)
                shap.dependence_plot(
                    feat_idx,
                    shap_values,
                    X_train_tree,
                    feature_names=feature_names,
                    ax=axes[i],
                    show=False,
                )
                axes[i].set_title(feat, fontsize=8)
            plt.tight_layout()
            shap_top5_path = os.path.join(base_dir, "shap_top5_v3.png")
            plt.savefig(shap_top5_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"SHAP top-5 dependence saved to {shap_top5_path}")

        except Exception as e:
            print(f"WARNING: SHAP computation failed: {e}")
    else:
        print("SHAP or XGBoost not available — skipping.")

    # =========================================================================
    # ETAPA 13: GENERATE SUBMISSION
    # =========================================================================
    print("\n" + "=" * 60)
    print("ETAPA 13: GENERACIÓN DE SUBMISSION")
    print("=" * 60)

    submission = pd.DataFrame(
        {"Id": test_id, "SalePrice": submission_prices}
    )

    # Validations
    assert len(submission) == 1459, f"Expected 1459 rows, got {len(submission)}"
    assert submission["SalePrice"].isnull().sum() == 0
    assert (submission["SalePrice"] > 0).all()
    assert submission["SalePrice"].min() > MIN_PRICE, (
        f"Min price {submission['SalePrice'].min():.0f} below threshold"
    )
    assert submission["SalePrice"].max() < MAX_PRICE, (
        f"Max price {submission['SalePrice'].max():.0f} above threshold"
    )

    print(f"Submission statistics:")
    print(f"  Count : {len(submission)}")
    print(f"  Mean  : {submission['SalePrice'].mean():.0f}")
    print(f"  Median: {submission['SalePrice'].median():.0f}")
    print(f"  Min   : {submission['SalePrice'].min():.0f}")
    print(f"  Max   : {submission['SalePrice'].max():.0f}")
    print(f"  Std   : {submission['SalePrice'].std():.0f}")

    sub_path = os.path.join(base_dir, "submission_v3.csv")
    submission.to_csv(sub_path, index=False)
    print(f"\nSubmission V3 saved to {sub_path}")

    # Final timing
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    print("\n" + "=" * 60)
    print("PIPELINE V3 COMPLETE")
    print(f"Final Blend OOF RMSLE : {final_blend_rmsle:.5f}")
    print(f"Stacking OOF RMSLE    : {stacking_oof_rmsle:.5f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
