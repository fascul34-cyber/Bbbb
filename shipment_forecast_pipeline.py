"""
Pipeline: распределение планового объёма по сцепкам
(origin, destination, cargo_type, speed).

Слои:
    A. Калибровка плана: per-origin регрессия fact = a + b * plan.
    B. Структура (доли сцепок внутри origin):
        - baseline_3m_expert        — старый метод
        - inverse_error_3m          — веса обратно ошибке plan vs fact
        - optimized_weights         — веса 3 месяцев через constrained LS
        - seasonal_12m              — 12-мес база * сезонный коэф
        - seasonal_12m_croston      — то же + Croston/SBA для редких сцепок
    Сборка: forecast = share * (откалиброванный) plan_volume.

Поверх:
    - boosted_residuals — LightGBM на остатках seasonal_croston + plan_cal
    - MinT-style reconciliation, чтобы прогнозы сцепок сходились с
      origin-итогами после остатков boosting'а.

Запуск: `python shipment_forecast_pipeline.py`
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    import lightgbm as lgb  # type: ignore
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

KEY = ["origin", "destination", "cargo_type", "speed"]
RNG = np.random.default_rng(42)


# ============================================================================
# 1. Метрики
# ============================================================================

def wmape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return np.abs(y_true - y_pred).sum() / denom


# ============================================================================
# 2. Подготовка данных
# ============================================================================

def add_period(df: pd.DataFrame, date_col: str = "data") -> pd.DataFrame:
    df = df.copy()
    df["period"] = pd.to_datetime(df[date_col]).dt.to_period("M")
    return df


def make_panel(fact: pd.DataFrame) -> pd.DataFrame:
    fact = add_period(fact)
    return fact.groupby(KEY + ["period"], as_index=False)["volume"].sum()


# ============================================================================
# 3. Калибровка плана (слой A)
# ============================================================================

def fit_plan_calibration(
    plans_history: pd.DataFrame,
    fact_panel: pd.DataFrame,
    min_obs: int = 4,
) -> pd.DataFrame:
    """fact_origin_total = a + b * plan, per-origin с глобальным fallback."""
    fact_origin = (
        fact_panel.groupby(["origin", "period"], as_index=False)["volume"]
        .sum().rename(columns={"volume": "fact"})
    )
    df = plans_history.merge(fact_origin, on=["origin", "period"], how="inner")

    if len(df) >= 3:
        b_g, a_g = np.polyfit(df["volume"].values, df["fact"].values, 1)
    else:
        a_g, b_g = 0.0, 1.0

    rows = []
    for origin, g in df.groupby("origin"):
        if len(g) >= min_obs:
            b, a = np.polyfit(g["volume"].values, g["fact"].values, 1)
        else:
            a, b = a_g, b_g
        rows.append((origin, a, b))

    cal = pd.DataFrame(rows, columns=["origin", "a", "b"])
    cal.attrs["global"] = (float(a_g), float(b_g))
    return cal


def apply_plan_calibration(plan: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    a_g, b_g = cal.attrs.get("global", (0.0, 1.0))
    out = plan.merge(cal, on="origin", how="left")
    out["a"] = out["a"].fillna(a_g)
    out["b"] = out["b"].fillna(b_g)
    out["volume_calibrated"] = (out["a"] + out["b"] * out["volume"]).clip(lower=0)
    return out[["origin", "volume", "volume_calibrated"]]


# ============================================================================
# 4. Структурные модели (слой B)
# ============================================================================

def _shares_from_weighted(panel_window: pd.DataFrame, period_weights: dict) -> pd.DataFrame:
    df = panel_window.copy()
    df["w"] = df["period"].map(period_weights).fillna(0.0)
    df["wv"] = df["volume"] * df["w"]
    agg = df.groupby(KEY, as_index=False)["wv"].sum()
    totals = agg.groupby("origin")["wv"].sum().rename("origin_total")
    agg = agg.merge(totals, on="origin")
    agg["share"] = agg["wv"] / agg["origin_total"].replace(0, np.nan)
    return agg[KEY + ["share"]].dropna()


def shares_baseline_3m(panel, target_period, weights=(0.5, 0.3, 0.2)) -> pd.DataFrame:
    history = [target_period - i for i in range(1, 4)]
    df = panel[panel["period"].isin(history)]
    return _shares_from_weighted(df, dict(zip(history, weights)))


def shares_inverse_error_weights(
    panel, plans_history, target_period, n_months=3, eps=1e-6,
) -> pd.DataFrame:
    """3-мес агрегация с весами обратно средней |plan - fact| по origin."""
    history = [target_period - i for i in range(1, n_months + 1)]
    fact_origin = (
        panel[panel["period"].isin(history)]
        .groupby(["origin", "period"], as_index=False)["volume"].sum()
        .rename(columns={"volume": "fact"})
    )
    plan_origin = plans_history[plans_history["period"].isin(history)][
        ["origin", "period", "volume"]
    ].rename(columns={"volume": "plan"})
    err = fact_origin.merge(plan_origin, on=["origin", "period"], how="inner")
    if not len(err):
        return shares_baseline_3m(panel, target_period)
    err["abs_err"] = (err["fact"] - err["plan"]).abs()
    monthly = err.groupby("period")["abs_err"].mean()
    inv = 1.0 / (monthly + eps)
    period_w = (inv / inv.sum()).to_dict()
    return _shares_from_weighted(panel[panel["period"].isin(history)], period_w)


# -------------------------- Optimized weights --------------------------------

def fit_optimized_weights(
    panel: pd.DataFrame,
    n_months: int = 3,
    inner_val_months: int = 3,
) -> np.ndarray:
    """
    Подбирает веса последних n_months через constrained LS на inner walk-forward
    валидации. Минимизирует суммарную |fact - share*actual_origin_total|
    по нескольким последним месяцам истории.

    sum(w) = 1, w_i >= 0.
    """
    periods = sorted(panel["period"].unique())
    if len(periods) < n_months + inner_val_months + 1:
        return np.array([0.5, 0.3, 0.2][:n_months])

    val_periods = periods[-inner_val_months:]

    def objective(w):
        total_err = 0.0
        for tp in val_periods:
            history = [tp - i for i in range(1, n_months + 1)]
            window = panel[panel["period"].isin(history)]
            shares = _shares_from_weighted(window, dict(zip(history, w)))

            actual = panel[panel["period"] == tp][KEY + ["volume"]]
            origin_actual = (
                actual.groupby("origin")["volume"].sum().rename("origin_total")
            )
            fc = shares.merge(origin_actual, on="origin", how="left")
            fc["forecast"] = fc["share"] * fc["origin_total"]
            merged = actual.merge(fc[KEY + ["forecast"]], on=KEY, how="outer").fillna(0)
            total_err += np.abs(merged["volume"] - merged["forecast"]).sum()
        return total_err

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n_months
    x0 = np.full(n_months, 1.0 / n_months)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else x0


def shares_optimized_weights(panel, target_period, n_months=3) -> pd.DataFrame:
    w = fit_optimized_weights(panel, n_months=n_months)
    history = [target_period - i for i in range(1, n_months + 1)]
    df = panel[panel["period"].isin(history)]
    return _shares_from_weighted(df, dict(zip(history, w)))


# -------------------------- Seasonal + Croston -------------------------------

def shares_seasonal(panel, target_period, base_window=12) -> pd.DataFrame:
    """12-мес база * сезонный коэф (отношение доли в SAM к базе)."""
    history_periods = [target_period - i for i in range(1, base_window + 1)]
    hist = panel[panel["period"].isin(history_periods)]

    base = hist.groupby(KEY, as_index=False)["volume"].sum()
    base_total = base.groupby("origin")["volume"].sum().rename("origin_total")
    base = base.merge(base_total, on="origin")
    base["base_share"] = base["volume"] / base["origin_total"].replace(0, np.nan)
    base = base[KEY + ["base_share"]]

    sam_period = target_period - 12
    sam = panel[panel["period"] == sam_period]
    if len(sam):
        sam_total = sam.groupby("origin")["volume"].sum().rename("origin_total")
        sam = sam.merge(sam_total, on="origin")
        sam["sam_share"] = sam["volume"] / sam["origin_total"].replace(0, np.nan)
        merged = base.merge(sam[KEY + ["sam_share"]], on=KEY, how="left")
        merged["seasonal_k"] = (
            merged["sam_share"] / merged["base_share"]
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.25, 4.0)
        merged["raw_share"] = merged["base_share"] * merged["seasonal_k"]
    else:
        merged = base.rename(columns={"base_share": "raw_share"})

    totals = merged.groupby("origin")["raw_share"].sum().rename("origin_total")
    merged = merged.merge(totals, on="origin")
    merged["share"] = merged["raw_share"] / merged["origin_total"].replace(0, np.nan)
    return merged[KEY + ["share"]].dropna()


def croston_per_pair_volume(
    panel: pd.DataFrame,
    target_period: pd.Period,
    window: int = 12,
    sba_alpha: float = 0.1,
) -> pd.DataFrame:
    """
    SBA (Syntetos-Boylan Approximation) — оценка ожидаемого помесячного объёма
    по разреженному ряду. Возвращает (origin, dest, cargo, speed, expected_volume).

        size_avg     — средний объём в месяцах с поставкой
        interval     — window / число месяцев с поставкой
        forecast     = size_avg / interval * (1 - alpha/2)

    Для плотных рядов работает как обычное среднее, для редких — без перекоса.
    """
    history = [target_period - i for i in range(1, window + 1)]
    h = panel[panel["period"].isin(history)].copy()

    if not len(h):
        return pd.DataFrame(columns=KEY + ["expected_volume", "n_obs"])

    g = h.groupby(KEY).agg(
        size_avg=("volume", lambda v: v[v > 0].mean() if (v > 0).any() else 0.0),
        n_obs=("volume", lambda v: int((v > 0).sum())),
    ).reset_index()
    g["interval"] = window / g["n_obs"].replace(0, np.nan)
    g["expected_volume"] = (
        g["size_avg"] / g["interval"] * (1 - sba_alpha / 2)
    ).fillna(0.0)
    return g[KEY + ["expected_volume", "n_obs"]]


def shares_seasonal_croston(
    panel: pd.DataFrame,
    target_period: pd.Period,
    window: int = 12,
    sparse_threshold: int = 4,
) -> pd.DataFrame:
    """
    Гибрид: для плотных сцепок (n_obs >= sparse_threshold) — seasonal_12m,
    для редких — SBA. Затем нормировка по origin.
    """
    seasonal = shares_seasonal(panel, target_period, base_window=window)
    croston = croston_per_pair_volume(panel, target_period, window=window)

    # Конвертируем seasonal-доли в "ожидаемый объём", чтобы сложить с Croston.
    # Для шкалы используем средний помесячный origin total за окно.
    history = [target_period - i for i in range(1, window + 1)]
    origin_avg = (
        panel[panel["period"].isin(history)]
        .groupby(["origin", "period"])["volume"].sum()
        .groupby("origin").mean().rename("origin_avg").reset_index()
    )

    seasonal = seasonal.merge(origin_avg, on="origin", how="left")
    seasonal["seasonal_volume"] = seasonal["share"] * seasonal["origin_avg"]
    seasonal = seasonal[KEY + ["seasonal_volume"]]

    merged = croston.merge(seasonal, on=KEY, how="outer").fillna(0.0)
    merged["expected"] = np.where(
        merged["n_obs"] >= sparse_threshold,
        merged["seasonal_volume"],
        merged["expected_volume"],
    )

    totals = merged.groupby("origin")["expected"].sum().rename("origin_total")
    merged = merged.merge(totals, on="origin")
    merged["share"] = merged["expected"] / merged["origin_total"].replace(0, np.nan)
    return merged[KEY + ["share"]].dropna()


# ============================================================================
# 5. Reconciliation (MinT-style, простая версия)
# ============================================================================

def reconcile_pairs_to_origin(
    pairs_forecast: pd.DataFrame,
    origin_target: pd.DataFrame,
    method: str = "wls",
) -> pd.DataFrame:
    """
    Подгоняет прогнозы сцепок так, чтобы их сумма внутри каждого origin
    точно совпадала с заданным origin-итогом (например, calibrated plan).

    method:
        "ols"  — равномерная корректировка: b_i = b̂_i + (T - sum)/n
        "wls"  — пропорционально базовому прогнозу (в духе MinT с
                 W = diag(b̂)): крупные сцепки получают большую правку
    """
    df = pairs_forecast.merge(
        origin_target.rename(columns={"volume": "target"})[["origin", "target"]],
        on="origin", how="left",
    )
    if method == "ols":
        sums = df.groupby("origin")["volume_forecast"].transform("sum")
        n = df.groupby("origin")["volume_forecast"].transform("size")
        df["volume_forecast"] = df["volume_forecast"] + (df["target"] - sums) / n
    else:  # wls / proportional
        sums = df.groupby("origin")["volume_forecast"].transform("sum")
        # Если sum == 0, fallback на равные доли
        n = df.groupby("origin")["volume_forecast"].transform("size")
        scale = np.where(sums > 0, df["target"] / sums.replace(0, np.nan), np.nan)
        equal_split = df["target"] / n
        df["volume_forecast"] = np.where(
            sums > 0, df["volume_forecast"] * scale, equal_split,
        )
    df["volume_forecast"] = df["volume_forecast"].clip(lower=0)
    return df[KEY + ["volume_forecast"]]


# ============================================================================
# 6. Сборка прогноза
# ============================================================================

def forecast(plan, shares, calibration=None) -> pd.DataFrame:
    if calibration is not None:
        cal = apply_plan_calibration(plan, calibration)
        plan_use = cal.rename(columns={"volume_calibrated": "vol"})[["origin", "vol"]]
    else:
        plan_use = plan.rename(columns={"volume": "vol"})[["origin", "vol"]]
    out = shares.merge(plan_use, on="origin", how="left")
    out["volume_forecast"] = (out["share"] * out["vol"]).fillna(0.0)
    return out[KEY + ["volume_forecast"]]


# ============================================================================
# 7. Boosting на остатках (LightGBM)
# ============================================================================

def _build_features(
    pairs_with_forecast: pd.DataFrame,
    target_period: pd.Period,
    panel: pd.DataFrame,
    plan: pd.DataFrame,
) -> pd.DataFrame:
    """Простой набор фич для каждой строки прогноза."""
    df = pairs_with_forecast.copy()
    df["month"] = target_period.month
    df["year"] = target_period.year
    df["period_idx"] = (target_period - pd.Period("2020-01", freq="M")).n

    # Лаги факта по сцепке
    for lag in (1, 3, 12):
        lag_p = target_period - lag
        lag_df = (
            panel[panel["period"] == lag_p]
            .groupby(KEY)["volume"].sum().rename(f"lag_{lag}").reset_index()
        )
        df = df.merge(lag_df, on=KEY, how="left")

    # Сколько месяцев из последних 12 сцепка появлялась
    last12 = [target_period - i for i in range(1, 13)]
    n_obs = (
        panel[panel["period"].isin(last12)]
        .groupby(KEY)["volume"].apply(lambda v: int((v > 0).sum()))
        .rename("n_obs_12").reset_index()
    )
    df = df.merge(n_obs, on=KEY, how="left")

    # План (origin-уровень)
    df = df.merge(
        plan.rename(columns={"volume": "plan_origin"})[["origin", "plan_origin"]],
        on="origin", how="left",
    )

    df = df.fillna(0.0)
    return df


def boosted_forecast(
    panel_hist: pd.DataFrame,
    plans_hist: pd.DataFrame,
    plan_target: pd.DataFrame,
    target_period: pd.Period,
    train_lookback: int = 8,
    base_method: Callable = shares_seasonal_croston,
) -> pd.DataFrame:
    """
    Базовая модель: seasonal_croston + plan_cal.
    Boosting предсказывает residual = fact - base, фичи задаются _build_features.
    Финальный прогноз reconciled под calibrated plan.
    """
    if not _HAS_LGB:
        raise RuntimeError("lightgbm не установлен")

    # ---- 1. Собираем training set по прошлым target_period ----
    history_periods = sorted(panel_hist["period"].unique())
    train_periods = [
        p for p in history_periods
        if (p - 13) >= history_periods[0]  # хватает истории на base + лаги
    ][-train_lookback:]

    if len(train_periods) < 2:
        # Слишком мало истории — отдаём базовый прогноз без boosting'а
        cal = fit_plan_calibration(plans_hist, panel_hist)
        shares = base_method(panel_hist, target_period)
        return forecast(plan_target, shares, calibration=cal)

    train_frames = []
    for tp in train_periods:
        sub_panel = panel_hist[panel_hist["period"] < tp]
        sub_plans = plans_hist[plans_hist["period"] < tp]
        sub_plan_tp = plans_hist[plans_hist["period"] == tp][["origin", "volume"]]
        if not len(sub_plan_tp):
            continue

        cal_sub = fit_plan_calibration(sub_plans, sub_panel)
        shares_sub = base_method(sub_panel, tp)
        base_fc = forecast(sub_plan_tp, shares_sub, calibration=cal_sub)

        actual = (
            panel_hist[panel_hist["period"] == tp]
            .groupby(KEY, as_index=False)["volume"].sum()
        )
        merged = base_fc.merge(actual, on=KEY, how="outer").fillna(0.0)
        merged["residual"] = merged["volume"] - merged["volume_forecast"]

        feats = _build_features(merged, tp, sub_panel, sub_plan_tp)
        train_frames.append(feats)

    train = pd.concat(train_frames, ignore_index=True)

    # ---- 2. Обучаем LightGBM ----
    feature_cols = [
        "month", "year", "period_idx",
        "lag_1", "lag_3", "lag_12",
        "n_obs_12", "plan_origin",
        "volume_forecast",  # base prediction как фича
    ]
    cat_cols = KEY  # категориальные через native LightGBM
    X = train[feature_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    y = train["residual"].values

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        verbosity=-1,
    )
    model.fit(X, y, categorical_feature=cat_cols)

    # ---- 3. Прогноз на target_period ----
    cal = fit_plan_calibration(plans_hist, panel_hist)
    shares = base_method(panel_hist, target_period)
    base_fc = forecast(plan_target, shares, calibration=cal)

    feats_target = _build_features(base_fc, target_period, panel_hist, plan_target)
    Xt = feats_target[feature_cols + cat_cols].copy()
    for c in cat_cols:
        Xt[c] = Xt[c].astype("category")
    residual_pred = model.predict(Xt)
    base_fc["volume_forecast"] = (base_fc["volume_forecast"] + residual_pred).clip(lower=0)

    # ---- 4. Reconciliation под calibrated plan ----
    cal_plan = apply_plan_calibration(plan_target, cal)[
        ["origin", "volume_calibrated"]
    ].rename(columns={"volume_calibrated": "volume"})
    return reconcile_pairs_to_origin(base_fc, cal_plan, method="wls")


# ============================================================================
# 8. Walk-forward валидация
# ============================================================================

# Метод = функция (panel_hist, plans_hist, plan_target, target_period) -> forecast_df
MethodFn = Callable[
    [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Period],
    pd.DataFrame,
]


@dataclass
class FoldResult:
    period: pd.Period
    method: str
    wmape_pairs: float
    wmape_origin: float


def walk_forward(
    panel: pd.DataFrame,
    plans: pd.DataFrame,
    eval_periods: list[pd.Period],
    methods: dict[str, MethodFn],
) -> tuple[pd.DataFrame, dict]:
    rows: list[FoldResult] = []
    out: dict = {}

    for tp in eval_periods:
        plan_tp = plans[plans["period"] == tp][["origin", "volume"]]
        fact_tp = panel[panel["period"] == tp][KEY + ["volume"]]
        panel_hist = panel[panel["period"] < tp]
        plans_hist = plans[plans["period"] < tp]

        for name, fn in methods.items():
            fc = fn(panel_hist, plans_hist, plan_tp, tp)
            merged = fact_tp.merge(fc, on=KEY, how="outer").fillna(0.0)

            origin_actual = merged.groupby("origin")["volume"].sum()
            origin_fc = merged.groupby("origin")["volume_forecast"].sum()

            rows.append(FoldResult(
                period=tp,
                method=name,
                wmape_pairs=wmape(merged["volume"], merged["volume_forecast"]),
                wmape_origin=wmape(origin_actual.values, origin_fc.values),
            ))
            out[(tp, name)] = merged

    return pd.DataFrame([r.__dict__ for r in rows]), out


# ============================================================================
# 9. Реестр методов
# ============================================================================

def m_baseline_3m(panel_hist, plans_hist, plan_tp, tp):
    return forecast(plan_tp, shares_baseline_3m(panel_hist, tp))


def m_inverse_error(panel_hist, plans_hist, plan_tp, tp):
    return forecast(plan_tp, shares_inverse_error_weights(panel_hist, plans_hist, tp))


def m_optimized_weights(panel_hist, plans_hist, plan_tp, tp):
    return forecast(plan_tp, shares_optimized_weights(panel_hist, tp))


def m_seasonal(panel_hist, plans_hist, plan_tp, tp):
    return forecast(plan_tp, shares_seasonal(panel_hist, tp))


def m_seasonal_calibrated(panel_hist, plans_hist, plan_tp, tp):
    cal = fit_plan_calibration(plans_hist, panel_hist)
    return forecast(plan_tp, shares_seasonal(panel_hist, tp), calibration=cal)


def m_seasonal_croston_calibrated(panel_hist, plans_hist, plan_tp, tp):
    cal = fit_plan_calibration(plans_hist, panel_hist)
    return forecast(plan_tp, shares_seasonal_croston(panel_hist, tp), calibration=cal)


def m_boosted(panel_hist, plans_hist, plan_tp, tp):
    return boosted_forecast(panel_hist, plans_hist, plan_tp, tp)


METHODS: dict[str, MethodFn] = {
    "baseline_3m_expert":          m_baseline_3m,
    "inverse_error_3m":            m_inverse_error,
    "optimized_weights_3m":        m_optimized_weights,
    "seasonal_12m":                m_seasonal,
    "seasonal_12m + plan_cal":     m_seasonal_calibrated,
    "seasonal_croston + plan_cal": m_seasonal_croston_calibrated,
}
if _HAS_LGB:
    METHODS["boosted_residuals"] = m_boosted


# ============================================================================
# 10. Синтетика
# ============================================================================

def synth_data(n_months: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = ["ST_A", "ST_B", "ST_C", "ST_D"]
    dests = ["DEST_1", "DEST_2", "DEST_3", "DEST_4", "DEST_5"]
    cargos = ["coal", "oil", "metal"]
    speeds = ["СПФ", "не СПФ"]

    # Структура: смесь плотных и редких сцепок
    pairs = []
    for o in origins:
        n = RNG.integers(5, 10)
        for _ in range(n):
            pairs.append((
                o,
                RNG.choice(dests),
                RNG.choice(cargos),
                RNG.choice(speeds),
                RNG.uniform(0.05, 1.0),
                RNG.uniform(0.4, 1.0),  # вероятность появления в месяце
            ))
    structure = pd.DataFrame(pairs, columns=KEY + ["w", "p_occur"])
    structure = structure.drop_duplicates(KEY)
    structure["share_true"] = structure.groupby("origin")["w"].transform(
        lambda s: s / s.sum()
    )

    end = pd.Period("2026-03", freq="M")
    periods = [end - i for i in range(n_months)][::-1]

    fact_rows, plan_rows = [], []
    for p in periods:
        season = 1.0 + 0.18 * np.sin(2 * np.pi * (p.month - 1) / 12)
        for o in origins:
            true_volume = RNG.normal(1000, 80) * season
            plan_volume = max(0, true_volume * RNG.normal(0.92, 0.07) + RNG.normal(0, 30))
            plan_rows.append((o, plan_volume, p))

            sub = structure[structure["origin"] == o]
            for row in sub.itertuples(index=False):
                if RNG.random() > row.p_occur:
                    continue  # пропуск месяца — будет sparse pattern
                drift = RNG.normal(1.0, 0.08)
                vol = true_volume * row.share_true * drift * RNG.normal(1.0, 0.05)
                if vol > 0:
                    fact_rows.append((
                        row.origin, row.destination, row.cargo_type, row.speed,
                        vol, p.to_timestamp(),
                    ))

    fact = pd.DataFrame(fact_rows, columns=KEY + ["volume", "data"])
    plan = pd.DataFrame(plan_rows, columns=["origin", "volume", "period"])
    return fact, plan


# ============================================================================
# 11. Демо
# ============================================================================

def main():
    fact, plans = synth_data(n_months=24)
    panel = make_panel(fact)

    all_periods = sorted(panel["period"].unique())
    eval_periods = all_periods[-6:]

    print(f"Месяцев истории: {len(all_periods)}, eval: {len(eval_periods)}")
    print(f"Уникальных сцепок: {panel[KEY].drop_duplicates().shape[0]}")
    print(f"LightGBM доступен: {_HAS_LGB}\n")

    results, _ = walk_forward(panel, plans, eval_periods, METHODS)

    print("=== По месяцам, WMAPE на сцепках ===")
    print(
        results.pivot(index="period", columns="method", values="wmape_pairs")
        .round(3).to_string()
    )

    print("\n=== По месяцам, WMAPE на origin-итогах ===")
    print(
        results.pivot(index="period", columns="method", values="wmape_origin")
        .round(3).to_string()
    )

    print("\n=== Среднее по eval-периоду ===")
    summary = (
        results.groupby("method")[["wmape_pairs", "wmape_origin"]]
        .mean().round(3).sort_values("wmape_pairs")
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
