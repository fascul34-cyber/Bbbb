"""
Pipeline скелет: распределение планового объёма по сцепкам
(origin, destination, cargo_type, speed).

Слои:
    A. Калибровка плана (per-origin linear: fact = a + b * plan).
    B. Структура: доли сцепок внутри origin (сезонная база 12 мес).
    Сборка: forecast = share * (calibrated) plan_volume.

Валидация: walk-forward по последним N месяцам.
Метрики: WMAPE на уровне сцепок и на уровне origin.

Чтобы запустить: `python shipment_forecast_pipeline.py`
(работает на встроенной синтетике; подмени `load_data()` на свои данные).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

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
    """Сворачивает сырой факт в панель (origin, dest, cargo, speed, period)."""
    fact = add_period(fact)
    return fact.groupby(KEY + ["period"], as_index=False)["volume"].sum()


# ============================================================================
# 3. Калибровка плана (слой A): bias плана vs факт на уровне origin
# ============================================================================

def fit_plan_calibration(
    plans_history: pd.DataFrame,
    fact_panel: pd.DataFrame,
    min_obs: int = 4,
) -> pd.DataFrame:
    """
    Подгоняет per-origin регрессию fact_origin_total = a + b * plan.
    Если по origin наблюдений мало — fallback на глобальные коэффициенты.
    """
    fact_origin = (
        fact_panel.groupby(["origin", "period"], as_index=False)["volume"]
        .sum()
        .rename(columns={"volume": "fact"})
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
# 4. Структурные модели (слой B): доли сцепок внутри origin
# ============================================================================

def shares_baseline_3m(
    panel: pd.DataFrame,
    target_period: pd.Period,
    weights=(0.5, 0.3, 0.2),
) -> pd.DataFrame:
    """Старый метод: средневзвешенные доли за 3 последних месяца."""
    history = [target_period - i for i in range(1, 4)]
    df = panel[panel["period"].isin(history)].copy()
    df["w"] = df["period"].map({p: w for p, w in zip(history, weights)})
    df["wv"] = df["volume"] * df["w"]
    agg = df.groupby(KEY, as_index=False)["wv"].sum()
    totals = agg.groupby("origin")["wv"].sum().rename("origin_total")
    agg = agg.merge(totals, on="origin")
    agg["share"] = agg["wv"] / agg["origin_total"].replace(0, np.nan)
    return agg[KEY + ["share"]].dropna()


def shares_inverse_error_weights(
    panel: pd.DataFrame,
    plans_history: pd.DataFrame,
    target_period: pd.Period,
    n_months: int = 3,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Та же 3-месячная агрегация, но веса месяцев обратно пропорциональны
    ошибке плана vs факт за этот месяц на уровне origin.
    Так логичнее, чем экспертные веса: «верим тому месяцу, где план не врал».
    """
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
    err["abs_err"] = (err["fact"] - err["plan"]).abs()

    # Один общий вес на месяц (можно делать per-origin, но на коротких рядах шумит).
    monthly = err.groupby("period")["abs_err"].mean()
    inv = 1.0 / (monthly + eps)
    weights = (inv / inv.sum()).to_dict()

    df = panel[panel["period"].isin(history)].copy()
    df["w"] = df["period"].map(weights).fillna(1.0 / n_months)
    df["wv"] = df["volume"] * df["w"]
    agg = df.groupby(KEY, as_index=False)["wv"].sum()
    totals = agg.groupby("origin")["wv"].sum().rename("origin_total")
    agg = agg.merge(totals, on="origin")
    agg["share"] = agg["wv"] / agg["origin_total"].replace(0, np.nan)
    return agg[KEY + ["share"]].dropna()


def shares_seasonal(
    panel: pd.DataFrame,
    target_period: pd.Period,
    base_window: int = 12,
) -> pd.DataFrame:
    """
    Сезонная модель долей:
        base_share  = средняя доля сцепки за последние `base_window` месяцев
        seasonal_k  = (доля сцепки в том же месяце год назад) / base_share
        share       = base_share * seasonal_k, нормированно по origin

    Если в SAM (same-as-month) нет данных — просто base_share.
    """
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
        sam = sam[KEY + ["sam_share"]]
        merged = base.merge(sam, on=KEY, how="left")
        merged["seasonal_k"] = (
            merged["sam_share"] / merged["base_share"]
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        # Подрежем, чтобы единичные выбросы не сносили долю
        merged["seasonal_k"] = merged["seasonal_k"].clip(0.25, 4.0)
        merged["raw_share"] = merged["base_share"] * merged["seasonal_k"]
    else:
        merged = base.rename(columns={"base_share": "raw_share"})

    totals = merged.groupby("origin")["raw_share"].sum().rename("origin_total")
    merged = merged.merge(totals, on="origin")
    merged["share"] = merged["raw_share"] / merged["origin_total"].replace(0, np.nan)
    return merged[KEY + ["share"]].dropna()


# ============================================================================
# 5. Сборка прогноза
# ============================================================================

def forecast(
    plan: pd.DataFrame,
    shares: pd.DataFrame,
    calibration: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """share * (опционально откалиброванный) плановый объём."""
    if calibration is not None:
        cal = apply_plan_calibration(plan, calibration)
        plan_use = cal.rename(columns={"volume_calibrated": "vol"})[["origin", "vol"]]
    else:
        plan_use = plan.rename(columns={"volume": "vol"})[["origin", "vol"]]

    out = shares.merge(plan_use, on="origin", how="left")
    out["volume_forecast"] = out["share"] * out["vol"]
    return out[KEY + ["volume_forecast"]]


# ============================================================================
# 6. Walk-forward валидация
# ============================================================================

# Метод = функция (panel_hist, plans_hist, target_period) -> (shares, cal_or_None)
MethodFn = Callable[
    [pd.DataFrame, pd.DataFrame, pd.Period],
    tuple[pd.DataFrame, Optional[pd.DataFrame]],
]


@dataclass
class FoldResult:
    period: pd.Period
    method: str
    wmape_pairs: float        # на уровне сцепок
    wmape_origin: float       # на уровне origin (сколько ошиблись по объёму отправок)


def walk_forward(
    panel: pd.DataFrame,
    plans: pd.DataFrame,
    eval_periods: list[pd.Period],
    methods: dict[str, MethodFn],
) -> tuple[pd.DataFrame, dict]:
    rows: list[FoldResult] = []
    forecasts: dict = {}

    for tp in eval_periods:
        plan_tp = plans[plans["period"] == tp][["origin", "volume"]]
        fact_tp = panel[panel["period"] == tp][KEY + ["volume"]]
        panel_hist = panel[panel["period"] < tp]
        plans_hist = plans[plans["period"] < tp]

        for name, fn in methods.items():
            shares, cal = fn(panel_hist, plans_hist, tp)
            fc = forecast(plan_tp, shares, calibration=cal)
            merged = fact_tp.merge(fc, on=KEY, how="outer").fillna(0.0)

            origin_actual = merged.groupby("origin")["volume"].sum()
            origin_fc = merged.groupby("origin")["volume_forecast"].sum()

            rows.append(FoldResult(
                period=tp,
                method=name,
                wmape_pairs=wmape(merged["volume"], merged["volume_forecast"]),
                wmape_origin=wmape(origin_actual.values, origin_fc.values),
            ))
            forecasts[(tp, name)] = merged

    return pd.DataFrame([r.__dict__ for r in rows]), forecasts


# ============================================================================
# 7. Реестр методов
# ============================================================================

def m_baseline_3m(panel_hist, plans_hist, tp):
    return shares_baseline_3m(panel_hist, tp), None


def m_inverse_error(panel_hist, plans_hist, tp):
    return shares_inverse_error_weights(panel_hist, plans_hist, tp), None


def m_seasonal(panel_hist, plans_hist, tp):
    return shares_seasonal(panel_hist, tp), None


def m_seasonal_calibrated(panel_hist, plans_hist, tp):
    cal = fit_plan_calibration(plans_hist, panel_hist)
    return shares_seasonal(panel_hist, tp), cal


METHODS: dict[str, MethodFn] = {
    "baseline_3m_expert":     m_baseline_3m,
    "inverse_error_3m":       m_inverse_error,
    "seasonal_12m":           m_seasonal,
    "seasonal_12m + plan_cal": m_seasonal_calibrated,
}


# ============================================================================
# 8. Синтетика, чтобы скрипт запускался из коробки
# ============================================================================

def synth_data(n_months: int = 18) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = ["ST_A", "ST_B", "ST_C", "ST_D"]
    dests = ["DEST_1", "DEST_2", "DEST_3", "DEST_4", "DEST_5"]
    cargos = ["coal", "oil", "metal"]
    speeds = ["СПФ", "не СПФ"]

    # Базовая «реальная» структура: для каждой origin свой набор сцепок и долей
    pairs = []
    for o in origins:
        n = RNG.integers(4, 8)
        for _ in range(n):
            pairs.append((
                o,
                RNG.choice(dests),
                RNG.choice(cargos),
                RNG.choice(speeds),
                RNG.uniform(0.05, 1.0),
            ))
    structure = pd.DataFrame(pairs, columns=KEY + ["w"])
    structure = structure.drop_duplicates(KEY)
    structure["share_true"] = structure.groupby("origin")["w"].transform(
        lambda s: s / s.sum()
    )

    end = pd.Period("2026-03", freq="M")
    periods = [end - i for i in range(n_months)][::-1]

    fact_rows = []
    plan_rows = []
    for p in periods:
        # Сезонность по месяцу года
        season = 1.0 + 0.15 * np.sin(2 * np.pi * (p.month - 1) / 12)
        for o in origins:
            true_volume = RNG.normal(1000, 80) * season
            # plan = смещённый шумный сигнал
            plan_volume = max(0, true_volume * RNG.normal(0.92, 0.07) + RNG.normal(0, 30))
            plan_rows.append((o, plan_volume, p))

            sub = structure[structure["origin"] == o].copy()
            # Лёгкий drift долей внутри origin
            drift = RNG.normal(1.0, 0.08, size=len(sub))
            sub_share = sub["share_true"].values * drift
            sub_share = sub_share / sub_share.sum()
            for (_, dst, cg, sp, *_), sh in zip(sub.itertuples(index=False), sub_share):
                vol = true_volume * sh * RNG.normal(1.0, 0.05)
                if vol > 0:
                    fact_rows.append((o, dst, cg, sp, vol, p.to_timestamp()))

    fact = pd.DataFrame(fact_rows, columns=KEY + ["volume", "data"])
    plan = pd.DataFrame(plan_rows, columns=["origin", "volume", "period"])
    return fact, plan


# ============================================================================
# 9. Демо запуск
# ============================================================================

def main():
    fact, plans = synth_data(n_months=18)
    panel = make_panel(fact)

    all_periods = sorted(panel["period"].unique())
    # Оцениваем на последних 6 месяцах (на ранних истории мало для seasonal_12m)
    eval_periods = all_periods[-6:]

    results, _ = walk_forward(panel, plans, eval_periods, METHODS)

    print("\n=== По месяцам (WMAPE на уровне сцепок) ===")
    print(
        results.pivot(index="period", columns="method", values="wmape_pairs")
        .round(3).to_string()
    )

    print("\n=== По месяцам (WMAPE на уровне origin) ===")
    print(
        results.pivot(index="period", columns="method", values="wmape_origin")
        .round(3).to_string()
    )

    print("\n=== Среднее по периоду оценки ===")
    print(
        results.groupby("method")[["wmape_pairs", "wmape_origin"]]
        .mean().round(3).to_string()
    )

    # ---------------- Куда расти ----------------
    # 1) Croston/SBA для редких сцепок (intermittent demand) — поверх shares_seasonal
    #    как fallback, когда в base_window сцепка появлялась < 3 раз.
    # 2) Подбор весов в shares_baseline через constrained LS на eval_periods,
    #    либо exponential smoothing с одним alpha (cross-validated).
    # 3) Boosting на остатках: фичи (origin, cargo, speed, month, plan, lag-фичи)
    #    -> таргет = (volume_actual - volume_forecast). LightGBM/CatBoost.
    # 4) MinT reconciliation для согласования прогнозов сцепок и origin-итогов.


if __name__ == "__main__":
    main()
