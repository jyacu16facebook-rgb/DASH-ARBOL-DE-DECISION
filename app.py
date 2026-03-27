# app.py
# ==========================================================
# STREAMLIT APP: Árbol de decisión univariable
# Fenología vs métricas productivas
# Optimizado para EXACTAMENTE 3 nodos finales (3 hojas)
# + Poda dinámica por tamaño mínimo de hoja (7% del N filtrado)
# + Boxplot por regla del árbol con ANOVA
# + Importancia de variables (Random Forest + Optuna)
# ==========================================================

import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import optuna

from sklearn.tree import DecisionTreeRegressor, export_text, _tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from scipy.stats import f_oneway

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(
    page_title="Árbol de decisión | Fenología vs rendimiento",
    layout="wide"
)

REQ_SHEET = "DATA"
DATA_FILE = "CONSOLIDADO 2022-2026.xlsb"

MAX_DEPTH = 3
MAX_LEAF_NODES = 3
RANDOM_STATE = 42

# PODA DINÁMICA
MIN_LEAF_RATIO = 0.07   # 7% del N filtrado
MIN_LEAF_FLOOR = 2      # mínimo absoluto para no permitir hojas de 1 observación

RF_N_ESTIMATORS = 500
OPTUNA_TRIALS = 15

# ==========================================================
# COLUMNAS REQUERIDAS
# ==========================================================
COLS_REQUIRED = [
    "AÑO", "CAMPAÑA", "SEMANA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
    "kilogramos",
    "Ha COSECHADA", "Ha TURNO", "DENSIDAD",
    "KG/HA",
    "PESO BAYA (g)", "CALIBRE BAYA (mm)",
    "MADERAS PRINCIPALES", "CORTES", "BROTES TOTALES", "TERMINALES",
    "EDAD PLANTA", "EDAD PLANTA FINAL",
    "BP_N_BROTES_ULT", "BP_LONG_ULT", "BP_DIAM_ULT",
    "BS_N_BROTES_ULT", "BS_LONG_ULT", "BS_DIAM_ULT",
    "BT_N_BROTES_ULT", "BT_LONG_ULT", "BT_DIAM_ULT",
    "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT",
    "SIEMBRA FINAL", "SEG DENSIDAD"
]

UNIT_COLS_BASE = ["CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]

Y_OPTIONS = {
    "KG/HA": "KG/HA",
    "KG/PLANTA": "KG/PLANTA",
    "PESO BAYA (g)": "PESO BAYA (g)",
    "CALIBRE BAYA (mm)": "CALIBRE BAYA (mm)",
}

X_OPTIONS = {
    "MADERAS PRINCIPALES": "MADERAS PRINCIPALES",
    "CORTES": "CORTES",
    "BROTES TOTALES": "BROTES TOTALES",
    "TERMINALES": "TERMINALES",
    "BP_N_BROTES_ULT": "BP_N_BROTES_ULT",
    "BP_LONG_ULT": "BP_LONG_ULT",
    "BP_DIAM_ULT": "BP_DIAM_ULT",
    "BS_N_BROTES_ULT": "BS_N_BROTES_ULT",
    "BS_LONG_ULT": "BS_LONG_ULT",
    "BS_DIAM_ULT": "BS_DIAM_ULT",
    "BT_N_BROTES_ULT": "BT_N_BROTES_ULT",
    "BT_LONG_ULT": "BT_LONG_ULT",
    "BT_DIAM_ULT": "BT_DIAM_ULT",
    "ALTURA_PLANTA_ULT": "ALTURA_PLANTA_ULT",
    "ANCHO_PLANTA_ULT": "ANCHO_PLANTA_ULT",
}

RF_FEATURES = [
    "MADERAS PRINCIPALES",
    "CORTES",
    "BROTES TOTALES",
    "TERMINALES",
    "BP_N_BROTES_ULT",
    "BP_LONG_ULT",
    "BP_DIAM_ULT",
    "BS_N_BROTES_ULT",
    "BS_LONG_ULT",
    "BS_DIAM_ULT",
    "BT_N_BROTES_ULT",
    "BT_LONG_ULT",
    "BT_DIAM_ULT",
    "ALTURA_PLANTA_ULT",
    "ANCHO_PLANTA_ULT",
]

X_LABELS = {
    "MADERAS PRINCIPALES": "MADERAS PRINCIPALES (CONTEO)",
    "CORTES": "CORTES (CONTEO)",
    "BROTES TOTALES": "BROTES TOTALES (CONTEO)",
    "TERMINALES": "TERMINALES (CONTEO)",
    "BP_N_BROTES_ULT": "BP_N_BROTES_ULT (CONTEO)",
    "BP_LONG_ULT": "BP_LONG_ULT (cm)",
    "BP_DIAM_ULT": "BP_DIAM_ULT (mm)",
    "BS_N_BROTES_ULT": "BS_N_BROTES_ULT (CONTEO)",
    "BS_LONG_ULT": "BS_LONG_ULT (cm)",
    "BS_DIAM_ULT": "BS_DIAM_ULT (mm)",
    "BT_N_BROTES_ULT": "BT_N_BROTES_ULT (CONTEO)",
    "BT_LONG_ULT": "BT_LONG_ULT (cm)",
    "BT_DIAM_ULT": "BT_DIAM_ULT (mm)",
    "ALTURA_PLANTA_ULT": "ALTURA_PLANTA_ULT (cm)",
    "ANCHO_PLANTA_ULT": "ANCHO_PLANTA_ULT (cm)",
}

# ==========================================================
# HELPERS
# ==========================================================
def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def first_valid(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if not s.empty else np.nan


def simple_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return np.nan
    return float(x.mean(skipna=True))


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))


def sum_numeric(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").sum(skipna=True))


def _sort_campaign_categories(campaign_series: pd.Series):
    uniq = campaign_series.dropna().astype(str).unique().tolist()

    def to_int_or_big(x):
        try:
            return int(str(x).strip())
        except Exception:
            return 10**9

    uniq_sorted = sorted(uniq, key=lambda x: (to_int_or_big(x), str(x)))
    return uniq_sorted


def ensure_categories_age(df: pd.DataFrame) -> pd.DataFrame:
    if "EDAD PLANTA FINAL" in df.columns:
        df["EDAD PLANTA FINAL"] = df["EDAD PLANTA FINAL"].astype(str).str.strip()
        df.loc[df["EDAD PLANTA FINAL"].isin(["3", "3.0", "3.00"]), "EDAD PLANTA FINAL"] = "3+"
        order = ["1", "2", "3+"]
        df["EDAD PLANTA FINAL"] = pd.Categorical(df["EDAD PLANTA FINAL"], categories=order, ordered=True)
    return df


@st.cache_data(show_spinner=False)
def read_excel_path(path: str, sheet: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsb"):
        return pd.read_excel(path, sheet_name=sheet, engine="pyxlsb")
    return pd.read_excel(path, sheet_name=sheet)


def validate_cols(df: pd.DataFrame) -> list:
    return [c for c in COLS_REQUIRED if c not in df.columns]


def build_unique_turno_table(df_subset: pd.DataFrame) -> pd.DataFrame:
    if df_subset.empty:
        return pd.DataFrame(columns=UNIT_COLS_BASE + ["Ha_TURNO_u", "DENSIDAD_u"])

    base = (
        df_subset.groupby(UNIT_COLS_BASE, dropna=False)
        .agg(
            Ha_TURNO_u=("Ha TURNO", first_valid),
            DENSIDAD_u=("DENSIDAD", first_valid),
        )
        .reset_index()
    )

    base["Ha_TURNO_u"] = to_numeric_safe(base["Ha_TURNO_u"])
    base["DENSIDAD_u"] = to_numeric_safe(base["DENSIDAD_u"])
    return base


def unique_turno_area_sum(df_subset: pd.DataFrame) -> float:
    base = build_unique_turno_table(df_subset)
    if base.empty:
        return 0.0
    return float(base["Ha_TURNO_u"].sum(skipna=True))


def unique_turno_plants_sum(df_subset: pd.DataFrame) -> float:
    base = build_unique_turno_table(df_subset)
    if base.empty:
        return 0.0
    base["PLANTAS_EST"] = base["Ha_TURNO_u"] * base["DENSIDAD_u"]
    return float(base["PLANTAS_EST"].sum(skipna=True))


def ratio_kg_over_unique_turno_area(df_subset: pd.DataFrame) -> float:
    kg_sum = sum_numeric(df_subset["kilogramos"])
    area_sum = unique_turno_area_sum(df_subset)
    if area_sum <= 0:
        return np.nan
    return kg_sum / area_sum


def ratio_kg_planta_over_unique_turno(df_subset: pd.DataFrame) -> float:
    kg_sum = sum_numeric(df_subset["kilogramos"])
    plantas_sum = unique_turno_plants_sum(df_subset)
    if plantas_sum <= 0:
        return np.nan
    return kg_sum / plantas_sum


def compute_metric_value(df_subset: pd.DataFrame, metric_name: str) -> float:
    if metric_name == "KG/HA":
        return ratio_kg_over_unique_turno_area(df_subset)
    if metric_name == "KG/PLANTA":
        return ratio_kg_planta_over_unique_turno(df_subset)
    if metric_name == "PESO BAYA (g)":
        return weighted_mean(df_subset["PESO BAYA (g)"], df_subset["kilogramos"])
    if metric_name == "CALIBRE BAYA (mm)":
        return weighted_mean(df_subset["CALIBRE BAYA (mm)"], df_subset["kilogramos"])
    return np.nan


def apply_filters(
    df: pd.DataFrame,
    camp, fundo, etapa, campo, turno, variedad, edad_final,
    siembra_final, seg_densidad,
    semana_min, semana_max
):
    dff = df.copy()

    if camp:
        dff = dff[dff["CAMPAÑA"].isin(camp)]
    if fundo:
        dff = dff[dff["FUNDO"].isin(fundo)]
    if etapa:
        dff = dff[dff["ETAPA"].isin(etapa)]
    if campo:
        dff = dff[dff["CAMPO"].isin(campo)]
    if turno:
        dff = dff[dff["TURNO"].isin(turno)]
    if variedad:
        dff = dff[dff["VARIEDAD"].isin(variedad)]
    if edad_final:
        dff = dff[dff["EDAD PLANTA FINAL"].isin(edad_final)]
    if siembra_final:
        dff = dff[dff["SIEMBRA FINAL"].isin(siembra_final)]
    if seg_densidad:
        dff = dff[dff["SEG DENSIDAD"].isin(seg_densidad)]

    dff = dff[(dff["SEMANA"] >= semana_min) & (dff["SEMANA"] <= semana_max)]
    return dff


def build_model_df(dff: pd.DataFrame, x_col: str, y_metric: str) -> pd.DataFrame:
    if dff.empty:
        return pd.DataFrame()

    rows = []
    for keys, g in dff.groupby(UNIT_COLS_BASE, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        rec = {col: keys[i] for i, col in enumerate(UNIT_COLS_BASE)}
        rec["X_VAL"] = simple_mean(g[x_col])
        rec["Y_VAL"] = compute_metric_value(g, y_metric)
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["CAMPAÑA"] = out["CAMPAÑA"].astype(str)
    out["X_VAL"] = to_numeric_safe(out["X_VAL"])
    out["Y_VAL"] = to_numeric_safe(out["Y_VAL"])
    out = out.dropna(subset=["X_VAL", "Y_VAL"]).copy()

    return out


def build_rf_model_df(dff: pd.DataFrame, y_metric: str) -> pd.DataFrame:
    if dff.empty:
        return pd.DataFrame()

    rows = []
    for keys, g in dff.groupby(UNIT_COLS_BASE, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        rec = {col: keys[i] for i, col in enumerate(UNIT_COLS_BASE)}
        rec["Y_VAL"] = compute_metric_value(g, y_metric)

        for feat in RF_FEATURES:
            rec[feat] = simple_mean(g[feat])

        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["CAMPAÑA"] = out["CAMPAÑA"].astype(str)
    for feat in RF_FEATURES:
        out[feat] = to_numeric_safe(out[feat])
    out["Y_VAL"] = to_numeric_safe(out["Y_VAL"])

    out = out.dropna(subset=["Y_VAL"]).copy()
    out = out.loc[out[RF_FEATURES].notna().any(axis=1)].copy()

    return out


def compute_axis_range(series: pd.Series, lower_zero: bool = True):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None

    real_min = float(s.min())
    real_max = float(s.max())

    if real_min == real_max:
        pad = max(abs(real_max) * 0.10, 1)
        low = real_min - pad
        high = real_max + pad
    else:
        pad = (real_max - real_min) * 0.08
        low = real_min - pad
        high = real_max + pad

    if lower_zero and high > 0:
        low = max(0, low)

    if high <= low:
        high = low + 1

    return [low, high]


def compute_dynamic_tree_params(n_obs: int) -> dict:
    """
    Poda dinámica basada en el N filtrado del modelo univariable.
    Regla principal:
    - min_samples_leaf = max(2, ceil(7% del N))
    - min_samples_split = al menos el doble del tamaño mínimo de hoja
    """
    min_leaf = max(MIN_LEAF_FLOOR, int(math.ceil(MIN_LEAF_RATIO * n_obs)))
    min_split = max(4, 2 * min_leaf)

    return {
        "max_depth": MAX_DEPTH,
        "max_leaf_nodes": MAX_LEAF_NODES,
        "random_state": RANDOM_STATE,
        "min_samples_leaf": min_leaf,
        "min_samples_split": min_split
    }


def fit_tree_and_metrics(model_df: pd.DataFrame):
    X = model_df[["X_VAL"]].values
    y = model_df["Y_VAL"].values
    n_obs = len(model_df)

    params = compute_dynamic_tree_params(n_obs)

    metric_mode = "full_sample"
    mae_ref = np.nan
    r2_ref = np.nan

    model_eval = DecisionTreeRegressor(**params)

    if len(model_df) >= 20 and model_df["X_VAL"].nunique() >= 4:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE
        )
        model_eval.fit(X_train, y_train)
        y_pred_test = model_eval.predict(X_test)
        mae_ref = mean_absolute_error(y_test, y_pred_test)
        r2_ref = r2_score(y_test, y_pred_test)
        metric_mode = "holdout_20"
    else:
        model_eval.fit(X, y)
        y_pred_full = model_eval.predict(X)
        mae_ref = mean_absolute_error(y, y_pred_full)
        r2_ref = r2_score(y, y_pred_full) if len(np.unique(y)) > 1 else np.nan

    model_final = DecisionTreeRegressor(**params)
    model_final.fit(X, y)

    model_df = model_df.copy()
    model_df["Y_PRED"] = model_final.predict(X)
    model_df["LEAF_ID"] = model_final.apply(X)

    return model_final, model_df, metric_mode, mae_ref, r2_ref, params


def get_thresholds(model: DecisionTreeRegressor) -> list:
    tree_ = model.tree_
    thresholds = []

    for node_id in range(tree_.node_count):
        feature = tree_.feature[node_id]
        threshold = tree_.threshold[node_id]
        if feature != _tree.TREE_UNDEFINED:
            thresholds.append(float(threshold))

    return sorted(set(thresholds))


def build_rule_text(low: float, high: float, feature_name: str = "X_VAL") -> str:
    if np.isneginf(low) and np.isposinf(high):
        return f"{feature_name}: todos los valores"
    if np.isneginf(low):
        return f"{feature_name} ≤ {high:.4f}"
    if np.isposinf(high):
        return f"{feature_name} > {low:.4f}"
    return f"{low:.4f} < {feature_name} ≤ {high:.4f}"


def human_range_label(low: float, high: float) -> str:
    if np.isneginf(low) and np.isposinf(high):
        return "Todos los valores"
    if np.isneginf(low):
        return f"≤ {high:,.4f}"
    if np.isposinf(high):
        return f"> {low:,.4f}"
    return f"({low:,.4f} ; {high:,.4f}]"


def extract_leaf_ranges(model: DecisionTreeRegressor) -> list:
    tree_ = model.tree_
    ranges = []

    def recurse(node_id: int, low: float, high: float):
        feature = tree_.feature[node_id]
        threshold = tree_.threshold[node_id]

        if feature != _tree.TREE_UNDEFINED:
            recurse(tree_.children_left[node_id], low, min(high, float(threshold)))
            recurse(tree_.children_right[node_id], max(low, float(threshold)), high)
        else:
            pred = float(tree_.value[node_id][0][0])
            samples = int(tree_.n_node_samples[node_id])
            ranges.append({
                "leaf_id": node_id,
                "x_low": low,
                "x_high": high,
                "pred": pred,
                "samples_tree": samples,
                "rule": build_rule_text(low, high, "X")
            })

    recurse(0, -np.inf, np.inf)
    return ranges


def assign_leaf_labels(range_df: pd.DataFrame) -> pd.DataFrame:
    range_df = range_df.sort_values("Y_PRED_RANGO", ascending=True).reset_index(drop=True)

    if len(range_df) == 1:
        range_df["CLASE_RANGO"] = ["Único"]
    elif len(range_df) == 2:
        range_df["CLASE_RANGO"] = ["Peor", "Mejor"]
    elif len(range_df) >= 3:
        labels = ["Peor", "Medio", "Mejor"]
        if len(range_df) > 3:
            extras = ["Mejor"] * (len(range_df) - 3)
            labels = labels + extras
        range_df["CLASE_RANGO"] = labels[:len(range_df)]

    return range_df


def build_ranges_table(model: DecisionTreeRegressor, model_df: pd.DataFrame) -> pd.DataFrame:
    leaf_ranges = extract_leaf_ranges(model)

    leaves_summary = (
        model_df.groupby("LEAF_ID", dropna=False)
        .agg(
            N=("Y_VAL", "size"),
            X_MIN=("X_VAL", "min"),
            X_MAX=("X_VAL", "max"),
            Y_REAL_PROM=("Y_VAL", "mean"),
            Y_REAL_MIN=("Y_VAL", "min"),
            Y_REAL_MAX=("Y_VAL", "max"),
            Y_PRED_RANGO=("Y_PRED", "mean"),
            CAMPAÑAS=("CAMPAÑA", lambda s: ", ".join(_sort_campaign_categories(pd.Series(s.astype(str).unique()))))
        )
        .reset_index()
        .rename(columns={"LEAF_ID": "leaf_id"})
    )

    range_df = pd.DataFrame(leaf_ranges).merge(leaves_summary, on="leaf_id", how="left")
    range_df["RANGO_X"] = range_df.apply(lambda r: human_range_label(r["x_low"], r["x_high"]), axis=1)
    range_df["REGLA"] = range_df.apply(lambda r: build_rule_text(r["x_low"], r["x_high"], "X"), axis=1)
    range_df = assign_leaf_labels(range_df)

    cols = [
        "CLASE_RANGO", "RANGO_X", "REGLA", "N",
        "X_MIN", "X_MAX",
        "Y_PRED_RANGO", "Y_REAL_PROM", "Y_REAL_MIN", "Y_REAL_MAX",
        "CAMPAÑAS", "leaf_id"
    ]

    return range_df[cols].sort_values("Y_PRED_RANGO", ascending=False).reset_index(drop=True)


def build_rules_text(model: DecisionTreeRegressor, feature_name: str) -> str:
    return export_text(model, feature_names=[feature_name], decimals=4)


def make_scatter_plot(model_df: pd.DataFrame, x_label: str, y_label: str, thresholds: list):
    title = (
        f"{x_label} vs {y_label} | "
        f"Árbol de decisión (profundidad ≤ {MAX_DEPTH}, hojas ≤ {MAX_LEAF_NODES})"
    )

    fig = px.scatter(
        model_df,
        x="X_VAL",
        y="Y_VAL",
        color="CAMPAÑA",
        hover_data=["CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD", "Y_PRED"],
        title=title
    )

    for thr in thresholds:
        fig.add_vline(
            x=thr,
            line_width=2,
            line_dash="dash",
            annotation_text=f"{thr:,.2f}",
            annotation_position="top"
        )

    x_range = compute_axis_range(model_df["X_VAL"], lower_zero=True)
    y_range = compute_axis_range(model_df["Y_VAL"], lower_zero=True)

    fig.update_layout(
        xaxis=dict(title=x_label, range=x_range),
        yaxis=dict(title=y_label, range=y_range),
        legend_title_text="CAMPAÑA",
        height=620,
    )

    return fig


def build_boxplot_df(model_df_pred: pd.DataFrame, ranges_table: pd.DataFrame) -> pd.DataFrame:
    leaf_map = ranges_table[["leaf_id", "REGLA", "CLASE_RANGO", "RANGO_X", "Y_PRED_RANGO"]].copy()

    box_df = model_df_pred.merge(
        leaf_map,
        left_on="LEAF_ID",
        right_on="leaf_id",
        how="left"
    ).copy()

    # ORDENAR SIEMPRE DE MEJOR A PEOR
    order_df = (
        ranges_table[["REGLA", "Y_PRED_RANGO"]]
        .drop_duplicates()
        .sort_values("Y_PRED_RANGO", ascending=False)
        .reset_index(drop=True)
    )
    rule_order = order_df["REGLA"].tolist()
    box_df["REGLA"] = pd.Categorical(box_df["REGLA"], categories=rule_order, ordered=True)

    return box_df


def make_boxplot(box_df: pd.DataFrame, y_label: str):
    fig = px.box(
        box_df,
        x="REGLA",
        y="Y_VAL",
        points="all",
        category_orders={"REGLA": list(box_df["REGLA"].cat.categories)}
    )

    fig.update_layout(
        title=f"{y_label} por REGLA",
        xaxis_title="REGLA (ordenado de mejor a peor)",
        yaxis_title=y_label,
        height=620,
        showlegend=False
    )

    return fig


def compute_anova_stats(box_df: pd.DataFrame):
    stats_rows = []

    grp = (
        box_df.groupby("REGLA", observed=False)["Y_VAL"]
        .agg(["count", "mean", "std", "var"])
        .reset_index()
    )

    grp = grp.dropna(subset=["REGLA"]).copy()

    for _, row in grp.iterrows():
        n = int(row["count"])
        mean_ = float(row["mean"]) if pd.notna(row["mean"]) else np.nan
        std_ = float(row["std"]) if pd.notna(row["std"]) else 0.0
        var_ = float(row["var"]) if pd.notna(row["var"]) else 0.0
        cv_ = (std_ / mean_ * 100) if pd.notna(mean_) and mean_ != 0 else np.nan

        stats_rows.append({
            "GRUPO": str(row["REGLA"]),
            "N": n,
            "MEDIA": mean_,
            "DESV_STD": std_,
            "VARIANZA": var_,
            "CV(%)": cv_
        })

    stats_df = pd.DataFrame(stats_rows)

    # Mantener orden de mejor a peor en tabla estadística
    if not stats_df.empty:
        cat_order = [str(x) for x in box_df["REGLA"].cat.categories] if hasattr(box_df["REGLA"], "cat") else None
        if cat_order:
            stats_df["GRUPO"] = pd.Categorical(stats_df["GRUPO"], categories=cat_order, ordered=True)
            stats_df = stats_df.sort_values("GRUPO").reset_index(drop=True)
            stats_df["GRUPO"] = stats_df["GRUPO"].astype(str)

    valid_groups = []
    for _, sub in box_df.groupby("REGLA", observed=False):
        vals = pd.to_numeric(sub["Y_VAL"], errors="coerce").dropna().values
        if len(vals) > 0:
            valid_groups.append(vals)

    anova_result = {
        "prueba": "ANOVA",
        "estadistico": np.nan,
        "pvalor": np.nan,
        "N": int(box_df["Y_VAL"].notna().sum()),
        "grupos": int(box_df["REGLA"].dropna().nunique())
    }

    if len(valid_groups) >= 2:
        stat, pval = f_oneway(*valid_groups)
        anova_result["estadistico"] = float(stat)
        anova_result["pvalor"] = float(pval)

    return stats_df, anova_result


def impute_medians(df_in: pd.DataFrame, cols: list) -> pd.DataFrame:
    df_out = df_in.copy()
    for c in cols:
        med = df_out[c].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        df_out[c] = df_out[c].fillna(med)
    return df_out


@st.cache_data(show_spinner=False)
def run_optuna_and_rf_importance(rf_df: pd.DataFrame):
    if rf_df.empty:
        return None, None, None, None

    data = rf_df.copy()
    data = impute_medians(data, RF_FEATURES)

    X = data[RF_FEATURES].copy()
    y = data["Y_VAL"].copy()

    if len(data) < 10 or y.nunique() < 2:
        return None, None, None, None

    cv_splits = min(5, max(3, len(data) // 20))
    cv_splits = max(3, min(cv_splits, len(data)))
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators": RF_N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, 1.0]),
        }

        model = RandomForestRegressor(**params)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=kf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        rmse_cv = -scores.mean()
        return rmse_cv

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params.copy()
    final_params = {
        "n_estimators": RF_N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        **best_params
    }

    model_final = RandomForestRegressor(**final_params)
    model_final.fit(X, y)

    y_pred = model_final.predict(X)
    rmse_full = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2_full = float(r2_score(y, y_pred)) if y.nunique() > 1 else np.nan

    imp_df = pd.DataFrame({
        "VARIABLE": RF_FEATURES,
        "IMPORTANCIA_ORIGINAL": model_final.feature_importances_
    })

    imp_df = imp_df[imp_df["IMPORTANCIA_ORIGINAL"] > 0].copy()
    imp_df = imp_df.sort_values("IMPORTANCIA_ORIGINAL", ascending=False).reset_index(drop=True)

    if imp_df.empty:
        return study.best_params, rmse_full, r2_full, imp_df

    total_non_zero = imp_df["IMPORTANCIA_ORIGINAL"].sum()
    imp_df["IMPORTANCIA_%"] = (imp_df["IMPORTANCIA_ORIGINAL"] / total_non_zero) * 100
    imp_df["RANK"] = np.arange(1, len(imp_df) + 1)
    imp_df["VARIABLE_LABEL"] = imp_df["VARIABLE"].map(lambda x: X_LABELS.get(x, x))

    diff = 100.0 - imp_df["IMPORTANCIA_%"].sum()
    if len(imp_df) > 0:
        imp_df.loc[imp_df.index[0], "IMPORTANCIA_%"] += diff

    return study.best_params, rmse_full, r2_full, imp_df


def make_importance_plot(imp_df: pd.DataFrame):
    plot_df = imp_df.sort_values("IMPORTANCIA_%", ascending=True).copy()

    fig = px.bar(
        plot_df,
        x="IMPORTANCIA_%",
        y="VARIABLE_LABEL",
        orientation="h",
        text="IMPORTANCIA_%"
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        title="Importancia de variables | Random Forest",
        xaxis_title="Importancia (%)",
        yaxis_title="Variable",
        height=max(500, 35 * len(plot_df) + 180),
        showlegend=False
    )
    return fig


def build_explicit_summary_table(
    imp_df: pd.DataFrame,
    y_label: str,
    n_univ: int,
    pvalor_anova,
    r2_univ,
    rmse_rf
) -> pd.DataFrame:
    """
    Tabla explícita e interpretable:
    - VARIABLE X: variable del modelo RF
    - VARIABLE Y: métrica seleccionada
    - N: N del análisis univariable actual
    - P-VALOR: p-valor del ANOVA de reglas
    - R2: R² del árbol univariable
    - RMSE: RMSE del RF multivariable
    - IMPORTANCIA %: importancia relativa de cada X
    - RANGO: ranking de importancia
    """
    if imp_df is None or imp_df.empty:
        return pd.DataFrame(columns=[
            "VARIABLE X", "VARIABLE Y", "N", "P-VALOR", "R2", "RMSE", "IMPORTANCIA %", "RANGO"
        ])

    summary = imp_df.copy()

    summary["VARIABLE X"] = summary["VARIABLE_LABEL"]
    summary["VARIABLE Y"] = y_label
    summary["N"] = int(n_univ) if pd.notna(n_univ) else np.nan
    summary["P-VALOR"] = pvalor_anova if pd.notna(pvalor_anova) else np.nan
    summary["R2"] = r2_univ if pd.notna(r2_univ) else np.nan
    summary["RMSE"] = rmse_rf if pd.notna(rmse_rf) else np.nan
    summary["IMPORTANCIA %"] = summary["IMPORTANCIA_%"]
    summary["RANGO"] = summary["RANK"]

    summary = summary[
        ["VARIABLE X", "VARIABLE Y", "N", "P-VALOR", "R2", "RMSE", "IMPORTANCIA %", "RANGO"]
    ].copy()

    summary = summary.sort_values(["VARIABLE Y", "RANGO"], ascending=[True, True]).reset_index(drop=True)
    return summary


# ==========================================================
# LOAD
# ==========================================================
st.title("🌳 Árbol de decisión | Fenología vs métricas productivas")
st.caption("Vista univariable e interpretable para identificar rangos fenológicos asociados al rendimiento.")

if not os.path.exists(DATA_FILE):
    st.error(
        f"No encuentro el archivo **{DATA_FILE}**.\n\n"
        "Verifica:\n"
        f"- que esté en el mismo folder que `app.py`\n"
        f"- que el nombre sea exactamente `{DATA_FILE}`\n"
        f"- que la hoja sea `{REQ_SHEET}`"
    )
    st.stop()

df_raw = read_excel_path(DATA_FILE, REQ_SHEET)
missing = validate_cols(df_raw)

if missing:
    st.error("Faltan columnas requeridas:")
    st.write(missing)
    st.stop()

df = df_raw.copy()

# ==========================================================
# LIMPIEZA BÁSICA
# ==========================================================
num_main = [
    "AÑO", "SEMANA", "kilogramos", "Ha COSECHADA", "Ha TURNO", "DENSIDAD",
    "KG/HA", "PESO BAYA (g)", "CALIBRE BAYA (mm)",
    "MADERAS PRINCIPALES", "CORTES", "BROTES TOTALES", "TERMINALES",
    "EDAD PLANTA",
    "BP_N_BROTES_ULT", "BP_LONG_ULT", "BP_DIAM_ULT",
    "BS_N_BROTES_ULT", "BS_LONG_ULT", "BS_DIAM_ULT",
    "BT_N_BROTES_ULT", "BT_LONG_ULT", "BT_DIAM_ULT",
    "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT"
]

for c in num_main:
    if c in df.columns:
        df[c] = to_numeric_safe(df[c])

for c in ["CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD", "SIEMBRA FINAL", "SEG DENSIDAD"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

df["SEMANA"] = to_numeric_safe(df["SEMANA"]).fillna(0).astype(int)
df = ensure_categories_age(df)

# ==========================================================
# SIDEBAR FILTROS
# ==========================================================
with st.sidebar:
    st.header("🎛️ Filtros")

    def ms(col):
        vals = sorted([v for v in df[col].dropna().unique().tolist()])
        return st.multiselect(col, vals, default=[])

    camp = ms("CAMPAÑA")
    fundo = ms("FUNDO")
    etapa = ms("ETAPA")
    campo = ms("CAMPO")
    turno = ms("TURNO")
    variedad = ms("VARIEDAD")

    edad_vals = [str(v) for v in df["EDAD PLANTA FINAL"].dropna().unique().tolist()]
    edad_order = [x for x in ["1", "2", "3+"] if x in edad_vals] + [x for x in sorted(edad_vals) if x not in ["1", "2", "3+"]]
    edad_final = st.multiselect("EDAD PLANTA FINAL", edad_order, default=[])

    siembra_final = ms("SIEMBRA FINAL")
    seg_densidad = ms("SEG DENSIDAD")

    sem_min = int(df["SEMANA"].min()) if not df["SEMANA"].dropna().empty else 1
    sem_max = int(df["SEMANA"].max()) if not df["SEMANA"].dropna().empty else 52
    semana_min, semana_max = st.slider(
        "SEMANA",
        min_value=sem_min,
        max_value=sem_max,
        value=(sem_min, sem_max)
    )

# ==========================================================
# DATA FILTRADA
# ==========================================================
dff = apply_filters(
    df=df,
    camp=camp,
    fundo=fundo,
    etapa=etapa,
    campo=campo,
    turno=turno,
    variedad=variedad,
    edad_final=edad_final,
    siembra_final=siembra_final,
    seg_densidad=seg_densidad,
    semana_min=semana_min,
    semana_max=semana_max
)

# ==========================================================
# CONTROLES PRINCIPALES
# ==========================================================
left_ctrl, right_ctrl = st.columns([0.28, 0.72])

with left_ctrl:
    y_pick = st.selectbox("Métrica Y", list(Y_OPTIONS.keys()), index=0)
    default_x_idx = list(X_OPTIONS.keys()).index("BROTES TOTALES") if "BROTES TOTALES" in X_OPTIONS else 0
    x_pick = st.selectbox("Variable X", list(X_OPTIONS.keys()), index=default_x_idx)

x_col = X_OPTIONS[x_pick]
x_label = X_LABELS.get(x_col, x_col)
y_label = y_pick

# ==========================================================
# MODEL DATA
# ==========================================================
model_df = build_model_df(dff, x_col=x_col, y_metric=y_pick)

if model_df.empty:
    st.warning("No hay datos suficientes con los filtros actuales.")
    st.stop()

if model_df["X_VAL"].nunique() < 2 or model_df["Y_VAL"].nunique() < 2:
    st.warning("No hay variabilidad suficiente en X o Y para ajustar el árbol.")
    st.stop()

# ==========================================================
# MODELO ÁRBOL
# ==========================================================
model, model_df_pred, metric_mode, mae_ref, r2_ref, tree_params = fit_tree_and_metrics(model_df)
thresholds = get_thresholds(model)
ranges_table = build_ranges_table(model, model_df_pred)
rules_text = build_rules_text(model, feature_name=x_label)

n_total_nodes = int(model.tree_.node_count)
n_leaf_nodes = int(model.get_n_leaves())
n_internal_nodes = n_total_nodes - n_leaf_nodes
real_depth = int(model.get_depth())

# ==========================================================
# KPIs
# ==========================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("N observaciones", f"{len(model_df_pred):,}")
c2.metric("Profundidad real", f"{real_depth}")
c3.metric("Hojas finales", f"{n_leaf_nodes}")
c4.metric("MAE", f"{mae_ref:,.4f}" if pd.notna(mae_ref) else "NA")
c5.metric("R²", f"{r2_ref:,.4f}" if pd.notna(r2_ref) else "NA")

st.caption(
    f"Nodos totales: {n_total_nodes} | "
    f"Nodos internos: {n_internal_nodes} | "
    f"Hojas finales: {n_leaf_nodes}"
)

st.caption(
    f"Poda dinámica activa: min_samples_leaf = {tree_params['min_samples_leaf']} "
    f"({MIN_LEAF_RATIO:.0%} del N filtrado, con piso {MIN_LEAF_FLOOR}) | "
    f"min_samples_split = {tree_params['min_samples_split']}"
)

if metric_mode == "holdout_20":
    st.caption("MAE y R² calculados sobre holdout 20%.")
else:
    st.caption("MAE y R² calculados sobre la muestra completa por tamaño limitado de datos.")

# ==========================================================
# SCATTER + CORTES
# ==========================================================
fig = make_scatter_plot(
    model_df=model_df_pred,
    x_label=x_label,
    y_label=y_label,
    thresholds=thresholds
)
st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# RESUMEN DE RANGOS
# ==========================================================
st.subheader("Rangos del árbol de decisión")

ranges_show = ranges_table.copy()
ranges_show = ranges_show.rename(columns={
    "CLASE_RANGO": "Clase",
    "RANGO_X": "Rango X",
    "REGLA": "Regla",
    "N": "N",
    "X_MIN": "X real min",
    "X_MAX": "X real max",
    "Y_PRED_RANGO": f"{y_label} esperado",
    "Y_REAL_PROM": f"{y_label} promedio real",
    "Y_REAL_MIN": f"{y_label} mínimo real",
    "Y_REAL_MAX": f"{y_label} máximo real",
    "CAMPAÑAS": "Campañas",
    "leaf_id": "Leaf ID"
})

st.dataframe(
    ranges_show.style.format({
        "X real min": "{:,.4f}",
        "X real max": "{:,.4f}",
        f"{y_label} esperado": "{:,.4f}",
        f"{y_label} promedio real": "{:,.4f}",
        f"{y_label} mínimo real": "{:,.4f}",
        f"{y_label} máximo real": "{:,.4f}",
    }),
    use_container_width=True
)

# ==========================================================
# BOXPLOT + ANOVA
# ==========================================================
st.subheader("Boxplot por regla del árbol")
st.caption(f"Los boxplots responden a la métrica general seleccionada: {y_label} y están ordenados de mejor a peor.")

box_df = build_boxplot_df(model_df_pred, ranges_table)
box_fig = make_boxplot(box_df, y_label=y_label)
st.plotly_chart(box_fig, use_container_width=True)

stats_df, anova_result = compute_anova_stats(box_df)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Prueba", anova_result["prueba"])
k2.metric("Estadístico (F)", f'{anova_result["estadistico"]:,.4f}' if pd.notna(anova_result["estadistico"]) else "NA")
k3.metric("p-valor", f'{anova_result["pvalor"]:,.6f}' if pd.notna(anova_result["pvalor"]) else "NA")
k4.metric("N", f'{anova_result["N"]:,}')
k5.metric("Grupos", f'{anova_result["grupos"]:,}')

st.dataframe(
    stats_df.style.format({
        "MEDIA": "{:,.4f}",
        "DESV_STD": "{:,.4f}",
        "VARIANZA": "{:,.4f}",
        "CV(%)": "{:,.0f}%"
    }),
    use_container_width=True
)

# ==========================================================
# REGLAS DEL ÁRBOL
# ==========================================================
st.subheader("Reglas del árbol")
st.code(rules_text, language="text")

# ==========================================================
# IMPORTANCIA DE VARIABLES | RANDOM FOREST + OPTUNA
# ==========================================================
st.subheader("Importancia de variables")
st.caption(
    "Modelo multivariable con Random Forest (500 árboles) optimizado con Optuna "
    f"({OPTUNA_TRIALS} trials, métrica objetivo = RMSE). "
    "Se omiten importancias iguales a 0 y las visibles se reescalan para sumar 100%."
)

rf_df = build_rf_model_df(dff, y_metric=y_pick)
best_params, rf_rmse, rf_r2, imp_df = run_optuna_and_rf_importance(rf_df)

if imp_df is None:
    st.warning("No hay datos suficientes para calcular importancia de variables con Random Forest.")
else:
    r1, r2c, r3, r4 = st.columns(4)
    r1.metric("N observaciones RF", f"{len(rf_df):,}")
    r2c.metric("Árboles RF", f"{RF_N_ESTIMATORS:,}")
    r3.metric("RMSE RF", f"{rf_rmse:,.4f}" if pd.notna(rf_rmse) else "NA")
    r4.metric("R² RF", f"{rf_r2:,.4f}" if pd.notna(rf_r2) else "NA")

    if best_params is not None:
        with st.expander("Ver mejores hiperparámetros de Optuna"):
            st.json(best_params)

    if imp_df.empty:
        st.warning("Todas las variables tuvieron importancia 0 después del ajuste.")
    else:
        imp_fig = make_importance_plot(imp_df)
        st.plotly_chart(imp_fig, use_container_width=True)

        # ==========================================================
        # NUEVA TABLA EXPLÍCITA RESUMEN
        # ==========================================================
        st.subheader("Tabla resumen explícita")

        explicit_summary = build_explicit_summary_table(
            imp_df=imp_df,
            y_label=y_label,
            n_univ=len(model_df_pred),
            pvalor_anova=anova_result["pvalor"],
            r2_univ=r2_ref,
            rmse_rf=rf_rmse
        )

        st.dataframe(
            explicit_summary.style.format({
                "N": "{:,.0f}",
                "P-VALOR": "{:,.6f}",
                "R2": "{:,.4f}",
                "RMSE": "{:,.4f}",
                "IMPORTANCIA %": "{:,.4f}%",
                "RANGO": "{:,.0f}",
            }),
            use_container_width=True
        )
