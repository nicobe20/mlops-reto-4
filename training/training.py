# training/training.py
# Entrenamiento: lee serie agregada de BigQuery (RAW) y publica métricas y predicciones en BQ.
import os
import datetime
import numpy as np
import pandas as pd

from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# ===== Config por entorno =====
PROJECT       = os.getenv("GCP_PROJECT_ID")
DATASET       = os.getenv("GCP_DATASET", "mlops_reto4")
LOCATION      = os.getenv("GCP_LOCATION", "us")
RAW_TABLE     = os.getenv("BQ_TABLE_RAW", "raw_parking")
METRICS_TABLE = os.getenv("BQ_TABLE_METRICS", "model_metrics")
PRED_TABLE    = os.getenv("BQ_TABLE_PRED", "predictions")

# Horizonte y ventana de historia configurables
H_FORECAST   = int(os.getenv("H_FORECAST", "24"))    # pasos a futuro (horas)
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "7"))   # días hacia atrás para entrenar

os.makedirs("monitoring", exist_ok=True)


# ===== Helpers BQ =====
def ensure_metrics_and_pred_tables(client: bigquery.Client):
    """Crea dataset/tablas (metrics, predictions) si no existen."""
    ds_ref = bigquery.Dataset(f"{PROJECT}.{DATASET}")
    try:
        client.get_dataset(ds_ref)
    except NotFound:
        ds_ref.location = LOCATION
        client.create_dataset(ds_ref)
        print(f"[BQ] Dataset creado: {DATASET}")

    metrics_id = f"{PROJECT}.{DATASET}.{METRICS_TABLE}"
    pred_id    = f"{PROJECT}.{DATASET}.{PRED_TABLE}"

    try:
        client.get_table(metrics_id)
    except NotFound:
        metrics_schema = [
            bigquery.SchemaField("run_at", "TIMESTAMP"),
            bigquery.SchemaField("train_points", "INTEGER"),
            bigquery.SchemaField("mae", "FLOAT"),
            bigquery.SchemaField("model_desc", "STRING"),
            bigquery.SchemaField("history_days", "INTEGER"),
            bigquery.SchemaField("h_forecast", "INTEGER"),
        ]
        client.create_table(bigquery.Table(metrics_id, schema=metrics_schema))
        print(f"[BQ] Tabla creada: {metrics_id}")

    try:
        client.get_table(pred_id)
    except NotFound:
        pred_schema = [
            bigquery.SchemaField("created_at", "TIMESTAMP"),
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("yhat", "FLOAT"),
        ]
        client.create_table(bigquery.Table(pred_id, schema=pred_schema))
        print(f"[BQ] Tabla creada: {pred_id}")


def read_series_from_bq(client: bigquery.Client, history_days: int) -> pd.Series:
    sql = f"""
    SELECT
      TIMESTAMP_TRUNC(timestamp, HOUR) AS ts_hour,
      SUM(CAST(free AS INT64)) AS total_free
    FROM `{PROJECT}.{DATASET}.{RAW_TABLE}`
    WHERE free IS NOT NULL
      AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {history_days} DAY)
    GROUP BY ts_hour
    ORDER BY ts_hour
    """
    df = client.query(sql).result().to_dataframe()
    if df.empty:
        raise RuntimeError("No hay datos en RAW para entrenar (consulta vacía).")

    df["ts_hour"] = pd.to_datetime(df["ts_hour"], utc=True)
    df = df.set_index("ts_hour").sort_index()

    # índice horario completo y relleno hacia adelante (last observation carried forward)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="H", tz="UTC")
    y = df["total_free"].reindex(full_idx).fillna(method="ffill")
    return y



# ===== Modelado =====
def baseline_hour_of_day(y: pd.Series, horizon: int):
    """
    Baseline estacional por hora del día: yhat(hora) = media histórica de esa hora.
    Respaldo con EWM si falta alguna hora.
    Retorna: (mae, yhat_future, model_desc)
    """
    y_filled = y.fillna(method="ffill").fillna(0)

    # Promedio por hora del día
    df = y_filled.to_frame("y")
    df["hour"] = df.index.hour
    hourly_means = df.groupby("hour")["y"].mean()

    # Futuro
    future_idx   = pd.date_range(y.index.max() + pd.Timedelta(hours=1),
                                 periods=horizon, freq="H")
    future_hours = future_idx.hour
    yhat_future  = pd.Series([hourly_means.get(h, np.nan) for h in future_hours],
                             index=future_idx, dtype="float64")

    # Backfill con EWM si hay NaN
    if yhat_future.isna().any():
        ewm_val = y_filled.ewm(span=12, adjust=False).mean().iloc[-1]
        yhat_future = yhat_future.fillna(float(ewm_val))

    # Pequeño backtest con últimas horas disponibles
    test_len = min(horizon, max(1, y_filled.dropna().shape[0] // 5))
    y_test   = y_filled.iloc[-test_len:]
    yhat_test_hours = y_test.index.hour
    yhat_test = pd.Series([hourly_means.get(h, y_filled.iloc[-1]) for h in yhat_test_hours],
                          index=y_test.index, dtype="float64")
    mae = float(mean_absolute_error(y_test, yhat_test))

    return mae, yhat_future, "baseline_hour_of_day + ewm_backfill"


def sarimax_daily(y: pd.Series, horizon: int):
    """
    SARIMAX con estacionalidad diaria (24) para cuando ya hay data suficiente.
    Retorna: (mae, yhat_future, model_desc)
    """
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
    y_train = y_train.fillna(method="ffill").fillna(0)
    y_test  = y_test.fillna(method="ffill").fillna(0)

    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    yhat_test   = res.forecast(steps=len(y_test))
    mae         = float(mean_absolute_error(y_test, yhat_test))
    yhat_future = res.forecast(steps=horizon)
    return mae, yhat_future, "SARIMAX(1,1,1)(1,1,1,24)"


def train_and_eval(y: pd.Series, horizon: int):
    """
    Lógica de selección de modelo:
    - Si hay <60 puntos efectivos → baseline por hora del día (no plano)
    - Si hay >=60 puntos → SARIMAX diario
    """
    n_effective = y.dropna().shape[0]
    if n_effective < 24:
        return baseline_hour_of_day(y, horizon)
    else:
        return sarimax_daily(y, horizon)


# ===== Salida a BQ/CSV =====
def write_metrics_and_preds(client: bigquery.Client, mae: float, model_desc: str,
                            yhat_future: pd.Series, n_train_points: int):
    ensure_metrics_and_pred_tables(client)

    run_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # Métricas
    metrics_rows = [{
        "run_at": run_at,
        "train_points": int(n_train_points),
        "mae": float(mae),
        "model_desc": model_desc,
        "history_days": int(HISTORY_DAYS),
        "h_forecast": int(H_FORECAST),
    }]
    client.insert_rows_json(f"{PROJECT}.{DATASET}.{METRICS_TABLE}", metrics_rows)

    # Predicciones
    pred_rows = [{
        "created_at": run_at,
        "timestamp": ts.to_pydatetime().replace(tzinfo=datetime.timezone.utc).isoformat(),
        "yhat": float(val) if val is not None and not np.isnan(val) else None
    } for ts, val in yhat_future.items()]
    client.insert_rows_json(f"{PROJECT}.{DATASET}.{PRED_TABLE}", pred_rows)

    # CSV local opcional (útil para versionado rápido)
    with open("monitoring/metrics.csv", "a", encoding="utf-8") as f:
        if f.tell() == 0:
            f.write("run_at,train_points,mae,model_desc,history_days,h_forecast\n")
        f.write(f"{run_at},{n_train_points},{mae},{model_desc},{HISTORY_DAYS},{H_FORECAST}\n")


# ===== Main =====
if __name__ == "__main__":
    assert PROJECT and DATASET, "Faltan variables de entorno GCP_PROJECT_ID / GCP_DATASET"
    bq = bigquery.Client(project=PROJECT, location=LOCATION)

    # 1) Leer serie (últimos HISTORY_DAYS)
    y = read_series_from_bq(bq, HISTORY_DAYS)
    n_train = int(y.dropna().shape[0])
    if n_train == 0:
        raise RuntimeError("Serie sin puntos válidos para entrenar.")

    # 2) Entrenar con selección de modelo
    mae, yhat_future, model_desc = train_and_eval(y, H_FORECAST)

    # 3) Escribir métricas y predicciones
    write_metrics_and_preds(bq, mae, model_desc, yhat_future, n_train)

    print(f"[TRAIN] listo | puntos_entrenamiento={n_train} | MAE={mae:.3f} | modelo={model_desc} | "
          f"forecast_h={H_FORECAST} | history_days={HISTORY_DAYS}")
