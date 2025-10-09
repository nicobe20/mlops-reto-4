# training/train_model.py
import os, datetime
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

PROJECT = os.getenv("GCP_PROJECT_ID")
DATASET = os.getenv("GCP_DATASET", "mlops_reto4")
LOCATION = os.getenv("GCP_LOCATION", "us")
RAW_TABLE = os.getenv("BQ_TABLE_RAW", "raw_parking")
METRICS_TABLE = os.getenv("BQ_TABLE_METRICS", "model_metrics")
PRED_TABLE = os.getenv("BQ_TABLE_PRED", "predictions")
H_FORECAST = int(os.getenv("H_FORECAST", "24"))  # horizonte 24 horas

os.makedirs("monitoring", exist_ok=True)

def ensure_metrics_and_pred_tables(client):
    ds_ref = bigquery.Dataset(f"{PROJECT}.{DATASET}")
    try:
        client.get_dataset(ds_ref)
    except NotFound:
        ds_ref.location = LOCATION
        client.create_dataset(ds_ref)

    metrics_id = f"{PROJECT}.{DATASET}.{METRICS_TABLE}"
    pred_id = f"{PROJECT}.{DATASET}.{PRED_TABLE}"

    try:
        client.get_table(metrics_id)
    except NotFound:
        metrics_schema = [
            bigquery.SchemaField("run_at", "TIMESTAMP"),
            bigquery.SchemaField("train_points", "INTEGER"),
            bigquery.SchemaField("mae", "FLOAT"),
            bigquery.SchemaField("model_desc", "STRING"),
        ]
        client.create_table(bigquery.Table(metrics_id, schema=metrics_schema))

    try:
        client.get_table(pred_id)
    except NotFound:
        pred_schema = [
            bigquery.SchemaField("created_at", "TIMESTAMP"),
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("yhat", "FLOAT"),
        ]
        client.create_table(bigquery.Table(pred_id, schema=pred_schema))

def read_series_from_bq(client):
    sql = f"""
    SELECT timestamp, SUM(CAST(free AS INT64)) AS total_free
    FROM `{PROJECT}.{DATASET}.{RAW_TABLE}`
    WHERE free IS NOT NULL
    GROUP BY timestamp
    ORDER BY timestamp
    """
    df = client.query(sql).result().to_dataframe(create_bqstorage_client=True)
    if df.empty:
        raise RuntimeError("No hay datos en RAW para entrenar.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df["total_free"].asfreq("H")  # fuerza frecuencia horaria

def train_and_eval(y):
    # split: último día para test si hay datos suficientes
    if len(y.dropna()) < 60:
        # pocos datos: usa naive last-value
        y_train, y_test = y.iloc[:-H_FORECAST], y.iloc[-H_FORECAST:]
        yhat_test = pd.Series(index=y_test.index, data=y_train.dropna().iloc[-1] if not y_train.dropna().empty else 0.0)
        mae = float(mean_absolute_error(y_test.fillna(0), yhat_test.fillna(0)))
        model_desc = "naive_last_value"
        # pronóstico futuro constante
        last_val = y.dropna().iloc[-1] if not y.dropna().empty else 0.0
        future_idx = pd.date_range(y.index.max() + pd.Timedelta(hours=1), periods=H_FORECAST, freq="H")
        yhat_future = pd.Series(index=future_idx, data=last_val)
        return mae, model_desc, yhat_future
    else:
        # SARIMAX simple con estacionalidad diaria (24)
        y_train, y_test = y.iloc[:-H_FORECAST], y.iloc[-H_FORECAST:]
        y_train = y_train.fillna(method="ffill").fillna(0)
        y_test = y_test.fillna(method="ffill").fillna(0)

        model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,24), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        yhat_test = res.forecast(steps=len(y_test))
        mae = float(mean_absolute_error(y_test, yhat_test))
        model_desc = "SARIMAX(1,1,1)(1,1,1,24)"

        # futuro
        yhat_future = res.forecast(steps=H_FORECAST)
        return mae, model_desc, yhat_future

def write_metrics_and_preds(client, mae, model_desc, yhat_future):
    ensure_metrics_and_pred_tables(client)

    run_at = datetime.datetime.utcnow().isoformat()+"Z"
    metrics_rows = [{"run_at": run_at, "train_points": int(len(yhat_future)), "mae": float(mae), "model_desc": model_desc}]
    client.insert_rows_json(f"{PROJECT}.{DATASET}.{METRICS_TABLE}", metrics_rows)

    pred_rows = [{"created_at": run_at, "timestamp": ts.isoformat(), "yhat": float(val)} for ts, val in yhat_future.items()]
    client.insert_rows_json(f"{PROJECT}.{DATASET}.{PRED_TABLE}", pred_rows)

    # también guardamos un CSV local opcional
    with open("monitoring/metrics.csv", "a", encoding="utf-8") as f:
        if f.tell() == 0:
            f.write("run_at,train_points,mae,model_desc\n")
        f.write(f"{run_at},{len(yhat_future)},{mae},{model_desc}\n")

if __name__ == "__main__":
    assert PROJECT and DATASET, "Faltan GCP_PROJECT_ID / GCP_DATASET"
    bq = bigquery.Client(project=PROJECT, location=LOCATION)
    y = read_series_from_bq(bq)
    mae, model_desc, yhat_future = train_and_eval(y)
    write_metrics_and_preds(bq, mae, model_desc, yhat_future)
    print("Entrenamiento listo. MAE:", mae, "| Modelo:", model_desc)
