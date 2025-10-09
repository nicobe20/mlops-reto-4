#data collector. collector/collector.py
import os, io, json, time, csv, datetime
import requests
from google.cloud import storage, bigquery
from google.api_core.exceptions import NotFound

# ===== Config =====
API_URL = os.getenv(
    "DONOSTIA_API_URL",
    "https://www.donostia.eus/info/ciudadano/camaras_trafico.nsf/getParkings.xsp"
)

PROJECT = os.getenv("GCP_PROJECT_ID")
DATASET = os.getenv("GCP_DATASET", "mlops_reto4")
LOCATION = os.getenv("GCP_LOCATION", "us")
BUCKET  = os.getenv("GCP_BUCKET")
RAW_TABLE = os.getenv("BQ_TABLE_RAW", "raw_parking")

LOCAL_DIR = "data"
os.makedirs(LOCAL_DIR, exist_ok=True)

# ===== Helpers =====
def safe_get(d, *candidates, default=None):
    for c in candidates:
        if c in d and d[c] not in (None, ""):
            return d[c]
    return default

def fetch_json(url, retries=3, timeout=15):
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 * (i+1))

def normalize_records(obj, timestamp_iso):
    """
    Intenta normalizar una lista de parqueaderos en filas:
    parking_id, name, free, total, lat, lon, timestamp
    """
    records = []
    if isinstance(obj, dict):
        data = obj.get("parkings") or obj.get("data") or obj.get("items") or []
    else:
        data = obj

    if not isinstance(data, list):
        # si el endpoint ya devuelve lista
        data = obj if isinstance(obj, list) else []

    for it in data:
        # Campos típicos (heurística defensiva)
        pid = safe_get(it, "id", "idParking", "id_parking", "codigo", "code")
        name = safe_get(it, "name", "nombre", "parkingName")
        free = safe_get(it, "free", "libre", "plazas_libres", "available", "slotsAvailable")
        total = safe_get(it, "total", "capacidad", "plazas_totales", "slotsTotal")
        lat = safe_get(it, "lat", "latitude", "y")
        lon = safe_get(it, "lon", "lng", "long", "x")

        # Si no hay 'free', intenta calcular a partir de ocupación
        if free is None and "occupied" in it and total:
            try:
                free = int(total) - int(it["occupied"])
            except:
                free = None

        # tipado básico
        def as_int(x):
            try: return int(float(x))
            except: return None

        row = {
            "parking_id": str(pid) if pid is not None else None,
            "name": None if name is None else str(name),
            "free": as_int(free),
            "total": as_int(total),
            "lat": None if lat is None else float(lat),
            "lon": None if lon is None else float(lon),
            "timestamp": timestamp_iso
        }
        # filtra filas totalmente vacías
        if any(v is not None for v in row.values()):
            records.append(row)
    return records

def write_local_csv(rows, path):
    header = ["timestamp","parking_id","name","free","total","lat","lon"]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({
                "timestamp": r["timestamp"],
                "parking_id": r["parking_id"],
                "name": r["name"],
                "free": r["free"],
                "total": r["total"],
                "lat": r["lat"],
                "lon": r["lon"]
            })

def upload_to_gcs(local_path, bucket_name, dest_blob):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(local_path)
    print(f"Subido a gs://{bucket_name}/{dest_blob}")

def ensure_bq_raw_table(project, dataset, table):
    client = bigquery.Client(project=project, location=LOCATION)
    ds_ref = bigquery.Dataset(f"{project}.{dataset}")
    try:
        client.get_dataset(ds_ref)
    except NotFound:
        ds_ref.location = LOCATION
        client.create_dataset(ds_ref)
        print(f"Dataset creado: {dataset}")

    table_id = f"{project}.{dataset}.{table}"
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("parking_id", "STRING"),
        bigquery.SchemaField("name", "STRING"),
        bigquery.SchemaField("free", "INTEGER"),
        bigquery.SchemaField("total", "INTEGER"),
        bigquery.SchemaField("lat", "FLOAT"),
        bigquery.SchemaField("lon", "FLOAT"),
    ]
    try:
        client.get_table(table_id)
    except NotFound:
        tbl = bigquery.Table(table_id, schema=schema)
        client.create_table(tbl)
        print(f"Tabla creada: {table_id}")

def load_csv_to_bq(gcs_uri, project, dataset, table):
    client = bigquery.Client(project=project, location=LOCATION)
    table_id = f"{project}.{dataset}.{table}"
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=False,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=[
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("parking_id", "STRING"),
            bigquery.SchemaField("name", "STRING"),
            bigquery.SchemaField("free", "INTEGER"),
            bigquery.SchemaField("total", "INTEGER"),
            bigquery.SchemaField("lat", "FLOAT"),
            bigquery.SchemaField("lon", "FLOAT"),
        ],
    )
    job = client.load_table_from_uri(gcs_uri, table_id, job_config=job_config)
    job.result()
    print(f"Cargado en BigQuery: {gcs_uri} -> {table_id}")

# ===== Main =====
def main():
    assert PROJECT and BUCKET and DATASET, "Faltan variables de entorno GCP_PROJECT_ID / GCP_BUCKET / GCP_DATASET"

    now = datetime.datetime.utcnow()
    ts_iso = now.replace(microsecond=0).isoformat() + "Z"
    stamp = now.strftime("%Y%m%d_%H%M%S")

    # 1) Descarga
    obj = fetch_json(API_URL)
    rows = normalize_records(obj, ts_iso)
    if not rows:
        print("Advertencia: sin filas normalizadas. Se guarda JSON crudo para depurar.")
        # guarda crudo por si acaso
        raw_path = os.path.join(LOCAL_DIR, f"raw_{stamp}.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        # continua sin cargar a BQ
        return

    # 2) CSV local
    local_csv = os.path.join(LOCAL_DIR, f"data_{stamp}.csv")
    write_local_csv(rows, local_csv)

    # 3) Subir a GCS
    dest = f"raw/data_{stamp}.csv"
    upload_to_gcs(local_csv, BUCKET, dest)

    # 4) Asegurar tabla RAW y cargar CSV a BigQuery
    ensure_bq_raw_table(PROJECT, DATASET, RAW_TABLE)
    load_csv_to_bq(f"gs://{BUCKET}/{dest}", PROJECT, DATASET, RAW_TABLE)

if __name__ == "__main__":
    main()
