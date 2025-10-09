# collector/collector.py
# Data collector para parqueaderos (Donostia) -> GCS -> BigQuery (RAW)
import os, io, json, time, csv, datetime
import requests
from google.cloud import storage, bigquery
from google.api_core.exceptions import NotFound

# ===== Config =====
API_URL = os.getenv(
    "DONOSTIA_API_URL",
    "https://www.donostia.eus/info/ciudadano/camaras_trafico.nsf/getParkings.xsp"
)

PROJECT  = os.getenv("GCP_PROJECT_ID")
DATASET  = os.getenv("GCP_DATASET", "mlops_reto4")
LOCATION = os.getenv("GCP_LOCATION", "us-east1")
BUCKET   = os.getenv("GCP_BUCKET")
RAW_TABLE = os.getenv("BQ_TABLE_RAW", "raw_parking")

LOCAL_DIR = "data"
os.makedirs(LOCAL_DIR, exist_ok=True)


# ===== Helpers =====
def fetch_json(url, retries=3, timeout=20):
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[fetch_json] Intento {i+1}/{retries} falló: {e}")
            if i == retries - 1:
                raise
            time.sleep(2 * (i+1))


def to_int_flexible(x):
    """Convierte strings con coma/punto a int. Devuelve None si no se puede."""
    if x is None:
        return None
    try:
        s = str(x).strip()
        # casos: "129", "1.234", "1,234", "1,23" (tomamos floor)
        s = s.replace(" ", "")
        # Primero intenta reemplazar separador de miles y decimal común
        # Estrategia simple: quitar miles y usar punto como decimal
        s = s.replace(".", "")        # elimina posibles miles
        s = s.replace(",", ".")       # coma -> punto
        val = float(s)
        return int(val)
    except:
        try:
            return int(float(x))
        except:
            return None


def normalize_records(obj, timestamp_iso):
    """
    Adaptado al JSON que compartiste:
    {
      "type": "FeatureCollection",
      "name": "parkings",
      "count": 16,
      "features": [
        {
          "type": "Feature",
          "geometry": { "type": "Point", "coordinates": [x, y] },
          "properties": {
            "tipo": "LIB",
            "plazasRotatorias": 210,
            "plazasResidentes": 0,
            "plazasResidentesLibres": 0,
            "libres": "129",
            "nombre": "Atotxa",
            "noteId": "...",
            "precios": [...]
          }
        }, ...
      ]
    }

    Mapeo de salida:
      parking_id -> properties.noteId
      name       -> properties.nombre
      free       -> properties.libres
      total      -> plazasRotatorias + plazasResidentes
      lon, lat   -> geometry.coordinates [x, y] (guardamos tal cual)
      timestamp  -> timestamp_iso (UTC)
    """
    rows = []

    if not isinstance(obj, dict) or "features" not in obj or not isinstance(obj["features"], list):
        print("[normalize] Estructura no esperada. Claves raíz:",
              list(obj.keys()) if isinstance(obj, dict) else type(obj))
        return rows

    feats = obj["features"]
    for feat in feats:
        if not isinstance(feat, dict):
            continue

        props = feat.get("properties", {}) or {}
        geom  = feat.get("geometry", {}) or {}

        # Coordenadas: JSON trae [x, y]; guardamos lon=x, lat=y
        lon_val, lat_val = None, None
        coords = geom.get("coordinates")
        if isinstance(coords, list) and len(coords) >= 2:
            try:
                lon_val = float(coords[0])
                lat_val = float(coords[1])
            except:
                lon_val, lat_val = None, None

        parking_id = props.get("noteId")
        name       = props.get("nombre")
        libres     = to_int_flexible(props.get("libres"))
        rot        = to_int_flexible(props.get("plazasRotatorias"))
        res        = to_int_flexible(props.get("plazasResidentes"))

        total = (rot or 0) + (res or 0)

        row = {
            "parking_id": str(parking_id) if parking_id is not None else None,
            "name": str(name) if name is not None else None,
            "free": libres,
            "total": total if total is not None else None,
            "lat": lat_val,
            "lon": lon_val,
            "timestamp": timestamp_iso
        }

        # descarta filas totalmente vacías
        if any(v is not None for v in row.values()):
            rows.append(row)

    print(f"[normalize] Filas generadas: {len(rows)}")
    if rows:
        print("[normalize] Ejemplo de fila:", rows[0])
    else:
        try:
            first = obj["features"][0]
            if isinstance(first, dict) and "properties" in first:
                print("[normalize] Claves properties (ejemplo):", list(first["properties"].keys()))
        except Exception as e:
            print("[normalize] No se pudo mostrar ejemplo de properties:", e)

    return rows


def write_local_csv(rows, path):
    header = ["timestamp", "parking_id", "name", "free", "total", "lat", "lon"]
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

    # Asegura dataset/tabla antes (para que existan aunque la primera corrida no produzca filas)
    ensure_bq_raw_table(PROJECT, DATASET, RAW_TABLE)

    # 1) Descarga JSON
    obj = fetch_json(API_URL)

    # 2) Normaliza a filas
    rows = normalize_records(obj, ts_iso)
    if not rows:
        print("Advertencia: sin filas normalizadas. Se guarda JSON crudo para depurar.")
        raw_path = os.path.join(LOCAL_DIR, f"raw_{stamp}.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        # No sube a GCS ni carga a BQ si no hay filas
        print(f"JSON crudo guardado en {raw_path}")
        return

    # 3) CSV local
    local_csv = os.path.join(LOCAL_DIR, f"data_{stamp}.csv")
    write_local_csv(rows, local_csv)

    # 4) Subir a GCS
    dest = f"raw/data_{stamp}.csv"
    upload_to_gcs(local_csv, BUCKET, dest)

    # 5) Cargar CSV a BigQuery (RAW)
    load_csv_to_bq(f"gs://{BUCKET}/{dest}", PROJECT, DATASET, RAW_TABLE)


if __name__ == "__main__":
    main()
