import os, json, datetime, requests
from google.cloud import storage

DATA_URL = os.environ.get("DATA_URL", "https://api.citybik.es/v2/networks/bilbao")  
# Puedes cambiar a otro endpoint si prefieres (ejemplo Donostia o Medellín)
BUCKET = os.environ["BUCKET"]

def fetch_and_store(request=None):
    """Descarga JSON y lo guarda en GCS"""
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    data = r.json()

    now = datetime.datetime.utcnow()
    path = f"raw/dt={now.strftime('%Y-%m-%d')}/hour={now.strftime('%H')}/{int(now.timestamp())}.json"

    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(path)
    blob.upload_from_string(json.dumps(data), content_type="application/json")

    print(f"Data saved to: {path}")
    return {"ok": True, "path": path}
