# MLOps – Reto 4 (Entrega)
**Universidad EAFIT – 2025-2**  
**Curso:** Redes Intensivas en Datos  
**Reto 4:** Nivel 3 + Monitoreo + Visualización + Reentrenamiento  
**Autor:** _[tu nombre]_  
**Proyecto:** Predicción de plazas libres en parqueaderos (Donostia)  
**Fecha:** 2025-10-09

---

## 1. Objetivo
Implementar un pipeline MLOps reproducible que:
1) Extrae datos periódicos desde una API pública (Donostia – parqueaderos).  
2) Almacena los datos crudos en **Google Cloud Storage (GCS)** y **BigQuery (RAW)**.  
3) Entrena automáticamente un modelo de series de tiempo (baseline / SARIMAX) y guarda métricas y predicciones en **BigQuery**.  
4) Habilita **visualización/monitoreo** (Grafana) y **reentrenamiento** mediante **GitHub Actions**.

---

## 2. Arquitectura (alto nivel)
1. **Ingesta**: `extract_data.py` descarga JSON, normaliza, guarda CSV local, sube a `gs://<BUCKET>/raw/` y hace **LOAD** a `BigQuery` (tabla `raw_parking`).  
2. **Entrenamiento**: `training.py` lee `raw_parking`, **agrega por hora** (evita duplicados sub-horarios), entrena `baseline` o `SARIMAX`, y escribe:
   - Tabla `model_metrics` (MAE, puntos de entrenamiento, descriptor de modelo).
   - Tabla `predictions` (timestamp futuro → `yhat`).  
3. **Orquestación**: `.github/workflows/pipeline.yml` (cron) + modo **“ráfaga”** (para simular lecturas cada 15 s dentro de un job).  
4. **Visualización**: Grafana con fuente de datos **BigQuery**.

---

## 3. Prerrequisitos
- Proyecto de GCP con facturación.  
- Habilitar APIs: BigQuery API, BigQuery Storage API, Cloud Storage API.
- Crear **bucket** GCS (ej. `mlops-reto4-datos`).  
- Crear **dataset** BigQuery (ej. `mlops_reto4`, región `us`).  
- Crear **Service Account** con roles:
  - `roles/storage.objectAdmin`
  - `roles/bigquery.dataEditor`
  - `roles/bigquery.jobUser`
  - Generar **key JSON** (se usará como secreto).

---

## 4. Variables/secrets (GitHub → Settings → Secrets → Actions)
- `GCP_PROJECT_ID` → ID del proyecto (p. ej. `eafit-mlops-1234`)  
- `GCP_BUCKET` → `mlops-reto4-datos`  
- `GCP_DATASET` → `mlops_reto4`  
- `GCP_LOCATION` → `us-east1`  
- `GCP_SA_KEY` → **Contenido completo** del JSON de la Service Account  

---

## 5. Estructura del repositorio
```
.
├── data_extraction/
│   └── extract_data.py
├── training/
│   └── training.py
├── monitoring/
│   └── (se generará metrics.csv)
├── .github/workflows/
│   └── pipeline.yml
├── requirements.txt
└── README.md
```

---

## 6. Instalación local (opcional para pruebas)
```bash
python -m venv .venv && source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

export GOOGLE_APPLICATION_CREDENTIALS="/ruta/a/sa-key.json"
export GCP_PROJECT_ID="<tu-proyecto>"
export GCP_BUCKET="mlops-reto4-datos"
export GCP_DATASET="mlops_reto4"
export GCP_LOCATION="us"

python data_extraction/extract_data.py
python training/training.py
```

---

## 7. Ingesta de datos (extractor)
**Archivo:** `data_extraction/extract_data.py`  
**Flujo:** descarga JSON → normaliza → guarda `data_YYYYMMDD_HHMMSS.csv` → sube a GCS → `LOAD` a `BigQuery` (`raw_parking`).

**Esquema de `raw_parking`:**
```sql
timestamp TIMESTAMP,
parking_id STRING,
name STRING,
free INT64,
total INT64,
lat FLOAT64,
lon FLOAT64
```

**Verificación rápida en BigQuery:**
```sql
SELECT COUNT(*) AS n_rows FROM `<PROYECTO>.<DATASET>.raw_parking`;

SELECT
  TIMESTAMP_TRUNC(timestamp, HOUR) AS ts_hour,
  SUM(CAST(free AS INT64)) AS total_free
FROM `<PROYECTO>.<DATASET>.raw_parking`
GROUP BY ts_hour
ORDER BY ts_hour DESC
LIMIT 24;
```

> **Nota:** Para datos sub-horarios, el entrenador **agrega por hora** con `TIMESTAMP_TRUNC(timestamp, HOUR)` y rellena huecos con last-observation-carried-forward.

---

## 8. Entrenamiento y predicción
**Archivo:** `training/training.py`  
- Agrega por hora la serie total (`SUM(free)`) por ciudad.  
- Si hay pocos puntos → `baseline` (naive/last-value o promedio por hora del día).  
- Con suficientes horas → `SARIMAX(1,1,1)(1,1,1,24)`.

**Tablas de salida:**
```sql
-- model_metrics
run_at TIMESTAMP,
train_points INT64,
mae FLOAT64,
model_desc STRING;

-- predictions
created_at TIMESTAMP,
timestamp TIMESTAMP,
yhat FLOAT64;
```

**Verificación:**
```sql
SELECT * FROM `<PROYECTO>.<DATASET>.model_metrics` ORDER BY run_at DESC LIMIT 10;

SELECT * FROM `<PROYECTO>.<DATASET>.predictions`
WHERE created_at = (SELECT MAX(created_at) FROM `<PROYECTO>.<DATASET>.predictions`)
ORDER BY timestamp;
```

> **¿Qué es `yhat`?** Es el símbolo `ŷ` (y gorrito): **valor predicho** de la variable objetivo para un timestamp futuro.

---

## 9. Orquestación – GitHub Actions
**Archivo:** `.github/workflows/pipeline.yml`

- Ejecuta cada 1 hora (cron).  

```yaml
- name: Extract Data
  run: 
    python collector/collector.py 
```

Luego:
```yaml
- name: Train model
  run: python training/training.py
```

---

## 10. Visualización en Grafana
1. **Data Source**: BigQuery (sube el JSON de la Service Account).  
2. **Panel de métricas (MAE):**
```sql
SELECT TIMESTAMP(run_at) AS run_at, mae, model_desc
FROM `<PROYECTO>.<DATASET>.model_metrics`
ORDER BY run_at;
```
3. **Panel de pronóstico (última corrida):**
```sql
SELECT TIMESTAMP(timestamp) AS ts, yhat
FROM `<PROYECTO>.<DATASET>.predictions`
WHERE created_at = (SELECT MAX(created_at) FROM `<PROYECTO>.<DATASET>.predictions`)
ORDER BY ts;
```

---

## 11. Reproducibilidad (Cómo reproducir el proyecto?)
1. Clonar repo.  
2. Crear secrets en Actions con valores de un proyecto GCP del profesor (o Service Account provista).  
3. Activar workflow manualmente en Github actions.  
4. Verificar tablas y paneles siguiendo las consultas indicadas.

