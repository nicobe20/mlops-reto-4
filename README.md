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
- Crear **bucket** GCS (ej. `mlops-reto4-datos`):

![screenshot 1760043090](https://github.com/user-attachments/assets/dfa9a1cb-4995-4b50-b431-65aa8111921c)

![screenshot 1760043107](https://github.com/user-attachments/assets/21a4785b-6fa9-403f-b3fe-18297f313deb)

![screenshot 1760043217](https://github.com/user-attachments/assets/ebcfcadc-f00f-459d-b2e6-bf3e286ca229)

![screenshot 1760043261](https://github.com/user-attachments/assets/b1f5f045-113d-4abc-b809-bcead2958b72)

![screenshot 1760043334](https://github.com/user-attachments/assets/94966a5f-11fc-40f3-a641-5a9d6c491f48)


- Crear **dataset** BigQuery (ej. `mlops_reto4`, región `us`).
  
![screenshot 1760043507](https://github.com/user-attachments/assets/b6acc7ea-59ec-41c3-9c6e-7ef9716b175f)

![screenshot 1760043682](https://github.com/user-attachments/assets/8b6eaed6-3c33-4ae1-b02b-6baecbcd5f8a)

![screenshot 1760043804](https://github.com/user-attachments/assets/bdc7da0a-e4f6-4666-b7e2-d10fd9f17ff0)


- Crear **Service Account** con roles:
  - `roles/storage.objectAdmin`
  - `roles/bigquery.dataEditor`
  - `roles/bigquery.jobUser`
  - Generar **key JSON** (se usará como secreto).

![screenshot 1760043857](https://github.com/user-attachments/assets/7f77a332-d2c8-4998-9539-1347a59908d2)

![screenshot 1760044062](https://github.com/user-attachments/assets/f4495d1a-538a-482e-bf5d-172086770a3f)

![screenshot 1760044283](https://github.com/user-attachments/assets/2e643ca1-1ad8-4992-b55f-d0387ce7e0c3)

![screenshot 1760044663](https://github.com/user-attachments/assets/13bcbdf8-466f-4a70-ad23-0f3588572d0e)

![screenshot 1760044679](https://github.com/user-attachments/assets/b470a498-947d-40e2-bb1e-fa0262a7c01b)


---

## 4. Variables/secrets (GitHub → Settings → Secrets → Actions)
- `GCP_PROJECT_ID` → ID del proyecto (p. ej. `eafit-mlops-1234`)  
- `GCP_BUCKET` → `mlops-reto4-datos`  
- `GCP_DATASET` → `mlops_reto4`  
- `GCP_LOCATION` → `us-east1`  
- `GCP_SA_KEY` → **Contenido completo** del JSON de la Service Account  

<img width="1892" height="843" alt="image" src="https://github.com/user-attachments/assets/616f2bb2-78ab-447a-a05e-2901bbee3d14" />

<img width="986" height="516" alt="image" src="https://github.com/user-attachments/assets/44d4ce09-7ab6-4b92-a99f-32e5e262dbc2" />


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
![screenshot 1760048140](https://github.com/user-attachments/assets/ddfe3564-d28c-48ff-bf03-8b50c07abe12)

![screenshot 1760049343](https://github.com/user-attachments/assets/e0ad108d-35a7-40a9-8d9b-b0dc089267cd)

![screenshot 1760048183](https://github.com/user-attachments/assets/130667f4-b39b-4868-9742-8ae022fa9800)


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

![screenshot 1760048288](https://github.com/user-attachments/assets/b9d416fa-1ae6-4c43-b90a-ba1c19b9be41)

<img width="1165" height="763" alt="image" src="https://github.com/user-attachments/assets/5ef61678-bcd4-4c58-9911-97af36c99056" />

<img width="1175" height="849" alt="image" src="https://github.com/user-attachments/assets/170b4266-2d3e-4d62-aa13-6650c3be7a74" />



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
<img width="1919" height="848" alt="screenshot 1760050400" src="https://github.com/user-attachments/assets/660ef1f8-b1c3-4b34-9d52-5cc17638a5d1" />

<img width="1431" height="806" alt="screenshot 1760050420" src="https://github.com/user-attachments/assets/9b4ff1dc-0172-4488-82e1-769e09b80cd9" />

![screenshot 1760055256](https://github.com/user-attachments/assets/297b1cc2-bde9-425b-83c8-e183a01b2c86)



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

