from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import math
from io import StringIO
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="AI CSV Analyzer", version="1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

PREVIEW_ROWS = 10
MAX_CATEGORIES = 20
HIST_BINS = 10

def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]

    if x is None:
        return None

    if isinstance(x, np.integer):
        return int(x)

    if isinstance(x, np.floating):
        return None if not np.isfinite(x) else float(x)

    if isinstance(x, float):
        return None if not math.isfinite(x) else x

    return x


def column_summary(df):
    n = len(df)
    return [
        {
            "column": c,
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "missing_percentage": round(df[c].isna().mean() * 100, 2) if n else 0,
            "unique": int(df[c].nunique())
        }
        for c in df.columns
    ]


def statistics(df, numeric):
    return df[numeric].describe().round(2).to_dict() if numeric else {}


def value_counts(df, categorical):
    result = {}

    for c in categorical:
        if df[c].nunique() <= MAX_CATEGORIES:
            result[c] = {
                str(k) if pd.notna(k) else "Missing": int(v)
                for k, v in df[c].value_counts(dropna=False).items()
            }

    return result


def correlations(df, numeric):
    return df[numeric].corr().round(2).to_dict() if len(numeric) > 1 else {}


def outliers(df, numeric):
    result = {}

    for c in numeric:
        s = df[c].dropna()

        if s.empty:
            result[c] = 0
            continue

        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        result[c] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

    return result


def histograms(df, numeric):
    result = {}

    for c in numeric:
        s = df[c].dropna()

        if s.empty:
            continue

        if s.nunique() == 1:
            result[c] = {
                "bins": [float(s.iloc[0])],
                "counts": [len(s)]
            }
            continue

        counts, bins = pd.cut(
            s,
            bins=HIST_BINS,
            retbins=True,
            include_lowest=True
        )

        result[c] = {
            "bins": [round(float(x), 2) for x in bins],
            "counts": [
                int(x)
                for x in counts.value_counts().sort_index().values
            ]
        }

    return result


@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file.")

    try:
        data = await file.read()
        df = pd.read_csv(StringIO(data.decode("utf-8")))
    except UnicodeDecodeError:
        raise HTTPException(400, "CSV must use UTF-8 encoding.")
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}")

    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(exclude="number").columns.tolist()

    result = {
        "filename": file.filename,
        "rows": len(df),
        "columns": df.columns.tolist(),

        "numeric_columns": numeric,
        "categorical_columns": categorical,

        "column_summary": column_summary(df),

        "missing_values": df.isna().sum().to_dict(),
        "unique_values": df.nunique().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),

        "statistics": statistics(df, numeric),
        "value_counts": value_counts(df, categorical),
        "correlation": correlations(df, numeric),
        "outliers": outliers(df, numeric),
        "histogram_data": histograms(df, numeric),

        "preview": df.head(PREVIEW_ROWS).to_dict(orient="records")
    }

    return JSONResponse(content=clean(result))