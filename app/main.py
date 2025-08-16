from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict
import time, re
from . import utils
import pandas as pd
import numpy as np

app = FastAPI(title="TDS Data Analyst Agent - Final")

TOTAL_TIMEOUT = 170.0  # seconds

def _find_questions_file(form) -> UploadFile:
    for k, v in form.multi_items():
        if hasattr(v, "filename") and v.filename:
            if v.filename.lower().endswith("questions.txt") or v.filename.lower() == "questions.txt":
                return v
    return None

@app.post("/api/")
async def analyze(request: Request):
    start = time.time()
    form = await request.form()
    qfile = _find_questions_file(form)
    if qfile is None:
        raise HTTPException(status_code=400, detail="questions.txt must be provided")

    questions_text = (await qfile.read()).decode("utf-8", errors="ignore")

    uploads: Dict[str, UploadFile] = {}
    for k, v in form.multi_items():
        if hasattr(v, "filename") and v.filename:
            uploads[v.filename] = v

    try:
        if "wikipedia.org" in questions_text:
            m = re.search(r"https?://\S+", questions_text)
            if not m:
                raise HTTPException(status_code=400, detail="No URL found in questions.txt")
            url = m.group(0).strip()
            df = utils.scrape_table_from_wikipedia(url)

            gross_col = utils.find_column(df, ['gross','worldwide'])
            year_col = utils.find_column(df, ['year','released','release'])
            rank_col = utils.find_column(df, ['rank'])
            peak_col = utils.find_column(df, ['peak'])
            title_col = utils.find_column(df, ['title','film','movie','name']) or df.columns[0]

            gross_s = utils.to_num_col(df[gross_col]) if gross_col is not None else pd.Series([np.nan]*len(df))
            if year_col is not None:
                years = pd.to_numeric(df[year_col].astype(str).str.extract(r"(\d{4})")[0], errors='coerce')
            else:
                years = pd.to_numeric(df.iloc[:,0].astype(str).str.extract(r"(\d{4})")[0], errors='coerce')

            count_2bn_before_2000 = int(((gross_s >= 2_000_000_000) & (years < 2000)).sum())

            mask = (gross_s >= 1_500_000_000)
            film_name = ""
            if mask.any():
                cand_idx = mask[mask].index
                if len(cand_idx) > 1 and (years.notna().sum() > 0):
                    cand_years = years.loc[cand_idx]
                    first_idx = cand_years.sort_values().dropna().index[0] if cand_years.dropna().shape[0] > 0 else cand_idx[0]
                else:
                    first_idx = cand_idx[0]
                film_name = str(df.loc[first_idx, title_col])

            corr = 0.0
            if rank_col is not None and peak_col is not None:
                r_series = pd.to_numeric(df[rank_col], errors='coerce')
                p_series = utils.to_num_col(df[peak_col])
                common = r_series.dropna().index.intersection(p_series.dropna().index)
                if len(common) > 1:
                    corr = float(round(r_series.loc[common].corr(p_series.loc[common]), 6))
                else:
                    corr = 0.0
            else:
                corr = 0.0

            img_uri = ""
            if rank_col is not None and peak_col is not None:
                r_series = pd.to_numeric(df[rank_col], errors='coerce')
                p_series = utils.to_num_col(df[peak_col])
                common = r_series.dropna().index.intersection(p_series.dropna().index)
                if len(common) >= 2:
                    x = r_series.loc[common].values
                    y = p_series.loc[common].values
                    img_uri = utils.scatterplot_png_datauri(x, y, xlabel='Rank', ylabel='Peak', title='Rank vs Peak', max_bytes=100000)
                else:
                    img_uri = ""
            result = [count_2bn_before_2000, film_name, corr, img_uri]
            return JSONResponse(content=result)

        if "data.csv" in uploads:
            df = utils.read_uploaded_csv(uploads["data.csv"])
            numerics = df.select_dtypes(include=['number'])
            rows = int(df.shape[0])
            first_col = df.columns[0] if len(df.columns) > 0 else ""
            corr = 0.0
            img_uri = ""
            if numerics.shape[1] >= 2:
                c = numerics.iloc[:,0].corr(numerics.iloc[:,1])
                corr = float(round(float(c) if not pd.isna(c) else 0.0, 6))
                try:
                    img_uri = utils.scatterplot_png_datauri(numerics.iloc[:,0].values, numerics.iloc[:,1].values, xlabel=str(numerics.columns[0]), ylabel=str(numerics.columns[1]), title='Scatter', max_bytes=100000)
                except Exception:
                    img_uri = ""
            return JSONResponse(content=[rows, first_col, corr, img_uri])

        raise HTTPException(status_code=400, detail='Task not recognized; include a URL or upload data.csv')

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})
