import requests, pandas as pd, io, re, base64
import numpy as np
from bs4 import BeautifulSoup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats

REQUEST_TIMEOUT = 25

def scrape_table_from_wikipedia(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={'User-Agent':'tds-agent/1.0'})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = None
    for t in soup.find_all('table'):
        cl = t.get('class') or []
        clstr = ' '.join(cl).lower()
        if 'wikitable' in clstr or 'sortable' in clstr:
            table = t
            break
    if table is None:
        table = soup.find('table')
    if table is None:
        raise ValueError('No table found on page')
    df = pd.read_html(str(table), flavor='bs4')[0]
    return df

def find_column(df: pd.DataFrame, keywords):
    for c in df.columns:
        lc = str(c).lower()
        for kw in keywords:
            if kw in lc:
                return c
    return None

def to_num_col(series: pd.Series) -> pd.Series:
    s = series.astype(str).fillna('')
    def parse(x):
        x = str(x).strip()
        if x == '' or x.lower() == 'nan':
            return float('nan')
        m = re.search(r'([0-9\.,]+)\s*billion', x, re.I)
        if m:
            return float(m.group(1).replace(',','')) * 1_000_000_000
        m = re.search(r'([0-9\.,]+)\s*million', x, re.I)
        if m:
            return float(m.group(1).replace(',','')) * 1_000_000
        cleaned = re.sub(r'[^0-9\.\-]', '', x)
        if cleaned == '':
            return float('nan')
        try:
            if '.' in cleaned:
                return float(cleaned)
            else:
                return int(cleaned)
        except:
            try:
                return float(cleaned)
            except:
                return float('nan')
    return s.map(parse).astype(float)

def read_uploaded_csv(upload):
    content = upload.file.read()
    upload.file.seek(0)
    return pd.read_csv(io.BytesIO(content))

def scatterplot_png_datauri(x, y, xlabel='x', ylabel='y', title='', max_bytes=100000):
    if len(x) == 0 or len(y) == 0:
        raise ValueError('Empty data for plotting')
    def render(dpi, w, h, palette=False):
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
        ax.scatter(x, y, s=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if len(x) >= 2:
            try:
                slope, intercept, r, p, se = stats.linregress(x, y)
                xs = np.array([min(x), max(x)])
                ys = intercept + slope * xs
                ax.plot(xs, ys, linestyle=':', linewidth=2, color='red')
            except Exception:
                pass
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, optimize=True)
        plt.close(fig)
        data = buf.getvalue()
        if palette:
            img = Image.open(io.BytesIO(data)).convert('RGBA')
            img = img.convert('P', palette=Image.ADAPTIVE, colors=128)
            buf2 = io.BytesIO()
            img.save(buf2, format='PNG', optimize=True)
            return buf2.getvalue()
        return data

    data = render(150, 800, 600, palette=False)
    if len(data) <= max_bytes:
        return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')
    data = render(150, 800, 600, palette=True)
    if len(data) <= max_bytes:
        return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')
    for dpi in [120,100,80,60,50,40]:
        for scale in [1.0,0.9,0.8,0.7,0.6,0.5]:
            w = int(800 * scale)
            h = int(600 * scale)
            data = render(dpi, w, h, palette=True)
            if len(data) <= max_bytes:
                return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')
    img = Image.open(io.BytesIO(data)).convert('RGBA')
    img = img.resize((400,300), Image.LANCZOS)
    img = img.convert('P', palette=Image.ADAPTIVE, colors=64)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    data = buf.getvalue()
    if len(data) <= max_bytes:
        return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')
    raise ValueError(f'Could not compress image under {max_bytes} bytes')
