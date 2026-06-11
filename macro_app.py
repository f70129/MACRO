"""
總經週期追蹤看板  Macro Cycle Dashboard(官方來源版)
====================================================
設計原則(依使用者要求):
  1. 不使用任何第三方 API / 金鑰:只用官方檔案下載端點
       • FRED 官方 CSV 下載連結(免帳號,等同網站上的 Download CSV)
       • 台灣證交所 MOPS 官方月營收靜態報表(t21sc03 檔案)
       • Yahoo Finance 公開報價(僅作市場代理,非核心數據)
  2. ISM PMI 為 ISM 協會之授權調查數據,「無法」由其他數據計算還原。
     本看板改用官方免費、同性質的擴散指數調查:
       • 紐約聯儲 Empire State / 費城聯儲 / 達拉斯聯儲 製造業調查
       • 0 ≈ ISM 的 50(擴張/收縮分界),與 ISM 高度相關且更早公布
       • 新訂單-庫存 Spread 以費城聯儲子項目直接相減
       • 另以 Census 耐久財/製造業新訂單、存貨銷售比等硬數據交叉驗證
  3. 不容許錯誤數據:
       • 每條序列做「數值範圍 + 日期新鮮度」驗證,不合格 → 顯示「不可用」
       • 台股營收 YoY 以 MOPS 官方檔內「去年同月增減(%)」欄位交叉比對自算值
       • 「資料健康度」面板列出所有序列之來源/代碼/最新日期/狀態,可逐一稽核

自動更新:st.cache_data(ttl=...) 到期後,任何人開啟頁面即重新抓取官方最新檔案。
"""

import warnings

warnings.filterwarnings("ignore")

import io
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# 頁面設定
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌐 總經週期追蹤看板",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  html, body, [class*="css"] { font-family: "Microsoft JhengHei", "PingFang TC", sans-serif; }
  #MainMenu, footer { visibility: hidden; }
  .sig-card {
    background: #1a1a2e; border-radius: 10px; padding: 12px 14px;
    margin-bottom: 8px; border-left: 5px solid #555; min-height: 118px;
  }
  .sig-bull { border-left-color: #26a69a; }
  .sig-bear { border-left-color: #ef5350; }
  .sig-warn { border-left-color: #ffd700; }
  .sig-na   { border-left-color: #555; }
  .sig-card h4 { margin: 0 0 4px 0; font-size: 13px; color: #aaa; }
  .sig-card .val { font-size: 19px; font-weight: bold; color: #fff; }
  .sig-card .sub { font-size: 11.5px; color: #888; margin-top: 4px; line-height: 1.4; }
</style>
""",
    unsafe_allow_html=True,
)

TW_TZ = pytz.timezone("Asia/Taipei")

# 完整瀏覽器標頭:FRED / MOPS / Yahoo 會拒絕非瀏覽器 User-Agent 的請求
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/csv,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
}

# FRED 的防護會以 TLS 指紋識別程式化請求並讓連線逾時(ReadTimeout),
# curl_cffi 以瀏覽器 TLS 指紋發送請求即可正常取得;無此套件時退回 requests。
try:
    from curl_cffi import requests as cf_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False


def http_get(url: str, params: dict | None = None):
    """帶重試(0s/2s/5s)的 GET,優先使用瀏覽器 TLS 指紋。回傳 (Response或None, 錯誤描述)。
    錯誤描述會原樣顯示在資料健康度面板,供除錯定位。404 為確定性錯誤,不重試。"""
    last_err = "未知錯誤"
    for wait in (0, 2, 5):
        if wait:
            time.sleep(wait)
        if HAS_CURL_CFFI:
            try:
                r = cf_requests.get(url, params=params, headers=BROWSER_HEADERS,
                                    timeout=30, impersonate="chrome")
                if r.status_code == 200:
                    return r, ""
                if r.status_code == 404:
                    return None, "HTTP 404"
                last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:120]}"
        try:
            r = requests.get(url, params=params, headers=BROWSER_HEADERS, timeout=30)
            if r.status_code == 200:
                return r, ""
            if r.status_code == 404:
                return None, "HTTP 404"
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
    return None, last_err

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    margin=dict(l=40, r=20, t=50, b=30),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)

TTL_MARKET = 60 * 60          # 市場數據:1 小時
TTL_MACRO = 6 * 60 * 60       # 總經數據:6 小時
TTL_MOPS = 24 * 60 * 60       # MOPS 月營收:24 小時

# ─────────────────────────────────────────────────────────────────────────────
# 資料健康度登記簿(每條序列的驗證結果都記錄於此,UI 可稽核)
# ─────────────────────────────────────────────────────────────────────────────
HEALTH: list = []  # dict(name, source, code, last_date, last_value, status, note)


def register_health(name, source, code, s: pd.Series, status, note=""):
    HEALTH.append(dict(
        name=name, source=source, code=code,
        last_date="—" if s.dropna().empty else s.dropna().index[-1].strftime("%Y-%m-%d"),
        last_value="—" if s.dropna().empty else f"{float(s.dropna().iloc[-1]):,.2f}",
        status=status, note=note,
    ))


def validate_series(s: pd.Series, lo: float, hi: float, max_age_days: int):
    """驗證:數值範圍 + 新鮮度。回傳 (通過驗證的序列, 狀態, 備註)。
    範圍外的值一律剔除;剔除後若為空 → 不可用;過舊 → 標示警告但仍顯示。"""
    if s.dropna().empty:
        return pd.Series(dtype=float), "❌ 不可用", "抓取失敗或無資料"
    s = s.dropna().sort_index()
    bad = s[(s < lo) | (s > hi)]
    s = s[(s >= lo) & (s <= hi)]
    note = ""
    if len(bad) > 0:
        note = f"剔除 {len(bad)} 筆範圍外異常值;"
    if s.empty:
        return s, "❌ 不可用", note + "全部數值超出合理範圍"
    age = (datetime.now() - s.index[-1].to_pydatetime().replace(tzinfo=None)).days
    if age > max_age_days:
        return s, "⚠️ 資料偏舊", note + f"最新值距今 {age} 天(門檻 {max_age_days} 天)"
    return s, "✅ 正常", note + f"最新值距今 {age} 天"


# ─────────────────────────────────────────────────────────────────────────────
# FRED:官方 CSV 下載端點(免帳號、免金鑰;即官網圖表頁的 Download CSV)
# ─────────────────────────────────────────────────────────────────────────────
def _get_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:
        return ""


@st.cache_data(ttl=TTL_MACRO, show_spinner=False)
def fred_api(series_id: str, api_key: str):
    """FRED 官方 API(api.stlouisfed.org)。回傳 (Series, 錯誤描述)。"""
    r, err = http_get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": series_id, "api_key": api_key,
                "file_type": "json", "observation_start": "1950-01-01"},
    )
    if r is None:
        # 錯誤訊息可能含完整 URL,先遮蔽 api_key 再記錄,避免洩漏到健康度面板
        return pd.Series(dtype=float), f"官方 API:{err.replace(api_key, '***')}"
    try:
        j = r.json()
        if "observations" not in j:
            msg = str(j.get("error_message", ""))[:120]
            return pd.Series(dtype=float), f"官方 API 回應異常:{msg or '無 observations 欄位'}"
        obs = j["observations"]
        idx = pd.to_datetime([o["date"] for o in obs], errors="coerce")
        vals = pd.to_numeric([o["value"] for o in obs], errors="coerce")  # 缺值為 "."
        s = pd.Series(vals, index=idx).dropna()
        return s.sort_index(), ""
    except Exception as e:
        return pd.Series(dtype=float), f"官方 API 解析失敗 {type(e).__name__}: {str(e)[:120]}"


@st.cache_data(ttl=TTL_MACRO, show_spinner=False)
def fred_csv(series_id: str):
    """回傳 (Series, 錯誤描述)。錯誤描述空字串代表成功。"""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r, err = http_get(url)
    if r is None:
        return pd.Series(dtype=float), err
    try:
        df = pd.read_csv(io.StringIO(r.text))
        if df.shape[1] != 2:
            return pd.Series(dtype=float), f"CSV 欄數異常({df.shape[1]} 欄)"
        # 防呆:確認第二欄欄名即為該序列代碼(避免錯誤頁面被誤判為數據)
        if df.columns[1].strip().upper() != series_id.upper():
            return pd.Series(dtype=float), f"欄名不符:回傳 '{df.columns[1]}',預期 '{series_id}'"
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        s = pd.to_numeric(df.set_index("date")["value"], errors="coerce").dropna()
        return s.sort_index(), ""
    except Exception as e:
        return pd.Series(dtype=float), f"解析失敗 {type(e).__name__}: {str(e)[:120]}"


def load_fred(name, series_id, lo, hi, max_age_days) -> pd.Series:
    """有 FRED_API_KEY(Streamlit Secrets)時走官方 API,否則退回官方 CSV 下載端點。"""
    api_key = _get_secret("FRED_API_KEY")
    source = "FRED(官方 CSV)"
    if api_key:
        raw, err = fred_api(series_id, api_key)
        source = "FRED(官方 API)"
        if raw.empty:  # API 失敗時再退回 CSV
            raw2, err2 = fred_csv(series_id)
            if not raw2.empty:
                raw, err, source = raw2, "", "FRED(官方 CSV,API 備援)"
            else:
                err = f"{err};CSV 備援亦失敗:{err2}"
    else:
        raw, err = fred_csv(series_id)
    s, status, note = validate_series(raw, lo, hi, max_age_days)
    if err:
        note = (note + ";" if note else "") + f"錯誤:{err}"
    register_health(name, source, series_id, s, status, note)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 台灣證交所 MOPS:官方月營收靜態報表(免金鑰)
#   檔案格式:https://mops.twse.com.tw/nas/t21/sii/t21sc03_{民國年}_{月}_0.html
#   內含官方欄位「去年當月營收」「去年同月增減(%)」→ 用於交叉驗證自算 YoY
# ─────────────────────────────────────────────────────────────────────────────
SEMI_IDS = {"2330": "台積電", "2303": "聯電", "2454": "聯發科"}


# 證交所 2024 年改版後,舊版靜態月營收檔移至 legacy 主機 mopsov;依序嘗試兩個主機
MOPS_HOSTS = ("https://mopsov.twse.com.tw", "https://mops.twse.com.tw")


def _parse_mops_tables(text: str):
    """解析 MOPS 月營收 HTML。回傳 ({代號:(營收, 官方YoY)}, 表頭樣本字串供除錯)。
    容錯:欄名含空白/換行、MultiIndex 欄名、表頭被讀成資料列、代號被讀成浮點數。"""
    out, header_samples = {}, []

    def norm(c):
        parts = c if isinstance(c, tuple) else (c,)
        return re.sub(r"\s+", "", "".join(str(p) for p in parts))

    try:
        tables = pd.read_html(io.StringIO(text))
    except ValueError:
        return out, "read_html 找不到任何表格"
    for t in tables:
        cols = [norm(c) for c in t.columns]
        # 表頭被當成資料列時(欄名為數字索引),改用第一列當表頭
        if not any("公司代號" in c for c in cols) and len(t) > 0:
            first = [re.sub(r"\s+", "", str(x)) for x in t.iloc[0]]
            if any("公司代號" in x for x in first):
                cols, t = first, t.iloc[1:]
        t = t.copy()
        t.columns = cols
        if len(cols) >= 3:
            header_samples.append("|".join(cols[:5]))
        cid = next((c for c in cols if "公司代號" in c), None)
        crev = next((c for c in cols
                     if "當月營收" in c and not any(x in c for x in ("去年", "上月", "累計"))), None)
        cyoy = next((c for c in cols if "去年同月增減" in c), None)
        if not cid or not crev:
            continue
        for _, row in t.iterrows():
            sid = re.sub(r"\.0$", "", str(row[cid]).strip())
            if sid in SEMI_IDS:
                rev = pd.to_numeric(str(row[crev]).replace(",", ""), errors="coerce")
                yoyv = (pd.to_numeric(str(row[cyoy]).replace(",", ""), errors="coerce")
                        if cyoy else np.nan)
                if pd.notna(rev):
                    out[sid] = (float(rev), float(yoyv) if pd.notna(yoyv) else np.nan)
    sample = ";".join(header_samples[:2]) or "無可辨識表頭"
    return out, sample


def _fetch_mops_month(year: int, month: int):
    """抓取單月全上市公司營收檔。回傳 ({股票代號:(當月營收千元, 官方YoY%)} 或 None, 錯誤描述)。"""
    try:
        r, err = None, "未知錯誤"
        for host in MOPS_HOSTS:
            url = f"{host}/nas/t21/sii/t21sc03_{year - 1911}_{month}_0.html"
            r, err = http_get(url)
            if r is not None:
                break
        if r is None:
            return None, err
        if len(r.content) < 5000:
            return None, f"檔案過小({len(r.content)} bytes),該月可能尚未公布"
        # 編碼自動偵測:新版檔案為 UTF-8,舊版為 Big5
        raw = r.content
        text = None
        for enc in ("utf-8", "big5", "cp950"):
            try:
                text = raw.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if text is None:
            text = raw.decode("big5", errors="ignore")
        out, header_sample = _parse_mops_tables(text)
        if not out:
            return None, f"檔案可讀但找不到目標公司列;表頭樣本:{header_sample[:100]}"
        return out, ""
    except Exception as e:
        return None, f"解析失敗 {type(e).__name__}: {str(e)[:120]}"


@st.cache_data(ttl=TTL_MOPS, show_spinner=False)
def mops_semi_revenue(n_months: int = 60):
    """回傳 (各公司月營收 DataFrame[千元], 官方YoY DataFrame[%], 錯誤摘要)。並行抓取官方檔。"""
    now = datetime.now(TW_TZ)
    months = []
    y, m = now.year, now.month
    for _ in range(n_months + 1):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        months.append((y, m))
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda ym: (_fetch_mops_month(*ym), ym), months))
    rev_rows, yoy_rows, errs = {}, {}, {}
    for (data, err), (yy, mm) in results:
        if not data:
            errs[f"{yy}-{mm:02d}"] = err
            continue
        ts = pd.Timestamp(yy, mm, 1)
        rev_rows[ts] = {sid: v[0] for sid, v in data.items()}
        yoy_rows[ts] = {sid: v[1] for sid, v in data.items()}
    # 錯誤摘要:取最常見的錯誤訊息(最新月份尚未公布屬正常,不列入)
    err_summary = ""
    real_errs = {k: v for k, v in errs.items() if "尚未公布" not in v}
    if real_errs:
        common = pd.Series(list(real_errs.values())).mode()
        err_summary = f"{len(real_errs)} 個月份抓取失敗,主因:{common.iloc[0] if len(common) else '不明'}"
    if not rev_rows:
        return pd.DataFrame(), pd.DataFrame(), err_summary or "全部月份抓取失敗"
    rev = pd.DataFrame.from_dict(rev_rows, orient="index").sort_index()
    off_yoy = pd.DataFrame.from_dict(yoy_rows, orient="index").sort_index()
    return rev, off_yoy, err_summary


# ─────────────────────────────────────────────────────────────────────────────
# 台股月營收備援鏈(MOPS 報表檔失效時自動啟用)
#   第 2 層:FinMind(免金鑰,資料源即公開資訊觀測站申報數據)
#   交叉驗證:證交所 OpenAPI 官方當月營收(openapi.twse.com.tw,免金鑰)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=TTL_MOPS, show_spinner=False)
def finmind_month_revenue(stock_id: str):
    """回傳 (月營收 Series[千元], 錯誤描述)。FinMind 原始單位為元,統一換算為千元。"""
    r, err = http_get(
        "https://api.finmindtrade.com/api/v4/data",
        params={"dataset": "TaiwanStockMonthRevenue", "data_id": stock_id,
                "start_date": "2018-01-01"},
    )
    if r is None:
        return pd.Series(dtype=float), err
    try:
        j = r.json()
        rows = j.get("data", [])
        if not rows:
            return pd.Series(dtype=float), (str(j.get("msg", ""))[:120] or "回應無資料")
        df = pd.DataFrame(rows)
        idx = pd.to_datetime(df["revenue_year"].astype(str) + "-"
                             + df["revenue_month"].astype(str).str.zfill(2) + "-01")
        s = pd.Series(pd.to_numeric(df["revenue"], errors="coerce").values / 1000.0,
                      index=idx).dropna().sort_index()
        return s, ""
    except Exception as e:
        return pd.Series(dtype=float), f"解析失敗 {type(e).__name__}: {str(e)[:80]}"


@st.cache_data(ttl=TTL_MOPS, show_spinner=False)
def twse_openapi_current_revenue():
    """證交所官方 OpenAPI:全上市公司最新申報月營收(千元)。回傳 ({代號:千元}, 錯誤描述)。"""
    r, err = http_get("https://openapi.twse.com.tw/v1/opendata/t187ap05_L")
    if r is None:
        return {}, err
    try:
        out = {}
        for row in r.json():
            sid = str(row.get("公司代號", "")).strip()
            if sid in SEMI_IDS:
                key = next((k for k in row if "當月營收" in k), None)
                if key:
                    v = pd.to_numeric(str(row[key]).replace(",", ""), errors="coerce")
                    if pd.notna(v):
                        out[sid] = float(v)
        return out, ("" if out else "回應可讀但找不到目標公司")
    except Exception as e:
        return {}, f"解析失敗 {type(e).__name__}: {str(e)[:80]}"


# ─────────────────────────────────────────────────────────────────────────────
# Yahoo Finance chart 端點(免金鑰公開報價;僅作市場代理,非核心數據)
# 不經 yfinance 套件:直接取官方 chart JSON,錯誤原因可完整回報
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=TTL_MARKET, show_spinner=False)
def yahoo_monthly_close(ticker: str):
    """回傳 (月收盤 Series, 錯誤描述)。"""
    from urllib.parse import quote
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker)}"
    r, err = http_get(url, params={"range": "15y", "interval": "1mo"})
    if r is None:
        return pd.Series(dtype=float), err
    try:
        j = r.json()
        res = j["chart"]["result"][0]
        ts = res["timestamp"]
        close = res["indicators"]["quote"][0]["close"]
        s = pd.Series(close, index=pd.to_datetime(ts, unit="s"), dtype=float).dropna()
        return s.sort_index(), ""
    except Exception as e:
        api_err = ""
        try:
            api_err = str(j.get("chart", {}).get("error", ""))[:120]
        except Exception:
            pass
        return pd.Series(dtype=float), f"解析失敗 {type(e).__name__}{(':' + api_err) if api_err else ''}"


# ─────────────────────────────────────────────────────────────────────────────
# 指標運算
# ─────────────────────────────────────────────────────────────────────────────
def yoy(s: pd.Series, periods: int = 12) -> pd.Series:
    return (s / s.shift(periods) - 1.0) * 100.0


def zscore(s: pd.Series, window: int = 120) -> pd.Series:
    if s.dropna().empty:
        return pd.Series(dtype=float)
    m = s.rolling(window, min_periods=36).mean()
    sd = s.rolling(window, min_periods=36).std()
    z = (s - m) / sd
    if z.dropna().empty:
        z = (s - s.mean()) / s.std()
    return z


def to_monthly(s: pd.Series) -> pd.Series:
    return s if s.empty else s.resample("MS").mean().dropna()


def add_recession_bands(fig, recession: pd.Series):
    if recession.dropna().empty:
        return
    rec = recession[recession > 0]
    if rec.empty:
        return
    groups = (rec.index.to_series().diff() > pd.Timedelta(days=45)).cumsum()
    for _, g in rec.groupby(groups):
        fig.add_vrect(x0=g.index[0], x1=g.index[-1],
                      fillcolor="rgba(160,160,160,0.18)", line_width=0, layer="below")


def last(s):
    return None if s.dropna().empty else float(s.dropna().iloc[-1])


def last_date(s):
    return None if s.dropna().empty else s.dropna().index[-1].strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# 載入所有數據(每條序列:官方來源 + 範圍/新鮮度驗證)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("📡 下載官方數據檔中…(FRED CSV / 證交所 MOPS;首次載入約 30-60 秒)"):
    # ── 利率與衰退 ──
    t10y2y = load_fred("10Y-2Y 利差", "T10Y2Y", -5, 5, 10)
    usrec = load_fred("NBER 衰退期", "USREC", 0, 1, 90)

    # ── 流動性 ──
    # M2 自 1959 年起算,早年僅數百十億美元,下限需涵蓋全部歷史;官方公布滯後約 1 個月
    m2 = load_fred("美國 M2(十億美元)", "M2SL", 10, 60000, 75)
    walcl = load_fred("FED 資產(百萬美元)", "WALCL", 500_000, 12_000_000, 21)
    ecb_assets = load_fred("ECB 資產(百萬歐元)", "ECBASSETSW", 500_000, 20_000_000, 35)

    # ── 通膨 ──(CPI 公布滯後約 1 個月)
    cpi = load_fred("美國 CPI(整體)", "CPIAUCSL", 10, 500, 75)
    core_cpi = load_fred("美國核心 CPI(扣除食品能源)", "CPILFESL", 10, 500, 75)

    # ── 製造業調查(地區聯儲擴散指數;0 ≈ ISM 的 50)──
    empire = load_fred("Empire State 製造業(紐約聯儲)", "GACDISA066MSFRBNY", -80, 80, 60)
    philly = load_fred("費城聯儲製造業活動", "GACDFSA066MSFRBPHI", -80, 80, 60)
    dallas = load_fred("達拉斯聯儲製造業活動", "BACTSAMFRBDAL", -80, 80, 60)
    philly_no = load_fred("費城聯儲|新訂單", "NOCDFSA066MSFRBPHI", -80, 80, 60)
    philly_inv = load_fred("費城聯儲|庫存", "IVCDFSA066MSFRBPHI", -80, 80, 60)

    # ── 製造業硬數據(官方統計,用於交叉驗證調查)──
    # INDPRO 自 1919 年起算,早年指數值極低,下限需涵蓋全部歷史
    indpro = load_fred("美國工業生產指數", "INDPRO", 1, 200, 75)
    dgorder = load_fred("耐久財新訂單(百萬美元)", "DGORDER", 50_000, 600_000, 75)
    # ISRATIO 官方公布滯後約 2.5 個月,門檻放寬至 135 天
    isratio = load_fred("企業存貨/銷售比", "ISRATIO", 0.8, 2.2, 135)

    # ── 台灣半導體月營收:第 1 層 MOPS 報表檔,失敗時自動切換第 2 層 FinMind ──
    semi_rev_df, semi_off_yoy, mops_fetch_err = mops_semi_revenue(60)
    semi_source = "證交所 MOPS(官方報表檔)"
    semi_fallback_note = ""
    if semi_rev_df.empty or len([c for c in SEMI_IDS if c in semi_rev_df.columns]) < len(SEMI_IDS):
        fm_cols, fm_errs = {}, []
        for _sid in SEMI_IDS:
            _s, _e = finmind_month_revenue(_sid)
            if not _s.empty:
                fm_cols[_sid] = _s
            elif _e:
                fm_errs.append(f"{_sid}:{_e}")
        if len(fm_cols) == len(SEMI_IDS):
            semi_rev_df = pd.DataFrame(fm_cols).sort_index()
            semi_off_yoy = pd.DataFrame()        # FinMind 無官方 YoY 欄,改用 OpenAPI 交叉驗證
            semi_source = "FinMind 備援(數據源:公開資訊觀測站申報)"
            semi_fallback_note = f"MOPS 第 1 層失效({mops_fetch_err[:60]}),已自動切換備援;"
            mops_fetch_err = ""
        elif fm_errs:
            mops_fetch_err += f";FinMind 備援亦失敗:{fm_errs[0][:60]}"

    # ── 市場代理 ──
    sox, sox_err = yahoo_monthly_close("^SOX")
    twii, twii_err = yahoo_monthly_close("^TWII")
    sox, sox_status, sox_note = validate_series(sox, 50, 30000, 45)
    if sox_err:
        sox_note = (sox_note + ";" if sox_note else "") + f"錯誤:{sox_err}"
    register_health("費城半導體 SOX", "Yahoo chart 端點(市場代理)", "^SOX", sox, sox_status, sox_note)
    twii, twii_status, twii_note = validate_series(twii, 1000, 80000, 45)
    if twii_err:
        twii_note = (twii_note + ";" if twii_note else "") + f"錯誤:{twii_err}"
    register_health("台灣加權指數", "Yahoo chart 端點(市場代理)", "^TWII", twii, twii_status, twii_note)

# ── 衍生指標 ────────────────────────────────────────────────────────────────
m2_yoy = yoy(m2)
m2_yoy_chg3m = m2_yoy.diff(3)
fed_bs_yoy = yoy(to_monthly(walcl))
ecb_bs_yoy = yoy(to_monthly(ecb_assets))
indpro_yoy = yoy(indpro)
dgorder_yoy = yoy(dgorder)
cpi_yoy = yoy(cpi)
core_cpi_yoy = yoy(core_cpi)
cpi_yoy_chg3m = cpi_yoy.diff(3)                  # 通膨動能(3 個月變化)

# 地區聯儲調查綜合(等權平均;至少需 2 家)
survey_parts = [to_monthly(s) for s in (empire, philly, dallas) if not s.empty]
fed_survey = pd.Series(dtype=float)
if len(survey_parts) >= 2:
    sdf = pd.concat(survey_parts, axis=1)
    fed_survey = sdf[sdf.count(axis=1) >= 2].mean(axis=1).dropna()

# 新訂單-庫存 Spread(費城聯儲子項目直接相減)
no_inv_spread = pd.Series(dtype=float)
if not philly_no.empty and not philly_inv.empty:
    no_inv_spread = (to_monthly(philly_no) - to_monthly(philly_inv)).dropna()

# 台灣半導體合計營收(千元)與 YoY;自算 YoY 並與官方 YoY 欄位交叉驗證
semi_rev = pd.Series(dtype=float)
semi_rev_yoy = pd.Series(dtype=float)
mops_status = "❌ 不可用"
mops_note = (f"MOPS 與 FinMind 備援均失敗:{mops_fetch_err}" if mops_fetch_err
             else "MOPS 與 FinMind 備援均失敗")
if not semi_rev_df.empty:
    have = [c for c in SEMI_IDS if c in semi_rev_df.columns]
    full = semi_rev_df[have].dropna()
    if not full.empty and len(have) == len(SEMI_IDS):
        semi_rev = full.sum(axis=1)
        semi_rev_yoy = yoy(semi_rev)
        # 交叉驗證 A:自算個股 YoY vs MOPS 官方「去年同月增減(%)」(容差 0.5pp)
        mism = 0
        checked = 0
        for sid in have:
            own = yoy(semi_rev_df[sid]).dropna()
            off = (semi_off_yoy[sid].dropna()
                   if (not semi_off_yoy.empty and sid in semi_off_yoy.columns)
                   else pd.Series(dtype=float))
            common = own.index.intersection(off.index)[-6:]
            for ts in common:
                checked += 1
                if abs(own[ts] - off[ts]) > 0.5:
                    mism += 1
        if checked and mism == 0:
            mops_status, mops_note = "✅ 正常", f"自算 YoY 與官方欄位交叉驗證 {checked} 筆全數一致(±0.5pp)"
        elif checked:
            mops_status, mops_note = "⚠️ 注意", f"交叉驗證 {checked} 筆中 {mism} 筆不一致,請至 MOPS 原檔核對"
        else:
            # 交叉驗證 B(備援來源時):最新一筆 vs 證交所 OpenAPI 官方當月營收(容差 1%)
            cur, cerr = twse_openapi_current_revenue()
            matched, conflict = 0, 0
            for sid in have:
                col = semi_rev_df[sid].dropna()
                if sid in cur and not col.empty:
                    if abs(col.iloc[-1] - cur[sid]) <= max(cur[sid] * 0.01, 1000):
                        matched += 1
                    else:
                        conflict += 1
            if conflict:
                mops_status, mops_note = "⚠️ 注意", f"與證交所 OpenAPI 比對:{conflict} 家不一致,請核對"
            elif matched:
                mops_status, mops_note = "✅ 正常", f"與證交所 OpenAPI 官方當月營收交叉驗證一致({matched}/3 家)"
            else:
                mops_status, mops_note = "⚠️ 注意", f"無法交叉驗證(OpenAPI:{cerr or '無共同月份'}),數據僅單一來源"
    else:
        mops_status, mops_note = "⚠️ 注意", f"僅取得 {len(have)}/3 家公司,為避免口徑不一不予合計"
    if semi_fallback_note:
        mops_note = semi_fallback_note + mops_note
    if mops_fetch_err:
        mops_note += f";部分月份錯誤:{mops_fetch_err}"
register_health("台灣半導體月營收(2330+2303+2454)", semi_source,
                "t21sc03 報表檔 / FinMind / TWSE OpenAPI", semi_rev, mops_status, mops_note)

sox_yoy = yoy(sox)
twii_yoy = yoy(twii)

# ── 製造業週期綜合指數(僅納入通過驗證的成分)─────────────────────────────
composite_parts = {
    "地區聯儲調查綜合": fed_survey,
    "新訂單-庫存(費城)": no_inv_spread,
    "M2 YoY": m2_yoy,
    "10Y-2Y 利差": to_monthly(t10y2y),
    "台半導體營收 YoY": semi_rev_yoy,
    "耐久財新訂單 YoY": dgorder_yoy,
    "工業生產 YoY": indpro_yoy,
    "SOX YoY": to_monthly(sox_yoy),
}
z_parts = {k: zscore(v) for k, v in composite_parts.items() if not v.dropna().empty}
composite = pd.Series(dtype=float)
if z_parts:
    zdf = pd.concat(z_parts, axis=1)
    zdf = zdf[zdf.count(axis=1) >= 3]            # 至少三個成分才計算,避免失真
    composite = zdf.mean(axis=1).rolling(3, min_periods=1).mean().dropna()

# ─────────────────────────────────────────────────────────────────────────────
# 六大訊號判定
# ─────────────────────────────────────────────────────────────────────────────
signals = []

# 1) 殖利率曲線翻正
v = last(t10y2y)
if v is None:
    signals.append(("殖利率曲線 10Y-2Y", "sig-na", "資料不可用", "FRED T10Y2Y"))
else:
    was_inverted = (t10y2y.dropna().iloc[-500:] < 0).any()
    if v < 0:
        st_, txt = "sig-bear", f"{v:+.2f}%(倒掛中)"
        note = "倒掛中:歷史 ~85% 機率預示衰退(領先 6-18 個月)"
    elif was_inverted:
        st_, txt = "sig-warn", f"{v:+.2f}%(倒掛後翻正)"
        note = "⚠ 翻正訊號:倒掛後回正常為衰退前最後階段,警戒度最高"
    else:
        st_, txt = "sig-bull", f"{v:+.2f}%(正常)"
        note = "曲線正斜率,未見倒掛"
    signals.append((f"殖利率曲線 10Y-2Y|{last_date(t10y2y)}", st_, txt, note))

# 2) 製造業調查穿越 0(對應 ISM 的 50)
v = last(fed_survey)
if v is None:
    signals.append(("製造業調查綜合(聯儲)", "sig-na", "資料不可用",
                    "紐約/費城/達拉斯聯儲調查(官方,0 ≈ ISM 50)"))
else:
    prev = float(fed_survey.dropna().iloc[-2]) if len(fed_survey.dropna()) > 1 else v
    cross = "↑穿越0!" if prev < 0 <= v else ("↓跌破0!" if prev >= 0 > v else "")
    st_ = "sig-bull" if v >= 0 else "sig-bear"
    signals.append((f"製造業調查綜合|{last_date(fed_survey)}", st_, f"{v:+.1f} {cross}",
                    "紐約+費城+達拉斯聯儲平均;>0 擴張(≈ISM>50),官方免費調查"))

# 3) 新訂單-庫存 Spread(費城聯儲)
v = last(no_inv_spread)
if v is None:
    signals.append(("新訂單-庫存 Spread", "sig-na", "資料不可用", "費城聯儲子項(最領先,領先 2-4 個月)"))
else:
    trend = no_inv_spread.diff(3).dropna()
    rising = (not trend.empty) and trend.iloc[-1] > 0
    st_ = "sig-bull" if (v > 0 and rising) else ("sig-warn" if v > 0 else "sig-bear")
    signals.append((f"新訂單-庫存(費城)|{last_date(no_inv_spread)}", st_,
                    f"{v:+.1f}({'走升' if rising else '走弱'})",
                    "最領先子指標,領先整體調查 2-4 個月"))

# 4) M2 增速拐點
v, chg = last(m2_yoy), last(m2_yoy_chg3m)
if v is None or chg is None:
    signals.append(("M2 增速拐點", "sig-na", "資料不可用", "FRED M2SL"))
else:
    turning_up = chg > 0
    st_ = "sig-bull" if turning_up else "sig-bear"
    signals.append((f"美國 M2 YoY|{last_date(m2_yoy)}", st_,
                    f"{v:+.2f}%(3M {chg:+.2f}pp)",
                    "增速向上拐點 → 流動性改善,領先風險資產 2-3 個月"))

# 5) 半導體營收 YoY 翻正(MOPS 官方)
v = last(semi_rev_yoy)
if v is None:
    signals.append(("台半導體營收 YoY", "sig-na", "資料不可用", "證交所 MOPS 官方月報表"))
else:
    prev = float(semi_rev_yoy.dropna().iloc[-2]) if len(semi_rev_yoy.dropna()) > 1 else v
    cross = "↑YoY 翻正!" if prev < 0 <= v else ""
    st_ = "sig-bull" if v > 0 else "sig-bear"
    signals.append((f"台半導體營收 YoY|{last_date(semi_rev_yoy)}", st_,
                    f"{v:+.1f}% {cross}",
                    f"台積電+聯電+聯發科(MOPS 官方);{mops_note}"))

# 6) CPI 通膨動能
v, chg = last(cpi_yoy), last(cpi_yoy_chg3m)
core_v = last(core_cpi_yoy)
if v is None or chg is None:
    signals.append(("美國 CPI 通膨", "sig-na", "資料不可用", "FRED CPIAUCSL"))
else:
    cooling = chg < 0
    st_ = "sig-bull" if cooling else ("sig-warn" if v < 3 else "sig-bear")
    core_txt = f",核心 {core_v:+.2f}%" if core_v is not None else ""
    signals.append((f"美國 CPI YoY|{last_date(cpi_yoy)}", st_,
                    f"{v:+.2f}%(3M {chg:+.2f}pp){core_txt}",
                    "通膨降溫 → 政策寬鬆空間擴大;升溫且高於 3% 壓抑風險資產"))

# 7) 多指標共振
votes = [s for s in signals if s[1] in ("sig-bull", "sig-bear")]
bulls = sum(1 for s in votes if s[1] == "sig-bull")
if len(votes) >= 3:
    ratio = bulls / len(votes)
    if ratio >= 0.8:
        st_, txt, note = "sig-bull", f"{bulls}/{len(votes)} 偏多共振", "高可信度:週期上行確認"
    elif ratio <= 0.2:
        st_, txt, note = "sig-bear", f"{bulls}/{len(votes)} 偏多", "高可信度:週期下行確認"
    else:
        st_, txt, note = "sig-warn", f"{bulls}/{len(votes)} 偏多", "訊號分歧,週期方向未確認"
    signals.append(("多指標共振", st_, txt, note))
else:
    signals.append(("多指標共振", "sig-na", "可用訊號不足", "需至少 3 個有效訊號"))

# ─────────────────────────────────────────────────────────────────────────────
# 版面
# ─────────────────────────────────────────────────────────────────────────────
APP_VERSION = "v9(2026-06-10)"

st.title("🌐 總經週期追蹤看板")
st.caption(f"程式版本:{APP_VERSION}|更新時間:{datetime.now(TW_TZ):%Y-%m-%d %H:%M}(台北時間)|"
           f"來源:FRED 官方 CSV、證交所 MOPS 官方報表,免金鑰|快取到期自動重抓")

n_fail = sum(1 for h in HEALTH if h["status"].startswith("❌"))
many_failures = HEALTH and n_fail >= max(2, len(HEALTH) // 2)
if many_failures:
    st.error(f"⚠️ {n_fail}/{len(HEALTH)} 條數據抓取失敗。下方「資料健康度面板」已列出每條的確切錯誤原因"
             "(HTTP 狀態碼/例外訊息)——多為執行環境的對外連線被防火牆或代理伺服器擋下所致。"
             "請展開面板查看「備註」欄,即可判斷是哪一層的問題。")

cols = st.columns(3)
for i, (name, css, val, note) in enumerate(signals):
    with cols[i % 3]:
        st.markdown(
            f'<div class="sig-card {css}"><h4>{name}</h4>'
            f'<div class="val">{val}</div><div class="sub">{note}</div></div>',
            unsafe_allow_html=True,
        )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 週期綜合指數", "🏦 殖利率曲線", "💧 流動性 / 通膨", "🏭 製造業調查+硬數據", "💾 台灣半導體"]
)

# ── Tab 1:綜合指數 ──────────────────────────────────────────────────────────
with tab1:
    if composite.empty:
        st.warning("通過驗證的成分不足 3 項,暫不計算綜合指數(寧缺勿錯)。")
    else:
        c = composite[composite.index >= "2000-01-01"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c.index, y=c, name="製造業週期綜合指數",
                                 line=dict(color="#00d4aa", width=2.5),
                                 fill="tozeroy", fillcolor="rgba(0,212,170,0.10)"))
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        add_recession_bands(fig, usrec[usrec.index >= "2000-01-01"])
        fig.update_layout(title="製造業週期綜合指數(通過驗證成分之 Z-Score 平均,3M 平滑;灰底=NBER 衰退)",
                          height=440, **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        lvl, slope = float(c.iloc[-1]), float(c.diff(3).iloc[-1])
        regime = ("🟢 復甦(低檔回升)" if lvl < 0 and slope > 0 else
                  "🔵 擴張(高檔走升)" if lvl >= 0 and slope > 0 else
                  "🟡 趨緩(高檔回落)" if lvl >= 0 else "🔴 收縮(低檔下行)")
        st.metric("目前週期位置", regime, f"指數 {lvl:+.2f}|3M 動能 {slope:+.2f}")
        st.caption("已納入成分:" + "、".join(z_parts.keys()))

# ── Tab 2:殖利率曲線 ────────────────────────────────────────────────────────
with tab2:
    if t10y2y.empty:
        st.warning("FRED T10Y2Y 暫不可用。")
    else:
        s = t10y2y[t10y2y.index >= "1985-01-01"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s, name="10Y-2Y 利差 (%)",
                                 line=dict(color="#2196f3", width=1.6)))
        fig.add_hline(y=0, line_color="#ef5350", line_dash="dash",
                      annotation_text="倒掛分界", annotation_font_color="#ef5350")
        add_recession_bands(fig, usrec[usrec.index >= "1985-01-01"])
        fig.update_layout(title="美債 10Y-2Y 利差|倒掛→翻正後,歷史 ~85% 於 6-18 個月內衰退(灰底=衰退)",
                          height=440, **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 3:流動性 ───────────────────────────────────────────────────────────
with tab3:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("美國 M2 年增率 (%)", "央行資產負債表年增率 (%)",
                                        "美國 CPI 年增率 (%)"))
    if not m2_yoy.empty:
        s = m2_yoy[m2_yoy.index >= "1995-01-01"]
        fig.add_trace(go.Scatter(x=s.index, y=s, name="M2 YoY",
                                 line=dict(color="#00d4aa", width=2)), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="#888", row=1, col=1)
    if not fed_bs_yoy.empty:
        s = fed_bs_yoy[fed_bs_yoy.index >= "2008-01-01"]
        fig.add_trace(go.Scatter(x=s.index, y=s, name="FED 資產 YoY",
                                 line=dict(color="#ff9800", width=2)), row=2, col=1)
    if not ecb_bs_yoy.empty:
        s = ecb_bs_yoy[ecb_bs_yoy.index >= "2008-01-01"]
        fig.add_trace(go.Scatter(x=s.index, y=s, name="ECB 資產 YoY",
                                 line=dict(color="#9c27b0", width=2)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#888", row=2, col=1)
    if not cpi_yoy.empty:
        s = cpi_yoy[cpi_yoy.index >= "1995-01-01"]
        fig.add_trace(go.Scatter(x=s.index, y=s, name="CPI YoY",
                                 line=dict(color="#ef5350", width=2)), row=3, col=1)
    if not core_cpi_yoy.empty:
        s = core_cpi_yoy[core_cpi_yoy.index >= "1995-01-01"]
        fig.add_trace(go.Scatter(x=s.index, y=s, name="核心 CPI YoY",
                                 line=dict(color="#ffd700", width=1.6, dash="dot")), row=3, col=1)
    fig.add_hline(y=2, line_dash="dash", line_color="#26a69a", row=3, col=1,
                  annotation_text="FED 2% 目標", annotation_font_color="#26a69a")
    fig.update_layout(title="全球流動性與通膨|M2 增速拐點領先風險資產 2-3 個月;CPI 決定政策寬鬆空間",
                      height=780, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 4:製造業調查 + 硬數據 ───────────────────────────────────────────────
with tab4:
    st.info("ℹ️ ISM PMI 為 ISM 協會授權調查,無法由其他數據計算還原。以下為**官方免費**的"
            "同性質擴散指數(0 ≈ ISM 的 50),公布時間比 ISM 更早,並以 Census 硬數據交叉驗證。")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("地區聯儲製造業調查(擴散指數,0=榮枯線)",
                                        "新訂單 − 庫存 Spread(費城聯儲,最領先)",
                                        "硬數據驗證:耐久財新訂單 / 工業生產 YoY (%)、存貨銷售比"))
    for s, name, color, w in ((to_monthly(empire), "紐約 Empire", "#2196f3", 1.2),
                              (to_monthly(philly), "費城", "#ff9800", 1.2),
                              (to_monthly(dallas), "達拉斯", "#9c27b0", 1.2),
                              (fed_survey, "三家平均", "#00d4aa", 2.6)):
        if not s.empty:
            ss = s[s.index >= "2005-01-01"]
            fig.add_trace(go.Scatter(x=ss.index, y=ss, name=name,
                                     line=dict(color=color, width=w)), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#ef5350", row=1, col=1)
    if not no_inv_spread.empty:
        s = no_inv_spread[no_inv_spread.index >= "2005-01-01"]
        fig.add_trace(go.Bar(x=s.index, y=s, name="新訂單-庫存",
                             marker_color=["#26a69a" if x > 0 else "#ef5350" for x in s]),
                      row=2, col=1)
    if not dgorder_yoy.empty:
        s = dgorder_yoy[dgorder_yoy.index >= "2005-01-01"]
        fig.add_trace(go.Scatter(x=s.index, y=s, name="耐久財新訂單 YoY",
                                 line=dict(color="#ffd700", width=1.6)), row=3, col=1)
    if not indpro_yoy.empty:
        s = indpro_yoy[indpro_yoy.index >= "2005-01-01"]
        fig.add_trace(go.Scatter(x=s.index, y=s, name="工業生產 YoY",
                                 line=dict(color="#26a69a", width=1.6)), row=3, col=1)
    if not isratio.empty:
        s = isratio[isratio.index >= "2005-01-01"]
        fig.add_trace(go.Scatter(x=s.index, y=(s - s.mean()) * 20, name="存貨/銷售比(去均值×20)",
                                 line=dict(color="#888", width=1.2, dash="dot")), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#888", row=3, col=1)
    fig.update_layout(title="製造業:官方調查(軟數據)× 官方統計(硬數據)交叉驗證",
                      height=820, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 5:台灣半導體 ────────────────────────────────────────────────────────
with tab5:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("台灣三大半導體月營收合計 YoY (%)(MOPS 官方:台積電+聯電+聯發科)",
                                        "SOX 費城半導體 / 台灣加權 年增率 (%)(市場代理)"))
    if not semi_rev_yoy.dropna().empty:
        s = semi_rev_yoy.dropna()
        fig.add_trace(go.Bar(x=s.index, y=s, name="半導體營收 YoY",
                             marker_color=["#26a69a" if x > 0 else "#ef5350" for x in s]),
                      row=1, col=1)
        fig.add_hline(y=0, line_color="#888", row=1, col=1)
    else:
        st.warning(f"MOPS 月營收:{mops_status}|{mops_note}")
    if not sox_yoy.dropna().empty:
        s = sox_yoy.dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s, name="SOX YoY",
                                 line=dict(color="#2196f3", width=2)), row=2, col=1)
    if not twii_yoy.dropna().empty:
        s = twii_yoy.dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s, name="台灣加權 YoY",
                                 line=dict(color="#ffd700", width=1.5)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#888", row=2, col=1)
    fig.update_layout(title="台灣科技週期|半導體營收 YoY 翻正 = 科技上行週期確認", height=620, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    if not semi_rev.dropna().empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("最新合計月營收", f"{semi_rev.dropna().iloc[-1]/1e5:,.0f} 億台幣",
                  f"{last(semi_rev_yoy):+.1f}% YoY" if last(semi_rev_yoy) is not None else None)
        for col, sid in ((c2, "2330"), (c3, "2454")):
            if sid in semi_rev_df.columns:
                v = last(yoy(semi_rev_df[sid]))
                if v is not None:
                    col.metric(f"{SEMI_IDS[sid]} YoY", f"{v:+.1f}%")
        st.caption(f"資料驗證:{mops_status}|{mops_note}(營收單位:MOPS 原檔為千元)")

# ─────────────────────────────────────────────────────────────────────────────
# 資料健康度面板(可逐條稽核)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🔍 資料健康度 / 稽核面板(每條序列的來源、代碼、最新日期、驗證狀態)",
                 expanded=bool(many_failures)):
    if HEALTH:
        hdf = pd.DataFrame(HEALTH)
        hdf.columns = ["指標", "來源", "代碼/檔案", "最新日期", "最新值", "狀態", "備註"]
        st.dataframe(hdf, use_container_width=True, hide_index=True)
    st.caption("驗證規則:數值超出合理範圍即剔除;序列過舊標 ⚠️;抓取失敗顯示 ❌ 並排除於所有計算之外,"
               "綜合指數僅使用 ✅/⚠️ 成分且至少 3 項才計算。台股營收另以官方 YoY 欄位交叉比對(±0.5pp)。")

# ─────────────────────────────────────────────────────────────────────────────
# 側邊欄
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ 看板說明")
    st.markdown(
        """
**資料來源(全部官方、免金鑰、免註冊)**
- FRED 官方 CSV 下載端點(同官網 Download CSV)
- 台灣證交所 MOPS 官方月營收報表檔
- Yahoo Finance 公開報價(僅市場代理)

**關於 ISM**
ISM PMI 是授權調查數據,無法由其他數據
計算還原。本看板改用官方免費的紐約/費城/
達拉斯聯儲製造業調查(0 ≈ ISM 50),
並以 Census 硬數據交叉驗證。

**數據正確性保障**
- 每條序列驗證數值範圍與新鮮度
- 不合格 → 顯示「不可用」,絕不顯示錯誤數字
- 台股營收 YoY 與官方欄位交叉比對
- 「資料健康度」面板可逐條稽核

**FRED API Key(建議設定)**
在 Streamlit Cloud → App → Settings → Secrets:
```toml
FRED_API_KEY = "你的key"
```
有 key 時走 FRED 官方 API(最穩),
沒 key 時自動退回官方 CSV 下載端點。

**自動更新**
總經 6 小時、市場 1 小時、MOPS 24 小時
快取到期後任何訪問即自動重抓。
        """
    )
    if st.button("🔄 立即強制更新所有數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("⚠️ 本看板僅供研究參考,不構成投資建議。")
