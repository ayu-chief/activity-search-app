# app.py
import time
import re
import unicodedata
from typing import List, Dict

import streamlit as st
import pandas as pd
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials

from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import numpy as np

# -----------------------------------------------------------------------------
# 基本設定
# -----------------------------------------------------------------------------
st.set_page_config(page_title="活動コンテンツ検索", layout="wide")

# 読み込み結果の控えめ表示用ログ（Expanderにまとめる）
if "OPENED_LOG" not in st.session_state:
    st.session_state.OPENED_LOG = []  # [(title, sid), ...]

# -----------------------------------------------------------------------------
# Google Sheets 接続
# -----------------------------------------------------------------------------
SERVICE_ACCOUNT_INFO = st.secrets["google_service_account"]  # Secretsに保存したJSON
SPREADSHEET_IDS = [
    "1GCenO3IlDFrSITj1r90G_Vz_11D66POc8ny9HMtdCcM",
    "1Rjkgc6whTpg4FKUNLVSdzFSya-_tg42Wg4e10p-MmmI",
    "1PFDBuFuqxC4OWMCPjErP8uYYRovE55t-0oWsXNMCMqc",
    "1p4utUR9of_uSQNpzwJpSXgKiPrNur5nSTgHZvrbwmuc",
    "1HULvSdUAdSNdXXhPshu4mfwraf-bNq6zakFRhKF4Yfg",
]
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
creds = Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
gc = gspread.authorize(creds)

# レート制限対策（広め推奨）
BASE_WAIT = 8.0   # スプレッドシート間の待機（秒）
MAX_RETRY = 6     # 429時の最大リトライ回数

# -----------------------------------------------------------------------------
# 正規化ヘルパ
# -----------------------------------------------------------------------------
def normalize(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -----------------------------------------------------------------------------
# シート1枚 → レコード化（ラベル取り出し）
# -----------------------------------------------------------------------------
LABELS = [
    "校舎名", "コンテンツ名", "テーマ", "対象生徒", "対象", "参加人数",
    "準備物", "実施方法", "子供たちの反応", "子どもたちの反応", "良かった点", "改善点",
]

def extract_value(values: List[List[str]], label: str) -> str:
    lab = normalize(label)
    for row in values:
        for j, cell in enumerate(row):
            if lab and lab in normalize(cell):
                right = row[j+1:] if j+1 < len(row) else []
                toks = [c for c in right if str(c).strip()]
                if toks:
                    return " ".join(normalize(c) for c in toks).strip()
    return ""

def parse_sheet(values: List[List[str]]) -> Dict[str, str]:
    rec = {}
    for lab in LABELS:
        rec[lab] = extract_value(values, lab)

    # 子ども/子供 を統一
    if not rec.get("子供たちの反応"):
        rec["子供たちの反応"] = rec.get("子どもたちの反応", "")

    out = {
        "校舎名": rec.get("校舎名", ""),
        "コンテンツ名": rec.get("コンテンツ名", ""),
        "テーマ": rec.get("テーマ", ""),
        "対象": rec.get("対象生徒", "") or rec.get("対象", ""),
        "参加人数": rec.get("参加人数", ""),
        "準備物": rec.get("準備物", ""),
        "実施方法": rec.get("実施方法", ""),
        "子供たちの反応": rec.get("子供たちの反応", ""),
        "良かった点": rec.get("良かった点", ""),
        "改善点": rec.get("改善点", ""),
    }

    # 主要フィールドが空なら保険として全セル連結
    if not any(out.values()):
        flat = " ".join([" ".join([str(c) for c in r if str(c).strip()]) for r in values])
        out["コンテンツ名"] = out["コンテンツ名"] or "(名称未設定)"
        out["テーマ"] = flat[:200]

    return out

# -----------------------------------------------------------------------------
# 429回避：worksheets() を使わず、metadata(title)→values.batchGet
# -----------------------------------------------------------------------------
def _short_id(sid: str) -> str:
    return f"{sid[:6]}…{sid[-4:]}" if len(sid) > 12 else sid

def open_sheet_by_id(sid: str):
    """開けたらログに追加（画面にはその場で出さない）"""
    for attempt in range(MAX_RETRY):
        try:
            sh = gc.open_by_key(sid)
            st.session_state.OPENED_LOG.append((sh.title, sid))
            return sh
        except APIError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 429 and attempt < MAX_RETRY - 1:
                wait = BASE_WAIT * (2 ** attempt)
                st.warning(f"⏳ Rate limit: open_by_key (retry {attempt+1}/{MAX_RETRY}) in {wait:.1f}s")
                time.sleep(wait)
                continue
            st.session_state.OPENED_LOG.append((f"❌ FAILED: {sid}", sid))
            return None
        except Exception:
            st.session_state.OPENED_LOG.append((f"❌ FAILED: {sid}", sid))
            return None

@st.cache_data(show_spinner=True, ttl=6*60*60)
def load_all_data_v2() -> pd.DataFrame:
    """worksheets()は使わず、タイトルだけ取得→values.batchGetで一括取得"""
    rows = []
    for sid in SPREADSHEET_IDS:
        sh = open_sheet_by_id(sid)
        if not sh:
            continue

        # 1) タイトルのみ軽量取得
        meta = None
        for attempt in range(MAX_RETRY):
            try:
                meta = sh.fetch_sheet_metadata(params={"fields": "sheets(properties(title))"})
                break
            except APIError as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code == 429 and attempt < MAX_RETRY - 1:
                    wait = BASE_WAIT * (2 ** attempt)
                    st.warning(f"⏳ Rate limit: fetch_sheet_metadata (retry {attempt+1}/{MAX_RETRY}) in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                meta = None
                break
            except Exception:
                meta = None
                break
        if not meta:
            continue

        titles = [s["properties"]["title"] for s in meta.get("sheets", []) if "properties" in s]
        # 列幅は必要に応じて狭める（A:N など）。狭いほど速い
        ranges = [f"'{t}'!A:Q" for t in titles]

        # 2) 一括取得（values.batchGet）
        time.sleep(BASE_WAIT)
        vals_resp = None
        for attempt in range(MAX_RETRY):
            try:
                vals_resp = sh.values_batch_get(ranges=ranges, params={"majorDimension": "ROWS"})
                break
            except APIError as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code == 429 and attempt < MAX_RETRY - 1:
                    wait = BASE_WAIT * (2 ** attempt)
                    st.warning(f"⏳ Rate limit: values.batchGet (retry {attempt+1}/{MAX_RETRY}) in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                vals_resp = None
                break
            except Exception:
                vals_resp = None
                break
        if not vals_resp:
            continue

        for vr in vals_resp.get("valueRanges", []):
            rng = vr.get("range", "")
            ws_title = rng.split("!")[0].strip("'")
            vals = vr.get("values", [])
            if not vals:
                continue

            rec = parse_sheet(vals)
            if not any(rec.values()):
                continue

            rec["スプレッドシート"] = sh.title
            rec["ファイルID"] = sid
            rec["シート名"] = ws_title
            rec["検索用テキスト"] = " ".join([
                rec.get("コンテンツ名",""), rec.get("テーマ",""), rec.get("対象",""),
                rec.get("準備物",""), rec.get("実施方法",""),
                rec.get("子供たちの反応",""), rec.get("良かった点",""), rec.get("改善点","")
            ]).strip()
            rows.append(rec)

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 検索準備（埋め込み＋BM25）
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_embedder():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_data(show_spinner=False)
def build_bm25(corpus_tokens: List[List[str]]):
    return BM25Okapi(corpus_tokens)

def tokenize_ja(text: str) -> List[str]:
    text = normalize(text)
    toks = re.split(r"[ \u3000、。・,./!?！？\-\n\r\t]+", text)
    return [t for t in toks if t]

SYNONYMS = {
    "発表": ["プレゼン", "スピーチ", "表現", "発表練習"],
    "表現": ["発表", "伝える", "アウトプット"],
    "協力": ["協働", "チーム", "グループ", "共同"],
    "創作": ["ものづくり", "制作", "工作", "クリエイティブ"],
    "読解": ["読み取り", "感想", "読書", "朗読"],
}

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("活動コンテンツ検索")

with st.sidebar:
    st.header("検索設定")
    alpha = st.slider("意味重視（1.0） ←→ 語一致重視（0.0）", 0.0, 1.0, 0.7, 0.05)
    top_k = st.slider("件数", 5, 50, 15)
    st.caption("※初回は読み込みに時間がかかります（キャッシュ後は速くなります）")

# データ読み込み
with st.spinner("シートを読み込んでいます…"):
    df = load_all_data_v2()

# --- データソース表示：ログが無くても df から復元して必ず出す ---
sources = st.session_state.get("OPENED_LOG", [])
if (not sources) and (len(df) > 0):
    tmp = (
        df[["スプレッドシート", "ファイルID"]]
        .dropna()
        .drop_duplicates()
        .values
        .tolist()
    )
    sources = [(title, sid) for title, sid in tmp]

with st.expander(f"データソース（{len(sources)}件）", expanded=False):
    st.caption(f"📄 読み込めたレコード数: {len(df)}")
    for title, sid in sources:
        if isinstance(title, str) and title.startswith("❌"):
            st.caption(title)
        else:
            st.caption(f"✅ {title} ({_short_id(sid)})")

# 検索コーパス
corpus_texts = (df["検索用テキスト"].fillna("") + " " + df["コンテンツ名"].fillna("")).tolist()
corpus_tokens = [tokenize_ja(t) for t in corpus_texts]
bm25 = build_bm25(corpus_tokens)
embedder = load_embedder()
corpus_emb = embedder.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=False)

# 検索フォーム
st.divider()
st.caption("🔎 キーワードを入力して検索")
q = st.text_input(
    label="キーワードを入力",
    value="",
    placeholder="例: 発表練習, グループ活動, 朗読, 工作, 表現力 など",
    label_visibility="collapsed",
)

if q:
    expanded = [q]
    for k, vs in SYNONYMS.items():
        if k in q:
            expanded += vs
    q_expanded = " ".join(expanded)

    q_toks = tokenize_ja(q_expanded)
    bm25_scores = bm25.get_scores(q_toks)

    q_emb = embedder.encode([q_expanded], normalize_embeddings=True, show_progress_bar=False)
    sem_scores = util.cos_sim(q_emb, corpus_emb).cpu().numpy()[0]

    def minmax(x):
        x = np.array(x, dtype=float)
        if x.max() - x.min() < 1e-9:
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    bm25_n = minmax(bm25_scores)
    sem_n  = minmax(sem_scores)
    final  = alpha * sem_n + (1 - alpha) * bm25_n
    idx = np.argsort(final)[::-1][:top_k]

    st.subheader("検索結果")
    for rank, i in enumerate(idx, start=1):
        row = df.iloc[i]
        url = f"https://docs.google.com/spreadsheets/d/{row['ファイルID']}/edit"
        with st.container(border=True):
            st.markdown(
                f"**{rank}. {row.get('コンテンツ名','(名称未設定)')}** 　"
                f"[{row['スプレッドシート']} / {row['シート名']}]({url})"
            )
            cols = st.columns(3)
            with cols[0]:
                st.write("**テーマ**:", row.get("テーマ",""))
                st.write("**対象**:", row.get("対象",""))
                st.write("**参加人数**:", row.get("参加人数",""))
            with cols[1]:
                st.write("**準備物**:", row.get("準備物",""))
                st.write("**実施方法**:", row.get("実施方法",""))
            with cols[2]:
                st.write("**子供たちの反応**:", row.get("子供たちの反応",""))
                st.write("**良かった点**:", row.get("良かった点",""))
                st.write("**改善点**:", row.get("改善点",""))
            st.caption(f"score={final[i]:.3f} / semantic={sem_n[i]:.3f} / bm25={bm25_n[i]:.3f}")
