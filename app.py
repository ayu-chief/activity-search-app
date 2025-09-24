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

# 検索系
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

# -----------------------------------------------------------------------------
# 基本設定
# -----------------------------------------------------------------------------
st.set_page_config(page_title="🎯 活動コンテンツ検索", layout="wide")

# -----------------------------------------------------------------------------
# Google Sheets 接続
# -----------------------------------------------------------------------------
SERVICE_ACCOUNT_INFO = st.secrets["google_service_account"]  # Secretsに入れたJSON
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
st.info(f"Service Account: {creds.service_account_email}")

# レート制限対策用
BASE_WAIT = 1.1   # シート間の最小待機（秒）
MAX_RETRY = 5     # 429時の最大リトライ回数


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
# 429を避ける：スプレッドシート単位で batchGet するローダ
# -----------------------------------------------------------------------------
def open_sheet_by_id(sid: str):
    for attempt in range(MAX_RETRY):
        try:
            sh = gc.open_by_key(sid)
            st.success(f"✅ Opened: {sh.title} ({sid})")
            return sh
        except APIError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 429 and attempt < MAX_RETRY - 1:
                wait = BASE_WAIT * (2 ** attempt)
                st.warning(f"⏳ Rate limit: open_by_key (retry {attempt+1}/{MAX_RETRY}) in {wait:.1f}s")
                time.sleep(wait)
                continue
            st.error(f"❌ Failed to open (status={code}): {sid}")
            st.code(getattr(getattr(e, "response", None), "text", str(e))[:2000])
            return None
        except Exception as e:
            st.error(f"❌ Failed to open (unexpected): {sid}")
            st.code(str(e))
            return None

@st.cache_data(show_spinner=True, ttl=6*60*60)  # 6時間キャッシュ
def load_all_data() -> pd.DataFrame:
    """
    各スプレッドシートについて values.batchGet で全ワークシートを一括取得。
    ＝ リクエスト回数を「シート数」→「スプレッドシート数」に圧縮。
    """
    rows = []
    for sid in SPREADSHEET_IDS:
        sh = open_sheet_by_id(sid)
        if not sh:
            continue

        # 使う列幅を必要に応じて狭める（A:Q など）
        ranges = [f"'{ws.title}'!A:Z" for ws in sh.worksheets()]

        time.sleep(BASE_WAIT)  # シート間の間隔

        vals_resp = None
        for attempt in range(MAX_RETRY):
            try:
                vals_resp = sh.values_batch_get(
                    ranges=ranges,
                    params={"majorDimension": "ROWS"}
                )
                break
            except APIError as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code == 429 and attempt < MAX_RETRY - 1:
                    wait = BASE_WAIT * (2 ** attempt)
                    st.warning(f"⏳ Rate limit: values.batchGet (retry {attempt+1}/{MAX_RETRY}) in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                st.error(f"❌ Failed batchGet (status={code}): {sh.title}")
                st.code(getattr(getattr(e, "response", None), "text", str(e))[:2000])
                vals_resp = None
                break
            except Exception as e:
                st.error(f"❌ Failed batchGet (unexpected): {sh.title}")
                st.code(str(e))
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

    df = pd.DataFrame(rows)
    return df


# -----------------------------------------------------------------------------
# 検索準備（埋め込み＋BM25）
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_embedder():
    # 小型・多言語
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_data(show_spinner=False)
def build_bm25(corpus_tokens: List[List[str]]):
    return BM25Okapi(corpus_tokens)

def tokenize_ja(text: str) -> List[str]:
    # シンプルな空白・句読点分割（形態素解析なしでもそこそこ動く）
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
st.title("🎯 活動コンテンツ検索")

with st.sidebar:
    st.header("検索設定")
    alpha = st.slider("意味重視（1.0） ←→ 語一致重視（0.0）", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    top_k = st.slider("件数", 5, 50, 15)
    st.caption("※初回は読み込みに時間がかかります（キャッシュ後は速くなります）")

# データ読み込み
with st.spinner("シートを読み込んでいます…"):
    df = load_all_data()

st.write(f"📄 読み込めたレコード数: {len(df)}")

if len(df) == 0:
    st.stop()

# 検索コーパス準備
corpus_texts = (df["検索用テキスト"].fillna("") + " " + df["コンテンツ名"].fillna("")).tolist()
corpus_tokens = [tokenize_ja(t) for t in corpus_texts]
bm25 = build_bm25(corpus_tokens)
embedder = load_embedder()
corpus_emb = embedder.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=False)

# 検索フォーム
q = st.text_input("キーワードを入力（例：発表練習 / グループ活動 / 表現力 など）", "")
if q:
    # 同義語展開
    expanded = [q]
    for k, vs in SYNONYMS.items():
        if k in q:
            expanded += vs
    q_expanded = " ".join(expanded)

    # BM25
    q_toks = tokenize_ja(q_expanded)
    bm25_scores = bm25.get_scores(q_toks)

    # 埋め込み類似
    q_emb = embedder.encode([q_expanded], normalize_embeddings=True, show_progress_bar=False)
    sem_scores = util.cos_sim(q_emb, corpus_emb).cpu().numpy()[0]

    # 正規化（0-1）
    import numpy as np
    def minmax(x):
        x = np.array(x, dtype=float)
        if x.max() - x.min() < 1e-9:
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    bm25_n = minmax(bm25_scores)
    sem_n = minmax(sem_scores)

    # 融合
    final = alpha * sem_n + (1 - alpha) * bm25_n
    idx = np.argsort(final)[::-1][:top_k]

    st.subheader("検索結果")
    for rank, i in enumerate(idx, start=1):
        row = df.iloc[i]
        url = f"https://docs.google.com/spreadsheets/d/{row['ファイルID']}/edit"
        with st.container(border=True):
            st.markdown(f"**{rank}. {row.get('コンテンツ名','(名称未設定)')}** 　[{row['スプレッドシート']} / {row['シート名']}]({url})")
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

else:
    st.info("検索語を入力してください。例：**発表練習**, **グループ活動**, **朗読**, **工作**, **表現力** など")
