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

# 一覧の段階表示用（さらに表示）
if "show_n" not in st.session_state:
    st.session_state.show_n = 15
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

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
# 429回避：metadata(title,sheetId) → values.batchGet
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
    """worksheets()は使わず、タイトル＆sheetIdを取得→values.batchGetで一括取得"""
    rows = []
    for sid in SPREADSHEET_IDS:
        sh = open_sheet_by_id(sid)
        if not sh:
            continue

        # 1) タイトルと sheetId (gid) を軽量取得
        meta = None
        for attempt in range(MAX_RETRY):
            try:
                meta = sh.fetch_sheet_metadata(
                    params={"fields": "sheets(properties(title,sheetId))"}
                )
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

        sheets_props = [s["properties"] for s in meta.get("sheets", []) if "properties" in s]
        title_to_gid = {p.get("title"): p.get("sheetId") for p in sheets_props}

        titles = [p.get("title") for p in sheets_props]
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
            rec["シートGID"] = title_to_gid.get(ws_title)
            rec["検索用テキスト"] = " ".join([
                rec.get("コンテンツ名",""), rec.get("テーマ",""), rec.get("対象",""),
                rec.get("準備物",""), rec.get("実施方法",""),
                rec.get("子供たちの反応",""), rec.get("良かった点",""), rec.get("改善点","")
            ]).strip()
            rows.append(rec)

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# UI（モード切替）
# -----------------------------------------------------------------------------
st.title("活動コンテンツ検索")

with st.sidebar:
    st.header("表示モード")
    mode = st.radio(
        "ページを選択",
        ["🔍 検索", "📑 シート別一覧"],
        index=0,
        help="検索ページと、スプレッドシート/シートごとの一覧を切り替えます。",
    )

    st.header("検索設定")
    alpha = st.slider("意味重視（1.0） ←→ 語一致重視（0.0）", 0.0, 1.0, 0.7, 0.05)
    # top_k は「最大計算件数」（段階表示でここまで出せる）
    top_k = st.slider("件数（最大計算件数）", 50, 500, 200, step=50)
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

# =============================================================================
# 📑 シート別一覧（一覧モード）
# =============================================================================
if mode == "📑 シート別一覧":
    st.subheader("シート別一覧")

    # スプレッドシートのプルダウン（すべて or 個別）
    ss_options = (
        df[["スプレッドシート", "ファイルID"]]
        .dropna()
        .drop_duplicates()
        .sort_values("スプレッドシート")
        .values
        .tolist()
    )
    ss_names = ["（すべて）"] + [name for name, _id in ss_options]
    selected_ss = st.selectbox("スプレッドシートを選択", ss_names, index=0)

    if selected_ss == "（すべて）":
        df_view = df
    else:
        df_view = df[df["スプレッドシート"] == selected_ss]

    # シート単位に集計（件数は計算するが表示には使わない）
    grp = (
        df_view.groupby(["スプレッドシート", "ファイルID", "シート名", "シートGID"])
              .size().reset_index(name="レコード数")
              .sort_values(["スプレッドシート", "シート名"])
    )

    if grp.empty:
        st.info("該当データがありません。")
        st.stop()

    # スプレッドシートごとにまとめて表示
    for (ss_name, file_id), chunk in grp.groupby(["スプレッドシート", "ファイルID"]):
        with st.expander(f"{ss_name}（{len(chunk)}シート）", expanded=False):

            # 各シートの行
            for _, r in chunk.iterrows():
                gid = r["シートGID"]
                if pd.notna(gid):
                    url = f"https://docs.google.com/spreadsheets/d/{file_id}/edit#gid={int(gid)}"
                else:
                    url = f"https://docs.google.com/spreadsheets/d/{file_id}/edit"

                # ▼ プレビューは「テーマ」を重複除去して上位表示
                themes_series = (
                    df[(df["スプレッドシート"] == ss_name) & (df["シート名"] == r["シート名"])]
                    ["テーマ"]
                    .fillna("")
                    .map(normalize)
                )
                # 空を除外して重複排除
                themes_uniq = []
                seen = set()
                for t in themes_series:
                    if t and t not in seen:
                        seen.add(t)
                        themes_uniq.append(t)
                    if len(themes_uniq) >= 3:   # 表示数は必要に応じて調整
                        break

                # 行の表示（リンク＋テーマのプレビュー）
                st.markdown(f"- [{r['シート名']}]({url})")
                if themes_uniq:
                    st.caption(" ／ ".join(themes_uniq))

    st.stop()  # 一覧モードではここで終了

# =============================================================================
# 🔍 検索（検索モード）
# =============================================================================
st.caption("🔎 キーワードを入力して検索")
q = st.text_input(
    label="キーワードを入力",
    value="",
    placeholder="例: 発表練習, グループ活動, 朗読, 工作, 表現力 など",
    label_visibility="collapsed",
)

# 検索語が変わったら表示件数をリセット
if q != st.session_state.last_query:
    st.session_state.show_n = 15
    st.session_state.last_query = q

if q:
    # 検索準備（埋め込み＋BM25）※検索モードでのみ計算
    def tokenize_ja(text: str) -> List[str]:
        text = normalize(text)
        toks = re.split(r"[ \u3000、。・,./!?！？\-\n\r\t]+", text)
        return [t for t in toks if t]

    @st.cache_data(show_spinner=False)
    def build_bm25(corpus_tokens: List[List[str]]):
        return BM25Okapi(corpus_tokens)

    @st.cache_resource(show_spinner=True)
    def load_embedder():
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    corpus_texts = (df["検索用テキスト"].fillna("") + " " + df["コンテンツ名"].fillna("")).tolist()
    corpus_tokens = [tokenize_ja(t) for t in corpus_texts]
    bm25 = build_bm25(corpus_tokens)
    embedder = load_embedder()
    corpus_emb = embedder.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=False)

    SYNONYMS = {
        "発表": ["プレゼン", "スピーチ", "表現", "発表練習"],
        "表現": ["発表", "伝える", "アウトプット"],
        "協力": ["協働", "チーム", "グループ", "共同"],
        "創作": ["ものづくり", "制作", "工作", "クリエイティブ"],
        "読解": ["読み取り", "感想", "読書", "朗読"],
    }

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

    # 上位 top_k まで取得（ここまで段階表示で増やせる）
    idx_all = np.argsort(final)[::-1][:top_k]

    # いま表示する件数（15件ずつ増える）
    show_n = min(st.session_state.show_n, len(idx_all))
    idx = idx_all[:show_n]

    st.subheader(f"検索結果（{len(idx_all)}件中 {show_n}件を表示）")

    for rank, i in enumerate(idx, start=1):
        row = df.iloc[i]

        # gid があれば特定タブへ直リンク
        gid = row.get("シートGID")
        fid = row["ファイルID"]
        if pd.notna(gid):
            url = f"https://docs.google.com/spreadsheets/d/{fid}/edit#gid={int(gid)}"
        else:
            url = f"https://docs.google.com/spreadsheets/d/{fid}/edit"

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

    # --- さらに表示ボタン ---
    if show_n < len(idx_all):
        c1, c2, _ = st.columns([1, 1, 6])
        if c1.button("さらに表示（+15）"):
            st.session_state.show_n = min(show_n + 15, len(idx_all))
            st.rerun()
        if c2.button("全件表示"):
            st.session_state.show_n = len(idx_all)
            st.rerun()


