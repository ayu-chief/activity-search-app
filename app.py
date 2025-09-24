import streamlit as st
import pandas as pd
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import re, unicodedata

st.set_page_config(page_title="🎯 活動コンテンツ検索", layout="wide")

# =========================
#  Google Sheets 接続設定（統一）
# =========================
SERVICE_ACCOUNT_INFO = st.secrets["google_service_account"]
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

# 認証メールを表示（確認用）
st.info(f"Service Account: {creds.service_account_email}")

@st.cache_data(show_spinner=False)
def load_all_data():
    rows = []
    for sid in SPREADSHEET_IDS:
        try:
            st.write(f"🔎 Trying to open: {sid}")
            sh = gc.open_by_key(sid)
            st.success(f"✅ Opened: {sh.title}")
        except APIError as e:
            # gspreadのHTTPレスポンスを表示（403/404 の切り分けに有用）
            resp = getattr(e, "response", None)
            code = getattr(resp, "status_code", "?")
            text = getattr(resp, "text", str(e))
            st.error(f"❌ Failed to open (status={code}): {sid}")
            st.code(text[:2000])
            continue
        except Exception as e:
            st.error(f"❌ Failed to open (unexpected): {sid}")
            st.code(str(e))
            continue

        for ws in sh.worksheets():
            try:
                vals = ws.get_all_values()
            except Exception as e:
                st.error(f"❌ Failed to read values: {sh.title} / {ws.title}")
                st.code(str(e))
                continue

            if not vals:
                continue
            rec = parse_sheet(vals)  # 既存の関数を利用
            if not any(rec.values()):
                continue

            rec["スプレッドシート"] = sh.title
            rec["ファイルID"] = sid
            rec["シート名"] = ws.title
            rec["検索用テキスト"] = " ".join([
                rec.get("コンテンツ名",""), rec.get("テーマ",""), rec.get("対象",""),
                rec.get("準備物",""), rec.get("実施方法",""),
                rec.get("子供たちの反応",""), rec.get("良かった点",""), rec.get("改善点","")
            ]).strip()
            rows.append(rec)

    return pd.DataFrame(rows)

# =========================
#  テキスト正規化/トークン化
# =========================
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_ja_simple(s: str):
    s = normalize_text(s)
    s = re.sub(r"[、。・,./!?:;()\[\]{}「」『』￥$%^&*<>＝=＋+－\-…~―ー]", " ", s)
    return [t for t in s.split(" ") if t]

# =========================
#  クエリ拡張（同義語）
# =========================
SYNONYMS = {
    "探究": ["総合学習", "課題研究", "pbl", "プロジェクト学習"],
    "発表": ["プレゼン", "スピーチ", "口頭発表", "ショーケース"],
    "振り返り": ["リフレクション", "感想", "ふりかえり", "内省"],
    "職業体験": ["キャリア教育", "職業講話", "職業探究", "ゲスト講演"],
    "グループ活動": ["協働学習", "チーム作業", "共同作業", "班活動"],
    "創作": ["制作", "ものづくり", "ワークショップ", "ハンドメイド"],
    "表現": ["演劇", "朗読", "セリフ", "身体表現", "コミュニケーション"],
}
def expand_query(q: str) -> list[str]:
    qn = normalize_text(q)
    qs = {qn}
    for k, alts in SYNONYMS.items():
        if k in qn or any(a in qn for a in alts):
            qs.update([normalize_text(x) for x in [k, *alts]])
    return list(qs)

# =========================
#  シート → 1レコード抽出
#   （ラベル行から値をかき集める）
# =========================
LABELS = [
    "校舎名","コンテンツ名","テーマ","対象生徒","参加人数",
    "準備物","実施方法","子供たちの反応","子どもたちの反応","良かった点","改善点"
]

def extract_value(values, label):
    # シート全体から label を含むセルを探し、その行の「右側セル群」を結合して返す
    for row in values:
        for j, cell in enumerate(row):
            if label in str(cell):
                right = row[j+1:] if j+1 < len(row) else []
                toks = [t for t in right if t and str(t).strip()]
                if toks:
                    return " ".join(toks).strip()
    return ""

def parse_sheet(values):
    rec = {}
    for lab in LABELS:
        rec[lab] = extract_value(values, lab)
    # 子ども/子供 どちらかで補完
    if not rec.get("子供たちの反応"):
        rec["子供たちの反応"] = rec.get("子どもたちの反応","")
    # 表示用・検索用のフィールド整形
    rec_out = {
        "校舎名": rec.get("校舎名",""),
        "コンテンツ名": rec.get("コンテンツ名",""),
        "テーマ": rec.get("テーマ",""),
        "対象": rec.get("対象生徒",""),
        "参加人数": rec.get("参加人数",""),
        "準備物": rec.get("準備物",""),
        "実施方法": rec.get("実施方法",""),
        "子供たちの反応": rec.get("子供たちの反応",""),
        "良かった点": rec.get("良かった点",""),
        "改善点": rec.get("改善点",""),
    }
    return rec_out

@st.cache_data(show_spinner=False)
def load_all_data():
    rows = []
    for sid in SPREADSHEET_IDS:
        sh = gc.open_by_key(sid)
        file_title = sh.title
        for ws in sh.worksheets():
            vals = ws.get_all_values()
            if not vals:
                continue
            rec = parse_sheet(vals)
            # ほぼ空のシートはスキップ
            if not any(rec.values()):
                continue
            rec["スプレッドシート"] = file_title
            rec["ファイルID"] = sid
            rec["シート名"] = ws.title
            # 検索用テキスト
            rec["検索用テキスト"] = " ".join([
                rec.get("コンテンツ名",""), rec.get("テーマ",""), rec.get("対象",""),
                rec.get("準備物",""), rec.get("実施方法",""),
                rec.get("子供たちの反応",""), rec.get("良かった点",""), rec.get("改善点","")
            ]).strip()
            rows.append(rec)
    return pd.DataFrame(rows)

# =========================
#  モデル&インデックス準備
# =========================
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
model = load_model()

df = load_all_data()
if df.empty:
    st.error("データを取得できませんでした。サービスアカウントの共有権限をご確認ください。")
    st.stop()

# 正規化コーパス
df["検索用テキスト_norm"] = df["検索用テキスト"].map(normalize_text)

# 1) セマンティック埋め込み
@st.cache_resource(show_spinner=True)
def build_embeddings(texts: list[str]):
    return model.encode(texts, convert_to_tensor=True)
corpus_embeddings = build_embeddings(df["検索用テキスト_norm"].tolist())

# 2) BM25
corpus_tokens = [tokenize_ja_simple(t) for t in df["検索用テキスト_norm"].tolist()]
bm25 = BM25Okapi(corpus_tokens)

# =========================
#  ハイブリッド検索（必ず返す）
# =========================
def hybrid_search(query: str, top_k: int = 8, alpha: float = 0.7):
    # A) セマンティック
    q_emb = model.encode(normalize_text(query), convert_to_tensor=True)
    sem_hits = util.semantic_search(q_emb, corpus_embeddings, top_k=max(50, top_k*5))[0]
    sem_scores = {h["corpus_id"]: float(h["score"]) for h in sem_hits}

    # B) クエリ拡張 + BM25（最大値で統合）
    bm25_scores = {}
    for qx in (expand_query(query) or [query]):
        toks = tokenize_ja_simple(qx)
        scores = bm25.get_scores(toks)
        for i, s in enumerate(scores):
            bm25_scores[i] = max(bm25_scores.get(i, 0.0), float(s))

    # C) BM25スコア正規化
    max_bm = max(bm25_scores.values(), default=1.0) or 1.0
    bm25_scores = {i: s / max_bm for i, s in bm25_scores.items()}

    # D) 結合（しきい値で弾かない）
    all_ids = set(sem_scores) | set(bm25_scores)
    merged = []
    for i in all_ids:
        s_sem = sem_scores.get(i, 0.0)
        s_bm  = bm25_scores.get(i, 0.0)
        score = alpha * s_sem + (1 - alpha) * s_bm
        merged.append((i, score, s_sem, s_bm))
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]

# =========================
#  UI
# =========================
st.markdown("""
<div style='background:#e3f3ec;padding:16px;border-radius:8px'>
  <h2 style='margin:0'>🎯 活動コンテンツ検索</h2>
  <p style='margin:4px 0 0'>語が含まれていなくても、意味が近い活動を必ず提示します。</p>
</div>
""", unsafe_allow_html=True)
st.write("")

col1, col2, col3 = st.columns([3,1,1])
with col1:
    query = st.text_input("キーワードや目的（例：発表練習 / 物語体験 / グループ活動 など）", "")
with col2:
    top_k = st.number_input("件数", 1, 20, 6)
with col3:
    alpha = st.slider("意味重視↔語一致", 0.0, 1.0, 0.7, 0.05)

# 対象フィルタ（任意）
targets = st.multiselect("対象で絞り込み（任意）", ["小", "中", "高"])

def pass_target_filter(text):
    if not targets:
        return True
    t = text or ""
    return all(any(tag in t for tag in targets) for tag in targets)

if query:
    results = hybrid_search(query, top_k=top_k, alpha=alpha)
    st.caption("検索方式: Semantic + BM25 + クエリ拡張（しきい値なし・必ず提案）")
else:
    # 未入力時はおすすめ表示
    results = [(i, 0.0, 0.0, 0.0) for i in df.sample(min(top_k, len(df)), random_state=0).index]
    st.caption("未入力のため、ランダムにおすすめを表示中")

shown = 0
for i, total, s_sem, s_bm in results:
    row = df.iloc[i]
    if not pass_target_filter(row.get("対象","")):
        continue
    st.markdown(f"### 📌 {row.get('コンテンツ名','(名称未設定)')}　—　{row.get('シート名','')}")
    if query:
        st.progress(min(1.0, total))
        st.caption(f"(semantic: {s_sem:.2f} / bm25: {s_bm:.2f} / total: {total:.2f})")
    cols = st.columns(3)
    with cols[0]:
        st.write(f"**テーマ:** {row.get('テーマ','')}")
        st.write(f"**対象:** {row.get('対象','')}")
        st.write(f"**参加人数:** {row.get('参加人数','')}")
    with cols[1]:
        st.write(f"**準備物:** {row.get('準備物','')}")
        st.write(f"**良かった点:** {row.get('良かった点','')}")
    with cols[2]:
        st.write(f"**実施方法:** {row.get('実施方法','')}")
        st.write(f"**子供たちの反応:** {row.get('子供たちの反応','')}")
        st.write(f"**改善点:** {row.get('改善点','')}")
    st.caption(f"📄 {row.get('スプレッドシート','')} / シート: {row.get('シート名','')}")
    st.divider()
    shown += 1

if shown == 0:
    st.info("該当がフィルタで除外されました。フィルタや件数を調整してください。")



