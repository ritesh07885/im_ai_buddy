import json
import random
import re
import sqlite3
import time
import urllib.error
import urllib.request
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import streamlit as st

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import (
        AdaBoostClassifier,
        AdaBoostRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        IsolationForest,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        silhouette_score,
    )
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        MinMaxScaler,
        OneHotEncoder,
        PowerTransformer,
        RobustScaler,
        StandardScaler,
    )
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

st.set_page_config(
    page_title="I'm AI Buddy — Complete Data Science Studio",
    page_icon="🤖",
    layout="wide",
)

# ==============================================================================
# BILINGUAL HELPER (HINDI <-> ENGLISH ONE-TAP SWITCH)
# ==============================================================================
if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "Hindi"


def T(hi_text, en_text):
    return en_text if st.session_state.get("ui_lang", "Hindi") == "English" else hi_text


def is_en():
    return st.session_state.get("ui_lang", "Hindi") == "English"


# ==============================================================================
# DATA SCIENCE FUNCTION & ALGORITHM CATALOG (LIST / BOX REFERENCE)
# ==============================================================================
DS_FUNCTION_CATALOG = {
    "1. Data Cleaning & Preprocessing": [
        "Drop Duplicates",
        "Trim Whitespace",
        "Missing Value Imputation (Mean / Median / Mode / Constant / KNN)",
        "Drop Rows/Columns with High Missing %",
        "Outlier Handling (IQR Clipping / Winsorization / Z-Score Filter)",
        "Data Type Casting (Numeric / String / Datetime / Category)",
        "Text Normalization (Lower / Upper / Title / Remove Special Chars)",
        "Rename Column / Drop Column / Reorder Columns",
        "Custom Formula / Calculated Column (Add / Edit)",
        "Row Filtering (Condition / Keyword / Range)",
        "USD Range to INR Conversion",
    ],
    "2. Feature Engineering & Selection": [
        "Log1p / Square Root / Power Transform",
        "Binning / Discretization (Equal Width / Quantile)",
        "Scaling (StandardScaler / MinMaxScaler / RobustScaler)",
        "Encoding (OneHotEncoder / OrdinalEncoder)",
        "Date/Time Feature Extraction (Year / Month / Day / DayOfWeek)",
        "Interaction / Ratio / Difference Feature Creation",
        "SelectKBest Statistical Feature Ranking",
        "PCA Dimensionality Reduction",
    ],
    "3. EDA & Statistical Analysis": [
        "Dataset Shape, Nulls & Duplicate Audit",
        "All-Columns Data Profile (Type, Unique, Missing %, Mean/Median/Top)",
        "Descriptive Statistics (Count, Mean, Std, Min, 25%, 50%, 75%, Max)",
        "Skewness & Kurtosis Check",
        "Correlation Matrix & Heatmap (Pearson / Spearman)",
        "Distribution Histogram + Boxplot",
        "Category Frequency Bar Chart",
        "Scatter / Pair Relationship Plot",
        "GroupBy Aggregations (Sum, Mean, Median, Min, Max, Count, Std)",
        "IQR Outlier Detection Table",
    ],
    "4. Regression Algorithms": [
        "Linear Regression",
        "Ridge Regression (L2)",
        "Lasso Regression (L1)",
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "Extra Trees Regressor",
        "Gradient Boosting Regressor",
        "HistGradientBoosting Regressor",
        "AdaBoost Regressor",
        "K-Nearest Neighbors (KNN) Regressor",
        "Support Vector Regressor (SVR)",
        "MLP Neural Network Regressor",
    ],
    "5. Classification Algorithms": [
        "Logistic Regression",
        "Decision Tree Classifier",
        "Random Forest Classifier",
        "Extra Trees Classifier",
        "Gradient Boosting Classifier",
        "HistGradientBoosting Classifier",
        "AdaBoost Classifier",
        "K-Nearest Neighbors (KNN) Classifier",
        "Support Vector Classifier (SVC)",
        "Gaussian Naive Bayes",
        "MLP Neural Network Classifier",
    ],
    "6. Clustering & Anomaly Detection": [
        "K-Means Clustering (with Silhouette Score)",
        "Agglomerative Hierarchical Clustering",
        "DBSCAN Density Clustering",
        "Isolation Forest Anomaly / Outlier Detection",
    ],
    "7. Model Evaluation & Gen AI": [
        "Train / Test Split & Cross-Validation",
        "Regression Metrics (R2, MAE, RMSE)",
        "Classification Metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)",
        "Feature Importance Ranking",
        "Live Single-Row Prediction",
        "One-Click AI Report + Download (.md)",
        "AI / ML Roadmap Generator + Download (.md)",
    ],
}

# ==============================================================================
# GLOBAL SESSION STATE
# ==============================================================================
defaults_state = {
    "uni_loaded_file": None,
    "uni_raw_df": None,
    "uni_df": None,
    "uni_uploaded_img": None,
    "uni_img_name": None,
    "img_proc_mode": "Original (Auto-Inspected)",
    "img_resize_w": 800,
    "img_bright": 1.0,
    "img_contrast": 1.0,
    "ds_action_log": [],
    "da_ai_summary": None,
    "sql_query_box_val": "SELECT * FROM dataset LIMIT 20;",
    "sql_text_area": "SELECT * FROM dataset LIMIT 20;",
    "sql_res_df": None,
    "sql_err_msg": None,
    "sql_ai_msg": None,
    "sql_last_preset": None,
    "bi_bg_img": None,
    "bi_bg_src": None,
    "bi_ai_review": None,
    "bi_dax_code": None,
    "ds_ml_report": None,
    "ds_full_ai_report": None,
    "ds_trained_bundle": None,
    "ds_cluster_res": None,
    "ds_feat_rank_df": None,
    "ds_aiml_roadmap": None,
    "dl_mlp_bundle": None,
    "dl_seq_res": None,
    "dl_ae_res": None,
    "dl_ai_report": None,
    "nlp_sent_df": None,
    "nlp_kw_df": None,
    "nlp_clf_bundle": None,
    "ts_fc_df": None,
    "nlp_ai_report": None,
}
for _k, _v in defaults_state.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ==============================================================================
# HELPER FUNCTIONS (INCLUDING AUTOMATIC IMAGE PROCESSING PIPELINE)
# ==============================================================================
def log_ds_action(msg):
    st.session_state.ds_action_log.append(msg)


def load_uploaded_table(file_obj):
    if file_obj.name.lower().endswith(".xlsx"):
        return pd.read_excel(file_obj)
    raw_bytes = file_obj.getvalue()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(BytesIO(raw_bytes), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(BytesIO(raw_bytes), encoding="latin1", errors="replace")


def load_uploaded_image(file_obj):
    img = Image.open(BytesIO(file_obj.getvalue()))
    return img.convert("RGB")


def inspect_image_metrics(pil_img):
    rgb_img = pil_img.convert("RGB")
    arr = np.asarray(rgb_img, dtype=np.float32)
    w, h = rgb_img.size
    means = arr.mean(axis=(0, 1))
    brightness = float(np.mean(means))
    contrast = float(np.std(arr))
    ch_names = ["Red", "Green", "Blue"]
    dom_ch = ch_names[int(np.argmax(means))]
    total_px = w * h
    if total_px >= 1_000_000:
        tier = "High Resolution (1MP+)"
    elif total_px >= 250_000:
        tier = "Medium Resolution"
    else:
        tier = "Low Resolution"
    return {
        "width": w,
        "height": h,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "dominant_channel": dom_ch,
        "resolution_tier": tier,
        "rgb_means": [round(float(x), 1) for x in means],
    }


def process_uploaded_image(pil_img, mode="Original (Auto-Inspected)", resize_w=800, bright_f=1.0, contrast_f=1.0):
    work = pil_img.convert("RGB").copy()
    if mode == "Resize (Keep Aspect Ratio)":
        ow, oh = work.size
        if ow > 0:
            nh = max(1, int(oh * (resize_w / float(ow))))
            work = work.resize((int(resize_w), nh), Image.Resampling.LANCZOS)
    elif mode == "Grayscale":
        work = ImageOps.grayscale(work).convert("RGB")
    elif mode == "Sharpen":
        work = work.filter(ImageFilter.SHARPEN)
    elif mode == "Edge Enhance":
        work = work.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif mode == "Smooth / Blur":
        work = work.filter(ImageFilter.SMOOTH_MORE)
    elif mode == "Auto Contrast":
        work = ImageOps.autocontrast(work)
    elif mode == "Flip Horizontal":
        work = ImageOps.mirror(work)

    if abs(bright_f - 1.0) > 0.01:
        work = ImageEnhance.Brightness(work).enhance(bright_f)
    if abs(contrast_f - 1.0) > 0.01:
        work = ImageEnhance.Contrast(work).enhance(contrast_f)
    return work


def render_auto_image_pipeline_ui(pil_orig, img_filename, key_prefix="auto_img"):
    metrics = inspect_image_metrics(pil_orig)
    st.success(
        T(
            f"🖼️ इमेज अपलोड और ऑटो-प्रोसेस हो गई है: **{img_filename}** ({metrics['width']} × {metrics['height']} px • {metrics['resolution_tier']})",
            f"🖼️ Image uploaded & auto-processed: **{img_filename}** ({metrics['width']} × {metrics['height']} px • {metrics['resolution_tier']})",
        )
    )

    p_c1, p_c2, p_c3, p_c4 = st.columns([1.4, 1.0, 1.0, 1.0])
    proc_modes = [
        "Original (Auto-Inspected)",
        "Resize (Keep Aspect Ratio)",
        "Grayscale",
        "Sharpen",
        "Edge Enhance",
        "Smooth / Blur",
        "Auto Contrast",
        "Flip Horizontal",
    ]
    sel_mode = p_c1.selectbox(
        T("ऑटो-प्रोसेसिंग मोड:", "Auto-Processing Mode:"),
        proc_modes,
        index=proc_modes.index(st.session_state.img_proc_mode) if st.session_state.img_proc_mode in proc_modes else 0,
        key=f"{key_prefix}_mode",
    )
    st.session_state.img_proc_mode = sel_mode

    sel_rw = p_c2.slider(
        T("Resize Width (px):", "Resize Width (px):"),
        128,
        1920,
        int(st.session_state.img_resize_w),
        32,
        key=f"{key_prefix}_rw",
    )
    st.session_state.img_resize_w = sel_rw

    sel_br = p_c3.slider(
        T("Brightness:", "Brightness:"),
        0.5,
        1.7,
        float(st.session_state.img_bright),
        0.05,
        key=f"{key_prefix}_br",
    )
    st.session_state.img_bright = sel_br

    sel_ct = p_c4.slider(
        T("Contrast:", "Contrast:"),
        0.5,
        1.7,
        float(st.session_state.img_contrast),
        0.05,
        key=f"{key_prefix}_ct",
    )
    st.session_state.img_contrast = sel_ct

    processed_img = process_uploaded_image(
        pil_orig,
        mode=sel_mode,
        resize_w=sel_rw,
        bright_f=sel_br,
        contrast_f=sel_ct,
    )
    proc_metrics = inspect_image_metrics(processed_img)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric(T("प्रोसेस्ड आकार", "Processed Size"), f"{proc_metrics['width']}×{proc_metrics['height']}")
    m_col2.metric(T("Dominant Channel", "Dominant Channel"), proc_metrics["dominant_channel"])
    m_col3.metric(T("Brightness", "Brightness"), proc_metrics["brightness"])
    m_col4.metric(T("Contrast", "Contrast"), proc_metrics["contrast"])

    v_left, v_right = st.columns(2)
    with v_left:
        st.caption(T("मूल इमेज (Original)", "Original Image"))
        st.image(pil_orig, use_container_width=True)
    with v_right:
        st.caption(T(f"प्रोसेस्ड इमेज ({sel_mode})", f"Processed Image ({sel_mode})"))
        st.image(processed_img, use_container_width=True)

    buf_dl = BytesIO()
    processed_img.save(buf_dl, format="PNG")
    st.download_button(
        T("📥 प्रोसेस्ड इमेज डाउनलोड करें (.png)", "📥 Download Processed Image (.png)"),
        data=buf_dl.getvalue(),
        file_name=f"processed_{img_filename.rsplit('.', 1)[0]}.png",
        mime="image/png",
        key=f"{key_prefix}_dl_btn",
    )
    return processed_img


def set_sql_editor_query(q_text):
    st.session_state.sql_query_box_val = q_text
    st.session_state.sql_text_area = q_text


def _sync_sql_from_area():
    st.session_state.sql_query_box_val = st.session_state.sql_text_area


def get_1d_series(df_in, col_name):
    if col_name not in df_in.columns:
        return pd.Series(dtype="float64", index=df_in.index)
    extracted = df_in[col_name]
    if isinstance(extracted, pd.DataFrame):
        return extracted.iloc[:, 0]
    return extracted


def parse_usd_range_to_inr(val, rate, mode, unk_zero):
    if pd.isna(val):
        return 0.0 if unk_zero else np.nan
    txt = str(val).strip()
    if not txt or "unknown" in txt.lower() or "non-applicable" in txt.lower():
        return 0.0 if unk_zero else np.nan

    txt_low = txt.lower()
    if re.search(r"\b(?:b|billion)\b|\d+(?:\.\d+)?\s*b\b", txt_low):
        mult = 1e9
    elif re.search(r"\b(?:m|million)\b|\d+(?:\.\d+)?\s*m\b", txt_low):
        mult = 1e6
    elif re.search(r"\b(?:k|thousand)\b|\d+(?:\.\d+)?\s*k\b", txt_low):
        mult = 1e3
    else:
        mult = 1.0

    nums = [float(x.replace(",", "")) for x in re.findall(r"\d+(?:\.\d+)?", txt)]
    if not nums:
        return 0.0 if unk_zero else np.nan

    if len(nums) >= 2:
        low_v, high_v = nums[0] * mult, nums[1] * mult
    else:
        low_v = high_v = nums[0] * mult

    if mode.startswith("Lower"):
        usd_amt = low_v
    elif mode.startswith("Upper"):
        usd_amt = high_v
    else:
        usd_amt = (low_v + high_v) / 2.0

    return round(usd_amt * rate, 2)


def build_sql_analysis_library(cat_col, num_col):
    return {
        "1. Preview Top 20 Rows": "SELECT * FROM dataset LIMIT 20;",
        "2. Total Row Count": "SELECT COUNT(*) AS total_rows FROM dataset;",
        f"3. Group By Summary ({cat_col} -> {num_col})": (
            f'SELECT "{cat_col}", COUNT(*) AS row_count, SUM("{num_col}") AS total_val, '
            f'ROUND(AVG("{num_col}"), 2) AS avg_val, MIN("{num_col}") AS min_val, MAX("{num_col}") AS max_val\n'
            f'FROM dataset\nGROUP BY "{cat_col}"\nORDER BY total_val DESC;'
        ),
        f"4. Top 10 Highest ({num_col})": (
            f'SELECT * FROM dataset\nWHERE "{num_col}" IS NOT NULL\nORDER BY "{num_col}" DESC\nLIMIT 10;'
        ),
        f"5. Top 3 Per {cat_col} (ROW_NUMBER Window)": (
            f'WITH ranked AS (\n  SELECT *, ROW_NUMBER() OVER (PARTITION BY "{cat_col}" ORDER BY "{num_col}" DESC) AS rn\n  FROM dataset\n)\nSELECT * FROM ranked\nWHERE rn <= 3;'
        ),
        f"6. Dense Rank by {num_col} (DENSE_RANK)": (
            f'SELECT *, DENSE_RANK() OVER (ORDER BY "{num_col}" DESC) AS rank_num\nFROM dataset\nORDER BY "{num_col}" DESC;'
        ),
        f"7. Running Total of {num_col}": (
            f'SELECT *, SUM("{num_col}") OVER (ORDER BY "{num_col}" DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total\nFROM dataset\nLIMIT 100;'
        ),
        f"8. Share % of Total ({cat_col} by {num_col})": (
            f'WITH totals AS (\n  SELECT "{cat_col}", SUM("{num_col}") AS grp_sum\n  FROM dataset\n  GROUP BY "{cat_col}"\n)\n'
            f'SELECT "{cat_col}", grp_sum, ROUND(100.0 * grp_sum / NULLIF(SUM(grp_sum) OVER (), 0), 2) AS pct_share\nFROM totals\nORDER BY grp_sum DESC;'
        ),
        f"9. Distinct Values in {cat_col}": (
            f'SELECT "{cat_col}", COUNT(*) AS freq\nFROM dataset\nGROUP BY "{cat_col}"\nORDER BY freq DESC;'
        ),
        f"10. Rows Above Average {num_col}": (
            f'SELECT * FROM dataset\nWHERE "{num_col}" > (SELECT AVG("{num_col}") FROM dataset)\nORDER BY "{num_col}" DESC;'
        ),
    }


def _compact_prompt_for_token_limit(prompt_text, max_chars=5500):
    if len(prompt_text) <= max_chars:
        return prompt_text
    lines = prompt_text.splitlines()
    kept = []
    total = 0
    for ln in lines:
        clean_ln = ln[:320] + ("..." if len(ln) > 320 else "")
        if total + len(clean_ln) + 1 > max_chars:
            kept.append("... [Prompt shortened to fit Groq/Provider token limit] ...")
            break
        kept.append(clean_ln)
        total += len(clean_ln) + 1
    return "\n".join(kept)


def call_openai_compatible_chat(
    api_key_val, base_url, model_name, prompt_text, provider_label="OpenAI-Compatible", extra_headers=None
):
    if not api_key_val:
        return None, T(f"{provider_label} API Key खाली है।", f"{provider_label} API Key is empty.")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key_val.strip()}",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
    }
    if extra_headers:
        headers.update(extra_headers)

    prompts_to_try = [prompt_text]
    if len(prompt_text) > 5500:
        prompts_to_try.append(_compact_prompt_for_token_limit(prompt_text, max_chars=5500))
    prompts_to_try.append(_compact_prompt_for_token_limit(prompt_text, max_chars=2400))

    last_err = None
    for idx_p, p_try in enumerate(prompts_to_try):
        payload = {
            "model": model_name.strip(),
            "messages": [
                {"role": "system", "content": "You are I'm AI Buddy, a concise data science and AI assistant."},
                {"role": "user", "content": p_try},
            ],
            "temperature": 0.3,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    txt = (msg.get("content") or "").strip()
                    if txt:
                        suffix = " *(Auto-compacted for 413 token limit)*" if idx_p > 0 else ""
                        return f"{txt}\n\n*(Provider: {provider_label} — `{model_name}`{suffix})*", None
            return None, T(f"{provider_label} से खाली जवाब आया।", f"Empty response received from {provider_label}.")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_err = f"{provider_label} HTTP {e.code}: {body[:240]}"
            if e.code == 413 and idx_p < len(prompts_to_try) - 1:
                continue
            return None, last_err
        except Exception as exc:
            return None, T(f"{provider_label} एरर: {exc}", f"{provider_label} Error: {exc}")
    return None, last_err or T(f"{provider_label} अनुरोध विफल रहा।", f"{provider_label} request failed.")


def call_ollama_fallback(prompt_text, ollama_model="qwen2.5:3b", ollama_url="http://localhost:11434"):
    payload = {
        "model": ollama_model,
        "prompt": prompt_text,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            txt = (data.get("response") or "").strip()
            if txt:
                return f"{txt}\n\n*(Provider: Local Ollama — `{ollama_model}`)*", None
        return None, T("Ollama से खाली जवाब आया।", "Empty response received from Ollama.")
    except urllib.error.URLError as e:
        return None, T(f"Ollama कनेक्ट नहीं हो सका ({e.reason})", f"Could not connect to Ollama ({e.reason})")
    except Exception as e:
        return None, T(f"Ollama एरर: {e}", f"Ollama Error: {e}")


def call_ai_provider(ai_cfg, prompt_text):
    provider = ai_cfg.get("provider", "Google Gemini")
    enable_ollama = ai_cfg.get("enable_ollama", False)
    ollama_model = ai_cfg.get("ollama_model", "qwen2.5:3b")
    ollama_url = ai_cfg.get("ollama_url", "http://localhost:11434")

    if provider == "ChatGPT (OpenAI)":
        res, err = call_openai_compatible_chat(
            ai_cfg.get("openai_key", ""),
            "https://api.openai.com/v1",
            ai_cfg.get("openai_model", "gpt-4o-mini"),
            prompt_text,
            provider_label="ChatGPT (OpenAI)",
        )
        if res:
            return res, None
        if enable_ollama:
            ol_res, ol_err = call_ollama_fallback(prompt_text, ollama_model=ollama_model, ollama_url=ollama_url)
            if ol_res:
                return ol_res, None
            return None, f"{err} | {ol_err}"
        return None, err

    if provider == "Groq":
        res, err = call_openai_compatible_chat(
            ai_cfg.get("groq_key", ""),
            "https://api.groq.com/openai/v1",
            ai_cfg.get("groq_model", "openai/gpt-oss-120b"),
            prompt_text,
            provider_label="Groq",
        )
        if res:
            return res, None
        if enable_ollama:
            ol_res, ol_err = call_ollama_fallback(prompt_text, ollama_model=ollama_model, ollama_url=ollama_url)
            if ol_res:
                return ol_res, None
            return None, f"{err} | {ol_err}"
        return None, err

    if provider == "OpenRouter":
        res, err = call_openai_compatible_chat(
            ai_cfg.get("openrouter_key", ""),
            "https://openrouter.ai/api/v1",
            ai_cfg.get("openrouter_model", "openrouter/free"),
            prompt_text,
            provider_label="OpenRouter",
            extra_headers={"X-Title": "I'm AI Buddy"},
        )
        if res:
            return res, None
        if enable_ollama:
            ol_res, ol_err = call_ollama_fallback(prompt_text, ollama_model=ollama_model, ollama_url=ollama_url)
            if ol_res:
                return ol_res, None
            return None, f"{err} | {ol_err}"
        return None, err

    if provider == "Local Ollama":
        ol_res, ol_err = call_ollama_fallback(prompt_text, ollama_model=ollama_model, ollama_url=ollama_url)
        if ol_res:
            return ol_res, None
        return None, ol_err

    gem_key = ai_cfg.get("gemini_key", "")
    selected_gem_model = ai_cfg.get("gemini_model", "gemini-2.5-flash")
    models_to_try = []
    for m in [selected_gem_model, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-3.8-flash"]:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last_err = None
    if HAS_GENAI and gem_key:
        client = genai.Client(api_key=gem_key)
        for mdl in models_to_try:
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(model=mdl, contents=prompt_text)
                    if resp and resp.text and resp.text.strip():
                        return f"{resp.text.strip()}\n\n*(Provider: Google Gemini — `{mdl}`)*", None
                except Exception as exc:
                    last_err = exc
                    err_str = str(exc)
                    is_transient = any(
                        code in err_str for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "504"]
                    )
                    if is_transient and attempt < 2:
                        time.sleep((2 ** attempt) + random.uniform(0.1, 0.5))
                        continue
                    if "401" in err_str or "API_KEY_INVALID" in err_str:
                        return None, f"Invalid Gemini API Key: {err_str}"
                    break

    if enable_ollama:
        ol_res, ol_err = call_ollama_fallback(prompt_text, ollama_model=ollama_model, ollama_url=ollama_url)
        if ol_res:
            return ol_res, None
        last_err = f"Gemini Err: {last_err} | {ol_err}"

    return None, str(last_err or T("AI सेवा अस्थायी रूप से उपलब्ध नहीं है।", "AI service is temporarily unavailable."))


def build_local_stats_summary(df_in):
    num_c = df_in.select_dtypes(include="number").columns.tolist()
    cat_c = df_in.select_dtypes(exclude="number").columns.tolist()
    if is_en():
        lines = [
            f"**Local Statistical Summary:** Total Rows = `{len(df_in)}`, Total Columns = `{len(df_in.columns)}`",
            f"- Numeric Columns ({len(num_c)}): {', '.join(num_c) if num_c else 'None'}",
            f"- Category/Text Columns ({len(cat_c)}): {', '.join(cat_c) if cat_c else 'None'}",
            f"- Missing Cells (Nulls): `{int(df_in.isna().sum().sum())}` | Duplicate Rows: `{int(df_in.duplicated().sum())}`",
        ]
    else:
        lines = [
            f"**लोकल स्टैटिस्टिकल समरी:** कुल पंक्तियाँ = `{len(df_in)}`, कुल कॉलम = `{len(df_in.columns)}`",
            f"- Numeric कॉलम्स ({len(num_c)}): {', '.join(num_c) if num_c else 'कोई नहीं'}",
            f"- Category/Text कॉलम्स ({len(cat_c)}): {', '.join(cat_c) if cat_c else 'कोई नहीं'}",
            f"- खाली सेल (Nulls): `{int(df_in.isna().sum().sum())}` | डुप्लीकेट पंक्तियाँ: `{int(df_in.duplicated().sum())}`",
        ]
    for c in num_c[:6]:
        s = pd.to_numeric(get_1d_series(df_in, c), errors="coerce").dropna()
        if not s.empty:
            lines.append(
                f"- **{c}**: Sum = `{s.sum():,.2f}`, Avg = `{s.mean():,.2f}`, "
                f"Min = `{s.min():,.2f}`, Max = `{s.max():,.2f}`"
            )
    return "\n".join(lines)


def build_all_columns_profile(df_in):
    rows = []
    total_r = max(len(df_in), 1)
    for col in df_in.columns:
        s = get_1d_series(df_in, col)
        null_cnt = int(s.isna().sum())
        uniq_cnt = int(s.nunique(dropna=True))
        is_num = pd.api.types.is_numeric_dtype(s)
        row_dict = {
            "Column": str(col),
            "Type": "Numeric" if is_num else "Category/Text",
            "Dtype": str(s.dtype),
            "Non-Null": len(s) - null_cnt,
            "Null Count": null_cnt,
            "Null %": round(100.0 * null_cnt / total_r, 1),
            "Unique": uniq_cnt,
            "Mean": "",
            "Median": "",
            "Min / Max": "",
        }
        if is_num:
            num_s = pd.to_numeric(s, errors="coerce").dropna()
            if not num_s.empty:
                row_dict["Mean"] = f"{num_s.mean():,.2f}"
                row_dict["Median"] = f"{num_s.median():,.2f}"
                row_dict["Min / Max"] = f"{num_s.min():,.2f} / {num_s.max():,.2f}"
        else:
            top_vals = s.dropna().astype(str).value_counts().head(2)
            if not top_vals.empty:
                row_dict["Min / Max"] = ", ".join([f"{k} ({v})" for k, v in top_vals.items()])
        rows.append(row_dict)
    return pd.DataFrame(rows)


def build_preprocessor(X_df):
    num_cols = X_df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]
    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", max_categories=25, sparse_output=False, dtype=np.float32)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def extract_feature_importances(pipe, X_df):
    try:
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            raw_imps = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef_arr = np.asarray(model.coef_)
            raw_imps = np.mean(np.abs(coef_arr), axis=0) if coef_arr.ndim > 1 else np.abs(coef_arr)
        else:
            return None

        feat_names = []
        num_cols = X_df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in X_df.columns if c not in num_cols]
        if num_cols:
            feat_names.extend([str(c) for c in num_cols])
        if cat_cols and "cat" in prep.named_transformers_:
            ohe = prep.named_transformers_["cat"].named_steps["ohe"]
            feat_names.extend(list(ohe.get_feature_names_out([str(c) for c in cat_cols])))

        if len(feat_names) != len(raw_imps):
            feat_names = [f"Feature_{i}" for i in range(len(raw_imps))]

        return (
            pd.DataFrame({"Feature": feat_names, "Importance": np.round(raw_imps, 4)})
            .sort_values("Importance", ascending=False)
            .head(12)
        )
    except Exception:
        return None


def prepare_ml_subset(df_in, cols_needed, max_rows=35000):
    sub = df_in[cols_needed].copy()
    for c in sub.select_dtypes(include=["object", "string"]).columns:
        s_txt = sub[c].astype(str)
        if s_txt.nunique(dropna=True) > 5000:
            top_k = s_txt.value_counts().head(50).index
            sub[c] = s_txt.where(s_txt.isin(top_k), "__OTHER__")
    if len(sub) > max_rows:
        sub = sub.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return sub


def get_regression_candidates():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=1.0),
        "Lasso Regression (L1)": Lasso(alpha=0.1, max_iter=2000),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=80, max_depth=16, n_jobs=-1, random_state=42),
        "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=80, max_depth=16, n_jobs=-1, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(random_state=42),
        "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "SVR (RBF)": SVR(C=1.0),
        "MLP Regressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    }


def get_classification_candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=80, max_depth=16, n_jobs=-1, random_state=42),
        "Extra Trees Classifier": ExtraTreesClassifier(n_estimators=80, max_depth=16, n_jobs=-1, random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
        "HistGradientBoosting Classifier": HistGradientBoostingClassifier(random_state=42),
        "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
        "SVC (RBF)": SVC(probability=True, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    }


def train_regression_suite(df_sub, target_col, feature_cols, selected_models, test_ratio, max_rows=35000):
    work = prepare_ml_subset(df_sub, feature_cols + [target_col], max_rows=max_rows)
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col])
    if len(work) < 6:
        return None, T(
            "Regression के लिए कम से कम 6 वैध पंक्तियाँ चाहिए।",
            "At least 6 valid rows are required for Regression.",
        )

    X = work[feature_cols]
    y = work[target_col].astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

    all_cands = get_regression_candidates()
    active_cands = {k: v for k, v in all_cands.items() if k in selected_models} or {
        "Random Forest Regressor": all_cands["Random Forest Regressor"]
    }

    metrics_rows = []
    trained_pipes = {}
    for name, est in active_cands.items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", est)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = float(r2_score(y_test, preds)) if len(y_test) >= 2 else 0.0
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        cv_r2 = np.nan
        if len(work) >= 15 and name not in ("MLP Regressor", "SVR (RBF)"):
            try:
                cv_folds = min(5, len(work) // 3)
                kf = KFold(n_splits=max(2, cv_folds), shuffle=True, random_state=42)
                cv_scores = cross_val_score(
                    Pipeline([("prep", build_preprocessor(X)), ("model", est)]),
                    X,
                    y,
                    cv=kf,
                    scoring="r2",
                )
                cv_r2 = round(float(np.mean(cv_scores)), 4)
            except Exception:
                cv_r2 = np.nan
        metrics_rows.append({
            "Model": name,
            "R2 Score": round(r2, 4),
            "CV R2": cv_r2,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
        })
        trained_pipes[name] = pipe

    best_row = sorted(metrics_rows, key=lambda r: r["R2 Score"], reverse=True)[0]
    best_name = best_row["Model"]
    best_pipe = trained_pipes[best_name]
    y_pred_best = best_pipe.predict(X_test)

    return {
        "task": "regression",
        "target": target_col,
        "features": feature_cols,
        "leaderboard": pd.DataFrame(metrics_rows),
        "best_name": best_name,
        "best_pipe": best_pipe,
        "y_test": y_test.reset_index(drop=True),
        "y_pred": pd.Series(y_pred_best),
        "imp_df": extract_feature_importances(best_pipe, X),
        "X_ref": X,
        "trained_rows": len(work),
    }, None


def train_classification_suite(df_sub, target_col, feature_cols, selected_models, test_ratio, max_rows=35000):
    work = prepare_ml_subset(df_sub, feature_cols + [target_col], max_rows=max_rows)
    work[target_col] = work[target_col].astype(str).str.strip()
    work = work[work[target_col] != ""]
    class_counts = work[target_col].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    work = work[work[target_col].isin(valid_classes)]

    if len(work) < 8 or work[target_col].nunique() < 2:
        return None, T(
            "Classification के लिए कम से कम 2 Classes (हर में ≥2 rows) और ≥8 पंक्तियाँ चाहिए।",
            "Classification requires at least 2 classes (≥2 rows each) and ≥8 total rows.",
        )

    X = work[feature_cols]
    y = work[target_col]
    strat = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42, stratify=strat)

    all_cands = get_classification_candidates()
    active_cands = {k: v for k, v in all_cands.items() if k in selected_models} or {
        "Random Forest Classifier": all_cands["Random Forest Classifier"]
    }

    metrics_rows = []
    trained_pipes = {}
    for name, est in active_cands.items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", est)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, preds, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        metrics_rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Accuracy %": f"{acc * 100:.1f}%",
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
        })
        trained_pipes[name] = pipe

    best_row = sorted(metrics_rows, key=lambda r: r["Accuracy"], reverse=True)[0]
    best_name = best_row["Model"]
    best_pipe = trained_pipes[best_name]
    y_pred_best = best_pipe.predict(X_test)
    labels_list = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred_best, labels=labels_list)
    cls_rep_txt = classification_report(y_test, y_pred_best, zero_division=0)

    return {
        "task": "classification",
        "target": target_col,
        "features": feature_cols,
        "leaderboard": pd.DataFrame(metrics_rows),
        "best_name": best_name,
        "best_pipe": best_pipe,
        "labels": labels_list,
        "cm": cm,
        "cls_report": cls_rep_txt,
        "imp_df": extract_feature_importances(best_pipe, X),
        "X_ref": X,
        "trained_rows": len(work),
    }, None


def render_heatmap_fig(corr_df):
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=120)
    im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticklabels(corr_df.index, fontsize=8.5)
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            val = corr_df.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="#111827", fontsize=8, fontweight="bold")
    ax.set_title("Correlation Heatmap", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def render_dist_fig(series_in, col_name):
    s_1d = series_in.iloc[:, 0] if isinstance(series_in, pd.DataFrame) else series_in
    s_num = pd.to_numeric(s_1d, errors="coerce").dropna()
    fig, (ax_h, ax_b) = plt.subplots(2, 1, figsize=(6.2, 4.6), dpi=120, gridspec_kw={"height_ratios": [3, 1]})
    if len(s_num) > 1:
        ax_h.hist(s_num, bins=15, color="#2563EB", alpha=0.85, edgecolor="white", linewidth=0.7)
        ax_h.axvline(s_num.mean(), color="#DC2626", linestyle="--", linewidth=1.5, label=f"Mean: {s_num.mean():,.2f}")
        ax_h.axvline(s_num.median(), color="#059669", linestyle="-.", linewidth=1.5, label=f"Median: {s_num.median():,.2f}")
        ax_h.set_title(f"Distribution & Spread — {col_name}", fontsize=10.5, fontweight="bold")
        ax_h.set_ylabel("Frequency")
        ax_h.legend(fontsize=8)
        ax_h.grid(True, linestyle="--", alpha=0.3)

        ax_b.boxplot(
            s_num,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="#93C5FD", color="#1D4ED8"),
            medianprops=dict(color="#0F172A", linewidth=2),
        )
        ax_b.set_xlabel(str(col_name))
        ax_b.set_yticks([])
    else:
        ax_h.text(0.5, 0.5, T("पर्याप्त Numeric डेटा नहीं है।", "Not enough numeric data."), ha="center", va="center")
        ax_b.axis("off")
    plt.tight_layout()
    return fig


def render_cat_bar_fig(series_in, col_name, top_n=8):
    s_1d = series_in.iloc[:, 0] if isinstance(series_in, pd.DataFrame) else series_in
    vc = s_1d.dropna().astype(str).value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=120)
    if not vc.empty:
        vc_plot = vc.sort_values(ascending=True)
        ax.barh(vc_plot.index, vc_plot.values, color="#0D9488", alpha=0.85, edgecolor="white")
        ax.set_title(f"Top {len(vc_plot)} — {col_name}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Count")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.text(0.5, 0.5, T("Category डेटा नहीं मिला।", "No category data found."), ha="center", va="center")
    plt.tight_layout()
    return fig


def build_local_soft_bg(width=900, height=520):
    img = Image.new("RGBA", (width, height), (246, 249, 253, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.ellipse([-120, -90, 460, 340], fill=(214, 232, 255, 140))
    draw.ellipse([420, 140, 1020, 620], fill=(225, 245, 238, 140))
    draw.rounded_rectangle([60, 50, width - 60, height - 50], radius=28, outline=(190, 206, 228, 120), width=3)
    return img.filter(ImageFilter.GaussianBlur(radius=14))


def generate_ai_bg_image(gem_key_val, prompt_text):
    if not HAS_GENAI or not gem_key_val:
        return build_local_soft_bg(), T("Local Fallback (Gemini API Key नहीं है)", "Local Fallback (No Gemini API Key)")
    try:
        client = genai.Client(api_key=gem_key_val)
        resp = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt_text,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        if resp and resp.candidates:
            for cand in resp.candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) or []
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        pil_img = Image.open(BytesIO(inline.data)).convert("RGBA")
                        return pil_img, "AI Generated Image"
    except Exception as exc:
        return build_local_soft_bg(), f"Local Fallback ({exc})"
    return build_local_soft_bg(), "Local Fallback"


def aggregate_series(temp_df, x_col, y_col, agg_func):
    if agg_func == "Sum":
        return temp_df.groupby(x_col)[y_col].sum()
    if agg_func == "Avg":
        return temp_df.groupby(x_col)[y_col].mean()
    if agg_func == "Max":
        return temp_df.groupby(x_col)[y_col].max()
    return temp_df.groupby(x_col)[y_col].count()


def render_bi_chart_with_bg(
    df_sub, chart_type, x_col, y_col, stack_col, agg_func, top_n, bg_img, use_bg, opacity_val
):
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=130)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    rendered_ok = False

    s_x_raw = get_1d_series(df_sub, x_col) if x_col else None
    s_y_raw = get_1d_series(df_sub, y_col) if y_col else None
    s_st_raw = get_1d_series(df_sub, stack_col) if stack_col else None

    if chart_type == "Histogram" and y_col:
        s_num = pd.to_numeric(s_y_raw, errors="coerce").dropna()
        if len(s_num) > 1:
            ax.hist(s_num, bins=15, color="#2563EB", alpha=0.84, edgecolor="white", linewidth=0.8)
            ax.set_title(f"Distribution of {y_col}", fontsize=11, fontweight="bold")
            ax.set_xlabel(str(y_col))
            ax.set_ylabel("Count")
            rendered_ok = True

    elif chart_type == "Box" and y_col:
        s_num = pd.to_numeric(s_y_raw, errors="coerce").dropna()
        if not s_num.empty:
            ax.boxplot(
                s_num,
                vert=True,
                patch_artist=True,
                boxprops=dict(facecolor="#93C5FD", color="#1D4ED8", linewidth=1.3),
                medianprops=dict(color="#0F172A", linewidth=2),
            )
            ax.set_title(f"Box Plot of {y_col}", fontsize=11, fontweight="bold")
            ax.set_ylabel(str(y_col))
            rendered_ok = True

    elif chart_type == "Scatter" and x_col and y_col:
        sx = pd.to_numeric(s_x_raw, errors="coerce")
        sy = pd.to_numeric(s_y_raw, errors="coerce")
        pair = pd.DataFrame({"x": sx, "y": sy}).dropna()
        if len(pair) > 4000:
            pair = pair.sample(n=4000, random_state=42)
        if not pair.empty:
            ax.scatter(pair["x"], pair["y"], color="#0D9488", alpha=0.75, edgecolors="white", linewidth=0.5, s=45)
            ax.set_title(f"{y_col} vs {x_col}", fontsize=11, fontweight="bold")
            ax.set_xlabel(str(x_col))
            ax.set_ylabel(str(y_col))
            rendered_ok = True

    elif chart_type == "Donut" and x_col and y_col:
        temp = pd.DataFrame({"x_k": s_x_raw.astype(str), "y_v": pd.to_numeric(s_y_raw, errors="coerce")}).dropna(
            subset=["y_v"]
        )
        temp = temp[temp["y_v"] >= 0]
        if not temp.empty:
            grp = aggregate_series(temp, "x_k", "y_v", agg_func)
            grp = grp[grp > 0].sort_values(ascending=False).head(min(top_n, 8))
            if not grp.empty:
                wedges, _ = ax.pie(
                    grp.values,
                    labels=grp.index,
                    autopct="%1.1f%%",
                    startangle=140,
                    pctdistance=0.78,
                    wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
                )
                for w in wedges:
                    w.set_alpha(0.88)
                ax.set_title(f"{agg_func} of {y_col} by {x_col} (Donut)", fontsize=11, fontweight="bold")
                rendered_ok = True

    elif chart_type == "Stacked Bar" and x_col and y_col and stack_col and stack_col != x_col:
        temp = pd.DataFrame({
            "x_k": s_x_raw.astype(str),
            "y_v": pd.to_numeric(s_y_raw, errors="coerce"),
            "s_k": s_st_raw.astype(str),
        }).dropna(subset=["y_v"])
        if not temp.empty:
            top_x = aggregate_series(temp, "x_k", "y_v", agg_func).sort_values(ascending=False).head(top_n).index
            top_s = temp[temp["x_k"].isin(top_x)]["s_k"].value_counts().head(6).index
            sub_st = temp[temp["x_k"].isin(top_x) & temp["s_k"].isin(top_s)]
            if not sub_st.empty:
                agg_map = {"Sum": "sum", "Avg": "mean", "Max": "max", "Count": "count"}
                piv = sub_st.pivot_table(
                    index="x_k",
                    columns="s_k",
                    values="y_v",
                    aggfunc=agg_map.get(agg_func, "sum"),
                    fill_value=0,
                ).reindex(top_x)
                piv.plot(kind="bar", stacked=True, ax=ax, alpha=0.86, edgecolor="white", linewidth=0.6)
                ax.set_title(f"Stacked {agg_func} of {y_col} by {x_col} ({stack_col})", fontsize=10.5, fontweight="bold")
                ax.set_xlabel(str(x_col))
                ax.set_ylabel(f"{agg_func} of {y_col}")
                plt.xticks(rotation=30, ha="right")
                ax.legend(title=str(stack_col), fontsize=8, title_fontsize=8)
                rendered_ok = True

    elif x_col and y_col:
        temp = pd.DataFrame({"x_k": s_x_raw.astype(str), "y_v": pd.to_numeric(s_y_raw, errors="coerce")}).dropna(
            subset=["y_v"]
        )
        if not temp.empty:
            grp = aggregate_series(temp, "x_k", "y_v", agg_func)
            grp = grp.sort_values(ascending=False).head(top_n)
            if not grp.empty:
                if chart_type == "Horizontal Bar":
                    grp_plot = grp.sort_values(ascending=True)
                    ax.barh(grp_plot.index, grp_plot.values, color="#1D4ED8", alpha=0.85, edgecolor="white")
                    ax.set_xlabel(f"{agg_func} of {y_col}")
                    ax.set_ylabel(str(x_col))
                elif chart_type == "Bar":
                    ax.bar(grp.index, grp.values, color="#2563EB", alpha=0.85, edgecolor="white")
                    ax.set_ylabel(f"{agg_func} of {y_col}")
                    ax.set_xlabel(str(x_col))
                    plt.xticks(rotation=30, ha="right")
                elif chart_type == "Line":
                    ax.plot(grp.index, grp.values, marker="o", linewidth=2.2, color="#0F766E")
                    ax.set_ylabel(f"{agg_func} of {y_col}")
                    ax.set_xlabel(str(x_col))
                    plt.xticks(rotation=30, ha="right")
                elif chart_type == "Area":
                    x_idx = np.arange(len(grp))
                    ax.fill_between(x_idx, grp.values, color="#3B82F6", alpha=0.35)
                    ax.plot(x_idx, grp.values, marker="o", linewidth=2.2, color="#1D4ED8")
                    ax.set_xticks(x_idx)
                    ax.set_xticklabels(grp.index, rotation=30, ha="right")
                    ax.set_ylabel(f"{agg_func} of {y_col}")
                    ax.set_xlabel(str(x_col))

                ax.set_title(f"{agg_func} of {y_col} by {x_col}", fontsize=11, fontweight="bold")
                rendered_ok = True

    if not rendered_ok:
        ax.text(
            0.5,
            0.5,
            T("इस कॉलम संयोजन के लिए उपयुक्त डेटा उपलब्ध नहीं है।", "No suitable data for this column combination."),
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    elif chart_type != "Donut":
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    chart_buf = BytesIO()
    fig.savefig(chart_buf, format="png", transparent=True, dpi=130)
    plt.close(fig)
    chart_buf.seek(0)
    chart_rgba = Image.open(chart_buf).convert("RGBA")

    if use_bg and bg_img is not None:
        bg_resized = bg_img.resize(chart_rgba.size, Image.Resampling.LANCZOS).convert("RGBA")
        bg_faded = bg_resized.copy()
        alpha_channel = bg_faded.split()[3]
        alpha_scaled = alpha_channel.point(lambda p: int(p * opacity_val))
        bg_faded.putalpha(alpha_scaled)
        base_white = Image.new("RGBA", chart_rgba.size, (255, 255, 255, 255))
        composed = Image.alpha_composite(base_white, bg_faded)
        return Image.alpha_composite(composed, chart_rgba)

    white_canvas = Image.new("RGBA", chart_rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white_canvas, chart_rgba)


def run_sql_engine(engine_name, df_main, query_str):
    q = query_str.strip()
    if not q:
        return None, T("कृपया SQL क्वेरी लिखें।", "Please enter an SQL query.")
    if engine_name.startswith("DuckDB") and HAS_DUCKDB:
        con = duckdb.connect(database=":memory:")
        try:
            con.register("dataset", df_main)
            res = con.execute(q).fetchdf()
            return res, None
        except Exception as exc:
            return None, str(exc)
        finally:
            con.close()
    else:
        conn = sqlite3.connect(":memory:")
        try:
            df_copy = df_main.copy()
            for col in df_copy.columns:
                if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].astype(str)
            df_copy.to_sql("dataset", conn, index=False, if_exists="replace")
            res = pd.read_sql_query(q, conn)
            return res, None
        except Exception as exc:
            return None, str(exc)
        finally:
            conn.close()


# ==============================================================================
# SIDEBAR (ONE-TAP LANGUAGE SWITCH + MULTI-PLATFORM AI SETTINGS)
# ==============================================================================
st.sidebar.header(T("🌐 भाषा / Language (One-Tap)", "🌐 Language / भाषा (One-Tap)"))
lang_choice = st.sidebar.radio(
    T("इंटरफ़ेस व AI आउटपुट की भाषा:", "Interface & AI Output Language:"),
    ["Hindi", "English"],
    index=0 if st.session_state.ui_lang == "Hindi" else 1,
    horizontal=True,
    key="sidebar_lang_radio",
)
if lang_choice != st.session_state.ui_lang:
    st.session_state.ui_lang = lang_choice
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header(T("🤖 I'm AI Buddy — सेटिंग्स", "🤖 I'm AI Buddy — Settings"))

selected_provider = st.sidebar.selectbox(
    T("AI Platform चुनें:", "Select AI Platform:"),
    ["Google Gemini", "ChatGPT (OpenAI)", "Groq", "OpenRouter", "Local Ollama"],
    key="ai_provider_sel",
)

gemini_key = ""
gemini_model = "gemini-2.5-flash"
openai_key = ""
openai_model = "gpt-4o-mini"
groq_key = ""
groq_model = "openai/gpt-oss-120b"
openrouter_key = ""
openrouter_model = "openrouter/free"
ollama_mdl_name = "qwen2.5:3b"
ollama_url = "http://localhost:11434"

if selected_provider == "Google Gemini":
    gemini_key = st.sidebar.text_input(
        T("Google Gemini API Key डालें:", "Enter Google Gemini API Key:"), type="password", key="gem_api_key_in"
    ).strip()
    gemini_model = st.sidebar.selectbox(
        T("Gemini Model चुनें:", "Select Gemini Model:"),
        ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-3.8-flash"],
        key="gem_model_sel",
    )
elif selected_provider == "ChatGPT (OpenAI)":
    openai_key = st.sidebar.text_input(
        T("OpenAI (ChatGPT) API Key डालें:", "Enter OpenAI (ChatGPT) API Key:"), type="password", key="oai_api_key_in"
    ).strip()
    openai_model = st.sidebar.selectbox(
        T("OpenAI Model चुनें:", "Select OpenAI Model:"),
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        key="oai_model_sel",
    )
elif selected_provider == "Groq":
    groq_key = st.sidebar.text_input(
        T("Groq API Key डालें:", "Enter Groq API Key:"), type="password", key="groq_api_key_in"
    ).strip()
    groq_preset = st.sidebar.selectbox(
        T("Groq Model चुनें:", "Select Groq Model:"),
        ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "Custom Model Name"],
        key="groq_model_sel",
    )
    if groq_preset == "Custom Model Name":
        groq_model = st.sidebar.text_input(
            T("Custom Groq Model ID लिखें:", "Enter Custom Groq Model ID:"),
            value="openai/gpt-oss-120b",
            key="groq_custom_mdl_in",
        ).strip()
    else:
        groq_model = groq_preset
elif selected_provider == "OpenRouter":
    openrouter_key = st.sidebar.text_input(
        T("OpenRouter API Key डालें:", "Enter OpenRouter API Key:"), type="password", key="or_api_key_in"
    ).strip()
    openrouter_model = st.sidebar.text_input("OpenRouter Model Name:", value="openrouter/free", key="or_model_in").strip()
else:
    ollama_mdl_name = st.sidebar.text_input("Ollama Model Name:", value="qwen2.5:3b", key="ol_mdl_in").strip()
    ollama_url = st.sidebar.text_input("Ollama URL:", value="http://localhost:11434", key="ol_url_in").strip()

st.sidebar.markdown("---")
st.sidebar.caption(T("अन्य उपलब्ध Keys (Optional Backup):", "Other Available Keys (Optional Backup):"))
with st.sidebar.expander(T("🔑 बाकी Platforms की Keys भी सेव करें", "🔑 Save Keys for Other Platforms"), expanded=False):
    if selected_provider != "Google Gemini":
        gemini_key = st.text_input("Gemini API Key:", value=gemini_key, type="password", key="bk_gem_key").strip()
    if selected_provider != "ChatGPT (OpenAI)":
        openai_key = st.text_input("OpenAI API Key:", value=openai_key, type="password", key="bk_oai_key").strip()
    if selected_provider != "Groq":
        groq_key = st.text_input("Groq API Key:", value=groq_key, type="password", key="bk_groq_key").strip()
    if selected_provider != "OpenRouter":
        openrouter_key = st.text_input("OpenRouter API Key:", value=openrouter_key, type="password", key="bk_or_key").strip()
    if selected_provider != "Local Ollama":
        ollama_mdl_name = st.text_input("Ollama Model:", value=ollama_mdl_name, key="bk_ol_mdl").strip()

use_ollama_fb = st.sidebar.checkbox(
    T("Local Ollama Fallback चालू रखें (केवल लोकल मशीन पर)", "Enable Local Ollama Fallback (Local machine only)"),
    value=(selected_provider == "Local Ollama"),
    key="ol_fb_chk",
)

ai_config = {
    "provider": selected_provider,
    "gemini_key": gemini_key,
    "gemini_model": gemini_model,
    "openai_key": openai_key,
    "openai_model": openai_model,
    "groq_key": groq_key,
    "groq_model": groq_model,
    "openrouter_key": openrouter_key,
    "openrouter_model": openrouter_model,
    "ollama_model": ollama_mdl_name,
    "ollama_url": ollama_url,
    "enable_ollama": use_ollama_fb,
}

if st.sidebar.button(T("🔄 मूल डेटा पर रीसेट करें", "🔄 Reset to Original Dataset")):
    if st.session_state.uni_raw_df is not None:
        st.session_state.uni_df = st.session_state.uni_raw_df.copy()
        st.session_state.ds_trained_bundle = None
        st.session_state.ds_cluster_res = None
        st.session_state.ds_feat_rank_df = None
        st.session_state.dl_mlp_bundle = None
        st.session_state.dl_seq_res = None
        st.session_state.dl_ae_res = None
        st.session_state.dl_ai_report = None
        st.session_state.nlp_sent_df = None
        st.session_state.nlp_kw_df = None
        st.session_state.nlp_clf_bundle = None
        st.session_state.ts_fc_df = None
        st.session_state.nlp_ai_report = None
        st.session_state.ds_action_log = []
        st.rerun()

# ==============================================================================
# TOP HEADER & SHARED FILE UPLOADER (CSV, XLSX, JPG, JPEG, PNG)
# ==============================================================================
st.title("🤖 I'm AI Buddy")
st.caption(
    T(
        f"सक्रिय AI इंजन: **{selected_provider}** — Complete Data Science Studio, Auto Image Processing, Data Cleaning, Feature Engineering, SQL Studio, Power BI AI & ML/Gen AI।",
        f"Active AI Engine: **{selected_provider}** — Complete Data Science Studio, Auto Image Processing, Data Cleaning, Feature Engineering, SQL Studio, Power BI AI & ML/Gen AI.",
    )
)

top_u1, top_u2 = st.columns([3, 1])
with top_u1:
    uploaded_file = st.file_uploader(
        T(
            "अपनी CSV, Excel (.xlsx) या Image (.jpg, .jpeg, .png) फ़ाइल अपलोड करें (Image अपलोड करते ही ऑटो-प्रोसेस होगी)",
            "Upload your CSV, Excel (.xlsx), or Image (.jpg, .jpeg, .png) file (Images auto-process on upload)",
        ),
        type=["csv", "xlsx", "jpg", "jpeg", "png"],
        key="uni_file_uploader",
    )

if uploaded_file is not None:
    ext_f = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
    if ext_f in ("jpg", "jpeg", "png"):
        if st.session_state.uni_img_name != uploaded_file.name:
            st.session_state.uni_img_name = uploaded_file.name
            loaded_img = load_uploaded_image(uploaded_file)
            st.session_state.uni_uploaded_img = loaded_img
            st.session_state.bi_bg_img = process_uploaded_image(
                loaded_img,
                mode=st.session_state.img_proc_mode,
                resize_w=st.session_state.img_resize_w,
                bright_f=st.session_state.img_bright,
                contrast_f=st.session_state.img_contrast,
            ).convert("RGBA")
            st.session_state.bi_bg_src = f"Auto-Processed Image ({uploaded_file.name})"
    else:
        if st.session_state.uni_loaded_file != uploaded_file.name:
            st.session_state.uni_loaded_file = uploaded_file.name
            loaded = load_uploaded_table(uploaded_file)
            st.session_state.uni_raw_df = loaded.copy()
            st.session_state.uni_df = loaded.copy()
            st.session_state.da_ai_summary = None
            st.session_state.sql_res_df = None
            st.session_state.bi_ai_review = None
            st.session_state.ds_ml_report = None
            st.session_state.ds_full_ai_report = None
            st.session_state.ds_trained_bundle = None
            st.session_state.ds_cluster_res = None
            st.session_state.ds_feat_rank_df = None
            st.session_state.ds_action_log = []
else:
    st.session_state.uni_loaded_file = None
    st.session_state.uni_raw_df = None
    st.session_state.uni_df = None
    if st.session_state.uni_uploaded_img is None:
        st.info(
            T(
                "📂 कृपया शुरू करने के लिए अपनी CSV, Excel या Image (JPG/PNG) फ़ाइल अपलोड करें। अभी कोई सैंपल डेटा लोड नहीं है।",
                "📂 Please upload your CSV, Excel, or Image (JPG/PNG) file to get started. No sample dataset is preloaded.",
            )
        )
        st.stop()

if st.session_state.uni_uploaded_img is not None and (st.session_state.uni_df is None or st.session_state.uni_df.empty):
    img_now = st.session_state.uni_uploaded_img
    proc_now = render_auto_image_pipeline_ui(img_now, st.session_state.uni_img_name or "image.png", key_prefix="solo_img")
    st.session_state.bi_bg_img = proc_now.convert("RGBA")
    st.info(
        T(
            "💡 Tabular Data Analysis, SQL, ML और Forecasting चलाने के लिए ऊपर CSV या Excel फ़ाइल भी अपलोड करें।",
            "💡 Upload a CSV or Excel file above to run Tabular Data Analysis, SQL, ML, and Forecasting.",
        )
    )
    st.stop()

active_df = st.session_state.uni_df
if active_df is None or active_df.empty:
    st.warning(T("⚠️ कृपया डेटा वाली CSV या Excel फ़ाइल अपलोड करें।", "⚠️ Please upload a non-empty CSV or Excel file."))
    st.stop()

with top_u2:
    st.metric(T("सक्रिय डेटा आकार", "Active Dataset Shape"), f"{active_df.shape[0]} × {active_df.shape[1]}")
    st.download_button(
        T("📥 वर्तमान डेटा (.csv)", "📥 Current Dataset (.csv)"),
        data=active_df.to_csv(index=False).encode("utf-8"),
        file_name="im_ai_buddy_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )

if st.session_state.uni_uploaded_img is not None:
    with st.expander(
        T(f"🖼️ ऑटो-प्रोसेस्ड इमेज: {st.session_state.uni_img_name}", f"🖼️ Auto-Processed Image: {st.session_state.uni_img_name}"),
        expanded=True,
    ):
        proc_with_df = render_auto_image_pipeline_ui(
            st.session_state.uni_uploaded_img,
            st.session_state.uni_img_name or "image.png",
            key_prefix="with_df_img",
        )
        st.session_state.bi_bg_img = proc_with_df.convert("RGBA")

# ==============================================================================
# TOP NAVIGATION TABS (6 TABS: 1 TO 6)
# ==============================================================================
tab_da, tab_sql, tab_pbi, tab_ds, tab_dl, tab_nlp = st.tabs([
    "📊 Data Analysis",
    "🗄️ SQL Studio",
    "📈 Power BI AI",
    "🧠 Data Science & Gen AI",
    "🧬 Deep Learning Lab",
    "💬 NLP & Forecasting",
])

# ------------------------------------------------------------------------------
# TAB 1: ADVANCE DATA ANALYSIS & CLEANING + USD TO INR CONVERTER + REPORT EXPORT
# ------------------------------------------------------------------------------
with tab_da:
    st.header(T("📊 Advance Data Analysis और डेटा क्लीनिंग", "📊 Advanced Data Analysis & Data Cleaning"))

    da_nums = active_df.select_dtypes(include="number").columns.tolist()
    da_cats = active_df.select_dtypes(exclude="number").columns.tolist()
    da_all = active_df.columns.tolist()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(T("कुल पंक्तियाँ", "Total Rows"), active_df.shape[0])
    m2.metric(T("कुल कॉलम", "Total Columns"), active_df.shape[1])
    m3.metric(T("खाली सेल (Nulls)", "Missing Cells (Nulls)"), int(active_df.isna().sum().sum()))
    m4.metric(T("डुप्लीकेट पंक्तियाँ", "Duplicate Rows"), int(active_df.duplicated().sum()))

    with st.expander(
        T("🧹 डेटा क्लीनिंग और कॉलम एडिटिंग (सभी Tabs पर लागू होगा)", "🧹 Data Cleaning & Column Editing (Applies across all tabs)"),
        expanded=False,
    ):
        c_clean1, c_clean2, c_clean3 = st.columns(3)
        with c_clean1:
            if st.button(T("🗑️ डुप्लीकेट पंक्तियाँ हटाएं", "🗑️ Drop Duplicate Rows"), key="da_drop_dups"):
                st.session_state.uni_df = active_df.drop_duplicates().reset_index(drop=True)
                log_ds_action("Data Analysis: Dropped duplicate rows")
                st.rerun()
            if st.button(T("✂️ टेक्स्ट से अतिरिक्त स्पेस हटाएं (Trim)", "✂️ Trim Whitespace from Text"), key="da_trim"):
                tmp = active_df.copy()
                for col in tmp.select_dtypes(include=["object", "string"]).columns:
                    tmp[col] = get_1d_series(tmp, col).apply(lambda x: x.strip() if isinstance(x, str) else x)
                st.session_state.uni_df = tmp
                log_ds_action("Data Analysis: Trimmed whitespace from text columns")
                st.rerun()

        with c_clean2:
            null_opts_hi = ["Numeric में Median, Text में 'Unknown'", "Null वाली पंक्तियाँ हटाएं"]
            null_opts_en = ["Fill Numeric with Median, Text with 'Unknown'", "Drop Rows with Nulls"]
            null_action = st.selectbox(
                T("Nulls संभालने का तरीका:", "Null Handling Method:"),
                null_opts_en if is_en() else null_opts_hi,
                key="da_null_sel",
            )
            if st.button(T("🩹 Nulls ठीक करें", "🩹 Fix Nulls"), key="da_fix_nulls"):
                tmp = active_df.copy()
                if null_action.startswith("Numeric") or null_action.startswith("Fill Numeric"):
                    for col in tmp.select_dtypes(include="number").columns:
                        s_n = pd.to_numeric(get_1d_series(tmp, col), errors="coerce")
                        tmp[col] = s_n.fillna(s_n.median())
                    for col in tmp.select_dtypes(exclude="number").columns:
                        tmp[col] = get_1d_series(tmp, col).fillna("Unknown")
                else:
                    tmp = tmp.dropna().reset_index(drop=True)
                st.session_state.uni_df = tmp
                log_ds_action(f"Data Analysis: Handled nulls ({null_action})")
                st.rerun()

        with c_clean3:
            ren_col = st.selectbox(T("कॉलम का नाम बदलें:", "Select Column to Rename:"), active_df.columns.tolist(), key="da_ren_col")
            ren_new = st.text_input(T("नया नाम:", "New Column Name:"), value=str(ren_col), key="da_ren_txt")
            if st.button(T("✏️ नाम अपडेट करें", "✏️ Update Name"), key="da_ren_btn"):
                if ren_new.strip() and ren_new.strip() != ren_col:
                    st.session_state.uni_df = active_df.rename(columns={ren_col: ren_new.strip()})
                    log_ds_action(f"Data Analysis: Renamed '{ren_col}' -> '{ren_new.strip()}'")
                    st.rerun()

    with st.expander(T("💱 USD Range कॉलम को Indian Rupee (INR) में बदलें", "💱 Convert USD Range Column to Indian Rupee (INR)"), expanded=False):
        c_fx1, c_fx2, c_fx3, c_fx4 = st.columns([1.4, 1.1, 1.2, 1.3])
        with c_fx1:
            none_lbl = "— Select —" if is_en() else "— चुनें —"
            fx_src_col = st.selectbox(T("USD कॉलम चुनें:", "Select USD Column:"), [none_lbl] + da_all, key="da_fx_col")
        with c_fx2:
            usd_inr_rate = st.number_input("1 USD = ₹ (Rate):", min_value=1.0, max_value=500.0, value=92.79, step=0.1, key="da_fx_rate")
        with c_fx3:
            range_opts = (
                ["Midpoint (Average)", "Lower Limit (Minimum)", "Upper Limit (Maximum)"]
                if is_en()
                else ["Midpoint (औसत)", "Lower Limit (न्यूनतम)", "Upper Limit (अधिकतम)"]
            )
            range_pick = st.selectbox(T("रेंज का कौन-सा मान लें?", "Range Value Extraction:"), range_opts, key="da_fx_mode")
        with c_fx4:
            unknown_as_zero = st.checkbox(T("Unknown को 0 मानें", "Treat Unknown as 0"), value=False, key="da_fx_unk")

        if fx_src_col != none_lbl:
            s_fx_raw = get_1d_series(active_df, fx_src_col)
            inr_vals = s_fx_raw.apply(lambda v: parse_usd_range_to_inr(v, usd_inr_rate, range_pick, unknown_as_zero))
            new_inr_col = f"{fx_src_col}_INR"

            prev_view = pd.DataFrame({fx_src_col: s_fx_raw.astype(str), new_inr_col: inr_vals}).drop_duplicates()
            prev_view["INR (K)"] = (prev_view[new_inr_col] / 1e3).round(2)
            prev_view["INR (Lakhs)"] = (prev_view[new_inr_col] / 1e5).round(2)
            st.dataframe(prev_view, use_container_width=True)

            if st.button(T("➕ नया INR कॉलम डेटासेट में जोड़ें", "➕ Add New INR Column to Dataset"), key="da_fx_add_btn"):
                tmp_df = active_df.copy()
                tmp_df[new_inr_col] = inr_vals.values
                st.session_state.uni_df = tmp_df
                log_ds_action(f"Data Analysis: Added INR column '{new_inr_col}'")
                st.rerun()

    st.subheader(T("🔎 इंटरएक्टिव फ़िल्टर और कॉलम एक्सप्लोरर", "🔎 Interactive Filter & Column Explorer"))
    f_col1, f_col2, f_col3 = st.columns([1.2, 1.4, 1.4])
    all_lbl = "All" if is_en() else "सभी"
    with f_col1:
        sel_cols_da = st.multiselect(T("देखने के लिए कॉलम चुनें:", "Select Columns to Display:"), da_all, default=da_all, key="da_show_cols")
    with f_col2:
        search_kw = st.text_input(
            T("कीवर्ड से पंक्तियाँ खोजें:", "Search Rows by Keyword:"),
            placeholder=T("किसी शब्द या नंबर से खोजें...", "Search by any word or number..."),
            key="da_search_kw",
        )
    with f_col3:
        if da_cats:
            flt_cat = st.selectbox(T("Category फ़िल्टर कॉलम:", "Category Filter Column:"), [all_lbl] + da_cats, key="da_flt_cat")
            if flt_cat != all_lbl:
                s_fcat = get_1d_series(active_df, flt_cat)
                cat_vals = sorted(s_fcat.dropna().astype(str).unique().tolist())
                flt_val = st.selectbox(T(f"{flt_cat} चुनें:", f"Select {flt_cat}:"), [all_lbl] + cat_vals, key="da_flt_val")
            else:
                flt_val = all_lbl
        else:
            flt_cat, flt_val = all_lbl, all_lbl

    da_view = active_df.copy()
    if flt_cat != all_lbl and flt_val != all_lbl:
        da_view = da_view[get_1d_series(da_view, flt_cat).astype(str) == flt_val]
    if search_kw.strip():
        kw_l = search_kw.strip().lower()
        mask = pd.Series(False, index=da_view.index)
        for c in da_view.columns:
            mask = mask | get_1d_series(da_view, c).astype(str).str.lower().str.contains(kw_l, na=False)
        da_view = da_view[mask]

    cols_to_show = sel_cols_da if sel_cols_da else da_all
    st.caption(
        T(
            f"फ़िल्टर परिणाम: **{len(da_view)}** पंक्तियाँ × **{len(cols_to_show)}** कॉलम",
            f"Filtered Results: **{len(da_view)}** rows × **{len(cols_to_show)}** columns",
        )
    )
    st.dataframe(da_view[cols_to_show].head(100), use_container_width=True)

    st.subheader(T("🩺 कॉलम स्वास्थ्य (Missing / Unique) और Numeric विवरण", "🩺 Column Health (Missing / Unique) & Numeric Summary"))
    st.dataframe(build_all_columns_profile(da_view), use_container_width=True)
    if da_nums:
        st.dataframe(da_view[da_nums].describe().round(2), use_container_width=True)

    st.subheader(T("📐 GroupBy विश्लेषण और चार्ट", "📐 GroupBy Analysis & Chart"))
    if da_all:
        gb_c1, gb_c2, gb_c3, gb_c4 = st.columns(4)
        gb_by = gb_c1.selectbox(T("Group By कॉलम:", "Group By Column:"), da_all, index=0, key="da_gb_by")
        val_choices = da_nums if da_nums else da_all
        gb_val = gb_c2.selectbox(T("Value कॉलम:", "Value Column:"), val_choices, index=0, key="da_gb_val")
        gb_fn = gb_c3.selectbox("Function:", ["sum", "mean", "count", "max", "min"], key="da_gb_fn")
        gb_ch = gb_c4.selectbox(T("Chart प्रकार:", "Chart Type:"), ["Bar Chart", "Line Chart"], key="da_gb_ch")

        s_grp_1d = get_1d_series(da_view, gb_by)
        s_val_1d = get_1d_series(da_view, gb_val)
        gb_temp = pd.DataFrame({
            "Group_Col": s_grp_1d.astype(str),
            "Val_Col": pd.to_numeric(s_val_1d, errors="coerce"),
        }).dropna(subset=["Val_Col"])

        if not gb_temp.empty:
            out_col_name = f"{gb_fn}_{gb_val}"
            grp_tbl = (
                gb_temp.groupby("Group_Col")["Val_Col"]
                .agg(gb_fn)
                .reset_index()
                .sort_values(by="Val_Col", ascending=False)
            )
            grp_tbl.columns = [str(gb_by), out_col_name]

            g_left, g_right = st.columns([1.1, 1.3])
            with g_left:
                st.dataframe(grp_tbl, use_container_width=True)
            with g_right:
                fig_gb, ax_gb = plt.subplots(figsize=(6.2, 3.8), dpi=120)
                plot_top = grp_tbl.head(10)
                if gb_ch == "Bar Chart":
                    ax_gb.bar(plot_top[str(gb_by)], plot_top[out_col_name], color="#2563EB", alpha=0.85, edgecolor="white")
                else:
                    ax_gb.plot(plot_top[str(gb_by)], plot_top[out_col_name], marker="o", linewidth=2.2, color="#0F766E")
                ax_gb.set_title(f"{gb_fn.upper()}({gb_val}) by {gb_by}", fontsize=10.5, fontweight="bold")
                plt.xticks(rotation=30, ha="right")
                ax_gb.grid(True, linestyle="--", alpha=0.3)
                ax_gb.spines["top"].set_visible(False)
                ax_gb.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_gb, use_container_width=True)
                plt.close(fig_gb)
        else:
            st.info(
                T(
                    f"'{gb_val}' कॉलम में Numeric डेटा नहीं मिला; 'count' चुनकर या दूसरा कॉलम चुनकर देखें।",
                    f"No numeric data found in '{gb_val}'; try selecting 'count' or another column.",
                )
            )

    if st.button(T("🤖 AI से Data Analysis रिपोर्ट जनरेट करें", "🤖 Generate AI Data Analysis Report"), key="da_ai_btn"):
        local_rep = build_local_stats_summary(da_view)
        if is_en():
            p_da = (
                "You are I'm AI Buddy — a senior Data Analyst & Personal Decision Advisor. Write a clear, concise summary report in English "
                "covering:\n"
                "1. **Key Findings & Patterns**\n"
                "2. **Data Quality & Cleaning Notes**\n"
                "3. **What Matters Most for Me & What I Should Do Next (Practical Action Plan)** — clearly tell the user what is most important for their work/decisions from this result and exact next steps to take.\n\n"
                + local_rep
            )
        else:
            p_da = (
                "तुम I'm AI Buddy हो — एक सीनियर डेटा एनालिस्ट और प्रैक्टिकल एडवाइज़र। इस डेटासेट के लिए सरल हिंदी में संक्षिप्त रिपोर्ट दो जिसमें ये 3 हिस्से हों:\n"
                "1. **मुख्य बातें और पैटर्न (Key Findings)**\n"
                "2. **डेटा क्वालिटी और सुधार (Data Quality Notes)**\n"
                "3. **इस डेटा से मेरे लिए क्या ज़रूरी है और मुझे आगे क्या करना चाहिए (Next Steps & Useful Takeaways)** — साफ-साफ बताओ कि इस रिज़ल्ट में मेरे काम की सबसे ज़रूरी चीज़ क्या है और अगला कदम क्या होना चाहिए।\n\n"
                + local_rep
            )
        with st.spinner(T(f"{selected_provider} से एनालिसिस रिपोर्ट बन रही है...", f"Generating analysis report via {selected_provider}...")):
            ans, err = call_ai_provider(ai_config, p_da)
        st.session_state.da_ai_summary = ans if ans else f"{local_rep}\n\n*(AI Note: {err})*"

    if st.session_state.da_ai_summary:
        st.markdown(st.session_state.da_ai_summary)

    st.divider()
    report_sections = [
        "# I'm AI Buddy — Data Analysis Report",
        f"- **Dataset Shape:** {da_view.shape[0]} rows × {da_view.shape[1]} columns",
        f"- **Active Columns:** {', '.join(map(str, cols_to_show))}",
        "",
        "## Statistical Summary",
        build_local_stats_summary(da_view),
    ]
    if st.session_state.da_ai_summary:
        report_sections.extend(["", "## AI Analysis Insights", st.session_state.da_ai_summary])

    st.download_button(
        T("📄 पूरी Analysis रिपोर्ट डाउनलोड करें (.md)", "📄 Download Full Analysis Report (.md)"),
        data="\n".join(report_sections).encode("utf-8"),
        file_name="im_ai_buddy_analysis_report.md",
        mime="text/markdown",
        key="da_dl_report_btn",
    )

# ------------------------------------------------------------------------------
# TAB 2: SQL STUDIO
# ------------------------------------------------------------------------------
with tab_sql:
    st.header(T("🗄️ SQL एनालिस्ट स्टूडियो (टेबल नाम: `dataset`)", "🗄️ SQL Analyst Studio (Table Name: `dataset`)"))

    sql_cats = active_df.select_dtypes(exclude="number").columns.tolist()
    sql_nums = active_df.select_dtypes(include="number").columns.tolist()
    fc = sql_cats[0] if sql_cats else active_df.columns[0]
    fn = sql_nums[0] if sql_nums else active_df.columns[0]
    analysis_queries_map = build_sql_analysis_library(fc, fn)

    eng_opts = ["DuckDB (Window + CTE)", "SQLite3 (Built-in)"] if HAS_DUCKDB else ["SQLite3 (Built-in)"]
    s_top1, s_top2 = st.columns([2, 3])
    with s_top1:
        chosen_engine = st.selectbox(T("SQL इंजन:", "SQL Engine:"), eng_opts, key="sql_eng_sel")
    with s_top2:
        nl_sql = st.text_input(
            T("हिंदी/इंग्लिश में सवाल लिखें → AI SQL बनाएगा:", "Ask in English/Hindi → AI will write SQL:"),
            placeholder=T(
                "जैसे: हर Category का कुल Revenue और औसत निकालो",
                "e.g., Calculate total revenue and average per category",
            ),
            key="sql_nl_box",
        )
        if st.button(T("✨ AI से SQL लिखवाएं", "✨ Generate SQL with AI"), key="sql_gen_btn") and nl_sql.strip():
            cols_desc = ", ".join([f"{c} ({str(active_df[c].dtype)})" for c in active_df.columns])
            p_sql = (
                f"Write ONLY a valid SQL query for table `dataset` with columns: {cols_desc}.\n"
                f"User request: {nl_sql.strip()}\nNo markdown fences, no explanation."
            )
            with st.spinner(T(f"{selected_provider} से SQL तैयार हो रहा है...", f"Generating SQL via {selected_provider}...")):
                q_gen, q_err = call_ai_provider(ai_config, p_sql)
            if q_gen:
                q_clean = re.sub(r"\n\n\*\(Provider:.*?\)\*$", "", q_gen.strip())
                q_clean = re.sub(r"^```(?:sql)?\s*", "", q_clean, flags=re.I)
                q_clean = re.sub(r"\s*```$", "", q_clean).strip()
                set_sql_editor_query(q_clean)
                st.session_state.sql_ai_msg = T(
                    f"{selected_provider} ने SQL क्वेरी नीचे एडिटर में भर दी है।",
                    f"{selected_provider} has populated the SQL query in the editor below.",
                )
                st.rerun()
            else:
                st.warning(T(f"AI SQL जनरेट नहीं हो सका: {q_err}", f"Could not generate AI SQL: {q_err}"))

    if st.session_state.sql_ai_msg:
        st.info(st.session_state.sql_ai_msg)

    st.subheader(T("📚 सभी Analysis SQL Queries (बॉक्स से चुनें)", "📚 All Analysis SQL Queries (Select from Box)"))
    q_box_c1, q_box_c2 = st.columns([3, 1])
    with q_box_c1:
        selected_q_name = st.selectbox(
            T("तैयार Analysis SQL Query चुनें:", "Select Ready-to-Run SQL Query:"),
            list(analysis_queries_map.keys()),
            key="sql_all_queries_box",
        )
    with q_box_c2:
        st.write("")
        if st.button(T("📥 क्वेरी एडिटर में डालें", "📥 Load into Editor"), key="sql_load_box_btn"):
            set_sql_editor_query(analysis_queries_map[selected_q_name])
            st.rerun()

    if st.session_state.sql_last_preset != selected_q_name:
        st.session_state.sql_last_preset = selected_q_name
        set_sql_editor_query(analysis_queries_map[selected_q_name])
        st.rerun()

    st.text_area(T("SQL क्वेरी एडिटर:", "SQL Query Editor:"), height=175, key="sql_text_area", on_change=_sync_sql_from_area)

    if st.button(T("▶️ SQL क्वेरी चलाएं", "▶️ Run SQL Query"), type="primary", key="sql_run_btn"):
        r_df, r_err = run_sql_engine(chosen_engine, active_df, st.session_state.sql_text_area)
        st.session_state.sql_res_df = r_df
        st.session_state.sql_err_msg = r_err

    if st.session_state.sql_err_msg:
        st.error(f"SQL Error: {st.session_state.sql_err_msg}")

    if st.session_state.sql_res_df is not None:
        out_sql = st.session_state.sql_res_df
        st.success(
            T(
                f"परिणाम: {len(out_sql)} पंक्तियाँ × {len(out_sql.columns)} कॉलम",
                f"Result: {len(out_sql)} rows × {len(out_sql.columns)} columns",
            )
        )
        st.dataframe(out_sql, use_container_width=True)
        st.download_button(
            T("📥 SQL रिजल्ट डाउनलोड करें (.csv)", "📥 Download SQL Result (.csv)"),
            data=out_sql.to_csv(index=False).encode("utf-8"),
            file_name="im_ai_buddy_sql_output.csv",
            mime="text/csv",
            key="sql_dl_btn",
        )

# ------------------------------------------------------------------------------
# TAB 3: POWER BI AI STUDIO
# ------------------------------------------------------------------------------
with tab_pbi:
    st.header(T("📈 Power BI AI डैशबोर्ड स्टूडियो (9 Charts + AI Background)", "📈 Power BI AI Dashboard Studio (9 Charts + AI Background)"))

    if st.session_state.bi_bg_img is None:
        im, src = generate_ai_bg_image(
            gemini_key,
            "Minimalist soft pastel corporate geometric wave background, light cream and sky blue tones, no text",
        )
        st.session_state.bi_bg_img = im
        st.session_state.bi_bg_src = src

    p_cfg1, p_cfg2, p_cfg3, p_cfg4 = st.columns([1.2, 1.2, 1.2, 1.4])
    with p_cfg1:
        bi_chart_count = st.slider(T("चार्ट्स की संख्या:", "Number of Charts:"), 1, 9, 4, key="bi_num_charts")
    with p_cfg2:
        bi_top_n = st.slider(T("Top N कैटेगरी:", "Top N Categories:"), 3, 15, 7, key="bi_topn")
    with p_cfg3:
        bi_use_bg = st.checkbox(T("बैकग्राउंड इमेज दिखाएं", "Show Background Image"), value=True, key="bi_bg_chk")
        bi_opacity = st.slider(T("इमेज Opacity:", "Image Opacity:"), 0.06, 0.45, 0.16, 0.02, key="bi_opac")
    with p_cfg4:
        bi_bg_prompt = st.text_input(
            T("AI बैकग्राउंड थीम:", "AI Background Theme:"),
            value="Minimalist soft pastel geometric presentation background, no text",
            key="bi_bg_prm",
        )
        if st.button(T("🎨 नई AI बैकग्राउंड इमेज", "🎨 Generate New AI Background"), key="bi_new_bg_btn"):
            with st.spinner(T("बैकग्राउंड बन रहा है...", "Generating background...")):
                nim, nsrc = generate_ai_bg_image(gemini_key, bi_bg_prompt)
                st.session_state.bi_bg_img = nim
                st.session_state.bi_bg_src = nsrc
            st.rerun()

    with st.expander(T("✏️ Power BI डेटा सेल एडिट करें (Live Update)", "✏️ Edit Power BI Data Cells (Live Update)"), expanded=False):
        edited_bi_grid = st.data_editor(active_df, num_rows="dynamic", use_container_width=True, key="bi_grid_edit")
        if st.button(T("💾 एडिट किया हुआ डेटा सेव करें", "💾 Save Edited Data"), key="bi_save_grid"):
            st.session_state.uni_df = edited_bi_grid
            st.rerun()

    bi_nums = active_df.select_dtypes(include="number").columns.tolist()
    bi_cats = active_df.select_dtypes(exclude="number").columns.tolist()
    bi_all = active_df.columns.tolist()

    bi_view = active_df.copy()
    slicer_cols = [c for c in bi_cats if 1 < active_df[c].nunique(dropna=True) <= 50]
    if slicer_cols:
        st.subheader(T("🎛️ Slicers / फ़िल्टर्स", "🎛️ Slicers / Filters"))
        sc_boxes = st.columns(min(3, len(slicer_cols)))
        for i, sc in enumerate(slicer_cols[:3]):
            s_sl = get_1d_series(active_df, sc)
            u_opts = sorted(s_sl.dropna().astype(str).unique().tolist())
            sel_opts = sc_boxes[i].multiselect(f"{sc}:", u_opts, default=u_opts, key=f"pbi_sl_{sc}")
            if sel_opts:
                bi_view = bi_view[get_1d_series(bi_view, sc).astype(str).isin(sel_opts)]

    st.subheader("📌 KPI Cards")
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric(T("कुल रिकॉर्ड्स", "Total Records"), f"{len(bi_view):,}")
    if bi_nums:
        kpi_num = st.selectbox(T("KPI Numeric कॉलम:", "KPI Numeric Column:"), bi_nums, key="pbi_kpi_num")
        s_kv = pd.to_numeric(get_1d_series(bi_view, kpi_num), errors="coerce").dropna()
        kc2.metric(T(f"कुल {kpi_num}", f"Total {kpi_num}"), f"{s_kv.sum():,.2f}" if not s_kv.empty else "0")
        kc3.metric(T(f"औसत {kpi_num}", f"Avg {kpi_num}"), f"{s_kv.mean():,.2f}" if not s_kv.empty else "0")
    else:
        kpi_num = None
        kc2.metric(T("Numeric कॉलम", "Numeric Columns"), "0")
        kc3.metric(T("Text कॉलम", "Text Columns"), str(len(bi_cats)))

    if bi_cats:
        kpi_cat = st.selectbox(T("KPI Category कॉलम:", "KPI Category Column:"), bi_cats, key="pbi_kpi_cat")
        vc_k = get_1d_series(bi_view, kpi_cat).dropna().astype(str).value_counts()
        kc4.metric(f"Top {kpi_cat}", f"{vc_k.index[0]} ({vc_k.iloc[0]})" if not vc_k.empty else "N/A")
    else:
        kc4.metric("Nulls", str(int(bi_view.isna().sum().sum())))

    st.subheader(T(f"📊 डैशबोर्ड विज़ुअल्स ({bi_chart_count} Charts)", f"📊 Dashboard Visuals ({bi_chart_count} Charts)"))
    all_9_charts = ["Horizontal Bar", "Bar", "Stacked Bar", "Line", "Area", "Donut", "Scatter", "Histogram", "Box"]
    agg_choices = ["Sum", "Avg", "Count", "Max"]

    for r_start in range(0, bi_chart_count, 2):
        grid_cols = st.columns(2)
        for offset in range(2):
            idx_c = r_start + offset
            if idx_c >= bi_chart_count:
                break
            with grid_cols[offset]:
                with st.container(border=True):
                    st.markdown(f"**Chart #{idx_c + 1}**")
                    ch_type = st.selectbox("Chart Type:", all_9_charts, index=idx_c % len(all_9_charts), key=f"pbi_ct_{idx_c}")
                    cc1, cc2, cc3 = st.columns(3)
                    def_x = bi_cats[idx_c % len(bi_cats)] if bi_cats else bi_all[idx_c % len(bi_all)]
                    def_y = bi_nums[idx_c % len(bi_nums)] if bi_nums else bi_all[idx_c % len(bi_all)]
                    x_c = cc1.selectbox("X / Category:", bi_all, index=bi_all.index(def_x), key=f"pbi_x_{idx_c}")
                    y_c = cc2.selectbox("Y / Metric:", bi_all, index=bi_all.index(def_y), key=f"pbi_y_{idx_c}")
                    ag_c = cc3.selectbox("Aggregation:", agg_choices, index=0, key=f"pbi_ag_{idx_c}")

                    st_c = None
                    if ch_type == "Stacked Bar":
                        st_opts = [c for c in bi_all if c != x_c]
                        if st_opts:
                            st_c = st.selectbox(T("Stack / Legend कॉलम:", "Stack / Legend Column:"), st_opts, key=f"pbi_st_{idx_c}")

                    img_out = render_bi_chart_with_bg(
                        bi_view,
                        chart_type=ch_type,
                        x_col=x_c,
                        y_col=y_c,
                        stack_col=st_c,
                        agg_func=ag_c,
                        top_n=bi_top_n,
                        bg_img=st.session_state.bi_bg_img,
                        use_bg=bi_use_bg,
                        opacity_val=bi_opacity,
                    )
                    st.image(img_out, use_container_width=True)

    st.divider()
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        dax_req = st.text_input(
            T("DAX फ़ॉर्मूला बनवाएं:", "Generate DAX Formula:"),
            placeholder=T("जैसे: Total Revenue, Profit Margin %, या Rank", "e.g., Total Revenue, Profit Margin %, or Rank"),
            key="pbi_dax_in",
        )
        if st.button(T("📐 DAX जनरेट करें", "📐 Generate DAX"), key="pbi_dax_btn") and dax_req.strip():
            lang_inst = "Give short English explanation + DAX code." if is_en() else "Give short Hindi explanation + DAX code."
            p_d = f"Write a clean Power BI DAX measure for columns {bi_all} and requirement: {dax_req.strip()}. {lang_inst}"
            with st.spinner(T(f"{selected_provider} से DAX बन रहा है...", f"Generating DAX via {selected_provider}...")):
                dt, de = call_ai_provider(ai_config, p_d)
            st.session_state.bi_dax_code = (
                dt
                if dt
                else T(
                    f"DAX जनरेट नहीं हो सका ({de})।\nउदाहरण: `Total {kpi_num or 'Value'} = SUM('Table'[{kpi_num or 'Amount'}])`",
                    f"Could not generate DAX ({de}).\nExample: `Total {kpi_num or 'Value'} = SUM('Table'[{kpi_num or 'Amount'}])`",
                )
            )
        if st.session_state.bi_dax_code:
            st.markdown(st.session_state.bi_dax_code)

    with d_col2:
        if st.button(T("🧠 AI डैशबोर्ड सुझाव लें", "🧠 Get AI Dashboard Insights"), key="pbi_rev_btn"):
            if is_en():
                p_r = (
                    f"You are I'm AI Buddy — a Power BI expert & Decision Advisor. Write 4 key dashboard KPI insights in English "
                    f"and end with a dedicated section **What Matters Most & What I Should Do Next** explaining the most useful takeaway and exact next action from this dashboard: "
                    f"columns = {bi_all}, rows = {len(bi_view)}"
                )
            else:
                p_r = (
                    f"तुम I'm AI Buddy — एक Power BI विशेषज्ञ हो। इस डैशबोर्ड डेटा के लिए सरल हिंदी में "
                    f"4 मुख्य KPI निष्कर्ष लिखो और अंत में **इस डैशबोर्ड से मेरे लिए क्या ज़रूरी है और आगे क्या करना चाहिए** शीर्षक से मेरे काम की सबसे महत्वपूर्ण बात और अगला प्रैक्टिकल कदम बताओ: "
                    f"कॉलम = {bi_all}, पंक्तियाँ = {len(bi_view)}"
                )
            with st.spinner(T(f"{selected_provider} से BI रिव्यू तैयार हो रहा है...", f"Generating BI review via {selected_provider}...")):
                rt, re_msg = call_ai_provider(ai_config, p_r)
            st.session_state.bi_ai_review = (
                rt
                if rt
                else T(
                    f"लोकल BI नोट: कुल पंक्तियाँ = {len(bi_view)} ({re_msg})",
                    f"Local BI Note: Total rows = {len(bi_view)} ({re_msg})",
                )
            )
        if st.session_state.bi_ai_review:
            st.markdown(st.session_state.bi_ai_review)
            pbi_rep_md = "# I'm AI Buddy — Power BI AI Report\n\n" + st.session_state.bi_ai_review
            if st.session_state.bi_dax_code:
                pbi_rep_md += "\n\n## DAX Formula\n" + st.session_state.bi_dax_code
            st.download_button(
                T("📥 Power BI AI Report डाउनलोड करें (.md)", "📥 Download Power BI AI Report (.md)"),
                data=pbi_rep_md.encode("utf-8"),
                file_name="im_ai_buddy_powerbi_ai_report.md",
                mime="text/markdown",
                key="pbi_dl_report_btn",
            )

# ------------------------------------------------------------------------------
# TAB 4: COMPLETE DATA SCIENCE STUDIO + ALL FUNCTIONS BOX + ONE-CLICK AI REPORT
# ------------------------------------------------------------------------------
with tab_ds:
    st.header(
        T(
            "🧠 Complete Data Science Studio (Cleaning, Features, Algorithms & One-Click AI Report)",
            "🧠 Complete Data Science Studio (Cleaning, Features, Algorithms & One-Click AI Report)",
        )
    )

    ds_nums = active_df.select_dtypes(include="number").columns.tolist()
    ds_cats = active_df.select_dtypes(exclude="number").columns.tolist()
    ds_all = active_df.columns.tolist()

    with st.expander(
        T("📦 Data Science Function & Algorithm Box (सभी उपलब्ध टूल्स की सूची)", "📦 Data Science Function & Algorithm Box (All Available Tools)"),
        expanded=True,
    ):
        all_fn_lbl = "All Functions (Show All)" if is_en() else "All Functions (सभी एक साथ)"
        cat_box_sel = st.selectbox(
            T("Data Science Category Box चुनें:", "Select Data Science Category Box:"),
            [all_fn_lbl] + list(DS_FUNCTION_CATALOG.keys()),
            key="ds_func_cat_box",
        )
        if cat_box_sel == all_fn_lbl:
            f_cols = st.columns(2)
            items_list = list(DS_FUNCTION_CATALOG.items())
            for idx_cat, (grp_title, fn_list) in enumerate(items_list):
                with f_cols[idx_cat % 2]:
                    st.markdown(f"**{grp_title}**")
                    st.selectbox(
                        f"{grp_title} — Function/Algorithm List:",
                        fn_list,
                        key=f"ds_fn_list_{idx_cat}",
                    )
        else:
            chosen_fn_item = st.selectbox(
                T(f"{cat_box_sel} — उपलब्ध Functions / Algorithms:", f"{cat_box_sel} — Available Functions / Algorithms:"),
                DS_FUNCTION_CATALOG[cat_box_sel],
                key="ds_fn_single_box",
            )
            st.info(
                T(
                    f"चयनित टूल: **{chosen_fn_item}** — नीचे संबंधित सेक्शन में जाकर इसे सीधे डेटा पर लागू करें।",
                    f"Selected Tool: **{chosen_fn_item}** — Use the corresponding section below to apply it directly to your data.",
                )
            )

        if st.session_state.ds_action_log:
            st.markdown(T("**📝 Applied Transformation Log:**", "**📝 Applied Transformation Log:**"))
            for act_item in st.session_state.ds_action_log[-8:]:
                st.caption(f"• {act_item}")

    st.subheader(
        T(
            "🛠️ 1. Data Cleaning & Custom Column/Row Builder (जो चाहें जोड़ें या बदलें)",
            "🛠️ 1. Data Cleaning & Custom Column/Row Builder (Add or Modify Anything)",
        )
    )
    with st.expander(
        T(
            "🧹 Cleaning, Type Casting, Outlier Clipping & Custom Column/Formula",
            "🧹 Cleaning, Type Casting, Outlier Clipping & Custom Column/Formula",
        ),
        expanded=True,
    ):
        cl_t1, cl_t2, cl_t3 = st.columns(3)

        with cl_t1:
            st.markdown("**A. Missing Value Imputation & Cleaning**")
            imp_cols = st.multiselect(T("कॉलम्स चुनें:", "Select Columns:"), ds_all, default=ds_all[: min(3, len(ds_all))], key="ds_imp_cols")
            imp_method = st.selectbox(
                T("Imputation / Cleaning तरीका:", "Imputation / Cleaning Method:"),
                [
                    "Median (Numeric)",
                    "Mean (Numeric)",
                    "Mode / Most Frequent",
                    "Constant Value",
                    "KNN Imputer (Numeric)",
                    "Drop Missing Rows",
                    "Drop Duplicates",
                    "Trim Text Spaces",
                ],
                key="ds_imp_method",
            )
            const_val = st.text_input(T("Constant मान (यदि Constant चुना है):", "Constant Value (if selected):"), value="0", key="ds_const_val")
            if st.button(T("✅ Cleaning लागू करें", "✅ Apply Cleaning"), key="ds_apply_clean_btn"):
                tmp = active_df.copy()
                if imp_method == "Drop Duplicates":
                    tmp = tmp.drop_duplicates().reset_index(drop=True)
                elif imp_method == "Trim Text Spaces":
                    target_c = imp_cols if imp_cols else tmp.select_dtypes(include=["object", "string"]).columns.tolist()
                    for c in target_c:
                        tmp[c] = get_1d_series(tmp, c).apply(lambda x: x.strip() if isinstance(x, str) else x)
                elif imp_method == "Drop Missing Rows":
                    tmp = tmp.dropna(subset=imp_cols if imp_cols else None).reset_index(drop=True)
                elif imp_method == "KNN Imputer (Numeric)" and HAS_SKLEARN:
                    num_c = [c for c in (imp_cols or ds_nums) if pd.api.types.is_numeric_dtype(tmp[c])]
                    if num_c:
                        knn_imp = KNNImputer(n_neighbors=5)
                        tmp[num_c] = knn_imp.fit_transform(tmp[num_c])
                else:
                    target_c = imp_cols if imp_cols else ds_all
                    for c in target_c:
                        s_c = get_1d_series(tmp, c)
                        if imp_method == "Median (Numeric)" and pd.api.types.is_numeric_dtype(s_c):
                            s_num = pd.to_numeric(s_c, errors="coerce")
                            tmp[c] = s_num.fillna(s_num.median())
                        elif imp_method == "Mean (Numeric)" and pd.api.types.is_numeric_dtype(s_c):
                            s_num = pd.to_numeric(s_c, errors="coerce")
                            tmp[c] = s_num.fillna(s_num.mean())
                        elif imp_method == "Mode / Most Frequent":
                            mode_v = s_c.mode(dropna=True)
                            if not mode_v.empty:
                                tmp[c] = s_c.fillna(mode_v.iloc[0])
                        elif imp_method == "Constant Value":
                            tmp[c] = s_c.fillna(const_val)
                st.session_state.uni_df = tmp
                log_ds_action(f"Cleaning: {imp_method} on {imp_cols or 'all'}")
                st.rerun()

        with cl_t2:
            st.markdown("**B. Data Type Casting & Outlier Handling**")
            cast_col = st.selectbox(T("कॉलम चुनें:", "Select Column:"), ds_all, key="ds_cast_col")
            cast_type = st.selectbox(
                T("नया Data Type:", "New Data Type:"),
                ["Numeric (float/int)", "String / Text", "Datetime", "Category"],
                key="ds_cast_type",
            )
            if st.button(T("🔄 Type बदलें", "🔄 Convert Type"), key="ds_apply_cast_btn"):
                tmp = active_df.copy()
                s_c = get_1d_series(tmp, cast_col)
                if cast_type.startswith("Numeric"):
                    tmp[cast_col] = pd.to_numeric(s_c, errors="coerce")
                elif cast_type.startswith("String"):
                    tmp[cast_col] = s_c.astype(str)
                elif cast_type == "Datetime":
                    tmp[cast_col] = pd.to_datetime(s_c, errors="coerce")
                elif cast_type == "Category":
                    tmp[cast_col] = s_c.astype("category")
                st.session_state.uni_df = tmp
                log_ds_action(f"Type Cast: '{cast_col}' -> {cast_type}")
                st.rerun()

            if ds_nums:
                out_col = st.selectbox(T("Outlier कॉलम:", "Outlier Column:"), ds_nums, key="ds_out_col")
                out_opts = ["IQR Clip (Cap to Bounds)", "Remove IQR Outliers"] if is_en() else ["IQR Clip (सीमा में रखें)", "Remove IQR Outliers"]
                out_act = st.selectbox("Outlier Action:", out_opts, key="ds_out_act")
                if st.button(T("📉 Outlier ठीक करें", "📉 Handle Outliers"), key="ds_apply_out_btn"):
                    tmp = active_df.copy()
                    s_o = pd.to_numeric(get_1d_series(tmp, out_col), errors="coerce")
                    q1, q3 = s_o.quantile(0.25), s_o.quantile(0.75)
                    iqr = q3 - q1
                    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    if out_act.startswith("IQR Clip"):
                        tmp[out_col] = s_o.clip(lower=low, upper=high)
                    else:
                        tmp = tmp[(s_o >= low) & (s_o <= high)].reset_index(drop=True)
                    st.session_state.uni_df = tmp
                    log_ds_action(f"Outlier ({out_act}): '{out_col}'")
                    st.rerun()

        with cl_t3:
            st.markdown("**C. Add Custom Column / Formula / Drop Columns**")
            new_col_name = st.text_input(
                T("नया कॉलम नाम:", "New Column Name:"),
                placeholder=T("जैसे: total_score या log_price", "e.g., total_score or log_price"),
                key="ds_new_col_name",
            )
            if len(ds_nums) >= 2:
                f_c1 = st.selectbox(T("कॉलम 1:", "Column 1:"), ds_nums, index=0, key="ds_f_c1")
                f_op = st.selectbox(T("ऑपरेटर:", "Operator:"), ["+", "-", "*", "/", "Percentage (c1/c2*100)", "Diff % ((c2-c1)/c1*100)"], key="ds_f_op")
                f_c2 = st.selectbox(T("कॉलम 2:", "Column 2:"), ds_nums, index=min(1, len(ds_nums) - 1), key="ds_f_c2")
                if st.button(T("➕ नया कॉलम जोड़ें", "➕ Add New Column"), key="ds_add_calc_btn") and new_col_name.strip():
                    tmp = active_df.copy()
                    a_s = pd.to_numeric(get_1d_series(tmp, f_c1), errors="coerce")
                    b_s = pd.to_numeric(get_1d_series(tmp, f_c2), errors="coerce")
                    if f_op == "+":
                        tmp[new_col_name.strip()] = a_s + b_s
                    elif f_op == "-":
                        tmp[new_col_name.strip()] = a_s - b_s
                    elif f_op == "*":
                        tmp[new_col_name.strip()] = a_s * b_s
                    elif f_op == "/":
                        tmp[new_col_name.strip()] = a_s / b_s.replace(0, np.nan)
                    elif f_op.startswith("Percentage"):
                        tmp[new_col_name.strip()] = (a_s / b_s.replace(0, np.nan)) * 100.0
                    else:
                        tmp[new_col_name.strip()] = ((b_s - a_s) / a_s.replace(0, np.nan)) * 100.0
                    st.session_state.uni_df = tmp
                    log_ds_action(f"Added Column '{new_col_name.strip()}' ({f_c1} {f_op} {f_c2})")
                    st.rerun()

            drop_c_sel = st.multiselect(T("हटाने योग्य कॉलम्स (Drop Columns):", "Columns to Drop:"), ds_all, key="ds_drop_cols_sel")
            if st.button(T("🗑️ चुने हुए कॉलम हटाएं", "🗑️ Drop Selected Columns"), key="ds_drop_cols_btn") and drop_c_sel:
                keep_c = [c for c in ds_all if c not in drop_c_sel]
                if keep_c:
                    st.session_state.uni_df = active_df[keep_c].copy()
                    log_ds_action(f"Dropped Columns: {drop_c_sel}")
                    st.rerun()

    st.subheader("⚙️ 2. Feature Engineering, Scaling, Encoding & PCA")
    with st.expander("🧬 Transform Features (Log, Scaling, Encoding, Binning, Date Parts, PCA)", expanded=False):
        if HAS_SKLEARN:
            fe_c1, fe_c2, fe_c3 = st.columns(3)
            with fe_c1:
                st.markdown("**Numeric Transform / Scaling**")
                fe_num_col = st.selectbox(T("Numeric कॉलम:", "Numeric Column:"), ds_nums if ds_nums else ds_all, key="ds_fe_num_col")
                fe_num_op = st.selectbox(
                    "Transformation:",
                    [
                        "StandardScaler (z-score)",
                        "MinMaxScaler (0 to 1)",
                        "RobustScaler (IQR)",
                        "Log1p Transform (log(1+x))",
                        "PowerTransformer (Yeo-Johnson)",
                        "Quantile Bins (4 Quartiles)",
                    ],
                    key="ds_fe_num_op",
                )
                if st.button(T("⚡ Feature लागू करें", "⚡ Apply Feature Transform"), key="ds_fe_num_btn") and ds_nums:
                    tmp = active_df.copy()
                    s_v = pd.to_numeric(get_1d_series(tmp, fe_num_col), errors="coerce")
                    arr_2d = s_v.fillna(s_v.median()).to_numpy().reshape(-1, 1)
                    out_n = f"{fe_num_col}_fe"
                    if fe_num_op.startswith("StandardScaler"):
                        tmp[out_n] = StandardScaler().fit_transform(arr_2d).ravel()
                    elif fe_num_op.startswith("MinMaxScaler"):
                        tmp[out_n] = MinMaxScaler().fit_transform(arr_2d).ravel()
                    elif fe_num_op.startswith("RobustScaler"):
                        tmp[out_n] = RobustScaler().fit_transform(arr_2d).ravel()
                    elif fe_num_op.startswith("Log1p"):
                        tmp[out_n] = np.log1p(np.clip(s_v, a_min=0, a_max=None))
                    elif fe_num_op.startswith("PowerTransformer"):
                        tmp[out_n] = PowerTransformer().fit_transform(arr_2d).ravel()
                    elif fe_num_op.startswith("Quantile Bins"):
                        tmp[out_n] = pd.qcut(s_v, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop").astype(str)
                    st.session_state.uni_df = tmp
                    log_ds_action(f"Feature Eng: {fe_num_op} on '{fe_num_col}' -> '{out_n}'")
                    st.rerun()

            with fe_c2:
                st.markdown("**Categorical Encoding / Date Parts**")
                fe_cat_col = st.selectbox(T("Category/Date कॉलम:", "Category/Date Column:"), ds_cats if ds_cats else ds_all, key="ds_fe_cat_col")
                fe_cat_op = st.selectbox(
                    "Encoding / Extraction:",
                    ["Ordinal Codes (0, 1, 2...)", "Frequency Encoding (Count)", "Extract Year / Month / Day"],
                    key="ds_fe_cat_op",
                )
                if st.button(T("🏷️ Encode / Extract करें", "🏷️ Encode / Extract"), key="ds_fe_cat_btn"):
                    tmp = active_df.copy()
                    s_ct = get_1d_series(tmp, fe_cat_col)
                    if fe_cat_op.startswith("Ordinal"):
                        codes, _ = pd.factorize(s_ct.astype(str))
                        tmp[f"{fe_cat_col}_code"] = codes
                    elif fe_cat_op.startswith("Frequency"):
                        vc_map = s_ct.astype(str).value_counts()
                        tmp[f"{fe_cat_col}_freq"] = s_ct.astype(str).map(vc_map)
                    elif fe_cat_op.startswith("Extract"):
                        dt_s = pd.to_datetime(s_ct, errors="coerce")
                        if dt_s.notna().sum() > 0:
                            tmp[f"{fe_cat_col}_year"] = dt_s.dt.year
                            tmp[f"{fe_cat_col}_month"] = dt_s.dt.month
                            tmp[f"{fe_cat_col}_day"] = dt_s.dt.day
                    st.session_state.uni_df = tmp
                    log_ds_action(f"Feature Eng: {fe_cat_op} on '{fe_cat_col}'")
                    st.rerun()

            with fe_c3:
                st.markdown("**PCA Dimensionality Reduction**")
                if len(ds_nums) >= 2:
                    pca_cols = st.multiselect(T("PCA Numeric कॉलम्स:", "PCA Numeric Columns:"), ds_nums, default=ds_nums[: min(4, len(ds_nums))], key="ds_pca_cols")
                    n_comp = st.slider("Components:", 1, min(4, len(pca_cols) if pca_cols else 2), min(2, len(pca_cols) if pca_cols else 2), key="ds_pca_n")
                    if st.button(T("🧭 PCA जोड़ें", "🧭 Add PCA Features"), key="ds_pca_btn") and len(pca_cols) >= n_comp:
                        tmp = active_df.copy()
                        mat_p = tmp[pca_cols].apply(pd.to_numeric, errors="coerce")
                        valid_p = mat_p.notna().all(axis=1)
                        if valid_p.sum() >= n_comp:
                            sc_p = StandardScaler().fit_transform(mat_p.loc[valid_p])
                            pca_m = PCA(n_components=n_comp, random_state=42)
                            comps = pca_m.fit_transform(sc_p)
                            for i in range(n_comp):
                                col_p_name = f"PCA_{i + 1}"
                                tmp[col_p_name] = np.nan
                                tmp.loc[valid_p, col_p_name] = comps[:, i]
                            st.session_state.uni_df = tmp
                            ev_str = ", ".join([f"PC{i+1}: {r*100:.1f}%" for i, r in enumerate(pca_m.explained_variance_ratio_)])
                            log_ds_action(f"PCA Added ({ev_str})")
                            st.rerun()
                else:
                    st.info(T("PCA के लिए कम से कम 2 Numeric कॉलम चाहिए।", "At least 2 numeric columns are required for PCA."))

    st.subheader(T("🔮 3. Predictive Modeling Studio (सभी Supervised Algorithms)", "🔮 3. Predictive Modeling Studio (All Supervised Algorithms)"))
    if not HAS_SKLEARN:
        st.warning(
            T(
                "असली ML Prediction मॉडल चलाने के लिए `scikit-learn` इंस्टॉल करें: `pip install scikit-learn`",
                "Install `scikit-learn` to run ML prediction models: `pip install scikit-learn`",
            )
        )
    else:
        ml_c1, ml_c2, ml_c3, ml_c4, ml_c5 = st.columns([1.2, 1.2, 1.5, 0.9, 1.0])
        with ml_c1:
            ml_task = st.selectbox(T("Prediction प्रकार चुनें:", "Select Prediction Task:"), ["Regression (Numeric)", "Classification (Category)"], key="ds_ml_task")
        with ml_c2:
            if ml_task.startswith("Regression"):
                tgt_opts = ds_nums if ds_nums else ds_all
            else:
                tgt_opts = [c for c in ds_all if 1 < active_df[c].nunique(dropna=True) <= 25] or ds_all
            ml_target = st.selectbox(T("Target कॉलम:", "Target Column:"), tgt_opts, key="ds_ml_target")
        with ml_c3:
            safe_feat_cands = [
                c for c in ds_all
                if c != ml_target and not (not pd.api.types.is_numeric_dtype(active_df[c]) and active_df[c].nunique(dropna=True) > 200)
            ]
            feat_cands = safe_feat_cands if safe_feat_cands else [c for c in ds_all if c != ml_target]
            ml_features = st.multiselect(
                T("Feature कॉलम्स (X):", "Feature Columns (X):"),
                feat_cands,
                default=feat_cands[: min(5, len(feat_cands))],
                key="ds_ml_feats",
            )
        with ml_c4:
            ml_test_sz = st.slider("Test Size:", 0.15, 0.40, 0.25, 0.05, key="ds_ml_test")
        with ml_c5:
            ml_max_rows = st.number_input("Max Rows (RAM Safe):", min_value=1000, max_value=200000, value=35000, step=5000, key="ds_ml_max_rows")

        if ml_task.startswith("Regression"):
            avail_algos = list(get_regression_candidates().keys())
            def_algos = ["Linear Regression", "Random Forest Regressor", "Gradient Boosting Regressor"]
        else:
            avail_algos = list(get_classification_candidates().keys())
            def_algos = ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier"]

        chosen_algos = st.multiselect(
            T("ट्रेन करने के लिए Algorithms चुनें (List/Box):", "Select Algorithms to Train (List/Box):"),
            avail_algos,
            default=def_algos,
            key="ds_chosen_algos",
        )

        if st.button(T("🚀 चुने हुए मॉडल ट्रेन करें और तुलना देखें", "🚀 Train Selected Models & Compare"), type="primary", key="ds_train_btn"):
            if not ml_features:
                st.error(T("कृपया कम से कम 1 Feature कॉलम चुनें।", "Please select at least 1 feature column."))
            elif not chosen_algos:
                st.error(T("कृपया कम से कम 1 Algorithm चुनें।", "Please select at least 1 algorithm."))
            else:
                with st.spinner(T("मॉडल ट्रेन हो रहे हैं...", "Training models...")):
                    if ml_task.startswith("Regression"):
                        res_b, err_b = train_regression_suite(
                            active_df, ml_target, ml_features, chosen_algos, ml_test_sz, max_rows=int(ml_max_rows)
                        )
                    else:
                        res_b, err_b = train_classification_suite(
                            active_df, ml_target, ml_features, chosen_algos, ml_test_sz, max_rows=int(ml_max_rows)
                        )
                if err_b:
                    st.error(err_b)
                else:
                    st.session_state.ds_trained_bundle = res_b
                    log_ds_action(f"Trained {ml_task}: Best='{res_b['best_name']}' on target '{ml_target}'")

        bundle = st.session_state.ds_trained_bundle
        if bundle is not None:
            st.success(
                T(
                    f"सर्वश्रेष्ठ मॉडल चुना गया: **{bundle['best_name']}** (ट्रेन किया गया डेटा: `{bundle.get('trained_rows', len(active_df)):,}` पंक्तियाँ)",
                    f"Best Model Selected: **{bundle['best_name']}** (Trained on `{bundle.get('trained_rows', len(active_df)):,}` rows)",
                )
            )
            m_left, m_right = st.columns([1.15, 1.25])
            with m_left:
                st.markdown(T("**Model Leaderboard (तुलना)**", "**Model Leaderboard (Comparison)**"))
                st.dataframe(bundle["leaderboard"], use_container_width=True)
                if bundle.get("imp_df") is not None and not bundle["imp_df"].empty:
                    st.markdown("**Top Feature Importances**")
                    st.dataframe(bundle["imp_df"], use_container_width=True)

            with m_right:
                if bundle["task"] == "regression":
                    fig_p, ax_p = plt.subplots(figsize=(6.0, 3.8), dpi=120)
                    ax_p.scatter(bundle["y_test"], bundle["y_pred"], color="#2563EB", alpha=0.8, edgecolors="white")
                    mn_v = min(bundle["y_test"].min(), bundle["y_pred"].min())
                    mx_v = max(bundle["y_test"].max(), bundle["y_pred"].max())
                    ax_p.plot([mn_v, mx_v], [mn_v, mx_v], linestyle="--", color="#DC2626")
                    ax_p.set_xlabel(f"Actual {bundle['target']}")
                    ax_p.set_ylabel(f"Predicted {bundle['target']}")
                    ax_p.set_title("Actual vs Predicted", fontsize=10.5, fontweight="bold")
                    ax_p.grid(True, linestyle="--", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_p, use_container_width=True)
                    plt.close(fig_p)
                else:
                    cm_arr = bundle["cm"]
                    lbls = bundle["labels"]
                    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 3.8), dpi=120)
                    ax_cm.imshow(cm_arr, cmap="Blues")
                    ax_cm.set_xticks(np.arange(len(lbls)))
                    ax_cm.set_yticks(np.arange(len(lbls)))
                    ax_cm.set_xticklabels(lbls, rotation=25, ha="right", fontsize=8)
                    ax_cm.set_yticklabels(lbls, fontsize=8)
                    for i in range(cm_arr.shape[0]):
                        for j in range(cm_arr.shape[1]):
                            ax_cm.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color="#111827", fontweight="bold")
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    ax_cm.set_title("Confusion Matrix", fontsize=10.5, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig_cm, use_container_width=True)
                    plt.close(fig_cm)
                    if bundle.get("cls_report"):
                        with st.expander("📄 Classification Report (Precision / Recall / F1)", expanded=False):
                            st.text(bundle["cls_report"])

            st.markdown(T("#### 🎯 Live Prediction (नए इनपुट पर अनुमान निकालें)", "#### 🎯 Live Prediction (Predict on New Input)"))
            X_ref = bundle["X_ref"]
            inp_dict = {}
            pred_cols_ui = st.columns(min(3, len(bundle["features"])))
            for idx_f, fc_name in enumerate(bundle["features"]):
                ui_b = pred_cols_ui[idx_f % len(pred_cols_ui)]
                s_ref = X_ref[fc_name]
                if pd.api.types.is_numeric_dtype(s_ref):
                    med_v = float(pd.to_numeric(s_ref, errors="coerce").median())
                    if np.isnan(med_v):
                        med_v = 0.0
                    inp_dict[fc_name] = ui_b.number_input(f"{fc_name}:", value=med_v, key=f"pred_in_{fc_name}")
                else:
                    opts_c = sorted(s_ref.dropna().astype(str).unique().tolist()) or ["Unknown"]
                    inp_dict[fc_name] = ui_b.selectbox(f"{fc_name}:", opts_c, key=f"pred_sel_{fc_name}")

            if st.button(T("🔮 Predict करें", "🔮 Run Prediction"), key="ds_do_predict_btn"):
                one_row_df = pd.DataFrame([inp_dict])
                p_out = bundle["best_pipe"].predict(one_row_df)[0]
                if bundle["task"] == "regression":
                    val_f = float(p_out)
                    st.metric(
                        f"Predicted {bundle['target']}",
                        f"{val_f:,.2f}",
                        delta=f"K: {val_f/1e3:,.2f}K | L: {val_f/1e5:,.2f}L",
                    )
                else:
                    st.metric(f"Predicted {bundle['target']}", str(p_out))
                    if hasattr(bundle["best_pipe"].named_steps["model"], "predict_proba"):
                        probs = bundle["best_pipe"].predict_proba(one_row_df)[0]
                        cls_list = bundle["best_pipe"].named_steps["model"].classes_
                        prob_df = pd.DataFrame({
                            "Class": [str(c) for c in cls_list],
                            "Probability": [round(float(pr) * 100, 2) for pr in probs],
                        }).sort_values("Probability", ascending=False)
                        st.dataframe(prob_df, use_container_width=True)

        with st.expander("🧩 Clustering & Anomaly Detection (K-Means, Agglomerative, DBSCAN, Isolation Forest)", expanded=False):
            if len(ds_nums) >= 2:
                cl_c1, cl_c2, cl_c3 = st.columns([2, 1.2, 1])
                cl_cols = cl_c1.multiselect(T("Numeric कॉलम्स:", "Numeric Columns:"), ds_nums, default=ds_nums[: min(3, len(ds_nums))], key="ds_km_cols")
                cl_algo = cl_c2.selectbox(
                    T("Algorithm चुनें:", "Select Algorithm:"),
                    ["K-Means Clustering", "Agglomerative Clustering", "DBSCAN Clustering", "Isolation Forest (Anomaly)"],
                    key="ds_cl_algo",
                )
                k_val = cl_c3.slider("Clusters (K):", 2, 6, 3, key="ds_km_k")
                if st.button("🧩 Run Clustering / Anomaly", key="ds_km_btn") and len(cl_cols) >= 2:
                    km_sub = active_df[cl_cols].apply(pd.to_numeric, errors="coerce")
                    valid_df_km = km_sub[km_sub.notna().all(axis=1)]
                    if len(valid_df_km) > 15000:
                        valid_df_km = valid_df_km.sample(n=15000, random_state=42)
                    if len(valid_df_km) >= k_val:
                        scaled_arr = StandardScaler().fit_transform(valid_df_km)
                        if cl_algo == "K-Means Clustering":
                            labels_km = KMeans(n_clusters=k_val, random_state=42, n_init=10).fit_predict(scaled_arr)
                        elif cl_algo == "Agglomerative Clustering":
                            labels_km = AgglomerativeClustering(n_clusters=k_val).fit_predict(scaled_arr)
                        elif cl_algo == "DBSCAN Clustering":
                            labels_km = DBSCAN(eps=0.8, min_samples=5).fit_predict(scaled_arr)
                        else:
                            labels_km = IsolationForest(contamination=0.05, random_state=42).fit_predict(scaled_arr)

                        res_km = valid_df_km.copy()
                        res_km["Cluster_Or_Flag"] = labels_km
                        st.session_state.ds_cluster_res = res_km.groupby("Cluster_Or_Flag").agg(["count", "mean"]).round(2)
                        if len(set(labels_km)) > 1 and cl_algo != "Isolation Forest (Anomaly)":
                            try:
                                sil_v = silhouette_score(scaled_arr, labels_km)
                                st.caption(f"Silhouette Score: `{sil_v:.4f}`")
                            except Exception:
                                pass
                        log_ds_action(f"{cl_algo} on {cl_cols}")

                if st.session_state.ds_cluster_res is not None:
                    st.dataframe(st.session_state.ds_cluster_res, use_container_width=True)
            else:
                st.info(T("Clustering के लिए कम से कम 2 Numeric कॉलम चाहिए।", "At least 2 numeric columns are required for clustering."))

    st.divider()

    st.subheader(T("📑 4. सभी कॉलम्स की Data Science प्रोफ़ाइल और चार्ट्स", "📑 4. All-Columns Data Science Profile & Visuals"))
    st.dataframe(build_all_columns_profile(active_df), use_container_width=True)

    vc_row1_l, vc_row1_r = st.columns(2)
    with vc_row1_l:
        st.markdown(T("**1. सभी Numeric कॉलम्स — Correlation Matrix & Heatmap**", "**1. All Numeric Columns — Correlation Matrix & Heatmap**"))
        if len(ds_nums) >= 2:
            corr_df = active_df[ds_nums].corr().round(3)
            st.dataframe(corr_df, use_container_width=True)
            fig_hm = render_heatmap_fig(corr_df)
            st.pyplot(fig_hm, use_container_width=True)
            plt.close(fig_hm)
        else:
            st.info(T("Heatmap के लिए कम से कम 2 Numeric कॉलम चाहिए।", "At least 2 numeric columns are required for Heatmap."))

    with vc_row1_r:
        st.markdown(T("**2. किसी भी Numeric कॉलम का Distribution चार्ट**", "**2. Distribution Chart for Any Numeric Column**"))
        if ds_nums:
            dist_col_sel = st.selectbox(T("Distribution के लिए Numeric कॉलम चुनें:", "Select Numeric Column for Distribution:"), ds_nums, key="ds_dist_col_sel")
            fig_dist = render_dist_fig(get_1d_series(active_df, dist_col_sel), dist_col_sel)
            st.pyplot(fig_dist, use_container_width=True)
            plt.close(fig_dist)
        else:
            st.info(T("Distribution चार्ट के लिए Numeric कॉलम नहीं है।", "No numeric column available for distribution chart."))

    vc_row2_l, vc_row2_r = st.columns(2)
    with vc_row2_l:
        st.markdown(T("**3. Category / Text कॉलम का वितरण चार्ट**", "**3. Category / Text Column Distribution Chart**"))
        if ds_cats:
            cat_col_sel = st.selectbox(T("Category कॉलम चुनें:", "Select Category Column:"), ds_cats, key="ds_cat_col_sel")
            fig_cat = render_cat_bar_fig(get_1d_series(active_df, cat_col_sel), cat_col_sel)
            st.pyplot(fig_cat, use_container_width=True)
            plt.close(fig_cat)
        else:
            st.info(T("Category कॉलम नहीं है।", "No category column available."))

    with vc_row2_r:
        st.markdown(T("**4. दो कॉलम्स का Scatter / Relationship चार्ट**", "**4. Two-Column Scatter / Relationship Chart**"))
        if len(ds_all) >= 2:
            sc_x = st.selectbox(T("X-Axis कॉलम:", "X-Axis Column:"), ds_all, index=0, key="ds_sc_x")
            sc_y_def = 1 if len(ds_nums) == 0 else ds_all.index(ds_nums[0])
            sc_y = st.selectbox(T("Y-Axis कॉलम:", "Y-Axis Column:"), ds_all, index=sc_y_def, key="ds_sc_y")
            s_sc_x = get_1d_series(active_df, sc_x)
            s_sc_y = get_1d_series(active_df, sc_y)
            sc_pair = pd.DataFrame({
                "x": s_sc_x.astype(str) if not pd.api.types.is_numeric_dtype(s_sc_x) else pd.to_numeric(s_sc_x, errors="coerce"),
                "y": pd.to_numeric(s_sc_y, errors="coerce"),
            }).dropna()
            if len(sc_pair) > 4000:
                sc_pair = sc_pair.sample(n=4000, random_state=42)
            fig_sc, ax_sc = plt.subplots(figsize=(6.2, 4.2), dpi=120)
            ax_sc.scatter(sc_pair["x"], sc_pair["y"], color="#7C3AED", alpha=0.75, edgecolors="white")
            ax_sc.set_title(f"{sc_y} vs {sc_x}", fontsize=10.5, fontweight="bold")
            ax_sc.set_xlabel(str(sc_x))
            ax_sc.set_ylabel(str(sc_y))
            plt.xticks(rotation=30, ha="right")
            ax_sc.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_sc, use_container_width=True)
            plt.close(fig_sc)

    st.divider()

    st.subheader(T("🗺️ AI / ML Roadmap Generator (डेटा व लक्ष्य के अनुसार रोडमैप)", "🗺️ AI / ML Roadmap Generator (Tailored to Your Data & Goal)"))
    with st.expander("🧭 Data-Driven & Career/Project AI/ML Roadmap", expanded=False):
        rm_c1, rm_c2, rm_c3 = st.columns([1.4, 1.2, 1.0])
        with rm_c1:
            rm_goal = st.text_input(
                T("आपका मुख्य लक्ष्य / प्रोजेक्ट क्या है?", "What is your main goal / project?"),
                value="End-to-End Predictive Analytics & Business Decision System",
                key="ds_rm_goal_in",
            )
        with rm_c2:
            rm_level = st.selectbox(
                T("स्तर (Level) चुनें:", "Select Level:"),
                ["Beginner to Practical", "Intermediate (Feature Eng + Tuning)", "Production / Interview Ready"],
                key="ds_rm_level_sel",
            )
        with rm_c3:
            rm_weeks = st.slider(T("समयसीमा (Weeks):", "Timeline (Weeks):"), 2, 24, 6, key="ds_rm_weeks_sl")

        if st.button(T("🗺️ AI / ML Roadmap जनरेट करें", "🗺️ Generate AI / ML Roadmap"), type="primary", key="ds_gen_roadmap_btn"):
            cols_info = ", ".join([f"{c} ({str(active_df[c].dtype)})" for c in active_df.columns[:25]])
            trained_note = ""
            if st.session_state.ds_trained_bundle is not None:
                b_rm = st.session_state.ds_trained_bundle
                trained_note = f"\nCurrent Best Trained Model: {b_rm['best_name']} (Task: {b_rm['task']}, Target: {b_rm['target']})"

            if is_en():
                fallback_roadmap = (
                    f"### 🗺️ AI / ML Roadmap ({rm_weeks} Weeks — {rm_level})\n"
                    f"**Goal:** {rm_goal}\n\n"
                    f"**Dataset Context:** `{active_df.shape[0]}` rows × `{active_df.shape[1]}` columns ({cols_info}){trained_note}\n\n"
                    "1. **Phase 1 — Data Audit & Cleaning:** Impute missing values, remove duplicate rows, fix data types, and handle outliers.\n"
                    "2. **Phase 2 — EDA & Feature Engineering:** Inspect correlations & distributions, apply scaling (`StandardScaler`/`RobustScaler`), encoding, and create engineered/PCA features.\n"
                    "3. **Phase 3 — Baseline & Model Comparison:** Train Regression/Classification/Clustering models and compare metrics (`R2`/`MAE`/`RMSE` or `Accuracy`/`F1`).\n"
                    "4. **Phase 4 — Validation & Error Analysis:** Run cross-validation, inspect confusion matrix/residuals, and analyze feature importances.\n"
                    "5. **Phase 5 — Deployment & Reporting:** Test live single-row predictions, export the One-Click AI Report, and present your dashboard."
                )
                rm_prompt = (
                    f"You are I'm AI Buddy — a senior AI/ML Mentor and Data Scientist.\n"
                    f"Goal: {rm_goal}\nLevel: {rm_level}\nTimeline: {rm_weeks} Weeks\n"
                    f"Dataset: {active_df.shape[0]} rows × {active_df.shape[1]} columns | Columns: {cols_info}{trained_note}\n\n"
                    "Create a clear, step-by-step AI/ML Roadmap in English specifying each phase's objective, "
                    "algorithms/techniques to use, and concrete deliverables."
                )
            else:
                fallback_roadmap = (
                    f"### 🗺️ AI / ML Roadmap ({rm_weeks} Weeks — {rm_level})\n"
                    f"**लक्ष्य:** {rm_goal}\n\n"
                    f"**डेटासेट संदर्भ:** `{active_df.shape[0]}` पंक्तियाँ × `{active_df.shape[1]}` कॉलम ({cols_info}){trained_note}\n\n"
                    "1. **Phase 1 — Data Audit & Cleaning:** Missing values भरें, duplicate rows हटाएं, data types सही करें, और outliers संभालें।\n"
                    "2. **Phase 2 — EDA & Feature Engineering:** Correlation, distribution, scaling (`StandardScaler`/`RobustScaler`), encoding और नए features/PCA बनाएं।\n"
                    "3. **Phase 3 — Baseline & Model Comparison:** Regression/Classification/Clustering मॉडल्स ट्रेन करें और metrics (`R2`/`MAE`/`RMSE` या `Accuracy`/`F1`) की तुलना करें।\n"
                    "4. **Phase 4 — Validation & Error Analysis:** Cross-validation, confusion matrix/residuals और feature importance जांचें।\n"
                    "5. **Phase 5 — Deployment & Reporting:** Live prediction जांचें, One-Click AI Report निकालें, और dashboard/app के रूप में प्रस्तुत करें।"
                )
                rm_prompt = (
                    f"तुम I'm AI Buddy — एक सीनियर AI/ML Mentor और Data Scientist हो।\n"
                    f"लक्ष्य: {rm_goal}\nस्तर: {rm_level}\nसमयसीमा: {rm_weeks} Weeks\n"
                    f"डेटासेट: {active_df.shape[0]} rows × {active_df.shape[1]} columns | Columns: {cols_info}{trained_note}\n\n"
                    "सरल हिंदी में एक साफ़, चरणबद्ध (step-by-step) AI/ML Roadmap बनाओ जिसमें हर phase का उद्देश्य, "
                    "इस्तेमाल होने वाले algorithms/techniques, और deliverables स्पष्ट हों।"
                )

            with st.spinner(T(f"{selected_provider} से AI/ML Roadmap तैयार हो रहा है...", f"Generating AI/ML Roadmap via {selected_provider}...")):
                rm_ans, rm_err = call_ai_provider(ai_config, rm_prompt)

            if rm_ans:
                st.session_state.ds_aiml_roadmap = rm_ans
            else:
                st.session_state.ds_aiml_roadmap = f"{fallback_roadmap}\n\n*(AI Note: {rm_err})*"
            log_ds_action(f"Generated AI/ML Roadmap ({rm_weeks}w, {rm_level})")
            st.rerun()

        if st.session_state.ds_aiml_roadmap:
            st.markdown(st.session_state.ds_aiml_roadmap)
            st.download_button(
                T("📥 AI / ML Roadmap डाउनलोड करें (.md)", "📥 Download AI / ML Roadmap (.md)"),
                data=st.session_state.ds_aiml_roadmap.encode("utf-8"),
                file_name="im_ai_buddy_aiml_roadmap.md",
                mime="text/markdown",
                key="ds_dl_roadmap_btn",
            )

    st.divider()

    ds_left, ds_right = st.columns([1.05, 1.35])

    with ds_left:
        st.subheader(T("🎯 Outlier जांच (IQR) व SelectKBest Ranking", "🎯 Outlier Check (IQR) & SelectKBest Ranking"))
        if ds_nums:
            outlier_rows = []
            for col in ds_nums:
                s = pd.to_numeric(get_1d_series(active_df, col), errors="coerce").dropna()
                if len(s) >= 4:
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    cnt = int(((s < low) | (s > high)).sum())
                    outlier_rows.append({
                        "Column": col,
                        "Outliers Count": cnt,
                        "Normal Range": f"{low:,.2f} to {high:,.2f}" if is_en() else f"{low:,.2f} से {high:,.2f}",
                    })
            if outlier_rows:
                st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)

            if HAS_SKLEARN and len(ds_nums) >= 2:
                skb_tgt = st.selectbox("SelectKBest Target:", ds_nums, key="ds_skb_tgt")
                skb_feats = [c for c in ds_nums if c != skb_tgt]
                if skb_feats and st.button(T("📊 Feature Score निकालें", "📊 Compute Feature Scores"), key="ds_skb_btn"):
                    sub_skb = active_df[skb_feats + [skb_tgt]].apply(pd.to_numeric, errors="coerce").dropna()
                    if len(sub_skb) >= 10:
                        sel_k = SelectKBest(score_func=f_regression, k="all").fit(sub_skb[skb_feats], sub_skb[skb_tgt])
                        st.session_state.ds_feat_rank_df = (
                            pd.DataFrame({"Feature": skb_feats, "F_Score": np.round(sel_k.scores_, 2), "P_Value": np.round(sel_k.pvalues_, 4)})
                            .sort_values("F_Score", ascending=False)
                        )
                if st.session_state.ds_feat_rank_df is not None:
                    st.dataframe(st.session_state.ds_feat_rank_df, use_container_width=True)
        else:
            st.info(T("Outlier जांच के लिए Numeric कॉलम नहीं है।", "No numeric column available for outlier detection."))

    with ds_right:
        st.subheader(T("📋 One-Click AI Report (पूरे डेटा का उत्तर)", "📋 One-Click AI Report (Complete Dataset Summary)"))
        st.caption(
            T(
                "एक क्लिक में पूरे डेटासेट की मुख्य बातें, पैटर्न, कॉलम विश्लेषण, और ML निष्कर्ष तैयार करें।",
                "Generate key dataset insights, patterns, column analysis, and ML takeaways in one click.",
            )
        )

        r_btn1, r_btn2 = st.columns([1.6, 1.0])
        with r_btn1:
            gen_one_click = st.button(
                T("⚡ One-Click AI Report जनरेट करें", "⚡ Generate One-Click AI Report"),
                type="primary",
                key="ds_one_click_report_btn",
                use_container_width=True,
            )
        with r_btn2:
            if st.button(T("🗑️ रिपोर्ट साफ़ करें", "🗑️ Clear Report"), key="ds_clear_report_btn", use_container_width=True):
                st.session_state.ds_full_ai_report = None
                st.rerun()

        if gen_one_click:
            local_summary = build_local_stats_summary(active_df)
            top_cats_lines = []
            for c in ds_cats[:5]:
                vc_top = get_1d_series(active_df, c).dropna().astype(str).value_counts().head(3)
                if not vc_top.empty:
                    top_cats_lines.append(f"- {c}: " + ", ".join([f"{k} ({v})" for k, v in vc_top.items()]))
            cats_block = "\n".join(top_cats_lines) if top_cats_lines else T("कोई Category कॉलम नहीं", "No category columns")

            ml_info_line = ""
            if st.session_state.ds_trained_bundle is not None:
                b_now = st.session_state.ds_trained_bundle
                ml_info_line = f"\nTrained Best Model: {b_now['best_name']} (Target: {b_now['target']})"

            cols_brief = ", ".join([f"{c} ({str(active_df[c].dtype)})" for c in active_df.columns[:25]])
            if len(active_df.columns) > 25:
                cols_brief += f" ... (+{len(active_df.columns) - 25} more cols)"
            sample_compact = active_df.iloc[:4, : min(10, len(active_df.columns))].to_string(index=False)
            if len(sample_compact) > 900:
                sample_compact = sample_compact[:900] + "..."

            if is_en():
                one_click_prompt = (
                    "You are I'm AI Buddy — a senior Data Scientist, Analyst, and Practical Decision Advisor. Based on the dataset summary below, "
                    "write a clear, well-structured, concise AI Report in English containing these 5 sections:\n"
                    "1. **Executive Overview**\n"
                    "2. **Key Numbers & Top Categories**\n"
                    "3. **Data Quality & Missing/Outlier Notes**\n"
                    "4. **ML & Analytical Insights**\n"
                    "5. **What Matters Most for Me & What I Should Do Next (My Action Plan)** — clearly highlight the most important findings for the user's real-world decisions/work, what is most useful in this result, and exact step-by-step actions to take next.\n\n"
                    f"Columns: {cols_brief}\n"
                    f"{local_summary[:1800]}\nTop Categories:\n{cats_block[:900]}{ml_info_line}\n"
                    f"Sample Rows:\n{sample_compact}"
                )
            else:
                one_click_prompt = (
                    "तुम I'm AI Buddy — एक सीनियर Data Scientist, Analyst और प्रैक्टिकल Decision Advisor हो। नीचे दिए गए "
                    "डेटासेट की जानकारी के आधार पर स्पष्ट, व्यवस्थित और संक्षिप्त AI Report सरल हिंदी में दो। "
                    "रिपोर्ट में ये 5 हिस्से शामिल करो:\n"
                    "1. **Executive Overview (डेटा का सार)**\n"
                    "2. **Key Numbers & Top Categories (मुख्य आंकड़े)**\n"
                    "3. **Data Quality & Missing/Outlier Notes (डेटा क्वालिटी व सुधार)**\n"
                    "4. **ML & Analytical Insights (मॉडल व ट्रेंड निष्कर्ष)**\n"
                    "5. **इस डेटा से मेरे लिए क्या ज़रूरी है और मुझे आगे क्या करना चाहिए (What Matters Most & Next Steps)** — रिज़ल्ट के बाद साफ-साफ बताओ कि मेरे काम की सबसे ज़रूरी चीज़ क्या है, किस चीज़ पर तुरंत ध्यान देना है, और मेरा अगला कदम (Action Plan) क्या होना चाहिए।\n\n"
                    f"Columns: {cols_brief}\n"
                    f"{local_summary[:1800]}\nTop Categories:\n{cats_block[:900]}{ml_info_line}\n"
                    f"Sample Rows:\n{sample_compact}"
                )

            with st.spinner(T(f"{selected_provider} से One-Click AI Report बन रही है...", f"Generating One-Click AI Report via {selected_provider}...")):
                rep_ans, rep_err = call_ai_provider(ai_config, one_click_prompt)

            if rep_ans:
                st.session_state.ds_full_ai_report = rep_ans
            else:
                st.session_state.ds_full_ai_report = (
                    f"### {T('लोकल डेटा रिपोर्ट', 'Local Data Report')} (AI Unavailable: {rep_err})\n\n"
                    f"{local_summary}\n\n**Top Categories:**\n{cats_block}"
                )
            st.rerun()

        if st.session_state.ds_full_ai_report:
            st.markdown(st.session_state.ds_full_ai_report)
            st.download_button(
                T("📥 AI Report डाउनलोड करें (.md)", "📥 Download AI Report (.md)"),
                data=st.session_state.ds_full_ai_report.encode("utf-8"),
                file_name="im_ai_buddy_ds_ai_report.md",
                mime="text/markdown",
                key="ds_dl_one_click_btn",
            )


# ------------------------------------------------------------------------------
# TAB 5: DEEP LEARNING LAB (NEURAL NETWORKS, 1D-CNN/RNN SEQUENCE & AUTOENCODER)
# ------------------------------------------------------------------------------
with tab_dl:
    st.header(
        T(
            "🧬 Deep Learning Lab (Deep MLP Neural Network, 1D-CNN / Sequence Net & Autoencoder)",
            "🧬 Deep Learning Lab (Deep MLP Neural Network, 1D-CNN / Sequence Net & Autoencoder)",
        )
    )
    st.caption(
        T(
            "यहाँ Deep Learning के सभी मुख्य आर्किटेक्चर (Multi-Layer Neural Network, 1D-CNN/RNN-style Sliding Sequence Neural Net, और Neural Autoencoder Anomaly Detector) दिए गए हैं।",
            "Includes core Deep Learning architectures: Multi-Layer Neural Network, 1D-CNN/RNN-style Sliding Sequence Neural Net, and Neural Autoencoder Anomaly Detector.",
        )
    )

    dl_nums = active_df.select_dtypes(include="number").columns.tolist()
    dl_all = active_df.columns.tolist()

    st.subheader(T("1. 🧠 Custom Deep Neural Network Builder (Multi-Layer Perceptron)", "1. 🧠 Custom Deep Neural Network Builder (Multi-Layer Perceptron)"))
    if not HAS_SKLEARN:
        st.warning(T("Deep Learning Lab के लिए `scikit-learn` आवश्यक है।", "`scikit-learn` is required for Deep Learning Lab."))
    else:
        with st.expander(T("⚙️ Neural Network Architecture & Hyperparameters कॉन्फ़िगर करें", "⚙️ Configure Neural Network Architecture & Hyperparameters"), expanded=True):
            dn1, dn2, dn3, dn4 = st.columns([1.2, 1.2, 1.5, 1.1])
            with dn1:
                dl_task = st.selectbox(T("Task चुनें:", "Select Task:"), ["Regression (Numeric)", "Classification (Category)"], key="dl_nn_task")
            with dn2:
                if dl_task.startswith("Regression"):
                    dl_tgt_opts = dl_nums if dl_nums else dl_all
                else:
                    dl_tgt_opts = [c for c in dl_all if 1 < active_df[c].nunique(dropna=True) <= 25] or dl_all
                dl_target = st.selectbox(T("Target कॉलम:", "Target Column:"), dl_tgt_opts, key="dl_nn_target")
            with dn3:
                dl_safe_feats = [
                    c for c in dl_all
                    if c != dl_target and not (not pd.api.types.is_numeric_dtype(active_df[c]) and active_df[c].nunique(dropna=True) > 200)
                ]
                dl_feat_cands = dl_safe_feats if dl_safe_feats else [c for c in dl_all if c != dl_target]
                dl_features = st.multiselect(
                    T("Input Features (X):", "Input Features (X):"),
                    dl_feat_cands,
                    default=dl_feat_cands[: min(6, len(dl_feat_cands))],
                    key="dl_nn_feats",
                )
            with dn4:
                dl_arch = st.selectbox(
                    T("Hidden Layers आर्किटेक्चर:", "Hidden Layers Architecture:"),
                    ["(64, 32)", "(128, 64)", "(128, 64, 32)", "(256, 128, 64)", "(64, 64, 32, 16)"],
                    index=2,
                    key="dl_nn_arch",
                )

            hp1, hp2, hp3, hp4 = st.columns(4)
            dl_act = hp1.selectbox("Activation Function:", ["relu", "tanh", "logistic"], index=0, key="dl_nn_act")
            dl_solver = hp2.selectbox("Optimizer:", ["adam", "sgd", "lbfgs"], index=0, key="dl_nn_opt")
            dl_lr = hp3.select_slider("Learning Rate:", options=[0.0005, 0.001, 0.005, 0.01, 0.05], value=0.001, key="dl_nn_lr")
            dl_epochs = hp4.slider("Max Epochs (Iterations):", 100, 1000, 350, 50, key="dl_nn_epochs")

            if st.button(T("🚀 Deep Neural Network ट्रेन करें", "🚀 Train Deep Neural Network"), type="primary", key="dl_train_nn_btn"):
                if not dl_features:
                    st.error(T("कम से कम 1 Feature चुनें।", "Please select at least 1 feature."))
                else:
                    layers_tuple = tuple(int(x.strip()) for x in dl_arch.strip("()").split(",") if x.strip())
                    work_dl = prepare_ml_subset(active_df, dl_features + [dl_target], max_rows=30000)
                    with st.spinner(T("Deep Neural Network ट्रेन हो रहा है...", "Training Deep Neural Network...")):
                        if dl_task.startswith("Regression"):
                            work_dl[dl_target] = pd.to_numeric(work_dl[dl_target], errors="coerce")
                            work_dl = work_dl.dropna(subset=[dl_target])
                            if len(work_dl) < 8:
                                st.error(T("कम से कम 8 वैध पंक्तियाँ चाहिए।", "At least 8 valid rows are required."))
                            else:
                                X_d = work_dl[dl_features]
                                y_d = work_dl[dl_target].astype(np.float32)
                                X_tr, X_te, y_tr, y_te = train_test_split(X_d, y_d, test_size=0.25, random_state=42)
                                mlp_m = MLPRegressor(
                                    hidden_layer_sizes=layers_tuple,
                                    activation=dl_act,
                                    solver=dl_solver,
                                    learning_rate_init=dl_lr,
                                    max_iter=dl_epochs,
                                    random_state=42,
                                )
                                pipe_dl = Pipeline([("prep", build_preprocessor(X_d)), ("model", mlp_m)])
                                pipe_dl.fit(X_tr, y_tr)
                                preds_d = pipe_dl.predict(X_te)
                                st.session_state.dl_mlp_bundle = {
                                    "task": "regression",
                                    "arch": dl_arch,
                                    "act": dl_act,
                                    "opt": dl_solver,
                                    "target": dl_target,
                                    "features": dl_features,
                                    "pipe": pipe_dl,
                                    "loss_curve": getattr(pipe_dl.named_steps["model"], "loss_curve_", []),
                                    "metrics": {
                                        "R2 Score": round(float(r2_score(y_te, preds_d)), 4),
                                        "MAE": round(float(mean_absolute_error(y_te, preds_d)), 2),
                                        "RMSE": round(float(np.sqrt(mean_squared_error(y_te, preds_d))), 2),
                                    },
                                    "y_test": y_te.reset_index(drop=True),
                                    "y_pred": pd.Series(preds_d),
                                    "X_ref": X_d,
                                }
                                log_ds_action(f"Deep Learning MLP {dl_arch} trained on '{dl_target}'")
                        else:
                            work_dl[dl_target] = work_dl[dl_target].astype(str).str.strip()
                            work_dl = work_dl[work_dl[dl_target] != ""]
                            vc_d = work_dl[dl_target].value_counts()
                            work_dl = work_dl[work_dl[dl_target].isin(vc_d[vc_d >= 2].index)]
                            if len(work_dl) < 8 or work_dl[dl_target].nunique() < 2:
                                st.error(T("कम से कम 2 Classes और 8 पंक्तियाँ चाहिए।", "At least 2 classes and 8 rows are required."))
                            else:
                                X_d = work_dl[dl_features]
                                y_d = work_dl[dl_target]
                                strat_d = y_d if y_d.value_counts().min() >= 2 else None
                                X_tr, X_te, y_tr, y_te = train_test_split(X_d, y_d, test_size=0.25, random_state=42, stratify=strat_d)
                                mlp_c = MLPClassifier(
                                    hidden_layer_sizes=layers_tuple,
                                    activation=dl_act,
                                    solver=dl_solver,
                                    learning_rate_init=dl_lr,
                                    max_iter=dl_epochs,
                                    random_state=42,
                                )
                                pipe_dl = Pipeline([("prep", build_preprocessor(X_d)), ("model", mlp_c)])
                                pipe_dl.fit(X_tr, y_tr)
                                preds_d = pipe_dl.predict(X_te)
                                st.session_state.dl_mlp_bundle = {
                                    "task": "classification",
                                    "arch": dl_arch,
                                    "act": dl_act,
                                    "opt": dl_solver,
                                    "target": dl_target,
                                    "features": dl_features,
                                    "pipe": pipe_dl,
                                    "loss_curve": getattr(pipe_dl.named_steps["model"], "loss_curve_", []),
                                    "metrics": {
                                        "Accuracy %": f"{accuracy_score(y_te, preds_d) * 100:.2f}%",
                                        "Precision": round(float(precision_score(y_te, preds_d, average="weighted", zero_division=0)), 4),
                                        "Recall": round(float(recall_score(y_te, preds_d, average="weighted", zero_division=0)), 4),
                                        "F1 Score": round(float(f1_score(y_te, preds_d, average="weighted", zero_division=0)), 4),
                                    },
                                    "X_ref": X_d,
                                }
                                log_ds_action(f"Deep Learning MLP {dl_arch} Classifier trained on '{dl_target}'")

            dl_b = st.session_state.dl_mlp_bundle
            if dl_b is not None:
                st.success(f"Neural Network `{dl_b['arch']}` ({dl_b['act']}, {dl_b['opt']}) — Target: `{dl_b['target']}`")
                m_cols_dl = st.columns(len(dl_b["metrics"]))
                for idx_m, (mk, mv) in enumerate(dl_b["metrics"].items()):
                    m_cols_dl[idx_m].metric(mk, str(mv))

                lc_col, pred_col_dl = st.columns([1.15, 1.25])
                with lc_col:
                    loss_vals = dl_b.get("loss_curve") or []
                    if len(loss_vals) > 1:
                        fig_lc, ax_lc = plt.subplots(figsize=(5.8, 3.4), dpi=120)
                        ax_lc.plot(range(1, len(loss_vals) + 1), loss_vals, color="#7C3AED", linewidth=2.0)
                        ax_lc.set_title("Neural Network Training Loss Curve", fontsize=10.5, fontweight="bold")
                        ax_lc.set_xlabel("Epoch / Iteration")
                        ax_lc.set_ylabel("Training Loss")
                        ax_lc.grid(True, linestyle="--", alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_lc, use_container_width=True)
                        plt.close(fig_lc)
                    else:
                        st.info(T("Loss curve केवल iterative solvers (adam/sgd) पर उपलब्ध है।", "Loss curve is available for iterative solvers (adam/sgd)."))

                with pred_col_dl:
                    st.markdown(T("**🎯 Neural Network Live Prediction**", "**🎯 Neural Network Live Prediction**"))
                    X_dl_ref = dl_b["X_ref"]
                    dl_inp = {}
                    sub_ui_dl = st.columns(2)
                    for idx_f, fc_n in enumerate(dl_b["features"]):
                        box_u = sub_ui_dl[idx_f % 2]
                        s_r = X_dl_ref[fc_n]
                        if pd.api.types.is_numeric_dtype(s_r):
                            mv_def = float(pd.to_numeric(s_r, errors="coerce").median())
                            dl_inp[fc_n] = box_u.number_input(f"{fc_n}:", value=0.0 if np.isnan(mv_def) else mv_def, key=f"dl_p_{fc_n}")
                        else:
                            opts_dl = sorted(s_r.dropna().astype(str).unique().tolist()) or ["Unknown"]
                            dl_inp[fc_n] = box_u.selectbox(f"{fc_n}:", opts_dl, key=f"dl_s_{fc_n}")

                    if st.button(T("⚡ Neural Predict करें", "⚡ Run Neural Prediction"), key="dl_live_pred_btn"):
                        out_dl = dl_b["pipe"].predict(pd.DataFrame([dl_inp]))[0]
                        st.metric(f"Neural Prediction ({dl_b['target']})", f"{float(out_dl):,.2f}" if dl_b["task"] == "regression" else str(out_dl))

    st.divider()

    st.subheader(T("2. 🌊 1D-CNN / Sequence Sliding-Window Neural Network (Sequential Pattern Prediction)", "2. 🌊 1D-CNN / Sequence Sliding-Window Neural Network (Sequential Pattern Prediction)"))
    if HAS_SKLEARN and dl_nums:
        with st.expander(T("📈 Sequential / Lag-Window Neural Network चलाएं", "📈 Run Sequential / Lag-Window Neural Network"), expanded=False):
            sq1, sq2, sq3 = st.columns([1.4, 1.0, 1.0])
            seq_col = sq1.selectbox(T("Numeric Sequence कॉलम:", "Numeric Sequence Column:"), dl_nums, key="dl_seq_col")
            win_sz = sq2.slider(T("Window / Kernel Size (पिछले Steps):", "Window / Kernel Size (Past Steps):"), 2, 15, 5, key="dl_seq_win")
            fut_steps = sq3.slider(T("अगले कितने Steps Predict करें:", "Future Steps to Predict:"), 3, 30, 10, key="dl_seq_fut")

            if st.button(T("🌊 Sequence Neural Net चलाएं", "🌊 Run Sequence Neural Net"), key="dl_seq_btn"):
                s_seq = pd.to_numeric(get_1d_series(active_df, seq_col), errors="coerce").dropna().to_numpy()
                if len(s_seq) <= win_sz + 5:
                    st.error(T("Sequence बहुत छोटी है; Window Size कम करें या बड़ा डेटा दें।", "Sequence is too short; reduce Window Size or provide more rows."))
                else:
                    X_w, y_w = [], []
                    for i in range(len(s_seq) - win_sz):
                        X_w.append(s_seq[i : i + win_sz])
                        y_w.append(s_seq[i + win_sz])
                    X_w = np.array(X_w, dtype=np.float32)
                    y_w = np.array(y_w, dtype=np.float32)

                    sc_seq = StandardScaler()
                    X_w_sc = sc_seq.fit_transform(X_w)
                    seq_nn = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=400, random_state=42)
                    seq_nn.fit(X_w_sc, y_w)
                    in_sample_pred = seq_nn.predict(X_w_sc)
                    seq_r2 = float(r2_score(y_w, in_sample_pred))

                    curr_win = s_seq[-win_sz:].copy().astype(np.float32)
                    future_preds = []
                    for _ in range(fut_steps):
                        nxt_v = float(seq_nn.predict(sc_seq.transform(curr_win.reshape(1, -1)))[0])
                        future_preds.append(nxt_v)
                        curr_win = np.append(curr_win[1:], nxt_v)

                    st.session_state.dl_seq_res = {
                        "col": seq_col,
                        "win": win_sz,
                        "r2": round(seq_r2, 4),
                        "history": s_seq[-min(60, len(s_seq)) :],
                        "future": future_preds,
                    }
                    log_ds_action(f"Sequence Neural Net trained on '{seq_col}' (window={win_sz}, steps={fut_steps})")

            if st.session_state.dl_seq_res is not None:
                sr = st.session_state.dl_seq_res
                st.metric(f"Sequence Neural Fit (R2) — {sr['col']}", f"{sr['r2']:.4f}")
                fig_sq, ax_sq = plt.subplots(figsize=(7.0, 3.6), dpi=120)
                hist_y = sr["history"]
                fut_y = sr["future"]
                ax_sq.plot(range(len(hist_y)), hist_y, label="Historical Sequence", color="#2563EB", linewidth=2)
                ax_sq.plot(range(len(hist_y), len(hist_y) + len(fut_y)), fut_y, label="Neural Forecast", color="#DC2626", marker="o", linestyle="--", linewidth=2)
                ax_sq.set_title(f"1D-Sliding Window Neural Forecast — {sr['col']}", fontsize=10.5, fontweight="bold")
                ax_sq.legend()
                ax_sq.grid(True, linestyle="--", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_sq, use_container_width=True)
                plt.close(fig_sq)
                fc_tbl_dl = pd.DataFrame({"Future_Step": range(1, len(fut_y) + 1), f"Predicted_{sr['col']}": np.round(fut_y, 2)})
                st.dataframe(fc_tbl_dl, use_container_width=True)
    else:
        st.info(T("Sequence Neural Net के लिए Numeric कॉलम आवश्यक है।", "At least 1 numeric column is required for Sequence Neural Net."))

    st.divider()

    st.subheader(T("3. 🛡️ Neural Autoencoder Anomaly Detector (Reconstruction Error)", "3. 🛡️ Neural Autoencoder Anomaly Detector (Reconstruction Error)"))
    if HAS_SKLEARN and len(dl_nums) >= 2:
        with st.expander(T("🔍 Autoencoder Bottleneck से Unusual / Fraud Rows खोजें", "🔍 Detect Unusual / Fraud Rows via Autoencoder Bottleneck"), expanded=False):
            ae_c1, ae_c2, ae_c3 = st.columns([2, 1, 1])
            ae_cols = ae_c1.multiselect(T("Numeric Features चुनें:", "Select Numeric Features:"), dl_nums, default=dl_nums[: min(5, len(dl_nums))], key="dl_ae_cols")
            ae_latent = ae_c2.slider(T("Bottleneck Neurons:", "Bottleneck Neurons:"), 1, max(1, len(ae_cols) - 1 if len(ae_cols) > 1 else 1), 1, key="dl_ae_lat")
            ae_top_k = ae_c3.slider(T("Top Anomalies दिखाएं:", "Top Anomalies to Show:"), 5, 30, 10, key="dl_ae_topk")

            if st.button(T("🛡️ Neural Autoencoder चलाएं", "🛡️ Run Neural Autoencoder"), key="dl_ae_btn") and len(ae_cols) >= 2:
                ae_sub = active_df[ae_cols].apply(pd.to_numeric, errors="coerce")
                valid_ae = ae_sub.dropna()
                if len(valid_ae) > 20000:
                    valid_ae = valid_ae.sample(n=20000, random_state=42)
                if len(valid_ae) >= 10:
                    sc_ae = StandardScaler()
                    X_ae_sc = sc_ae.fit_transform(valid_ae)
                    mid_dim = max(2, len(ae_cols) * 2)
                    ae_net = MLPRegressor(
                        hidden_layer_sizes=(mid_dim, ae_latent, mid_dim),
                        activation="tanh",
                        max_iter=400,
                        random_state=42,
                    )
                    ae_net.fit(X_ae_sc, X_ae_sc)
                    recon = ae_net.predict(X_ae_sc)
                    mse_err = np.mean((X_ae_sc - recon) ** 2, axis=1)
                    out_ae = active_df.loc[valid_ae.index].copy()
                    out_ae["Autoencoder_Anomaly_Score"] = np.round(mse_err, 4)
                    st.session_state.dl_ae_res = out_ae.sort_values("Autoencoder_Anomaly_Score", ascending=False).head(ae_top_k)
                    log_ds_action(f"Neural Autoencoder detected top {ae_top_k} anomalies on {ae_cols}")

            if st.session_state.dl_ae_res is not None:
                st.dataframe(st.session_state.dl_ae_res, use_container_width=True)
    else:
        st.info(T("Neural Autoencoder के लिए कम से कम 2 Numeric कॉलम चाहिए।", "At least 2 numeric columns are required for Neural Autoencoder."))

    st.divider()
    if st.button(T("🤖 Deep Learning AI Report जनरेट करें", "🤖 Generate Deep Learning AI Report"), key="dl_ai_rep_btn"):
        dl_ctx_lines = [f"Dataset: {active_df.shape[0]} rows × {active_df.shape[1]} columns"]
        if st.session_state.dl_mlp_bundle:
            b_dl = st.session_state.dl_mlp_bundle
            dl_ctx_lines.append(f"Trained Neural Network: {b_dl['arch']} ({b_dl['act']}, {b_dl['opt']}) on target '{b_dl['target']}' -> Metrics: {b_dl['metrics']}")
        if st.session_state.dl_seq_res:
            s_dl = st.session_state.dl_seq_res
            dl_ctx_lines.append(f"Sequence Neural Net on '{s_dl['col']}' (window={s_dl['win']}, R2={s_dl['r2']}) -> Next forecast values: {np.round(s_dl['future'][:5], 2).tolist()}")
        if st.session_state.dl_ae_res is not None:
            dl_ctx_lines.append(f"Autoencoder Top Anomaly Score: {st.session_state.dl_ae_res['Autoencoder_Anomaly_Score'].iloc[0]}")

        dl_summary_txt = "\n".join(dl_ctx_lines)
        if is_en():
            p_dl = (
                "You are I'm AI Buddy — a senior Deep Learning Engineer & Practical Advisor. Based on these Deep Learning results, "
                "write a clear summary in English with:\n"
                "1. **Neural Model & Architecture Insights**\n"
                "2. **What Matters Most for Me & What I Should Do Next (Practical Action Plan)** — clearly tell the user what is most useful here and what action to take next.\n\n"
                + dl_summary_txt
            )
        else:
            p_dl = (
                "तुम I'm AI Buddy — एक सीनियर Deep Learning Engineer और प्रैक्टिकल एडवाइज़र हो। नीचे दिए गए Deep Learning रिज़ल्ट के आधार पर सरल हिंदी में रिपोर्ट लिखो:\n"
                "1. **Neural Model व आर्किटेक्चर के मुख्य निष्कर्ष**\n"
                "2. **इस रिज़ल्ट से मेरे लिए क्या ज़रूरी है और मुझे आगे क्या करना चाहिए (Next Steps & Useful Takeaways)** — साफ-साफ बताओ कि मेरे काम की सबसे ज़रूरी बात क्या है और अगला कदम क्या होना चाहिए।\n\n"
                + dl_summary_txt
            )
        with st.spinner(T(f"{selected_provider} से Deep Learning रिपोर्ट बन रही है...", f"Generating Deep Learning report via {selected_provider}...")):
            dl_ans, dl_err = call_ai_provider(ai_config, p_dl)
        st.session_state.dl_ai_report = dl_ans if dl_ans else f"### Deep Learning Summary\n{dl_summary_txt}\n\n*(AI Note: {dl_err})*"

    if st.session_state.dl_ai_report:
        st.markdown(st.session_state.dl_ai_report)
        st.download_button(
            T("📥 Deep Learning AI Report डाउनलोड करें (.md)", "📥 Download Deep Learning AI Report (.md)"),
            data=st.session_state.dl_ai_report.encode("utf-8"),
            file_name="im_ai_buddy_deep_learning_report.md",
            mime="text/markdown",
            key="dl_dl_report_btn",
        )


# ------------------------------------------------------------------------------
# TAB 6: NLP & TIME-SERIES FORECASTING (NON-DEEP-LEARNING / CLASSICAL & TEXT)
# ------------------------------------------------------------------------------
with tab_nlp:
    st.header(
        T(
            "💬 NLP Studio & Time-Series Forecasting (Text Analytics, Sentiment & Trend Models)",
            "💬 NLP Studio & Time-Series Forecasting (Text Analytics, Sentiment & Trend Models)",
        )
    )
    st.caption(
        T(
            "यहाँ Deep Learning से अलग वाले मॉडल्स और टूल्स हैं: Text/Reviews पर NLP (Sentiment Analysis, TF-IDF Keywords, Text Classification) और Date/Numeric डेटा पर Time-Series Forecasting।",
            "Contains non-deep-learning text & forecasting tools: NLP on Text/Reviews (Sentiment Analysis, TF-IDF Keywords, Text Classification) and Time-Series Forecasting.",
        )
    )

    nlp_cats = active_df.select_dtypes(exclude="number").columns.tolist()
    nlp_nums = active_df.select_dtypes(include="number").columns.tolist()
    nlp_all = active_df.columns.tolist()

    st.subheader(T("1. 📝 NLP Text Analyzer (Sentiment Analysis & TF-IDF Top Keywords)", "1. 📝 NLP Text Analyzer (Sentiment Analysis & TF-IDF Top Keywords)"))
    if nlp_cats:
        n_c1, n_c2, n_c3 = st.columns([1.5, 1.0, 1.0])
        txt_col_sel = n_c1.selectbox(T("Text / Review कॉलम चुनें:", "Select Text / Review Column:"), nlp_cats, key="nlp_txt_col")
        top_kw_n = n_c2.slider(T("Top Keywords संख्या:", "Top Keywords Count:"), 5, 25, 12, key="nlp_kw_n")
        ngram_opt = n_c3.selectbox("N-Gram Range:", ["Unigrams (1 word)", "Unigrams + Bigrams (1-2 words)"], key="nlp_ngram_sel")

        if st.button(T("🔍 NLP Analysis चलाएं (Sentiment + Keywords)", "🔍 Run NLP Analysis (Sentiment + Keywords)"), type="primary", key="nlp_run_btn"):
            s_txt = get_1d_series(active_df, txt_col_sel).fillna("").astype(str)
            pos_words = {
                "good", "great", "excellent", "amazing", "awesome", "love", "best", "happy", "fast",
                "nice", "perfect", "helpful", "super", "fantastic", "clean", "worth", "profit", "high", "success",
            }
            neg_words = {
                "bad", "poor", "worst", "terrible", "slow", "hate", "issue", "problem", "broken",
                "late", "waste", "disappointed", "wrong", "fail", "error", "loss", "low", "refund", "cancel",
            }

            sent_labels = []
            sent_scores = []
            for line_t in s_txt:
                tokens = re.findall(r"[a-zA-Z]{2,}", line_t.lower())
                p_cnt = sum(1 for w in tokens if w in pos_words)
                n_cnt = sum(1 for w in tokens if w in neg_words)
                sc_val = p_cnt - n_cnt
                sent_scores.append(sc_val)
                if sc_val > 0:
                    sent_labels.append("Positive")
                elif sc_val < 0:
                    sent_labels.append("Negative")
                else:
                    sent_labels.append("Neutral")

            res_sent = pd.DataFrame({
                txt_col_sel: s_txt,
                "Sentiment": sent_labels,
                "Score": sent_scores,
                "Word_Count": s_txt.apply(lambda x: len(x.split())),
            })
            st.session_state.nlp_sent_df = res_sent

            if HAS_SKLEARN:
                from sklearn.feature_extraction.text import TfidfVectorizer
                non_empty_docs = [t for t in s_txt.tolist() if len(t.strip()) > 1]
                if len(non_empty_docs) >= 2:
                    ng_r = (1, 2) if ngram_opt.startswith("Unigrams +") else (1, 1)
                    vec = TfidfVectorizer(stop_words="english", max_features=300, ngram_range=ng_r)
                    try:
                        mat_tf = vec.fit_transform(non_empty_docs)
                        scores_tf = np.asarray(mat_tf.mean(axis=0)).ravel()
                        terms_tf = vec.get_feature_names_out()
                        kw_df = (
                            pd.DataFrame({"Keyword / Phrase": terms_tf, "TFIDF_Score": np.round(scores_tf, 4)})
                            .sort_values("TFIDF_Score", ascending=False)
                            .head(top_kw_n)
                        )
                        st.session_state.nlp_kw_df = kw_df
                    except ValueError:
                        st.session_state.nlp_kw_df = None
            log_ds_action(f"NLP Sentiment & TF-IDF analyzed on '{txt_col_sel}'")

        if st.session_state.nlp_sent_df is not None:
            sdf = st.session_state.nlp_sent_df
            sc_vc = sdf["Sentiment"].value_counts()
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Positive", int(sc_vc.get("Positive", 0)))
            sm2.metric("Neutral", int(sc_vc.get("Neutral", 0)))
            sm3.metric("Negative", int(sc_vc.get("Negative", 0)))
            sm4.metric("Avg Word Count", f"{sdf['Word_Count'].mean():.1f}")

            nl_l, nl_r = st.columns([1.2, 1.0])
            with nl_l:
                st.dataframe(sdf.head(50), use_container_width=True)
            with nl_r:
                if st.session_state.nlp_kw_df is not None and not st.session_state.nlp_kw_df.empty:
                    st.markdown("**Top TF-IDF Keywords**")
                    st.dataframe(st.session_state.nlp_kw_df, use_container_width=True)
    else:
        st.info(T("NLP के लिए डेटासेट में कम से कम 1 Text/Category कॉलम होना चाहिए।", "At least 1 text/category column is required for NLP."))

    st.divider()

    st.subheader(T("2. 🏷️ Text Classification Model (TF-IDF + Logistic / Naive Bayes / Linear SVM)", "2. 🏷️ Text Classification Model (TF-IDF + Logistic / Naive Bayes / Linear SVM)"))
    if HAS_SKLEARN and len(nlp_cats) >= 2:
        with st.expander(T("📚 Text से Category Predict करने का मॉडल ट्रेन करें", "📚 Train a Model to Predict Category from Text"), expanded=False):
            from sklearn.feature_extraction.text import TfidfVectorizer

            tc1, tc2, tc3 = st.columns(3)
            tc_text_col = tc1.selectbox(T("Input Text कॉलम:", "Input Text Column:"), nlp_cats, index=0, key="nlp_tc_x")
            tc_tgt_opts = [c for c in nlp_cats if c != tc_text_col]
            tc_target_col = tc2.selectbox(T("Target Category कॉलम:", "Target Category Column:"), tc_tgt_opts, key="nlp_tc_y")
            tc_model_name = tc3.selectbox("NLP Classifier:", ["Logistic Regression", "Multinomial Naive Bayes"], key="nlp_tc_mdl")

            if st.button(T("🚀 Text Classifier ट्रेन करें", "🚀 Train Text Classifier"), key="nlp_tc_train_btn"):
                sub_tc = pd.DataFrame({
                    "txt": get_1d_series(active_df, tc_text_col).fillna("").astype(str).str.strip(),
                    "lbl": get_1d_series(active_df, tc_target_col).fillna("").astype(str).str.strip(),
                })
                sub_tc = sub_tc[(sub_tc["txt"] != "") & (sub_tc["lbl"] != "")]
                vc_tc = sub_tc["lbl"].value_counts()
                sub_tc = sub_tc[sub_tc["lbl"].isin(vc_tc[vc_tc >= 2].index)]
                if len(sub_tc) < 8 or sub_tc["lbl"].nunique() < 2:
                    st.error(T("कम से कम 2 Categories (प्रत्येक में ≥2 पंक्तियाँ) चाहिए।", "At least 2 categories (≥2 rows each) are required."))
                else:
                    X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(sub_tc["txt"], sub_tc["lbl"], test_size=0.25, random_state=42)
                    clf_obj = LogisticRegression(max_iter=500) if tc_model_name == "Logistic Regression" else MultinomialNB()
                    txt_pipe = Pipeline([
                        ("tfidf", TfidfVectorizer(stop_words="english", max_features=1500, ngram_range=(1, 2))),
                        ("clf", clf_obj),
                    ])
                    txt_pipe.fit(X_tr_t, y_tr_t)
                    acc_tc = accuracy_score(y_te_t, txt_pipe.predict(X_te_t))
                    st.session_state.nlp_clf_bundle = {
                        "pipe": txt_pipe,
                        "acc": round(float(acc_tc) * 100, 2),
                        "x_col": tc_text_col,
                        "y_col": tc_target_col,
                        "mdl": tc_model_name,
                    }
                    log_ds_action(f"NLP Text Classifier ({tc_model_name}) trained -> Acc: {acc_tc*100:.1f}%")

            if st.session_state.nlp_clf_bundle is not None:
                tb = st.session_state.nlp_clf_bundle
                st.success(f"{tb['mdl']} — Validation Accuracy: **{tb['acc']}%** (`{tb['x_col']}` → `{tb['y_col']}`)")
                sample_sentence = st.text_input(T("नया वाक्य लिखकर टेस्ट करें:", "Enter a new sentence to classify:"), key="nlp_live_txt_in")
                if st.button(T("🔮 Text Classify करें", "🔮 Classify Text"), key="nlp_live_txt_btn") and sample_sentence.strip():
                    pred_lbl = tb["pipe"].predict([sample_sentence.strip()])[0]
                    st.metric(f"Predicted {tb['y_col']}", str(pred_lbl))

    st.divider()

    st.subheader(T("3. 📅 Time-Series & Moving-Average Trend Forecasting", "3. 📅 Time-Series & Moving-Average Trend Forecasting"))
    if nlp_nums:
        with st.expander(T("📈 Date / Index के आधार पर Future Trend Forecast निकालें", "📈 Generate Future Trend Forecast by Date / Index"), expanded=True):
            ts1, ts2, ts3, ts4 = st.columns([1.2, 1.2, 1.0, 1.0])
            ts_date_col = ts1.selectbox(T("Date / Order कॉलम (Optional):", "Date / Order Column (Optional):"), ["Row Order (Index)"] + nlp_all, key="ts_date_col")
            ts_val_col = ts2.selectbox(T("Forecast Numeric कॉलम:", "Forecast Numeric Column:"), nlp_nums, key="ts_val_col")
            ts_ma_win = ts3.slider(T("Moving Average Window:", "Moving Average Window:"), 2, 20, 4, key="ts_ma_win")
            ts_horizon = ts4.slider(T("Future Steps:", "Future Steps:"), 3, 30, 8, key="ts_horizon")

            if st.button(T("📊 Trend Forecast चलाएं", "📊 Run Trend Forecast"), key="ts_run_btn"):
                y_series = pd.to_numeric(get_1d_series(active_df, ts_val_col), errors="coerce")
                work_ts = pd.DataFrame({"val": y_series})
                if ts_date_col != "Row Order (Index)":
                    work_ts["dt"] = pd.to_datetime(get_1d_series(active_df, ts_date_col), errors="coerce")
                    if work_ts["dt"].notna().sum() >= 5:
                        work_ts = work_ts.dropna(subset=["dt", "val"]).groupby("dt", as_index=False)["val"].sum().sort_values("dt")
                    else:
                        work_ts = work_ts.dropna(subset=["val"]).reset_index(drop=True)
                else:
                    work_ts = work_ts.dropna(subset=["val"]).reset_index(drop=True)

                if len(work_ts) < 5:
                    st.error(T("Forecasting के लिए कम से कम 5 वैध मान चाहिए।", "At least 5 valid values are required for forecasting."))
                else:
                    vals = work_ts["val"].to_numpy(dtype=float)
                    ma_vals = pd.Series(vals).rolling(window=min(ts_ma_win, len(vals)), min_periods=1).mean().to_numpy()
                    x_idx = np.arange(len(vals), dtype=float)
                    slope, intercept = np.polyfit(x_idx, vals, 1)
                    fut_x = np.arange(len(vals), len(vals) + ts_horizon, dtype=float)
                    fut_trend = intercept + slope * fut_x
                    st.session_state.ts_fc_df = {
                        "col": ts_val_col,
                        "vals": vals[-min(80, len(vals)) :],
                        "ma": ma_vals[-min(80, len(ma_vals)) :],
                        "fut": fut_trend,
                        "slope": round(float(slope), 4),
                    }
                    log_ds_action(f"Time-Series Trend Forecast run on '{ts_val_col}' ({ts_horizon} steps)")

            if st.session_state.ts_fc_df is not None:
                tfd = st.session_state.ts_fc_df
                st.caption(f"Trend Slope per Step: `{tfd['slope']:+.4f}`")
                fig_ts, ax_ts = plt.subplots(figsize=(7.0, 3.6), dpi=120)
                hv = tfd["vals"]
                mv = tfd["ma"]
                fv = tfd["fut"]
                ax_ts.plot(range(len(hv)), hv, label="Actual", color="#2563EB", alpha=0.75)
                ax_ts.plot(range(len(mv)), mv, label="Moving Average", color="#059669", linewidth=2.0)
                ax_ts.plot(range(len(hv), len(hv) + len(fv)), fv, label="Linear Trend Forecast", color="#D97706", linestyle="--", marker="o")
                ax_ts.set_title(f"Time-Series Moving Average & Trend Forecast — {tfd['col']}", fontsize=10.5, fontweight="bold")
                ax_ts.legend()
                ax_ts.grid(True, linestyle="--", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_ts, use_container_width=True)
                plt.close(fig_ts)
                st.dataframe(
                    pd.DataFrame({"Future_Step": range(1, len(fv) + 1), f"Forecast_{tfd['col']}": np.round(fv, 2)}),
                    use_container_width=True,
                )

    st.divider()
    if st.button(T("🤖 NLP & Forecasting AI Report जनरेट करें", "🤖 Generate NLP & Forecasting AI Report"), key="nlp_ai_rep_btn"):
        nlp_ctx = [f"Dataset: {active_df.shape[0]} rows × {active_df.shape[1]} columns"]
        if st.session_state.nlp_sent_df is not None:
            vc_s = st.session_state.nlp_sent_df["Sentiment"].value_counts().to_dict()
            nlp_ctx.append(f"Sentiment Distribution: {vc_s}")
        if st.session_state.nlp_kw_df is not None and not st.session_state.nlp_kw_df.empty:
            nlp_ctx.append("Top Keywords: " + ", ".join(st.session_state.nlp_kw_df["Keyword / Phrase"].head(8).tolist()))
        if st.session_state.ts_fc_df is not None:
            nlp_ctx.append(f"Time-Series Forecast on '{st.session_state.ts_fc_df['col']}': slope={st.session_state.ts_fc_df['slope']}, next={np.round(st.session_state.ts_fc_df['fut'][:5], 2).tolist()}")

        nlp_summary_txt = "\n".join(nlp_ctx)
        if is_en():
            p_nlp = (
                "You are I'm AI Buddy — a senior NLP & Forecasting Specialist. Based on these results, write a clear report in English with:\n"
                "1. **NLP & Trend Findings**\n"
                "2. **What Matters Most for Me & What I Should Do Next (Practical Action Plan)** — highlight the most useful takeaway and exact next step.\n\n"
                + nlp_summary_txt
            )
        else:
            p_nlp = (
                "तुम I'm AI Buddy — एक सीनियर NLP और Forecasting विशेषज्ञ हो। नीचे दिए गए रिज़ल्ट के आधार पर सरल हिंदी में रिपोर्ट लिखो:\n"
                "1. **NLP और Trend के मुख्य निष्कर्ष**\n"
                "2. **इस रिज़ल्ट से मेरे लिए क्या ज़रूरी है और मुझे आगे क्या करना चाहिए (Next Steps & Useful Takeaways)** — साफ-साफ बताओ कि मेरे काम की सबसे ज़रूरी बात क्या है और अगला कदम क्या होना चाहिए।\n\n"
                + nlp_summary_txt
            )
        with st.spinner(T(f"{selected_provider} से NLP रिपोर्ट बन रही है...", f"Generating NLP report via {selected_provider}...")):
            n_ans, n_err = call_ai_provider(ai_config, p_nlp)
        st.session_state.nlp_ai_report = n_ans if n_ans else f"### NLP & Forecasting Summary\n{nlp_summary_txt}\n\n*(AI Note: {n_err})*"

    if st.session_state.nlp_ai_report:
        st.markdown(st.session_state.nlp_ai_report)
        st.download_button(
            T("📥 NLP & Forecasting Report डाउनलोड करें (.md)", "📥 Download NLP & Forecasting Report (.md)"),
            data=st.session_state.nlp_ai_report.encode("utf-8"),
            file_name="im_ai_buddy_nlp_forecast_report.md",
            mime="text/markdown",
            key="nlp_dl_report_btn",
        )

# ------------------------------------------------------------------------------
# MASTER COMBINED AI REPORT DOWNLOAD (ALL TABS IN ONE FILE)
# ------------------------------------------------------------------------------
st.divider()
master_parts = ["# I'm AI Buddy — Complete Master AI Report", f"- **Dataset:** `{st.session_state.uni_loaded_file}` ({active_df.shape[0]} rows × {active_df.shape[1]} columns)"]
if st.session_state.da_ai_summary:
    master_parts.extend(["", "## 1. Data Analysis AI Report", st.session_state.da_ai_summary])
if st.session_state.bi_ai_review:
    master_parts.extend(["", "## 2. Power BI AI Dashboard Report", st.session_state.bi_ai_review])
if st.session_state.ds_full_ai_report:
    master_parts.extend(["", "## 3. Data Science & Gen AI Report", st.session_state.ds_full_ai_report])
if st.session_state.dl_ai_report:
    master_parts.extend(["", "## 4. Deep Learning Lab Report", st.session_state.dl_ai_report])
if st.session_state.nlp_ai_report:
    master_parts.extend(["", "## 5. NLP & Forecasting Report", st.session_state.nlp_ai_report])

if len(master_parts) > 2:
    st.download_button(
        T("📦 सभी AI Reports एक साथ डाउनलोड करें (Master Report .md)", "📦 Download All Generated AI Reports Together (Master Report .md)"),
        data="\n".join(master_parts).encode("utf-8"),
        file_name="im_ai_buddy_master_ai_report.md",
        mime="text/markdown",
        key="master_all_ai_reports_dl_btn",
                              )
