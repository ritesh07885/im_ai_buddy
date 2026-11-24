from io import BytesIO
import json
import random
import re
import sqlite3
import time
import urllib.error
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
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
  from sklearn.cluster import KMeans
  from sklearn.compose import ColumnTransformer
  from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
  from sklearn.impute import SimpleImputer
  from sklearn.linear_model import LinearRegression, LogisticRegression
  from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
  from sklearn.model_selection import train_test_split
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import OneHotEncoder, StandardScaler

  HAS_SKLEARN = True
except ImportError:
  HAS_SKLEARN = False

st.set_page_config(
    page_title="I'm AI Buddy",
    page_icon="🤖",
    layout="wide",
)

# --- GLOBAL SESSION STATE ---
if "uni_loaded_file" not in st.session_state:
  st.session_state.uni_loaded_file = None
if "uni_raw_df" not in st.session_state:
  st.session_state.uni_raw_df = None
if "uni_df" not in st.session_state:
  st.session_state.uni_df = None

# Tab 1 (Data Analysis) state
if "da_ai_summary" not in st.session_state:
  st.session_state.da_ai_summary = None

# Tab 2 (SQL Studio) state
if "sql_query_box_val" not in st.session_state:
  st.session_state.sql_query_box_val = "SELECT * FROM dataset LIMIT 20;"
if "sql_text_area" not in st.session_state:
  st.session_state.sql_text_area = st.session_state.sql_query_box_val
if "sql_res_df" not in st.session_state:
  st.session_state.sql_res_df = None
if "sql_err_msg" not in st.session_state:
  st.session_state.sql_err_msg = None
if "sql_ai_msg" not in st.session_state:
  st.session_state.sql_ai_msg = None
if "sql_last_preset" not in st.session_state:
  st.session_state.sql_last_preset = None

# Tab 3 (Power BI AI) state
if "bi_bg_img" not in st.session_state:
  st.session_state.bi_bg_img = None
if "bi_bg_src" not in st.session_state:
  st.session_state.bi_bg_src = None
if "bi_ai_review" not in st.session_state:
  st.session_state.bi_ai_review = None
if "bi_dax_code" not in st.session_state:
  st.session_state.bi_dax_code = None

# Tab 4 (Data Science & Gen AI) state
if "ds_chat_history" not in st.session_state:
  st.session_state.ds_chat_history = []
if "ds_ml_report" not in st.session_state:
  st.session_state.ds_ml_report = None
if "ds_trained_bundle" not in st.session_state:
  st.session_state.ds_trained_bundle = None


# --- HELPER FUNCTIONS ---
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

  nums = [
      float(x.replace(",", "")) for x in re.findall(r"\d+(?:\.\d+)?", txt)
  ]
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


def build_default_dataset():
  return pd.DataFrame({
      "Region": [
          "Jalgaon",
          "Pune",
          "Mumbai",
          "Nashik",
          "Jalgaon",
          "Pune",
          "Mumbai",
          "Nashik",
          "Pune",
          "Mumbai",
          "Jalgaon",
          "Nashik",
      ],
      "Category": [
          "Masale",
          "Audio",
          "Analytics",
          "Masale",
          "Audio",
          "Analytics",
          "Masale",
          "Audio",
          "Masale",
          "Audio",
          "Analytics",
          "Analytics",
      ],
      "Month": [
          "Jan",
          "Jan",
          "Jan",
          "Feb",
          "Feb",
          "Feb",
          "Mar",
          "Mar",
          "Mar",
          "Apr",
          "Apr",
          "Apr",
      ],
      "Revenue": [
          42000,
          68000,
          95000,
          38000,
          74000,
          102000,
          51000,
          81000,
          49000,
          88000,
          91000,
          64000,
      ],
      "Cost": [
          26000,
          41000,
          58000,
          24000,
          45000,
          61000,
          31000,
          49000,
          29500,
          53000,
          55000,
          39000,
      ],
      "Units": [210, 12, 18, 190, 14, 20, 250, 15, 225, 17, 19, 16],
  })


def build_sql_analysis_library(cat_col, num_col):
  return {
      "1. Preview Top 20 Rows": "SELECT * FROM dataset LIMIT 20;",
      "2. Total Row Count": "SELECT COUNT(*) AS total_rows FROM dataset;",
      (f"3. Group By Summary ({cat_col} -> {num_col})"): (
          f'SELECT "{cat_col}", COUNT(*) AS row_count,'
          f' SUM("{num_col}") AS total_val, ROUND(AVG("{num_col}"), 2) AS'
          f' avg_val, MIN("{num_col}") AS min_val, MAX("{num_col}") AS'
          f' max_val\nFROM dataset\nGROUP BY "{cat_col}"\nORDER BY total_val'
          " DESC;"
      ),
      (f"4. Top 10 Highest ({num_col})"): (
          f'SELECT * FROM dataset\nWHERE "{num_col}" IS NOT NULL\nORDER BY'
          f' "{num_col}" DESC\nLIMIT 10;'
      ),
      (f"5. Top 3 Per {cat_col} (ROW_NUMBER Window)"): (
          "WITH ranked AS (\n  SELECT *,\n    ROW_NUMBER() OVER ("
          f' PARTITION BY "{cat_col}" ORDER BY "{num_col}" DESC'
          " ) AS rn\n  FROM dataset\n)\nSELECT * FROM ranked\nWHERE rn <= 3;"
      ),
      (f"6. Dense Rank by {num_col} (DENSE_RANK)"): (
          "SELECT *,\n"
          f'  DENSE_RANK() OVER (ORDER BY "{num_col}" DESC) AS rank_num\n'
          "FROM dataset\n"
          f'ORDER BY "{num_col}" DESC;'
      ),
      (f"7. Running Total of {num_col}"): (
          "SELECT *,\n"
          f'  SUM("{num_col}") OVER (ORDER BY "{num_col}" DESC ROWS BETWEEN'
          " UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total\n"
          "FROM dataset\n"
          "LIMIT 100;"
      ),
      (f"8. Share % of Total ({cat_col} by {num_col})"): (
          "WITH totals AS (\n"
          f'  SELECT "{cat_col}", SUM("{num_col}") AS grp_sum\n'
          "  FROM dataset\n"
          f'  GROUP BY "{cat_col}"\n'
          ")\n"
          f'SELECT "{cat_col}", grp_sum,\n'
          "  ROUND(100.0 * grp_sum / NULLIF(SUM(grp_sum) OVER (), 0), 2) AS"
          " pct_share\n"
          "FROM totals\n"
          "ORDER BY grp_sum DESC;"
      ),
      (f"9. Distinct Values in {cat_col}"): (
          f'SELECT "{cat_col}", COUNT(*) AS freq\nFROM dataset\nGROUP BY'
          f' "{cat_col}"\nORDER BY freq DESC;'
      ),
      (f"10. Rows Above Average {num_col}"): (
          "SELECT * FROM dataset\n"
          f'WHERE "{num_col}" > (SELECT AVG("{num_col}") FROM dataset)\n'
          f'ORDER BY "{num_col}" DESC;'
      ),
  }


def call_ollama_fallback(
    prompt_text, ollama_model="qwen2.5:3b", ollama_url="http://localhost:11434"
):
  """अगर Gemini API 503 या अन्य एरर दे तो लोकल Ollama से जवाब लेने का बैकअप।"""
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
    with urllib.request.urlopen(req, timeout=25) as resp:
      data = json.loads(resp.read().decode("utf-8"))
      txt = (data.get("response") or "").strip()
      if txt:
        return f"{txt}\n\n*(Source: Local Ollama — `{ollama_model}`)*", None
    return None, "Ollama से खाली जवाब आया।"
  except urllib.error.URLError as e:
    return None, f"Ollama कनेक्ट नहीं हो सका ({e.reason})"
  except Exception as e:
    return None, f"Ollama एरर: {e}"


def call_gemini_safe(
    api_key_val,
    prompt_text,
    enable_ollama=True,
    ollama_model="qwen2.5:3b",
):
  """
    Advanced resilient AI caller:
    1. Multiple Gemini models fallback
    2. Exponential backoff + jitter on 503 / 429
    3. Optional local Ollama fallback if cloud is unavailable
    """
  models_to_try = [
      "gemini-2.5-flash",
      "gemini-2.0-flash",
      "gemini-3.8-flash",
  ]
  last_err = None

  if HAS_GENAI and api_key_val:
    client = genai.Client(api_key=api_key_val)
    for mdl in models_to_try:
      for attempt in range(3):
        try:
          resp = client.models.generate_content(
              model=mdl,
              contents=prompt_text,
          )
          if resp and resp.text and resp.text.strip():
            return resp.text.strip(), None
        except Exception as exc:
          last_err = exc
          err_str = str(exc)
          is_transient = any(
              code in err_str
              for code in [
                  "503",
                  "UNAVAILABLE",
                  "429",
                  "RESOURCE_EXHAUSTED",
                  "500",
                  "504",
              ]
          )
          if is_transient and attempt < 2:
            sleep_sec = (2**attempt) + random.uniform(0.1, 0.5)
            time.sleep(sleep_sec)
            continue
          if "401" in err_str or "API_KEY_INVALID" in err_str:
            return None, f"Invalid API Key: {err_str}"
          break

  if enable_ollama:
    ol_res, ol_err = call_ollama_fallback(
        prompt_text, ollama_model=ollama_model
    )
    if ol_res:
      return ol_res, None
    last_err = f"Cloud Err: {last_err} | {ol_err}"

  return None, str(last_err or "AI सेवा अस्थायी रूप से उपलब्ध नहीं है।")


def build_local_stats_summary(df_in):
  num_c = df_in.select_dtypes(include="number").columns.tolist()
  cat_c = df_in.select_dtypes(exclude="number").columns.tolist()
  lines = [
      (
          "**लोकल स्टैटिस्टिकल समरी:** कुल पंक्तियाँ ="
          f" `{len(df_in)}`, कुल कॉलम = `{len(df_in.columns)}`"
      ),
      (
          f"- Numeric कॉलम्स ({len(num_c)}):"
          f" {', '.join(num_c) if num_c else 'कोई नहीं'}"
      ),
      (
          f"- Category/Text कॉलम्स ({len(cat_c)}):"
          f" {', '.join(cat_c) if cat_c else 'कोई नहीं'}"
      ),
      (
          f"- खाली सेल (Nulls): `{int(df_in.isna().sum().sum())}` | डुप्लीकेट"
          f" पंक्तियाँ: `{int(df_in.duplicated().sum())}`"
      ),
  ]
  for c in num_c[:5]:
    s = pd.to_numeric(get_1d_series(df_in, c), errors="coerce").dropna()
    if not s.empty:
      lines.append(
          f"- **{c}**: Sum = `{s.sum():,.2f}`, Avg = `{s.mean():,.2f}`,"
          f" Min = `{s.min():,.2f}`, Max = `{s.max():,.2f}`"
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
        row_dict["Min / Max"] = ", ".join(
            [f"{k} ({v})" for k, v in top_vals.items()]
        )
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
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers.append(("cat", cat_pipe, cat_cols))

  return ColumnTransformer(transformers=transformers, remainder="drop")


def extract_feature_importances(pipe, X_df):
  try:
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
      return None

    feat_names = []
    num_cols = X_df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]
    if num_cols:
      feat_names.extend([str(c) for c in num_cols])
    if cat_cols and "cat" in prep.named_transformers_:
      ohe = prep.named_transformers_["cat"].named_steps["ohe"]
      feat_names.extend(
          list(ohe.get_feature_names_out([str(c) for c in cat_cols]))
      )

    imps = model.feature_importances_
    if len(feat_names) != len(imps):
      feat_names = [f"Feature_{i}" for i in range(len(imps))]

    imp_df = (
        pd.DataFrame({"Feature": feat_names, "Importance": imps})
        .sort_values("Importance", ascending=False)
        .head(10)
    )
    return imp_df
  except Exception:
    return None


def train_regression_suite(df_sub, target_col, feature_cols, test_ratio):
  work = df_sub[feature_cols + [target_col]].copy()
  work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
  work = work.dropna(subset=[target_col])
  if len(work) < 6:
    return None, "Regression के लिए कम से कम 6 वैध पंक्तियाँ चाहिए।"

  X = work[feature_cols]
  y = work[target_col]
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_ratio, random_state=42
  )

  candidates = {
      "Random Forest Regressor": RandomForestRegressor(
          n_estimators=120, random_state=42
      ),
      "Gradient Boosting Regressor": GradientBoostingRegressor(
          random_state=42
      ),
      "Linear Regression": LinearRegression(),
  }

  metrics_rows = []
  trained_pipes = {}
  for name, est in candidates.items():
    pipe = Pipeline([("prep", build_preprocessor(X)), ("model", est)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    r2 = float(r2_score(y_test, preds)) if len(y_test) >= 2 else 0.0
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    metrics_rows.append({
        "Model": name,
        "R2 Score": round(r2, 4),
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
  }, None


def train_classification_suite(df_sub, target_col, feature_cols, test_ratio):
  work = df_sub[feature_cols + [target_col]].copy()
  work[target_col] = work[target_col].astype(str).str.strip()
  work = work[work[target_col] != ""]
  class_counts = work[target_col].value_counts()
  valid_classes = class_counts[class_counts >= 2].index
  work = work[work[target_col].isin(valid_classes)]

  if len(work) < 8 or work[target_col].nunique() < 2:
    return (
        None,
        "Classification के लिए कम से कम 2 Classes (हर में ≥2 rows) और ≥8"
        " पंक्तियाँ चाहिए।",
    )

  X = work[feature_cols]
  y = work[target_col]
  strat = y if y.value_counts().min() >= 2 else None
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_ratio, random_state=42, stratify=strat
  )

  candidates = {
      "Random Forest Classifier": RandomForestClassifier(
          n_estimators=120, random_state=42
      ),
      "Gradient Boosting Classifier": GradientBoostingClassifier(
          random_state=42
      ),
      "Logistic Regression": LogisticRegression(max_iter=500),
  }

  metrics_rows = []
  trained_pipes = {}
  for name, est in candidates.items():
    pipe = Pipeline([("prep", build_preprocessor(X)), ("model", est)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    metrics_rows.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Accuracy %": f"{acc * 100:.1f}%",
    })
    trained_pipes[name] = pipe

  best_row = sorted(metrics_rows, key=lambda r: r["Accuracy"], reverse=True)[0]
  best_name = best_row["Model"]
  best_pipe = trained_pipes[best_name]
  y_pred_best = best_pipe.predict(X_test)
  labels_list = sorted(y.unique().tolist())
  cm = confusion_matrix(y_test, y_pred_best, labels=labels_list)

  return {
      "task": "classification",
      "target": target_col,
      "features": feature_cols,
      "leaderboard": pd.DataFrame(metrics_rows),
      "best_name": best_name,
      "best_pipe": best_pipe,
      "labels": labels_list,
      "cm": cm,
      "imp_df": extract_feature_importances(best_pipe, X),
      "X_ref": X,
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
      ax.text(
          j,
          i,
          f"{val:.2f}",
          ha="center",
          va="center",
          color="#111827",
          fontsize=8,
          fontweight="bold",
      )
  ax.set_title("Correlation Heatmap", fontsize=11, fontweight="bold")
  fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
  plt.tight_layout()
  return fig


def render_dist_fig(series_in, col_name):
  s_1d = series_in.iloc[:, 0] if isinstance(series_in, pd.DataFrame) else series_in
  s_num = pd.to_numeric(s_1d, errors="coerce").dropna()
  fig, (ax_h, ax_b) = plt.subplots(
      2, 1, figsize=(6.2, 4.6), dpi=120, gridspec_kw={"height_ratios": [3, 1]}
  )
  if len(s_num) > 1:
    ax_h.hist(
        s_num,
        bins=15,
        color="#2563EB",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
    )
    ax_h.axvline(
        s_num.mean(),
        color="#DC2626",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {s_num.mean():,.2f}",
    )
    ax_h.axvline(
        s_num.median(),
        color="#059669",
        linestyle="-.",
        linewidth=1.5,
        label=f"Median: {s_num.median():,.2f}",
    )
    ax_h.set_title(
        f"Distribution & Spread — {col_name}", fontsize=10.5, fontweight="bold"
    )
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
    ax_h.text(
        0.5,
        0.5,
        "पर्याप्त Numeric डेटा नहीं है।",
        ha="center",
        va="center",
    )
    ax_b.axis("off")
  plt.tight_layout()
  return fig


def render_cat_bar_fig(series_in, col_name, top_n=8):
  s_1d = series_in.iloc[:, 0] if isinstance(series_in, pd.DataFrame) else series_in
  vc = s_1d.dropna().astype(str).value_counts().head(top_n)
  fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=120)
  if not vc.empty:
    vc_plot = vc.sort_values(ascending=True)
    ax.barh(
        vc_plot.index,
        vc_plot.values,
        color="#0D9488",
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_title(
        f"Top {len(vc_plot)} — {col_name}", fontsize=10.5, fontweight="bold"
    )
    ax.set_xlabel("Count")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
  else:
    ax.text(0.5, 0.5, "Category डेटा नहीं मिला।", ha="center", va="center")
  plt.tight_layout()
  return fig


def build_local_soft_bg(width=900, height=520):
  img = Image.new("RGBA", (width, height), (246, 249, 253, 255))
  draw = ImageDraw.Draw(img, "RGBA")
  draw.ellipse([-120, -90, 460, 340], fill=(214, 232, 255, 140))
  draw.ellipse([420, 140, 1020, 620], fill=(225, 245, 238, 140))
  draw.rounded_rectangle(
      [60, 50, width - 60, height - 50],
      radius=28,
      outline=(190, 206, 228, 120),
      width=3,
  )
  return img.filter(ImageFilter.GaussianBlur(radius=14))


def generate_ai_bg_image(api_key_val, prompt_text):
  if not HAS_GENAI or not api_key_val:
    return build_local_soft_bg(), "Local Fallback (API Key नहीं है)"
  try:
    client = genai.Client(api_key=api_key_val)
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
    df_sub,
    chart_type,
    x_col,
    y_col,
    stack_col,
    agg_func,
    top_n,
    bg_img,
    use_bg,
    opacity_val,
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
      ax.hist(
          s_num,
          bins=15,
          color="#2563EB",
          alpha=0.84,
          edgecolor="white",
          linewidth=0.8,
      )
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
      ax.scatter(
          pair["x"],
          pair["y"],
          color="#0D9488",
          alpha=0.75,
          edgecolors="white",
          linewidth=0.5,
          s=45,
      )
      ax.set_title(f"{y_col} vs {x_col}", fontsize=11, fontweight="bold")
      ax.set_xlabel(str(x_col))
      ax.set_ylabel(str(y_col))
      rendered_ok = True

  elif chart_type == "Donut" and x_col and y_col:
    temp = pd.DataFrame({
        "x_k": s_x_raw.astype(str),
        "y_v": pd.to_numeric(s_y_raw, errors="coerce"),
    }).dropna(subset=["y_v"])
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
        ax.set_title(
            f"{agg_func} of {y_col} by {x_col} (Donut)",
            fontsize=11,
            fontweight="bold",
        )
        rendered_ok = True

  elif (
      chart_type == "Stacked Bar"
      and x_col
      and y_col
      and stack_col
      and stack_col != x_col
  ):
    temp = pd.DataFrame({
        "x_k": s_x_raw.astype(str),
        "y_v": pd.to_numeric(s_y_raw, errors="coerce"),
        "s_k": s_st_raw.astype(str),
    }).dropna(subset=["y_v"])
    if not temp.empty:
      top_x = (
          aggregate_series(temp, "x_k", "y_v", agg_func)
          .sort_values(ascending=False)
          .head(top_n)
          .index
      )
      top_s = temp[temp["x_k"].isin(top_x)]["s_k"].value_counts().head(6).index
      sub_st = temp[temp["x_k"].isin(top_x) & temp["s_k"].isin(top_s)]
      if not sub_st.empty:
        agg_map = {
            "Sum": "sum",
            "Avg": "mean",
            "Max": "max",
            "Count": "count",
        }
        piv = sub_st.pivot_table(
            index="x_k",
            columns="s_k",
            values="y_v",
            aggfunc=agg_map.get(agg_func, "sum"),
            fill_value=0,
        ).reindex(top_x)
        piv.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            alpha=0.86,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_title(
            f"Stacked {agg_func} of {y_col} by {x_col} ({stack_col})",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xlabel(str(x_col))
        ax.set_ylabel(f"{agg_func} of {y_col}")
        plt.xticks(rotation=30, ha="right")
        ax.legend(title=str(stack_col), fontsize=8, title_fontsize=8)
        rendered_ok = True

  elif x_col and y_col:
    temp = pd.DataFrame({
        "x_k": s_x_raw.astype(str),
        "y_v": pd.to_numeric(s_y_raw, errors="coerce"),
    }).dropna(subset=["y_v"])
    if not temp.empty:
      grp = aggregate_series(temp, "x_k", "y_v", agg_func)
      grp = grp.sort_values(ascending=False).head(top_n)
      if not grp.empty:
        if chart_type == "Horizontal Bar":
          grp_plot = grp.sort_values(ascending=True)
          ax.barh(
              grp_plot.index,
              grp_plot.values,
              color="#1D4ED8",
              alpha=0.85,
              edgecolor="white",
          )
          ax.set_xlabel(f"{agg_func} of {y_col}")
          ax.set_ylabel(str(x_col))
        elif chart_type == "Bar":
          ax.bar(
              grp.index,
              grp.values,
              color="#2563EB",
              alpha=0.85,
              edgecolor="white",
          )
          ax.set_ylabel(f"{agg_func} of {y_col}")
          ax.set_xlabel(str(x_col))
          plt.xticks(rotation=30, ha="right")
        elif chart_type == "Line":
          ax.plot(
              grp.index,
              grp.values,
              marker="o",
              linewidth=2.2,
              color="#0F766E",
          )
          ax.set_ylabel(f"{agg_func} of {y_col}")
          ax.set_xlabel(str(x_col))
          plt.xticks(rotation=30, ha="right")
        elif chart_type == "Area":
          x_idx = np.arange(len(grp))
          ax.fill_between(x_idx, grp.values, color="#3B82F6", alpha=0.35)
          ax.plot(
              x_idx, grp.values, marker="o", linewidth=2.2, color="#1D4ED8"
          )
          ax.set_xticks(x_idx)
          ax.set_xticklabels(grp.index, rotation=30, ha="right")
          ax.set_ylabel(f"{agg_func} of {y_col}")
          ax.set_xlabel(str(x_col))

        ax.set_title(
            f"{agg_func} of {y_col} by {x_col}", fontsize=11, fontweight="bold"
        )
        rendered_ok = True

  if not rendered_ok:
    ax.text(
        0.5,
        0.5,
        "इस कॉलम संयोजन के लिए उपयुक्त डेटा उपलब्ध नहीं है।",
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
    bg_resized = bg_img.resize(
        chart_rgba.size, Image.Resampling.LANCZOS
    ).convert("RGBA")
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
    return None, "कृपया SQL क्वेरी लिखें।"
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


# --- SIDEBAR (COMMON SETTINGS) ---
st.sidebar.header("🤖 I'm AI Buddy — सेटिंग्स")
api_key = st.sidebar.text_input("Gemini API Key डालें:", type="password")
use_ollama_fb = st.sidebar.checkbox(
    "Local Ollama Fallback चालू रखें", value=True
)
ollama_mdl_name = st.sidebar.text_input("Ollama Model Name:", value="qwen2.5:3b")

if st.sidebar.button("🔄 मूल डेटा पर रीसेट करें"):
  if st.session_state.uni_raw_df is not None:
    st.session_state.uni_df = st.session_state.uni_raw_df.copy()
    st.session_state.ds_trained_bundle = None
    st.rerun()

# --- TOP HEADER & SHARED FILE UPLOADER ---
st.title("🤖 I'm AI Buddy")
st.caption(
    "मैं आपका Data Analysis, SQL Studio, Power BI AI और Data Science & Gen"
    " AI साथी हूँ — ऊपर से अपना काम चुनें।"
)

top_u1, top_u2 = st.columns([3, 1])
with top_u1:
  uploaded_file = st.file_uploader(
      "अपनी CSV या Excel फ़ाइल अपलोड करें (सभी Tabs में यही डेटा अपने-आप"
      " उपलब्ध रहेगा)",
      type=["csv", "xlsx"],
      key="uni_file_uploader",
  )

if uploaded_file is not None:
  if st.session_state.uni_loaded_file != uploaded_file.name:
    st.session_state.uni_loaded_file = uploaded_file.name
    loaded = load_uploaded_table(uploaded_file)
    st.session_state.uni_raw_df = loaded.copy()
    st.session_state.uni_df = loaded.copy()
    st.session_state.da_ai_summary = None
    st.session_state.sql_res_df = None
    st.session_state.bi_ai_review = None
    st.session_state.ds_ml_report = None
    st.session_state.ds_trained_bundle = None
else:
  if st.session_state.uni_loaded_file != "__default_sample__":
    st.session_state.uni_loaded_file = "__default_sample__"
    sample = build_default_dataset()
    st.session_state.uni_raw_df = sample.copy()
    st.session_state.uni_df = sample.copy()

active_df = st.session_state.uni_df
with top_u2:
  st.metric("सक्रिय डेटा आकार", f"{active_df.shape[0]} × {active_df.shape[1]}")
  st.download_button(
      "📥 वर्तमान डेटा (.csv)",
      data=active_df.to_csv(index=False).encode("utf-8"),
      file_name="im_ai_buddy_dataset.csv",
      mime="text/csv",
      use_container_width=True,
  )

# ==============================================================================
# TOP NAVIGATION TABS (ऊपर स्क्रीन पर मुख्य विकल्प)
# ==============================================================================
tab_da, tab_sql, tab_pbi, tab_ds = st.tabs([
    "📊 Data Analysis",
    "🗄️ SQL Studio",
    "📈 Power BI AI",
    "🧠 Data Science & Gen AI",
])

# ------------------------------------------------------------------------------
# TAB 1: ADVANCE DATA ANALYSIS & CLEANING + USD TO INR CONVERTER + REPORT EXPORT
# ------------------------------------------------------------------------------
with tab_da:
  st.header("📊 Advance Data Analysis और डेटा क्लीनिंग")

  da_nums = active_df.select_dtypes(include="number").columns.tolist()
  da_cats = active_df.select_dtypes(exclude="number").columns.tolist()
  da_all = active_df.columns.tolist()

  m1, m2, m3, m4 = st.columns(4)
  m1.metric("कुल पंक्तियाँ", active_df.shape[0])
  m2.metric("कुल कॉलम", active_df.shape[1])
  m3.metric("खाली सेल (Nulls)", int(active_df.isna().sum().sum()))
  m4.metric("डुप्लीकेट पंक्तियाँ", int(active_df.duplicated().sum()))

  with st.expander(
      "🧹 डेटा क्लीनिंग और कॉलम एडिटिंग (सभी Tabs पर लागू होगा)", expanded=False
  ):
    c_clean1, c_clean2, c_clean3 = st.columns(3)
    with c_clean1:
      if st.button("🗑️ डुप्लीकेट पंक्तियाँ हटाएं", key="da_drop_dups"):
        st.session_state.uni_df = active_df.drop_duplicates().reset_index(
            drop=True
        )
        st.rerun()
      if st.button("✂️ टेक्स्ट से अतिरिक्त स्पेस हटाएं (Trim)", key="da_trim"):
        tmp = active_df.copy()
        for col in tmp.select_dtypes(include=["object", "string"]).columns:
          s_tr = get_1d_series(tmp, col).apply(
              lambda x: x.strip() if isinstance(x, str) else x
          )
          tmp[col] = s_tr
        st.session_state.uni_df = tmp
        st.rerun()

    with c_clean2:
      null_action = st.selectbox(
          "Nulls संभालने का तरीका:",
          ["Numeric में Median, Text में 'Unknown'", "Null वाली पंक्तियाँ हटाएं"],
          key="da_null_sel",
      )
      if st.button("🩹 Nulls ठीक करें", key="da_fix_nulls"):
        tmp = active_df.copy()
        if null_action.startswith("Numeric"):
          for col in tmp.select_dtypes(include="number").columns:
            s_n = pd.to_numeric(get_1d_series(tmp, col), errors="coerce")
            tmp[col] = s_n.fillna(s_n.median())
          for col in tmp.select_dtypes(exclude="number").columns:
            tmp[col] = get_1d_series(tmp, col).fillna("Unknown")
        else:
          tmp = tmp.dropna().reset_index(drop=True)
        st.session_state.uni_df = tmp
        st.rerun()

    with c_clean3:
      ren_col = st.selectbox(
          "कॉलम का नाम बदलें:", active_df.columns.tolist(), key="da_ren_col"
      )
      ren_new = st.text_input("नया नाम:", value=str(ren_col), key="da_ren_txt")
      if st.button("✏️ नाम अपडेट करें", key="da_ren_btn"):
        if ren_new.strip() and ren_new.strip() != ren_col:
          st.session_state.uni_df = active_df.rename(
              columns={ren_col: ren_new.strip()}
          )
          st.rerun()

  with st.expander(
      "💱 USD Range कॉलम को Indian Rupee (INR) में बदलें", expanded=False
  ):
    c_fx1, c_fx2, c_fx3, c_fx4 = st.columns([1.4, 1.1, 1.2, 1.3])
    with c_fx1:
      fx_src_col = st.selectbox(
          "USD कॉलम चुनें:",
          ["— चुनें —"] + da_all,
          key="da_fx_col",
      )
    with c_fx2:
      usd_inr_rate = st.number_input(
          "1 USD = ₹ (Rate):",
          min_value=1.0,
          max_value=500.0,
          value=92.79,
          step=0.1,
          key="da_fx_rate",
      )
    with c_fx3:
      range_pick = st.selectbox(
          "रेंज का कौन-सा मान लें?",
          ["Midpoint (औसत)", "Lower Limit (न्यूनतम)", "Upper Limit (अधिकतम)"],
          key="da_fx_mode",
      )
    with c_fx4:
      unknown_as_zero = st.checkbox(
          "Unknown को 0 मानें", value=False, key="da_fx_unk"
      )

    if fx_src_col != "— चुनें —":
      s_fx_raw = get_1d_series(active_df, fx_src_col)
      inr_vals = s_fx_raw.apply(
          lambda v: parse_usd_range_to_inr(
              v, usd_inr_rate, range_pick, unknown_as_zero
          )
      )
      new_inr_col = f"{fx_src_col}_INR"

      prev_view = pd.DataFrame({
          fx_src_col: s_fx_raw.astype(str),
          new_inr_col: inr_vals,
      }).drop_duplicates()
      prev_view["INR (K / हज़ार)"] = (prev_view[new_inr_col] / 1e3).round(2)
      prev_view["INR (Lakhs / लाख)"] = (prev_view[new_inr_col] / 1e5).round(2)
      st.dataframe(prev_view, use_container_width=True)

      if st.button("➕ नया INR कॉलम डेटासेट में जोड़ें", key="da_fx_add_btn"):
        tmp_df = active_df.copy()
        tmp_df[new_inr_col] = inr_vals.values
        st.session_state.uni_df = tmp_df
        st.rerun()

  st.subheader("🔎 इंटरएक्टिव फ़िल्टर और कॉलम एक्सप्लोरर")
  f_col1, f_col2, f_col3 = st.columns([1.2, 1.4, 1.4])
  with f_col1:
    sel_cols_da = st.multiselect(
        "देखने के लिए कॉलम चुनें:",
        da_all,
        default=da_all,
        key="da_show_cols",
    )
  with f_col2:
    search_kw = st.text_input(
        "कीवर्ड से पंक्तियाँ खोजें:",
        placeholder="किसी शब्द या नंबर से खोजें...",
        key="da_search_kw",
    )
  with f_col3:
    if da_cats:
      flt_cat = st.selectbox(
          "Category फ़िल्टर कॉलम:", ["सभी"] + da_cats, key="da_flt_cat"
      )
      if flt_cat != "सभी":
        s_fcat = get_1d_series(active_df, flt_cat)
        cat_vals = sorted(s_fcat.dropna().astype(str).unique().tolist())
        flt_val = st.selectbox(
            f"{flt_cat} चुनें:", ["सभी"] + cat_vals, key="da_flt_val"
        )
      else:
        flt_val = "सभी"
    else:
      flt_cat, flt_val = "सभी", "सभी"

  da_view = active_df.copy()
  if flt_cat != "सभी" and flt_val != "सभी":
    da_view = da_view[get_1d_series(da_view, flt_cat).astype(str) == flt_val]
  if search_kw.strip():
    kw_l = search_kw.strip().lower()
    mask = pd.Series(False, index=da_view.index)
    for c in da_view.columns:
      mask = mask | get_1d_series(da_view, c).astype(str).str.lower().str.contains(
          kw_l, na=False
      )
    da_view = da_view[mask]

  cols_to_show = sel_cols_da if sel_cols_da else da_all
  st.caption(
      f"फ़िल्टर परिणाम: **{len(da_view)}** पंक्तियाँ ×"
      f" **{len(cols_to_show)}** कॉलम"
  )
  st.dataframe(da_view[cols_to_show].head(100), use_container_width=True)

  st.subheader("🩺 कॉलम स्वास्थ्य (Missing / Unique) और Numeric विवरण")
  st.dataframe(build_all_columns_profile(da_view), use_container_width=True)
  if da_nums:
    st.dataframe(da_view[da_nums].describe().round(2), use_container_width=True)

  st.subheader("📐 GroupBy विश्लेषण और चार्ट")
  if da_all:
    gb_c1, gb_c2, gb_c3, gb_c4 = st.columns(4)
    gb_by = gb_c1.selectbox("Group By कॉलम:", da_all, index=0, key="da_gb_by")
    val_choices = da_nums if da_nums else da_all
    gb_val = gb_c2.selectbox(
        "Value कॉलम:", val_choices, index=0, key="da_gb_val"
    )
    gb_fn = gb_c3.selectbox(
        "Function:", ["sum", "mean", "count", "max", "min"], key="da_gb_fn"
    )
    gb_ch = gb_c4.selectbox(
        "Chart प्रकार:", ["Bar Chart", "Line Chart"], key="da_gb_ch"
    )

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
          ax_gb.bar(
              plot_top[str(gb_by)],
              plot_top[out_col_name],
              color="#2563EB",
              alpha=0.85,
              edgecolor="white",
          )
        else:
          ax_gb.plot(
              plot_top[str(gb_by)],
              plot_top[out_col_name],
              marker="o",
              linewidth=2.2,
              color="#0F766E",
          )
        ax_gb.set_title(
            f"{gb_fn.upper()}({gb_val}) by {gb_by}",
            fontsize=10.5,
            fontweight="bold",
        )
        plt.xticks(rotation=30, ha="right")
        ax_gb.grid(True, linestyle="--", alpha=0.3)
        ax_gb.spines["top"].set_visible(False)
        ax_gb.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_gb, use_container_width=True)
        plt.close(fig_gb)
    else:
      st.info(
          f"'{gb_val}' कॉलम में Numeric डेटा नहीं मिला; 'count' चुनकर या दूसरा"
          " कॉलम चुनकर देखें।"
      )

  if st.button("🤖 AI से Data Analysis रिपोर्ट जनरेट करें", key="da_ai_btn"):
    local_rep = build_local_stats_summary(da_view)
    if not api_key and not use_ollama_fb:
      st.session_state.da_ai_summary = local_rep
    else:
      p_da = (
          "तुम I'm AI Buddy हो — एक सीनियर डेटा एनालिस्ट। इस डेटासेट की मुख्य"
          " बातें, पैटर्न, और डेटा क्वालिटी के बारे में सरल हिंदी में संक्षिप्त"
          " रिपोर्ट दो:\n"
          + local_rep
      )
      with st.spinner("एनालिसिस रिपोर्ट बन रही है..."):
        ans, err = call_gemini_safe(
            api_key,
            p_da,
            enable_ollama=use_ollama_fb,
            ollama_model=ollama_mdl_name,
        )
      st.session_state.da_ai_summary = (
          ans if ans else f"{local_rep}\n\n*(AI नोट: {err})*"
      )

  if st.session_state.da_ai_summary:
    st.markdown(st.session_state.da_ai_summary)

  # --- DOWNLOAD ANALYSIS REPORT (.md) ---
  st.divider()
  report_sections = [
      "# I'm AI Buddy — Data Analysis Report",
      (
          f"- **Dataset Shape:** {da_view.shape[0]} rows ×"
          f" {da_view.shape[1]} columns"
      ),
      f"- **Active Columns:** {', '.join(map(str, cols_to_show))}",
      "",
      "## Statistical Summary",
      build_local_stats_summary(da_view),
  ]

  if (
      "fx_src_col" in locals()
      and fx_src_col != "— चुनें —"
      and "new_inr_col" in locals()
      and new_inr_col in da_view.columns
  ):
    s_inr_rep = pd.to_numeric(
        get_1d_series(da_view, new_inr_col), errors="coerce"
    ).dropna()
    if not s_inr_rep.empty:
      report_sections.extend([
          "",
          f"## Currency Conversion ({fx_src_col} → {new_inr_col})",
          f"- **Rate Used:** 1 USD = ₹{usd_inr_rate}",
          f"- **Range Mode:** {range_pick}",
          (
              f"- **Total ({new_inr_col}):** ₹{s_inr_rep.sum():,.2f}"
              f" (₹{s_inr_rep.sum()/1e3:,.2f} K | ₹{s_inr_rep.sum()/1e5:,.2f} L)"
          ),
          f"- **Average ({new_inr_col}):** ₹{s_inr_rep.mean():,.2f}",
      ])

  if st.session_state.da_ai_summary:
    report_sections.extend([
        "",
        "## AI Analysis Insights",
        st.session_state.da_ai_summary,
    ])

  st.download_button(
      "📄 पूरी Analysis रिपोर्ट डाउनलोड करें (.md)",
      data="\n".join(report_sections).encode("utf-8"),
      file_name="im_ai_buddy_analysis_report.md",
      mime="text/markdown",
      key="da_dl_report_btn",
  )

# ------------------------------------------------------------------------------
# TAB 2: SQL STUDIO (FIXED EDITOR SYNC + ADVANCED RESILIENCE)
# ------------------------------------------------------------------------------
with tab_sql:
  st.header("🗄️ SQL एनालिस्ट स्टूडियो (टेबल नाम: `dataset`)")

  sql_cats = active_df.select_dtypes(exclude="number").columns.tolist()
  sql_nums = active_df.select_dtypes(include="number").columns.tolist()
  fc = sql_cats[0] if sql_cats else active_df.columns[0]
  fn = sql_nums[0] if sql_nums else active_df.columns[0]
  analysis_queries_map = build_sql_analysis_library(fc, fn)

  eng_opts = (
      ["DuckDB (Window + CTE)", "SQLite3 (Built-in)"]
      if HAS_DUCKDB
      else ["SQLite3 (Built-in)"]
  )
  s_top1, s_top2 = st.columns([2, 3])
  with s_top1:
    chosen_engine = st.selectbox("SQL इंजन:", eng_opts, key="sql_eng_sel")
  with s_top2:
    nl_sql = st.text_input(
        "हिंदी/इंग्लिश में सवाल लिखें → AI SQL बनाएगा:",
        placeholder="जैसे: हर Category का कुल Revenue और औसत निकालो",
        key="sql_nl_box",
    )
    if st.button("✨ AI से SQL लिखवाएं", key="sql_gen_btn") and nl_sql.strip():
      cols_desc = ", ".join(
          [f"{c} ({str(active_df[c].dtype)})" for c in active_df.columns]
      )
      p_sql = (
          "Write ONLY a valid SQL query for table `dataset` with columns:"
          f" {cols_desc}.\nUser request: {nl_sql.strip()}\nNo markdown fences,"
          " no explanation."
      )
      with st.spinner("SQL तैयार हो रहा है..."):
        q_gen, q_err = call_gemini_safe(
            api_key,
            p_sql,
            enable_ollama=use_ollama_fb,
            ollama_model=ollama_mdl_name,
        )
      if q_gen:
        q_clean = re.sub(r"^```(?:sql)?\s*", "", q_gen.strip(), flags=re.I)
        q_clean = re.sub(r"\s*```$", "", q_clean).strip()
        set_sql_editor_query(q_clean)
        st.session_state.sql_ai_msg = "AI ने SQL क्वेरी नीचे एडिटर में भर दी है।"
        st.rerun()
      else:
        st.warning(f"AI SQL जनरेट नहीं हो सका: {q_err}")

  if st.session_state.sql_ai_msg:
    st.info(st.session_state.sql_ai_msg)

  st.subheader("📚 सभी Analysis SQL Queries (बॉक्स से चुनें)")
  q_box_c1, q_box_c2 = st.columns([3, 1])
  with q_box_c1:
    selected_q_name = st.selectbox(
        "तैयार Analysis SQL Query चुनें:",
        list(analysis_queries_map.keys()),
        key="sql_all_queries_box",
    )
  with q_box_c2:
    st.write("")
    if st.button("📥 क्वेरी एडिटर में डालें", key="sql_load_box_btn"):
      set_sql_editor_query(analysis_queries_map[selected_q_name])
      st.rerun()

  if st.session_state.sql_last_preset != selected_q_name:
    st.session_state.sql_last_preset = selected_q_name
    set_sql_editor_query(analysis_queries_map[selected_q_name])
    st.rerun()

  qt1, qt2, qt3 = st.columns(3)
  if qt1.button("📊 Group By Summary", key="sql_t1"):
    set_sql_editor_query(
        f'SELECT "{fc}", COUNT(*) AS rows_cnt, ROUND(AVG("{fn}"), 2) AS'
        f' avg_val, SUM("{fn}") AS total_val\nFROM dataset\nGROUP BY'
        f' "{fc}"\nORDER BY total_val DESC;'
    )
    st.rerun()
  if qt2.button("🏆 Top-3 Per Group (Window)", key="sql_t2"):
    set_sql_editor_query(
        f'WITH ranked AS (\n  SELECT *, ROW_NUMBER() OVER (PARTITION BY "{fc}"'
        f' ORDER BY "{fn}" DESC) AS rn\n  FROM dataset\n)\nSELECT * FROM ranked'
        " WHERE rn <= 3;"
    )
    st.rerun()
  if qt3.button("📈 Running Total", key="sql_t3"):
    set_sql_editor_query(
        f'SELECT *, SUM("{fn}") OVER (ORDER BY "{fn}" DESC ROWS BETWEEN'
        ' UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total\nFROM dataset'
        " LIMIT 100;"
    )
    st.rerun()

  st.text_area(
      "SQL क्वेरी एडिटर:",
      height=175,
      key="sql_text_area",
      on_change=_sync_sql_from_area,
  )

  if st.button("▶️ SQL क्वेरी चलाएं", type="primary", key="sql_run_btn"):
    r_df, r_err = run_sql_engine(
        chosen_engine, active_df, st.session_state.sql_text_area
    )
    st.session_state.sql_res_df = r_df
    st.session_state.sql_err_msg = r_err

  if st.session_state.sql_err_msg:
    st.error(f"SQL Error: {st.session_state.sql_err_msg}")

  if st.session_state.sql_res_df is not None:
    out_sql = st.session_state.sql_res_df
    st.success(
        f"परिणाम: {len(out_sql)} पंक्तियाँ × {len(out_sql.columns)} कॉलम"
    )
    st.dataframe(out_sql, use_container_width=True)
    st.download_button(
        "📥 SQL रिजल्ट डाउनलोड करें (.csv)",
        data=out_sql.to_csv(index=False).encode("utf-8"),
        file_name="im_ai_buddy_sql_output.csv",
        mime="text/csv",
        key="sql_dl_btn",
    )

# ------------------------------------------------------------------------------
# TAB 3: POWER BI AI STUDIO
# ------------------------------------------------------------------------------
with tab_pbi:
  st.header("📈 Power BI AI डैशबोर्ड स्टूडियो (9 Charts + AI Background)")

  if st.session_state.bi_bg_img is None:
    im, src = generate_ai_bg_image(
        api_key,
        "Minimalist soft pastel corporate geometric wave background, light"
        " cream and sky blue tones, no text",
    )
    st.session_state.bi_bg_img = im
    st.session_state.bi_bg_src = src

  p_cfg1, p_cfg2, p_cfg3, p_cfg4 = st.columns([1.2, 1.2, 1.2, 1.4])
  with p_cfg1:
    bi_chart_count = st.slider(
        "चार्ट्स की संख्या:", 1, 9, 4, key="bi_num_charts"
    )
  with p_cfg2:
    bi_top_n = st.slider("Top N कैटेगरी:", 3, 15, 7, key="bi_topn")
  with p_cfg3:
    bi_use_bg = st.checkbox(
        "बैकग्राउंड इमेज दिखाएं", value=True, key="bi_bg_chk"
    )
    bi_opacity = st.slider(
        "इमेज Opacity:", 0.06, 0.45, 0.16, 0.02, key="bi_opac"
    )
  with p_cfg4:
    bi_bg_prompt = st.text_input(
        "AI बैकग्राउंड थीम:",
        value="Minimalist soft pastel geometric presentation background, no text",
        key="bi_bg_prm",
    )
    if st.button("🎨 नई AI बैकग्राउंड इमेज", key="bi_new_bg_btn"):
      with st.spinner("बैकग्राउंड बन रहा है..."):
        nim, nsrc = generate_ai_bg_image(api_key, bi_bg_prompt)
        st.session_state.bi_bg_img = nim
        st.session_state.bi_bg_src = nsrc
      st.rerun()

  with st.expander(
      "✏️ Power BI डेटा सेल एडिट करें (Live Update)", expanded=False
  ):
    edited_bi_grid = st.data_editor(
        active_df,
        num_rows="dynamic",
        use_container_width=True,
        key="bi_grid_edit",
    )
    if st.button("💾 एडिट किया हुआ डेटा सेव करें", key="bi_save_grid"):
      st.session_state.uni_df = edited_bi_grid
      st.rerun()

  bi_nums = active_df.select_dtypes(include="number").columns.tolist()
  bi_cats = active_df.select_dtypes(exclude="number").columns.tolist()
  bi_all = active_df.columns.tolist()

  bi_view = active_df.copy()
  slicer_cols = [
      c for c in bi_cats if 1 < active_df[c].nunique(dropna=True) <= 50
  ]
  if slicer_cols:
    st.subheader("🎛️ Slicers / फ़िल्टर्स")
    sc_boxes = st.columns(min(3, len(slicer_cols)))
    for i, sc in enumerate(slicer_cols[:3]):
      s_sl = get_1d_series(active_df, sc)
      u_opts = sorted(s_sl.dropna().astype(str).unique().tolist())
      sel_opts = sc_boxes[i].multiselect(
          f"{sc}:", u_opts, default=u_opts, key=f"pbi_sl_{sc}"
      )
      if sel_opts:
        bi_view = bi_view[get_1d_series(bi_view, sc).astype(str).isin(sel_opts)]

  st.subheader("📌 KPI Cards")
  kc1, kc2, kc3, kc4 = st.columns(4)
  kc1.metric("कुल रिकॉर्ड्स", f"{len(bi_view):,}")
  if bi_nums:
    kpi_num = st.selectbox("KPI Numeric कॉलम:", bi_nums, key="pbi_kpi_num")
    s_kv = pd.to_numeric(
        get_1d_series(bi_view, kpi_num), errors="coerce"
    ).dropna()
    kc2.metric(
        f"कुल {kpi_num}", f"{s_kv.sum():,.2f}" if not s_kv.empty else "0"
    )
    kc3.metric(
        f"औसत {kpi_num}", f"{s_kv.mean():,.2f}" if not s_kv.empty else "0"
    )
  else:
    kpi_num = None
    kc2.metric("Numeric कॉलम", "0")
    kc3.metric("Text कॉलम", str(len(bi_cats)))

  if bi_cats:
    kpi_cat = st.selectbox("KPI Category कॉलम:", bi_cats, key="pbi_kpi_cat")
    vc_k = get_1d_series(bi_view, kpi_cat).dropna().astype(str).value_counts()
    kc4.metric(
        f"Top {kpi_cat}",
        f"{vc_k.index[0]} ({vc_k.iloc[0]})" if not vc_k.empty else "N/A",
    )
  else:
    kc4.metric("Nulls", str(int(bi_view.isna().sum().sum())))

  st.subheader(f"📊 डैशबोर्ड विज़ुअल्स ({bi_chart_count} Charts)")
  all_9_charts = [
      "Horizontal Bar",
      "Bar",
      "Stacked Bar",
      "Line",
      "Area",
      "Donut",
      "Scatter",
      "Histogram",
      "Box",
  ]
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
          ch_type = st.selectbox(
              "Chart Type:",
              all_9_charts,
              index=idx_c % len(all_9_charts),
              key=f"pbi_ct_{idx_c}",
          )
          cc1, cc2, cc3 = st.columns(3)
          def_x = (
              bi_cats[idx_c % len(bi_cats)]
              if bi_cats
              else bi_all[idx_c % len(bi_all)]
          )
          def_y = (
              bi_nums[idx_c % len(bi_nums)]
              if bi_nums
              else bi_all[idx_c % len(bi_all)]
          )
          x_c = cc1.selectbox(
              "X / Category:",
              bi_all,
              index=bi_all.index(def_x),
              key=f"pbi_x_{idx_c}",
          )
          y_c = cc2.selectbox(
              "Y / Metric:",
              bi_all,
              index=bi_all.index(def_y),
              key=f"pbi_y_{idx_c}",
          )
          ag_c = cc3.selectbox(
              "Aggregation:", agg_choices, index=0, key=f"pbi_ag_{idx_c}"
          )

          st_c = None
          if ch_type == "Stacked Bar":
            st_opts = [c for c in bi_all if c != x_c]
            if st_opts:
              st_c = st.selectbox(
                  "Stack / Legend कॉलम:", st_opts, key=f"pbi_st_{idx_c}"
              )

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
        "DAX फ़ॉर्मूला बनवाएं:",
        placeholder="जैसे: Total Revenue, Profit Margin %, या Rank",
        key="pbi_dax_in",
    )
    if st.button("📐 DAX जनरेट करें", key="pbi_dax_btn") and dax_req.strip():
      p_d = (
          f"Write a clean Power BI DAX measure for columns {bi_all} and"
          f" requirement: {dax_req.strip()}. Give short Hindi explanation +"
          " DAX code."
      )
      with st.spinner("DAX बन रहा है..."):
        dt, de = call_gemini_safe(
            api_key,
            p_d,
            enable_ollama=use_ollama_fb,
            ollama_model=ollama_mdl_name,
        )
      st.session_state.bi_dax_code = (
          dt
          if dt
          else (
              f"DAX जनरेट नहीं हो सका ({de})।\nउदाहरण: `Total {kpi_num or 'Value'}"
              f" = SUM('Table'[{kpi_num or 'Amount'}])`"
          )
      )
    if st.session_state.bi_dax_code:
      st.markdown(st.session_state.bi_dax_code)

  with d_col2:
    if st.button("🧠 AI डैशबोर्ड सुझाव लें", key="pbi_rev_btn"):
      p_r = (
          "तुम I'm AI Buddy — एक Power BI विशेषज्ञ हो। इस डैशबोर्ड डेटा के लिए"
          " सरल हिंदी में 4 मुख्य सुझाव और KPI निष्कर्ष लिखो: कॉलम ="
          f" {bi_all}, पंक्तियाँ = {len(bi_view)}"
      )
      with st.spinner("BI रिव्यू तैयार हो रहा है..."):
        rt, re_msg = call_gemini_safe(
            api_key,
            p_r,
            enable_ollama=use_ollama_fb,
            ollama_model=ollama_mdl_name,
        )
      st.session_state.bi_ai_review = (
          rt if rt else f"लोकल BI नोट: कुल पंक्तियाँ = {len(bi_view)} ({re_msg})"
      )
    if st.session_state.bi_ai_review:
      st.markdown(st.session_state.bi_ai_review)

# ------------------------------------------------------------------------------
# TAB 4: DATA SCIENCE, PREDICTIVE MODELING & GEN AI ASSISTANT
# ------------------------------------------------------------------------------
with tab_ds:
  st.header("🧠 Data Science, Predictive Modeling और Gen AI असिस्टेंट")

  ds_nums = active_df.select_dtypes(include="number").columns.tolist()
  ds_cats = active_df.select_dtypes(exclude="number").columns.tolist()
  ds_all = active_df.columns.tolist()

  # --- SECTION A: PREDICTIVE MODELING STUDIO (REGRESSION + CLASSIFICATION + CLUSTERING) ---
  st.subheader("🔮 Predictive Modeling Studio (Train, Evaluate & Predict)")
  if not HAS_SKLEARN:
    st.warning(
        "असली ML Prediction मॉडल चलाने के लिए `scikit-learn` इंस्टॉल करें: `pip"
        " install scikit-learn`"
    )
  else:
    ml_c1, ml_c2, ml_c3, ml_c4 = st.columns([1.2, 1.3, 1.6, 1.0])
    with ml_c1:
      ml_task = st.selectbox(
          "Prediction प्रकार चुनें:",
          ["Regression (Numeric)", "Classification (Category)"],
          key="ds_ml_task",
      )
    with ml_c2:
      if ml_task.startswith("Regression"):
        tgt_opts = ds_nums if ds_nums else ds_all
      else:
        tgt_opts = [
            c for c in ds_all if 1 < active_df[c].nunique(dropna=True) <= 25
        ] or ds_all
      ml_target = st.selectbox("Target कॉलम:", tgt_opts, key="ds_ml_target")
    with ml_c3:
      feat_cands = [c for c in ds_all if c != ml_target]
      ml_features = st.multiselect(
          "Feature कॉलम्स (X):",
          feat_cands,
          default=feat_cands[: min(5, len(feat_cands))],
          key="ds_ml_feats",
      )
    with ml_c4:
      ml_test_sz = st.slider(
          "Test Size:", 0.15, 0.40, 0.25, 0.05, key="ds_ml_test"
      )

    if st.button(
        "🚀 सभी मॉडल ट्रेन करें और तुलना देखें",
        type="primary",
        key="ds_train_btn",
    ):
      if not ml_features:
        st.error("कृपया कम से कम 1 Feature कॉलम चुनें।")
      else:
        with st.spinner("मॉडल ट्रेन हो रहे हैं..."):
          if ml_task.startswith("Regression"):
            res_b, err_b = train_regression_suite(
                active_df, ml_target, ml_features, ml_test_sz
            )
          else:
            res_b, err_b = train_classification_suite(
                active_df, ml_target, ml_features, ml_test_sz
            )
        if err_b:
          st.error(err_b)
        else:
          st.session_state.ds_trained_bundle = res_b

    bundle = st.session_state.ds_trained_bundle
    if bundle is not None:
      st.success(f"सर्वश्रेष्ठ मॉडल चुना गया: **{bundle['best_name']}**")
      m_left, m_right = st.columns([1.15, 1.25])
      with m_left:
        st.markdown("**Model Leaderboard (तुलना)**")
        st.dataframe(bundle["leaderboard"], use_container_width=True)
        if bundle.get("imp_df") is not None and not bundle["imp_df"].empty:
          st.markdown("**Top Feature Importances**")
          st.dataframe(bundle["imp_df"], use_container_width=True)

      with m_right:
        if bundle["task"] == "regression":
          fig_p, ax_p = plt.subplots(figsize=(6.0, 3.8), dpi=120)
          ax_p.scatter(
              bundle["y_test"],
              bundle["y_pred"],
              color="#2563EB",
              alpha=0.8,
              edgecolors="white",
          )
          mn_v = min(bundle["y_test"].min(), bundle["y_pred"].min())
          mx_v = max(bundle["y_test"].max(), bundle["y_pred"].max())
          ax_p.plot(
              [mn_v, mx_v], [mn_v, mx_v], linestyle="--", color="#DC2626"
          )
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
              ax_cm.text(
                  j,
                  i,
                  str(cm_arr[i, j]),
                  ha="center",
                  va="center",
                  color="#111827",
                  fontweight="bold",
              )
          ax_cm.set_xlabel("Predicted")
          ax_cm.set_ylabel("Actual")
          ax_cm.set_title("Confusion Matrix", fontsize=10.5, fontweight="bold")
          plt.tight_layout()
          st.pyplot(fig_cm, use_container_width=True)
          plt.close(fig_cm)

      # --- LIVE SINGLE-RECORD PREDICTOR ---
      st.markdown("#### 🎯 Live Prediction (नए इनपुट पर अनुमान निकालें)")
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
          inp_dict[fc_name] = ui_b.number_input(
              f"{fc_name}:", value=med_v, key=f"pred_in_{fc_name}"
          )
        else:
          opts_c = sorted(s_ref.dropna().astype(str).unique().tolist())
          if not opts_c:
            opts_c = ["Unknown"]
          inp_dict[fc_name] = ui_b.selectbox(
              f"{fc_name}:", opts_c, key=f"pred_sel_{fc_name}"
          )

      if st.button("🔮 Predict करें", key="ds_do_predict_btn"):
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

    # --- K-MEANS CLUSTERING (UNSUPERVISED SEGMENTATION) ---
    with st.expander(
        "🧩 K-Means Clustering / Customer-Record Segmentation", expanded=False
    ):
      if len(ds_nums) >= 2:
        cl_c1, cl_c2 = st.columns([2, 1])
        cl_cols = cl_c1.multiselect(
            "Clustering के लिए Numeric कॉलम्स:",
            ds_nums,
            default=ds_nums[: min(3, len(ds_nums))],
            key="ds_km_cols",
        )
        k_val = cl_c2.slider("Clusters (K):", 2, 6, 3, key="ds_km_k")
        if st.button("🧩 Clusters बनाएं", key="ds_km_btn") and len(cl_cols) >= 2:
          km_sub = active_df[cl_cols].apply(pd.to_numeric, errors="coerce")
          valid_mask = km_sub.notna().all(axis=1)
          if int(valid_mask.sum()) >= k_val:
            scaler_km = StandardScaler()
            scaled_arr = scaler_km.fit_transform(km_sub[valid_mask])
            km_model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            labels_km = km_model.fit_predict(scaled_arr)
            res_km = km_sub[valid_mask].copy()
            res_km["Cluster"] = labels_km
            st.dataframe(
                res_km.groupby("Cluster").agg(["count", "mean"]).round(2),
                use_container_width=True,
            )
          else:
            st.info("Clustering के लिए पर्याप्त Numeric डेटा नहीं मिला।")
      else:
        st.info("K-Means Clustering के लिए कम से कम 2 Numeric कॉलम चाहिए।")

  st.divider()

  # --- SECTION B: DATA SCIENCE PROFILES & VISUALS ---
  st.subheader("📑 सभी कॉलम्स की Data Science प्रोफ़ाइल")
  st.dataframe(build_all_columns_profile(active_df), use_container_width=True)

  st.subheader("📊 Data Science चार्ट्स (Heatmap, Distribution & Categories)")
  vc_row1_l, vc_row1_r = st.columns(2)

  with vc_row1_l:
    st.markdown("**1. सभी Numeric कॉलम्स — Correlation Matrix & Heatmap**")
    if len(ds_nums) >= 2:
      corr_df = active_df[ds_nums].corr().round(3)
      st.dataframe(corr_df, use_container_width=True)
      fig_hm = render_heatmap_fig(corr_df)
      st.pyplot(fig_hm, use_container_width=True)
      plt.close(fig_hm)
    else:
      st.info("Heatmap के लिए कम से कम 2 Numeric कॉलम चाहिए।")

  with vc_row1_r:
    st.markdown("**2. किसी भी Numeric कॉलम का Distribution चार्ट**")
    if ds_nums:
      dist_col_sel = st.selectbox(
          "Distribution के लिए Numeric कॉलम चुनें:",
          ds_nums,
          key="ds_dist_col_sel",
      )
      fig_dist = render_dist_fig(
          get_1d_series(active_df, dist_col_sel), dist_col_sel
      )
      st.pyplot(fig_dist, use_container_width=True)
      plt.close(fig_dist)
    else:
      st.info("Distribution चार्ट के लिए Numeric कॉलम नहीं है।")

  vc_row2_l, vc_row2_r = st.columns(2)
  with vc_row2_l:
    st.markdown("**3. Category / Text कॉलम का वितरण चार्ट**")
    if ds_cats:
      cat_col_sel = st.selectbox(
          "Category कॉलम चुनें:", ds_cats, key="ds_cat_col_sel"
      )
      fig_cat = render_cat_bar_fig(
          get_1d_series(active_df, cat_col_sel), cat_col_sel
      )
      st.pyplot(fig_cat, use_container_width=True)
      plt.close(fig_cat)
    else:
      st.info("Category कॉलम नहीं है।")

  with vc_row2_r:
    st.markdown("**4. दो कॉलम्स का Scatter / Relationship चार्ट**")
    if len(ds_all) >= 2:
      sc_x = st.selectbox("X-Axis कॉलम:", ds_all, index=0, key="ds_sc_x")
      sc_y_def = 1 if len(ds_nums) == 0 else ds_all.index(ds_nums[0])
      sc_y = st.selectbox("Y-Axis कॉलम:", ds_all, index=sc_y_def, key="ds_sc_y")
      s_sc_x = get_1d_series(active_df, sc_x)
      s_sc_y = get_1d_series(active_df, sc_y)
      fig_sc, ax_sc = plt.subplots(figsize=(6.2, 4.2), dpi=120)
      ax_sc.scatter(
          s_sc_x.astype(str)
          if not pd.api.types.is_numeric_dtype(s_sc_x)
          else pd.to_numeric(s_sc_x, errors="coerce"),
          pd.to_numeric(s_sc_y, errors="coerce"),
          color="#7C3AED",
          alpha=0.75,
          edgecolors="white",
      )
      ax_sc.set_title(f"{sc_y} vs {sc_x}", fontsize=10.5, fontweight="bold")
      ax_sc.set_xlabel(str(sc_x))
      ax_sc.set_ylabel(str(sc_y))
      plt.xticks(rotation=30, ha="right")
      ax_sc.grid(True, linestyle="--", alpha=0.3)
      plt.tight_layout()
      st.pyplot(fig_sc, use_container_width=True)
      plt.close(fig_sc)

  st.divider()
  ds_left, ds_right = st.columns([1.1, 1.3])

  with ds_left:
    st.subheader("🎯 Outlier जांच (IQR Method) और ML रोडमैप")
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
              "Normal Range": f"{low:,.2f} से {high:,.2f}",
          })
      if outlier_rows:
        st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)
    else:
      st.info("Outlier जांच के लिए Numeric कॉलम नहीं है।")

    if st.button("🤖 AI से ML / Modeling रोडमैप लें", key="ds_ml_btn"):
      base_info = build_local_stats_summary(active_df)
      p_ds = (
          "तुम I'm AI Buddy — एक Data Scientist हो। इस डेटासेट के आधार पर सरल"
          " हिंदी में बताओ:\n1) कौन-सा ML प्रॉब्लम (Regression /"
          " Classification / Clustering) बन सकता है,\n2) कौन-से फीचर्स और"
          " मॉडल सबसे अच्छे रहेंगे,\n3) डेटा प्री-प्रोसेसिंग में क्या ध्यान"
          " रखें।\n"
          + base_info
      )
      with st.spinner("Data Science प्लान तैयार हो रहा है..."):
        m_txt, m_err = call_gemini_safe(
            api_key,
            p_ds,
            enable_ollama=use_ollama_fb,
            ollama_model=ollama_mdl_name,
        )
      st.session_state.ds_ml_report = (
          m_txt if m_txt else f"{base_info}\n\n*(AI उपलब्ध नहीं: {m_err})*"
      )

    if st.session_state.ds_ml_report:
      st.markdown(st.session_state.ds_ml_report)

  with ds_right:
    st.subheader("💬 अपने डेटा के साथ Gen AI चैट")
    for msg in st.session_state.ds_chat_history:
      with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

    user_q = st.text_input(
        "अपने डेटासेट के बारे में कोई भी सवाल पूछें:",
        placeholder="जैसे: किस Region का प्रदर्शन सबसे अच्छा है और क्यों?",
        key="ds_chat_in",
    )
    c_send1, c_send2 = st.columns([1, 1])
    with c_send1:
      send_q = st.button("📨 सवाल भेजें", type="primary", key="ds_send_btn")
    with c_send2:
      if st.button("🧹 चैट साफ़ करें", key="ds_clear_btn"):
        st.session_state.ds_chat_history = []
        st.rerun()

    if send_q and user_q.strip():
      st.session_state.ds_chat_history.append(
          {"role": "user", "content": user_q.strip()}
      )
      context_snippet = (
          f"Dataset shape: {active_df.shape}\n"
          f"Columns: {list(active_df.columns)}\n"
          f"Sample rows:\n{active_df.head(10).to_string(index=False)}\n"
          f"Summary:\n{build_local_stats_summary(active_df)}"
      )
      chat_prompt = (
          "तुम I'm AI Buddy हो — एक सहायक AI डेटा एनालिस्ट। नीचे दिए गए"
          " डेटासेट के आधार पर यूज़र के सवाल का सटीक जवाब सरल हिंदी में"
          f" दो।\n\n{context_snippet}\n\nसवाल: {user_q.strip()}"
      )
      with st.spinner("जवाब तैयार हो रहा है..."):
        reply, r_err = call_gemini_safe(
            api_key,
            chat_prompt,
            enable_ollama=use_ollama_fb,
            ollama_model=ollama_mdl_name,
        )
      if not reply:
        reply = (
            f"AI इस समय उपलब्ध नहीं है ({r_err})।\n\n"
            + build_local_stats_summary(active_df)
        )
      st.session_state.ds_chat_history.append(
          {"role": "assistant", "content": reply}
      )
      st.rerun()