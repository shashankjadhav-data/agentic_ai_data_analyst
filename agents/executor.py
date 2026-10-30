# agentic_ai_analyst/agents/executor.py

import json
import textwrap
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from .base import Agent

try:
    from scipy import sparse as sp
except Exception:  # pragma: no cover
    sp = None


# ---------------- Prompts ----------------

MODELSPEC_SYSTEM = """You are an ML pipeline designer for ANY tabular dataset.

Return ONLY valid JSON for EXACTLY THIS schema (one object, top-level keys must match).
Do not wrap it in "modelspecs" or any other container. Do not output arrays at the top level.
No prose, no code, no markdown. Use DOUBLE QUOTES.

Required top-level keys (must all exist):
target, problemtype, split, features, preprocessing, models, selection, confusionmatrix, reasoningshort

Schema:
{
  "target": str,
  "problemtype": "classification" | "regression",
  "split": {"testsize": 0.2, "randomstate": 42, "stratify": boolean},
  "features": {"numeric": [str,...], "categorical": [str,...], "text": [str,...], "drop": [str,...]},
  "preprocessing": {
    "numericimputer": "median",
    "numericscaler": "standard",
    "categoricalencoder": "onehot",
    "onehotminfrequency": int,
    "textvectorizer": "HashingVectorizer" | "TfidfVectorizer",
    "textngram": [int,int],
    "maxtextfeatures": int
  },
  "models": [
    {"tag": "baseline", "name": str, "hyperparams": {"k":"v"}, "classweight": "balanced" | null},
    {"tag": "advanced", "name": str, "hyperparams": {"k":"v"}, "classweight": "balanced" | null}
  ],
  "selection": {"metric": "accuracy" | "f1weighted" | "r2" | "rmse", "direction": "maximize" | "minimize"},
  "confusionmatrix": {"enabled": boolean, "maxclasses": int},
  "reasoningshort": str
}

Rules:
- Never include the target column in features.numeric/categorical/text.
- Put ID-like columns into features.drop.
- Required preprocessing fields must NOT be null:
  textvectorizer, textngram, maxtextfeatures, onehotminfrequency.
- If categorical/text features are used, do NOT choose tree/boosting models.
Allowed classification models: LogisticRegression, LinearSVC, SGDClassifier, RandomForestClassifier, HistGradientBoostingClassifier
Allowed regression models: Ridge, SGDRegressor, RandomForestRegressor, HistGradientBoostingRegressor
"""

MODELSPEC_FIXER_SYSTEM = """You are a strict JSON repair agent.

Input: candidate_spec + validation_errors.
Output: ONLY corrected JSON for the EXACT schema below (one object).
Do not output "modelspecs". Do not output a list. No prose.

Top-level keys required:
target, problemtype, split, features, preprocessing, models, selection, confusionmatrix, reasoningshort

Hard requirements:
- preprocessing.textvectorizer must be "HashingVectorizer" or "TfidfVectorizer"
- preprocessing.textngram must be [int,int]
- preprocessing.maxtextfeatures must be int >= 2000
- preprocessing.onehotminfrequency must be int >= 1
- models must be exactly two objects with tag baseline and advanced
- models.name must be valid for the given problemtype (no classifier names in regression and vice versa)
- Do not include target in any feature list
- If categorical/text features exist, advanced model must be sparse-friendly:
  classification -> LogisticRegression, LinearSVC, SGDClassifier
  regression -> Ridge, SGDRegressor
"""


# ---------------- Allowed models ----------------

ALLOWEDMODELS = {
    "classification": {
        "LogisticRegression": LogisticRegression,
        "LinearSVC": LinearSVC,
        "SGDClassifier": SGDClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    },
    "regression": {
        "Ridge": Ridge,
        "SGDRegressor": SGDRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    },
}

TREEMODELS = {
    "RandomForestClassifier",
    "HistGradientBoostingClassifier",
    "RandomForestRegressor",
    "HistGradientBoostingRegressor",
}

ALLOWEDVECTORIZERS = {"HashingVectorizer", "TfidfVectorizer"}


# ---------------- JSON helpers ----------------

def extractfirstbalancedjsonobject(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    start = s.find("{")
    if start == -1:
        return None

    instring = False
    escape = False
    depth = 0

    for i in range(start, len(s)):
        ch = s[i]

        if instring:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                instring = False
            continue

        if ch == '"':
            instring = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return None


def parsejsonlenient(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty JSON text")

    if "```" in text:
        parts = text.split("```")
        fenced = [p for p in parts if "{" in p and "}" in p]
        if fenced:
            text = max(fenced, key=len).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    cand = extractfirstbalancedjsonobject(text)
    if not cand:
        raise ValueError("No JSON object found in LLM output")

    return json.loads(cand)


def islistofstr(x) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)


def ensurelist(spec: Dict[str, Any], path: List[str]) -> List[str]:
    cur = spec
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    leaf = path[-1]
    if leaf not in cur or cur[leaf] is None:
        cur[leaf] = []
    if not isinstance(cur[leaf], list):
        cur[leaf] = []
    cur[leaf] = [x for x in cur[leaf] if isinstance(x, str)]
    return cur[leaf]


# ---------------- Robust transformers ----------------

class CountFrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Safe encoding for high-cardinality categoricals: maps category -> frequency (train).
    Output is dense numeric matrix (n_rows, n_cols).
    """

    def __init__(self, normalize: bool = True):
        self.normalize = bool(normalize)
        self.maps: List[Dict[Any, float]] = []
        self.ntrain: int = 1

    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            Xdf = X.to_frame()
        elif isinstance(X, pd.DataFrame):
            Xdf = X
        else:
            Xdf = pd.DataFrame(X)

        self.ntrain = max(int(len(Xdf)), 1)
        self.maps = []

        for col in Xdf.columns:
            s = Xdf[col].astype("object")
            vc = s.value_counts(dropna=False)
            if self.normalize:
                mp = {k: float(v) / float(self.ntrain) for k, v in vc.items()}
            else:
                mp = {k: float(v) for k, v in vc.items()}
            self.maps.append(mp)

        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            Xdf = X.to_frame()
        elif isinstance(X, pd.DataFrame):
            Xdf = X
        else:
            Xdf = pd.DataFrame(X)

        out = np.zeros((len(Xdf), Xdf.shape[1]), dtype=np.float32)
        for j, col in enumerate(Xdf.columns):
            mp = self.maps[j] if j < len(self.maps) else {}
            s = Xdf[col].astype("object")
            out[:, j] = s.map(mp).fillna(0.0).astype(np.float32).to_numpy()

        return out


class DateTimeFeaturizer(BaseEstimator, TransformerMixin):
    """
    Turns a single column into numeric time features: year, month, day, dayofweek, hour.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            s = X
        elif isinstance(X, pd.DataFrame) and X.shape[1] == 1:
            s = X.iloc[:, 0]
        else:
            arr = np.asarray(X).reshape(-1)
            s = pd.Series(arr)

        dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

        year = dt.dt.year.fillna(0).astype(np.int32).to_numpy()
        month = dt.dt.month.fillna(0).astype(np.int32).to_numpy()
        day = dt.dt.day.fillna(0).astype(np.int32).to_numpy()
        dow = dt.dt.dayofweek.fillna(0).astype(np.int32).to_numpy()
        hour = dt.dt.hour.fillna(0).astype(np.int32).to_numpy()

        return np.vstack([year, month, day, dow, hour]).T.astype(np.float32)


# ---------------- Feature heuristics ----------------

TEXTNAMEHINTS = ("desc", "description", "text", "comment", "notes", "message", "review", "title", "summary", "body")
DATENAMEHINTS = ("date", "time", "timestamp", "datetime", "created", "updated")


def istextname(col: str) -> bool:
    c = (col or "").lower()
    return any(h in c for h in TEXTNAMEHINTS)


def isdatename(col: str) -> bool:
    c = (col or "").lower()
    return any(h in c for h in DATENAMEHINTS)


def safenunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except Exception:
        return int(pd.Series(s).nunique(dropna=True))


def safeuniqueratio(s: pd.Series) -> float:
    nonnull = int(pd.Series(s).notna().sum())
    nun = safenunique(pd.Series(s))
    return float(nun / nonnull) if nonnull else 0.0


def idlikecolumns(df: pd.DataFrame, target: Optional[str]) -> List[str]:
    drop: List[str] = []
    n = int(len(df))

    for c in df.columns:
        if c == target:
            continue

        name_low = c.lower()
        s = df[c]

        if ("id" in name_low) or ("uuid" in name_low) or ("key" in name_low):
            drop.append(c)
            continue

        if n >= 200 and (s.dtype == "object" or pd.api.types.is_categorical_dtype(s)):
            if safeuniqueratio(s) >= 0.98:
                drop.append(c)

    return list(dict.fromkeys(drop))


def numericlikecols(df: pd.DataFrame, target: Optional[str]) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c == target:
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
            continue

        if df[c].dtype == "object":
            s = pd.to_numeric(df[c].astype(str).replace({"\\N": np.nan}), errors="coerce")
            rate = float(s.notna().mean())
            if rate >= 0.85 and int(s.nunique(dropna=True)) >= 5:
                cols.append(c)

    return cols


def lowcardcategoricals(df: pd.DataFrame, target: Optional[str], maxclasses: int = 80) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c == target:
            continue
        if df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c]):
            nun = safenunique(df[c])
            if 2 <= nun <= maxclasses:
                cols.append(c)

    cols.sort(key=lambda x: safenunique(df[x]))
    return cols


def textlike(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = df[col]
    if s.dtype != "object":
        return False

    ss = s.dropna().astype(str)
    if len(ss) < 50:
        return istextname(col)

    sample = ss.head(1000)
    avglen = float(sample.str.len().mean()) if len(sample) else 0.0
    nun = int(sample.nunique())
    uniquer = float(nun / max(len(sample), 1))

    if istextname(col) and avglen >= 5:
        return True

    return (avglen >= 10.0 and nun >= 100 and uniquer >= 0.10) or (avglen >= 25.0)


def datetimelikeseries(s: pd.Series, namehint: bool = False) -> bool:
    if s is None or len(s) == 0:
        return False
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype != "object":
        return False

    ss = s.dropna().astype(str)
    if len(ss) < 50 and not namehint:
        return False

    sample = ss.head(1000)
    dt = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
    okrate = float(dt.notna().mean()) if len(dt) else 0.0

    return okrate >= (0.60 if namehint else 0.85)


# ---------------- Model spec sanitation ----------------

def sanitizemodelsinplace(spec: Dict[str, Any]) -> None:
    pt = spec.get("problemtype")
    if pt not in ("classification", "regression"):
        return

    allowed = set(ALLOWEDMODELS[pt].keys())

    models = spec.get("models")
    if not isinstance(models, list):
        models = []
        spec["models"] = models

    def defaultmodel(tag: str) -> Dict[str, Any]:
        if pt == "classification":
            if tag == "baseline":
                return {"tag": "baseline", "name": "LogisticRegression", "hyperparams": {"C": 1.0}, "classweight": None}
            return {"tag": "advanced", "name": "LinearSVC", "hyperparams": {"C": 1.0}, "classweight": "balanced"}
        else:
            if tag == "baseline":
                return {"tag": "baseline", "name": "Ridge", "hyperparams": {"alpha": 1.0}, "classweight": None}
            return {"tag": "advanced", "name": "SGDRegressor", "hyperparams": {"alpha": 0.0001}, "classweight": None}

    keep: List[Dict[str, Any]] = []
    for tag in ("baseline", "advanced"):
        found = None
        for m in models:
            if isinstance(m, dict) and m.get("tag") == tag:
                found = m
                break
        if found is None:
            found = defaultmodel(tag)
        keep.append(found)

    models[:] = keep

    for m in models:
        if not isinstance(m, dict):
            continue
        tag = m.get("tag") if isinstance(m.get("tag"), str) else "baseline"
        name = m.get("name")
        if not isinstance(name, str) or name not in allowed:
            repl = defaultmodel(tag)
            m["name"] = repl["name"]
            m["hyperparams"] = repl.get("hyperparams", {})
            m["classweight"] = repl.get("classweight", None)

        if pt == "regression":
            m["classweight"] = None

        hp = m.get("hyperparams")
        if not isinstance(hp, dict):
            hp = {}
            m["hyperparams"] = hp

        if pt == "regression":
            hp.pop("C", None)
            hp.pop("classweight", None)


def enforcesparsesafemodels(spec: Dict[str, Any]) -> None:
    feats = spec.get("features") or {}
    if not isinstance(feats, dict):
        return
    uses_sparse = bool(feats.get("categorical")) or bool(feats.get("text"))
    if not uses_sparse:
        return

    pt = spec.get("problemtype")
    if pt not in ("classification", "regression"):
        return

    models = spec.get("models")
    if not isinstance(models, list) or len(models) != 2:
        return

    for m in models:
        if not isinstance(m, dict):
            continue
        if m.get("name") in TREEMODELS:
            if pt == "classification":
                m["name"] = "LinearSVC"
                m["hyperparams"] = {"C": 1.0}
                m["classweight"] = "balanced"
            else:
                m["name"] = "SGDRegressor"
                m["hyperparams"] = {"alpha": 0.0001}
                m["classweight"] = None


def normalizespecinplace(spec: Dict[str, Any], df: pd.DataFrame) -> None:
    if not isinstance(spec, dict):
        return

    target = spec.get("target")
    if not isinstance(target, str) or target not in df.columns:
        return

    if spec.get("problemtype") not in ("classification", "regression"):
        spec["problemtype"] = "classification"

    split = spec.setdefault("split", {})
    if not isinstance(split, dict):
        split = {}
        spec["split"] = split
    split.setdefault("testsize", 0.2)
    split.setdefault("randomstate", 42)
    split.setdefault("stratify", True)

    if spec["problemtype"] == "regression":
        split["stratify"] = False
    else:
        split["stratify"] = bool(split.get("stratify", True))

    feats = spec.setdefault("features", {})
    if not isinstance(feats, dict):
        feats = {}
        spec["features"] = feats

    numeric = ensurelist(spec, ["features", "numeric"])
    categorical = ensurelist(spec, ["features", "categorical"])
    text = ensurelist(spec, ["features", "text"])
    drop = ensurelist(spec, ["features", "drop"])

    for arr in (numeric, categorical, text, drop):
        if target in arr:
            arr[:] = [c for c in arr if c != target]

    numeric[:] = [c for c in numeric if c in df.columns and c != target]
    categorical[:] = [c for c in categorical if c in df.columns and c != target]
    text[:] = [c for c in text if c in df.columns and c != target]
    drop[:] = [c for c in drop if c in df.columns and c != target]

    for c in list(categorical):
        if textlike(df, c):
            categorical.remove(c)
            if c not in text:
                text.append(c)

    idlike = set(idlikecolumns(df, target))
    for c in list(numeric):
        if c in idlike:
            numeric.remove(c)
            drop.append(c)
    for c in list(categorical):
        if c in idlike:
            categorical.remove(c)
            drop.append(c)
    for c in list(text):
        if c in idlike:
            text.remove(c)
            drop.append(c)

    drop[:] = list(dict.fromkeys([c for c in drop if c in df.columns and c != target]))

    if len(numeric) == 0:
        numeric[:] = [c for c in numericlikecols(df, target) if c not in set(drop)]

    if len(text) == 0:
        candidates = [c for c in df.columns if c != target and textlike(df, c) and c not in set(drop)]
        text[:] = candidates[:3]

    if len(categorical) == 0:
        catcandidates = lowcardcategoricals(df, target, maxclasses=80)
        categorical[:] = [c for c in catcandidates if c not in set(drop)]

    pre = spec.setdefault("preprocessing", {})
    if not isinstance(pre, dict):
        pre = {}
        spec["preprocessing"] = pre

    pre.setdefault("numericimputer", "median")
    pre.setdefault("numericscaler", "standard")
    pre.setdefault("categoricalencoder", "onehot")

    if pre.get("onehotminfrequency") is None or not isinstance(pre.get("onehotminfrequency"), int):
        pre["onehotminfrequency"] = 10
    if int(pre["onehotminfrequency"]) < 1:
        pre["onehotminfrequency"] = 1

    if pre.get("textvectorizer") not in ALLOWEDVECTORIZERS:
        pre["textvectorizer"] = "TfidfVectorizer" if len(text) > 0 else "HashingVectorizer"

    ngram = pre.get("textngram")
    if not (isinstance(ngram, list) and len(ngram) == 2 and all(isinstance(i, int) for i in ngram)):
        pre["textngram"] = [1, 2]

    if pre.get("maxtextfeatures") is None or not isinstance(pre.get("maxtextfeatures"), int):
        pre["maxtextfeatures"] = 2000
    if int(pre["maxtextfeatures"]) < 2000:
        pre["maxtextfeatures"] = 2000

    cm = spec.setdefault("confusionmatrix", {})
    if not isinstance(cm, dict):
        spec["confusionmatrix"] = {"enabled": True, "maxclasses": 15}
    else:
        if "enabled" not in cm or not isinstance(cm.get("enabled"), bool):
            cm["enabled"] = True
        if "maxclasses" not in cm or not isinstance(cm.get("maxclasses"), int):
            cm["maxclasses"] = 15

    if spec["problemtype"] == "regression":
        spec["confusionmatrix"]["enabled"] = False

    sel = spec.setdefault("selection", {})
    if not isinstance(sel, dict):
        sel = {}
        spec["selection"] = sel
    sel.setdefault("metric", "f1weighted" if spec["problemtype"] == "classification" else "r2")
    sel.setdefault("direction", "maximize")

    if "reasoningshort" not in spec or not isinstance(spec.get("reasoningshort"), str):
        spec["reasoningshort"] = "Auto-normalized spec."

    sanitizemodelsinplace(spec)
    enforcesparsesafemodels(spec)


def validatespec(spec: Dict[str, Any], df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    required = ["target", "problemtype", "split", "features", "preprocessing", "models", "selection", "confusionmatrix", "reasoningshort"]
    for k in required:
        if k not in spec:
            errors.append(f"Missing key: {k}")
    if errors:
        return False, errors

    target = spec.get("target")
    if not isinstance(target, str) or target not in df.columns:
        errors.append("target must be a column name present in df")

    pt = spec.get("problemtype")
    if pt not in ("classification", "regression"):
        errors.append("problemtype must be classification|regression")

    split = spec.get("split")
    if not isinstance(split, dict):
        errors.append("split must be an object")
    else:
        if not isinstance(split.get("testsize"), (int, float)) or not (0.05 <= float(split["testsize"]) <= 0.5):
            errors.append("split.testsize must be between 0.05 and 0.5")
        if not isinstance(split.get("randomstate"), int):
            errors.append("split.randomstate must be int")
        if not isinstance(split.get("stratify"), bool):
            errors.append("split.stratify must be boolean")
        if pt == "regression" and split.get("stratify", False):
            errors.append("split.stratify must be false for regression")

    feats = spec.get("features")
    if not isinstance(feats, dict):
        errors.append("features must be an object")
    else:
        for group in ["numeric", "categorical", "text", "drop"]:
            if group not in feats:
                errors.append(f"features.{group} missing")
            elif not islistofstr(feats[group]):
                errors.append(f"features.{group} must be list[str]")
        for group in ["numeric", "categorical", "text"]:
            for c in feats.get(group, []):
                if c not in df.columns:
                    errors.append(f"features.{group} contains unknown column: {c}")
                if c == target:
                    errors.append(f"features.{group} must not include target: {c}")

    pre = spec.get("preprocessing")
    if not isinstance(pre, dict):
        errors.append("preprocessing must be an object")
    else:
        if pre.get("textvectorizer") not in ALLOWEDVECTORIZERS:
            errors.append("preprocessing.textvectorizer must be HashingVectorizer|TfidfVectorizer (no null)")
        ngram = pre.get("textngram")
        if not (isinstance(ngram, list) and len(ngram) == 2 and all(isinstance(i, int) for i in ngram)):
            errors.append("preprocessing.textngram must be [int,int] (no null)")
        if not isinstance(pre.get("maxtextfeatures"), int) or int(pre["maxtextfeatures"]) < 2000:
            errors.append("preprocessing.maxtextfeatures must be int >= 2000 (no null)")
        if not isinstance(pre.get("onehotminfrequency"), int) or int(pre["onehotminfrequency"]) < 1:
            errors.append("preprocessing.onehotminfrequency must be int >= 1 (no null)")

    models = spec.get("models")
    if not isinstance(models, list) or len(models) != 2:
        errors.append("models must be a list of exactly 2 model objects")
    else:
        tags = [m.get("tag") for m in models if isinstance(m, dict)]
        if sorted(tags) != ["advanced", "baseline"]:
            errors.append("models must contain exactly one baseline and one advanced")
        for m in models:
            if not isinstance(m, dict):
                errors.append("each models[] entry must be an object")
                continue
            name = m.get("name")
            if pt in ALLOWEDMODELS and (not isinstance(name, str) or name not in ALLOWEDMODELS[pt]):
                errors.append(f"models.name '{name}' not allowed for {pt}")

    cm = spec.get("confusionmatrix")
    if pt == "regression" and isinstance(cm, dict) and cm.get("enabled", False):
        errors.append("confusionmatrix.enabled must be false for regression")

    return (len(errors) == 0), errors


# ---------------- Executor agent ----------------

class AnalysisExecutorAgent(Agent):
    def __init__(self, name, llm_client, bus, figures_dir: Path):
        super().__init__(name, llm_client, bus)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.high_card_threshold = 400
        self.max_text_columns = 3

    # ---------- small helpers ----------

    def _resolve_column_name(self, df: pd.DataFrame, name: Optional[str]) -> Optional[str]:
        if not name or not isinstance(name, str):
            return None
        if name in df.columns:
            return name
        n = name.strip()
        if n in df.columns:
            return n
        low = n.lower()
        for c in df.columns:
            if isinstance(c, str) and c.lower() == low:
                return c
        return None

    def _get_plan_dict(self, plan: Any, *keys: str) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            return {}
        for k in keys:
            v = plan.get(k)
            if isinstance(v, dict):
                return v
        return {}

    def _get_plan_bool(self, plan: Any, *keys: str) -> Optional[bool]:
        if not isinstance(plan, dict):
            return None
        for k in keys:
            if k in plan:
                return bool(plan.get(k))
        return None

    # ---------- plotting ----------

    def _savefig(self, filename: str) -> None:
        plt.tight_layout()
        plt.savefig(self.figures_dir / filename, dpi=120)
        plt.close()

    def _plot_text_card(self, title: str, text: str, filename: str) -> None:
        txt = (text or "").strip()
        if not txt:
            return
        wrapped = "\n".join(textwrap.fill(line, width=110) for line in txt.splitlines())
        plt.figure(figsize=(13, 7))
        plt.axis("off")
        plt.text(0.01, 0.97, title, fontsize=16, fontweight="bold", va="top")
        plt.text(0.01, 0.92, wrapped, fontsize=11, va="top", family="monospace")
        self._savefig(filename)

    def _eda_schema_overview_figure(self, df: pd.DataFrame, max_rows: int = 35) -> None:
        rows = []
        for c in df.columns:
            s = df[c]
            rows.append(
                {
                    "column": c,
                    "dtype": str(s.dtype),
                    "nunique": int(s.nunique(dropna=True)),
                    "missing_%": round(float(s.isna().mean() * 100.0), 2),
                    "sample": ", ".join(s.dropna().astype(str).head(2).tolist()),
                }
            )
        summ = pd.DataFrame(rows).sort_values(["missing_%", "nunique"], ascending=[False, False]).head(max_rows)

        fig, ax = plt.subplots(figsize=(14, 0.35 * (len(summ) + 3)))
        ax.axis("off")
        tbl = ax.table(cellText=summ.values, colLabels=summ.columns, loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.2)
        plt.title("Schema overview (top columns by missingness/cardinality)", pad=12)
        self._savefig("eda_table_schema_overview.png")

    def _eda_missingness_bar(self, df: pd.DataFrame) -> None:
        miss_pct = df.isna().mean().sort_values(ascending=False) * 100.0
        miss_pct = miss_pct[miss_pct > 0].head(40)
        if len(miss_pct) == 0:
            return
        plt.figure(figsize=(10, 6))
        sns.barplot(x=miss_pct.values, y=miss_pct.index)
        plt.title("Missingness (%), top columns")
        plt.xlabel("Percent missing")
        plt.ylabel("Column")
        self._savefig("eda_bar_missingness.png")

    def _eda_missingness_heatmap(self, df: pd.DataFrame, max_cols: int = 30) -> None:
        cols = list(df.columns[:max_cols])
        if len(cols) == 0:
            return
        m = df[cols].isna().astype(int)
        if m.values.sum() == 0:
            return
        plt.figure(figsize=(12, 5))
        sns.heatmap(m.sample(min(len(m), 2000), random_state=42), cbar=False)
        plt.title("Missingness heatmap (sample rows)")
        plt.xlabel("Columns")
        plt.ylabel("Rows (sample)")
        self._savefig("eda_heatmap_missingness.png")

    def _eda_target_distribution(self, df: pd.DataFrame, target: str) -> None:
        if target not in df.columns:
            return
        s = df[target].dropna()
        if pd.api.types.is_numeric_dtype(s):
            plt.figure(figsize=(10, 5))
            sns.histplot(s, bins=40)
            plt.title(f"Target distribution: {target}")
            plt.xlabel(target)
            plt.ylabel("Count")
            self._savefig("eda_hist_target_distribution.png")
            return

        s2 = df[target].astype(str).fillna("NA")
        vc = s2.value_counts().head(25)
        if len(vc) < 2:
            return
        plt.figure(figsize=(10, 5))
        sns.barplot(x=vc.values, y=vc.index)
        plt.title(f"Target distribution (top classes): {target}")
        plt.xlabel("Count")
        plt.ylabel(target)
        self._savefig("eda_bar_target_distribution.png")

    def _eda_numeric_distributions(self, df: pd.DataFrame, numeric_cols: List[str], max_cols: int = 6) -> None:
        cols = [c for c in numeric_cols if c in df.columns][:max_cols]
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) < 30:
                continue
            s = s.sample(min(len(s), 50000), random_state=42)
            plt.figure(figsize=(10, 5))
            sns.histplot(s, bins=40, kde=True)
            plt.title(f"Histogram: {c}")
            plt.xlabel(c)
            plt.ylabel("Count")
            self._savefig(f"eda_hist_{c}.png")

            plt.figure(figsize=(10, 3))
            sns.boxplot(x=s)
            plt.title(f"Box plot: {c}")
            plt.xlabel(c)
            self._savefig(f"eda_box_{c}.png")

    def _eda_categorical_topk(self, df: pd.DataFrame, cat_cols: List[str], topk: int = 20, max_cols: int = 6) -> None:
        cols = [c for c in cat_cols if c in df.columns][:max_cols]
        work = df
        if len(work) > 50000:
            work = work.sample(50000, random_state=42)

        for c in cols:
            s = work[c].astype(str).fillna("NA")
            vc = s.value_counts().head(topk)
            if len(vc) < 2:
                continue
            plt.figure(figsize=(10, 6))
            sns.barplot(x=vc.values, y=vc.index)
            plt.title(f"Top {topk} categories: {c}")
            plt.xlabel("Count")
            plt.ylabel(c)
            self._savefig(f"eda_bar_top_{c}.png")

    def _eda_corr_heatmap(self, df: pd.DataFrame, numeric_cols: List[str]) -> None:
        cols = [c for c in numeric_cols if c in df.columns]
        if len(cols) < 2:
            return
        work = df[cols].apply(pd.to_numeric, errors="coerce")
        corr = work.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
        plt.title("Correlation heatmap (numeric)")
        self._savefig("eda_heatmap_numeric_corr.png")

    def _eda_target_relationships(self, df: pd.DataFrame, target: str, numeric_cols: List[str], max_cols: int = 4) -> None:
        if target not in df.columns:
            return
        if not pd.api.types.is_numeric_dtype(df[target]):
            return

        y = pd.to_numeric(df[target], errors="coerce")
        cols = [c for c in numeric_cols if c != target and c in df.columns][:max_cols]

        for c in cols:
            x = pd.to_numeric(df[c], errors="coerce")
            tmp = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(tmp) < 100:
                continue
            tmp = tmp.sample(min(len(tmp), 5000), random_state=42)
            plt.figure(figsize=(7.5, 6))
            sns.scatterplot(data=tmp, x="x", y="y", s=12, alpha=0.4)
            plt.title(f"{target} vs {c}")
            plt.xlabel(c)
            plt.ylabel(target)
            self._savefig(f"eda_scatter_{target}_vs_{c}.png")

    # ---------- LLM helpers ----------

    def _llm_json_with_retry(
        self,
        system_prompt: str,
        payload: Dict[str, Any],
        time_key: str,
        raw_key: str,
        retry_time_key: str,
        retry_raw_key: str,
        max_tokens: int,
    ) -> Tuple[str, Dict[str, Any]]:
        msg = json.dumps(payload, indent=2)
        text, duration = self.llm.chat(system_prompt, msg, max_tokens=max_tokens)
        self.bus.publish(self.name, {time_key: duration, raw_key: text})
        try:
            return text, parsejsonlenient(text)
        except Exception:
            strict_system = system_prompt + "\n\nCRITICAL: Return ONLY one JSON object. No backticks. No extra text."
            text2, duration2 = self.llm.chat(strict_system, msg, max_tokens=max_tokens)
            self.bus.publish(self.name, {retry_time_key: duration2, retry_raw_key: text2})
            return text2, parsejsonlenient(text2)

    def _call_model_spec(self, df: pd.DataFrame, profile: Dict[str, Any], modeling_request: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        cols = []
        for c in df.columns:
            s = df[c]
            nonnull = int(s.notna().sum())
            nun = int(s.nunique(dropna=True))
            unique_ratio = float(nun / nonnull) if nonnull else 0.0
            cols.append(
                {
                    "name": c,
                    "dtype": str(s.dtype),
                    "nunique": nun,
                    "missing_ratio": float(s.isna().mean()),
                    "unique_ratio": unique_ratio,
                    "sample_values": s.dropna().astype(str).head(3).tolist(),
                }
            )

        payload = {
            "target_request": modeling_request,
            "profile": profile,
            "sample_shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": cols,
        }

        return self._llm_json_with_retry(
            MODELSPEC_SYSTEM,
            payload,
            time_key="modelspecllmtime",
            raw_key="rawmodelspec",
            retry_time_key="modelspecllmtime_retry",
            retry_raw_key="rawmodelspec_retry",
            max_tokens=1200,
        )

    def _fix_model_spec(self, candidate_text: str, candidate_spec: Dict[str, Any], errors: List[str]) -> Tuple[str, Dict[str, Any]]:
        payload = {"validationerrors": errors, "candidatespec": candidate_spec, "candidatetext": candidate_text}
        return self._llm_json_with_retry(
            MODELSPEC_FIXER_SYSTEM,
            payload,
            time_key="modelspecfixertime",
            raw_key="rawmodelspecfixed",
            retry_time_key="modelspecfixertime_retry",
            retry_raw_key="rawmodelspecfixed_retry",
            max_tokens=900,
        )

    def _llm_dataset_description(self, df: pd.DataFrame, profile: Dict[str, Any], plan: Dict[str, Any]) -> str:
        rows = []
        for c in df.columns[:40]:
            s = df[c]
            rows.append(
                {
                    "name": c,
                    "dtype": str(s.dtype),
                    "nunique": int(s.nunique(dropna=True)),
                    "missing_%": round(float(s.isna().mean() * 100.0), 2),
                }
            )

        tr = profile.get("target_recommendation") or profile.get("targetrecommendation") or {}
        payload = {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns_preview": rows,
            "analysis_plan_keys": list(plan.keys()) if isinstance(plan, dict) else [],
            "target_recommendation": tr if isinstance(tr, dict) else {},
        }

        system = (
            "You are a data analyst writing a report.\n"
            "Write a concise dataset description:\n"
            "- Max 12 bullet points\n"
            "- No code\n"
            "- Plain text only\n"
            "- Mention what the dataset appears to represent, key columns, missingness, cardinality, and modeling target context."
        )

        msg = json.dumps(payload, indent=2)
        try:
            text, _dur = self.llm.chat(system, msg, max_tokens=450)
            return (text or "").strip()
        except Exception:
            return ""

    # ---------- preprocessing ----------

    def _build_text_vectorizer(self, name: str, ngram: List[int], max_features: int):
        if name == "HashingVectorizer":
            n_features = 2 ** int(np.clip(int(np.log2(max_features)), 12, 20))
            return HashingVectorizer(
                n_features=n_features,
                alternate_sign=False,
                analyzer="char_wb",
                ngram_range=(int(ngram[0]), int(ngram[1])),
            )
        return TfidfVectorizer(
            max_features=int(max_features),
            lowercase=True,
            ngram_range=(int(ngram[0]), int(ngram[1])),
        )

    def _make_preprocessor(self, spec: Dict[str, Any], X_train: pd.DataFrame):
        feats = spec["features"]
        pre = spec["preprocessing"]

        num_cols = [c for c in feats.get("numeric", []) if c in X_train.columns]
        cat_cols = [c for c in feats.get("categorical", []) if c in X_train.columns]
        text_cols = [c for c in feats.get("text", []) if c in X_train.columns]

        dt_cols: List[str] = []
        for c in list(cat_cols):
            if datetimelikeseries(X_train[c], namehint=isdatename(c)):
                dt_cols.append(c)
                cat_cols.remove(c)

        low_card: List[str] = []
        high_card: List[str] = []
        for c in cat_cols:
            nun = int(X_train[c].nunique(dropna=True))
            if nun > self.high_card_threshold:
                high_card.append(c)
            else:
                low_card.append(c)

        def _ravel_to_1d(x):
            return np.asarray(x).ravel().astype(str)

        transformers = []

        if num_cols:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy=pre.get("numericimputer", "median"))),
                            ("scaler", StandardScaler() if pre.get("numericscaler", "standard") == "standard" else "passthrough"),
                        ]
                    ),
                    num_cols,
                )
            )

        if low_card:
            transformers.append(
                (
                    "cat_low",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=int(pre.get("onehotminfrequency", 1)))),
                        ]
                    ),
                    low_card,
                )
            )

        if high_card:
            transformers.append(
                (
                    "cat_high_freq",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("freq", CountFrequencyEncoder(normalize=True)),
                        ]
                    ),
                    high_card,
                )
            )

        if dt_cols:
            for c in dt_cols:
                transformers.append(
                    (
                        f"dt_{c}",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("dt", DateTimeFeaturizer()),
                            ]
                        ),
                        [c],
                    )
                )

        if text_cols:
            vec = self._build_text_vectorizer(
                pre.get("textvectorizer", "TfidfVectorizer"),
                pre.get("textngram", [1, 2]),
                int(pre.get("maxtextfeatures", 2000)),
            )
            text_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                    ("to_1d", FunctionTransformer(_ravel_to_1d, validate=False)),
                    ("vec", vec),
                ]
            )
            for c in text_cols[: self.max_text_columns]:
                transformers.append((f"txt_{c}", text_pipe, [c]))

        if not transformers:
            raise RuntimeError("No preprocessing transformers created (no usable feature columns).")

        coltf = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

        used = {
            "numeric": num_cols,
            "categoricallow": low_card,
            "categoricalhighfreq": high_card,
            "datetime": dt_cols,
            "text": text_cols[: self.max_text_columns],
        }

        return coltf, used

    def _wrap_regressor_if_skewed(self, model, y_train: pd.Series):
        y = pd.to_numeric(y_train, errors="coerce").dropna()
        if len(y) < 200:
            return model
        try:
            skew = float(y.skew())
        except Exception:
            return model
        if abs(skew) < 1.0:
            return model

        def fwd(a):
            a = np.asarray(a, dtype=float)
            return np.sign(a) * np.log1p(np.abs(a))

        def inv(a):
            a = np.asarray(a, dtype=float)
            return np.sign(a) * (np.expm1(np.abs(a)))

        return TransformedTargetRegressor(regressor=model, func=fwd, inverse_func=inv)

    # ---------- model fitting ----------

    def _fit_one_model(self, spec: Dict[str, Any], model_cfg: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        target = spec["target"]
        pt = spec["problemtype"]

        df2 = df.dropna(subset=[target]).copy()
        if len(df2) < 30:
            raise RuntimeError(f"Not enough rows after dropping missing target ({len(df2)} rows).")

        y = df2[target]
        X = df2.drop(columns=[target])

        drop_cols = [c for c in spec.get("features", {}).get("drop", []) if c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)

        split = spec["split"]
        stratify = None
        if pt == "classification" and split.get("stratify", False) and y.nunique() > 1:
            vc = y.value_counts()
            if int(vc.min()) >= 2:
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(split.get("testsize", 0.2)),
            random_state=int(split.get("randomstate", 42)),
            stratify=stratify,
        )

        if pt == "classification":
            n_classes = int(pd.Series(y_train).nunique(dropna=True))
            if n_classes < 2:
                raise RuntimeError("Training split has <2 classes; cannot fit a classifier.")

        preproc, used = self._make_preprocessor(spec, X_train)

        model_name = model_cfg["name"]
        model_cls = ALLOWEDMODELS[pt][model_name]
        hp = dict(model_cfg.get("hyperparams") or {})
        cw = model_cfg.get("classweight")

        Xt_train = preproc.fit_transform(X_train, y_train)
        if not hasattr(Xt_train, "shape") or int(Xt_train.shape[1]) == 0:
            raise RuntimeError("No usable features produced by preprocessing (0 columns after transform).")

        is_sparse = False
        if sp is not None:
            try:
                is_sparse = sp.issparse(Xt_train)
            except Exception:
                is_sparse = False

        if pt == "classification" and cw == "balanced" and model_name in ("LogisticRegression", "SGDClassifier", "LinearSVC"):
            hp["class_weight"] = "balanced"
        else:
            hp.pop("class_weight", None)

        if model_name == "LogisticRegression":
            default_solver = "saga" if is_sparse else "lbfgs"
            hp = {"max_iter": 2000, "tol": 1e-3, "solver": default_solver, **hp}
        elif model_name == "SGDClassifier":
            hp = {"loss": "log_loss", "max_iter": 3000, "tol": 1e-3, **hp}
        elif model_name == "SGDRegressor":
            hp = {"max_iter": 3000, "tol": 1e-3, **hp}

        model = model_cls(**hp)
        if pt == "regression":
            model = self._wrap_regressor_if_skewed(model, y_train)

        model.fit(Xt_train, y_train)

        Xt_test = preproc.transform(X_test)
        y_pred = model.predict(Xt_test)

        out = {
            "stepid": f"model_{model_cfg.get('tag', '')}",
            "model": model_name,
            "problemtype": pt,
            "metrics": {
                "accuracy": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "f1": "N/A",
                "r2": "N/A",
                "rmse": "N/A",
                "mae": "N/A",
            },
            "ntrain": int(len(X_train)),
            "ntest": int(len(X_test)),
            "usedfeatures": {
                "numeric": used["numeric"],
                "categoricallow": used["categoricallow"],
                "categoricalhighfreq": used["categoricalhighfreq"],
                "datetime": used["datetime"],
                "text": used["text"],
                "dropped": drop_cols,
            },
            "tag": model_cfg.get("tag"),
        }

        if pt == "classification":
            out["metrics"]["accuracy"] = float(accuracy_score(y_test, y_pred))
            out["metrics"]["precision"] = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            out["metrics"]["recall"] = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            out["metrics"]["f1"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

            cm_cfg = spec.get("confusionmatrix", {})
            if cm_cfg.get("enabled", True):
                max_classes = int(cm_cfg.get("maxclasses", 15))

                labels, counts = np.unique(np.asarray(y_test), return_counts=True)
                if len(labels) > max_classes:
                    top = labels[np.argsort(counts)[::-1][:max_classes]]
                    mask = np.isin(y_test, top) & np.isin(y_pred, top)
                    y_t = np.asarray(y_test)[mask]
                    y_p = np.asarray(y_pred)[mask]
                    labels = top
                else:
                    y_t = np.asarray(y_test)
                    y_p = np.asarray(y_pred)

                cm = confusion_matrix(y_t, y_p, labels=labels)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
                disp.plot(include_values=False, cmap="Blues", xticks_rotation=45)
                plt.title(f"Confusion matrix ({model_name}, {model_cfg.get('tag')})")
                self._savefig(f"ml_confusion_matrix_{model_name.lower()}_{model_cfg.get('tag', '')}.png")

        else:
            out["metrics"]["r2"] = float(r2_score(y_test, y_pred))
            out["metrics"]["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            out["metrics"]["mae"] = float(mean_absolute_error(y_test, y_pred))

        return out

    def _metric_alias(self, metric: str) -> str:
        m = (metric or "").strip().lower()
        if m in ("f1weighted", "f1_weighted", "f1"):
            return "f1"
        return m

    def _select_best(self, spec: Dict[str, Any], metrics_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not metrics_list:
            return None

        sel = spec.get("selection") or {}
        metric = self._metric_alias(sel.get("metric", "r2"))
        direction = sel.get("direction", "maximize")

        def score(m):
            val = (m.get("metrics") or {}).get(metric)
            try:
                return float(val)
            except Exception:
                return -np.inf if direction == "maximize" else np.inf

        return max(metrics_list, key=score) if direction == "maximize" else min(metrics_list, key=score)

    # ---------- fallback spec ----------

    def _basic_fallback_spec(self, df: pd.DataFrame, target: str, problemtype: str) -> Dict[str, Any]:
        drop = idlikecolumns(df, target)

        preferred_numeric = [c for c in ["Quantity", "UnitPrice"] if c in df.columns and c != target and c not in set(drop)]
        numeric = preferred_numeric if preferred_numeric else [c for c in numericlikecols(df, target) if c not in set(drop)]

        categorical: List[str] = []
        text: List[str] = []

        if len(numeric) == 0:
            categorical = [c for c in lowcardcategoricals(df, target, maxclasses=80) if c not in set(drop)][:6]
        if len(numeric) == 0 and len(categorical) == 0:
            text = [c for c in df.columns if c != target and textlike(df, c) and c not in set(drop)][:1]

        spec: Dict[str, Any] = {
            "target": target,
            "problemtype": problemtype,
            "split": {"testsize": 0.2, "randomstate": 42, "stratify": True if problemtype == "classification" else False},
            "features": {"numeric": numeric, "categorical": categorical, "text": text, "drop": drop},
            "preprocessing": {
                "numericimputer": "median",
                "numericscaler": "standard",
                "categoricalencoder": "onehot",
                "onehotminfrequency": 10,
                "textvectorizer": "TfidfVectorizer",
                "textngram": [1, 2],
                "maxtextfeatures": 2000,
            },
            "models": [],
            "selection": {"metric": "f1weighted" if problemtype == "classification" else "r2", "direction": "maximize"},
            "confusionmatrix": {"enabled": True if problemtype == "classification" else False, "maxclasses": 15},
            "reasoningshort": "Basic fallback models used because LLM ModelSpec was invalid or produced no successful models.",
        }

        if problemtype == "classification":
            spec["models"] = [
                {"tag": "baseline", "name": "LogisticRegression", "hyperparams": {"C": 1.0}, "classweight": None},
                {"tag": "advanced", "name": "LinearSVC", "hyperparams": {"C": 1.0}, "classweight": "balanced"},
            ]
        else:
            spec["models"] = [
                {"tag": "baseline", "name": "Ridge", "hyperparams": {"alpha": 1.0}, "classweight": None},
                {"tag": "advanced", "name": "SGDRegressor", "hyperparams": {"alpha": 0.0001}, "classweight": None},
            ]

        normalizespecinplace(spec, df)
        return spec

    # ---------- main entrypoint ----------

    def run(self):
        profile = self.bus.get_state("data_profile") or {}
        plan = self.bus.get_state("analysis_plan") or {}
        df = self.bus.get_state("df")

        if df is None or len(df) == 0:
            raise RuntimeError("Executor: df sample is missing/empty.")

        # Robust plan parsing (supports both snake_case and legacy keys)
        need_modeling = self._get_plan_bool(plan, "need_modeling", "needmodeling", "needModeling")
        modeling_request = self._get_plan_dict(plan, "modeling_request", "modelingrequest", "modelingRequest")

        # If planner didn't set need_modeling explicitly, infer it: model if we can find a target.
        if need_modeling is None:
            need_modeling = True if isinstance(modeling_request, dict) and len(modeling_request) > 0 else False

        target = None
        if isinstance(modeling_request, dict):
            target = modeling_request.get("target")

        # Fallback: target recommendation from profiler
        tr = profile.get("target_recommendation") or profile.get("targetrecommendation") or {}
        if not target and isinstance(tr, dict):
            target = tr.get("target")

        # Resolve column name (handles case mismatch)
        resolved_target = self._resolve_column_name(df, target)
        if resolved_target:
            target = resolved_target
            if isinstance(modeling_request, dict):
                modeling_request["target"] = target

        # --- richer EDA figures ---
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if (df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c]))]

        self._eda_schema_overview_figure(df)
        self._eda_missingness_bar(df)
        self._eda_missingness_heatmap(df)
        self._eda_numeric_distributions(df, numeric_cols, max_cols=6)
        self._eda_categorical_topk(df, cat_cols, topk=20, max_cols=6)
        self._eda_corr_heatmap(df, numeric_cols)

        if target and target in df.columns:
            self._eda_target_distribution(df, target)
            self._eda_target_relationships(df, target, numeric_cols, max_cols=4)

        # LLM dataset description as figure + state
        desc = self._llm_dataset_description(df, profile, plan if isinstance(plan, dict) else {})
        if desc:
            self.bus.set_state("llm_dataset_description", desc)
            self._plot_text_card("LLM dataset description", desc, "eda_text_dataset_description.png")

        metrics_list: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        if need_modeling and target and target in df.columns:
            # Resolve problem type
            problemtype = None
            if isinstance(modeling_request, dict):
                problemtype = modeling_request.get("problemtype") or modeling_request.get("problem_type")
            if not problemtype and isinstance(tr, dict):
                problemtype = tr.get("problemtype") or tr.get("problem_type")
            if problemtype not in ("classification", "regression"):
                problemtype = "classification"

            # 1) LLM spec
            try:
                raw_text, llm_spec = self._call_model_spec(df, profile, modeling_request or {"target": target, "problemtype": problemtype})
                spec = llm_spec if isinstance(llm_spec, dict) else {}
                normalizespecinplace(spec, df)
                self.bus.set_state("model_spec", spec)

                ok, val_errors = validatespec(spec, df)
                fixer_tries = 0
                while (not ok) and fixer_tries < 2:
                    raw_text, fixed_obj = self._fix_model_spec(raw_text, spec, val_errors)
                    spec = fixed_obj if isinstance(fixed_obj, dict) else {}
                    normalizespecinplace(spec, df)
                    self.bus.set_state("model_spec", spec)
                    ok, val_errors = validatespec(spec, df)
                    fixer_tries += 1

                if ok:
                    for mcfg in spec["models"]:
                        try:
                            met = self._fit_one_model(spec, mcfg, df)
                            met["stepid"] = f"llm_model_{mcfg.get('tag', '')}"
                            metrics_list.append(met)
                        except Exception:
                            continue

                    if metrics_list:
                        best = self._select_best(spec, metrics_list)
                        if best is not None:
                            self.bus.set_state("best_model_summary", best)

            except Exception as e:
                errors.append({"step": "llm_modeling", "error": repr(e)})

            # 2) Fallback if nothing trained
            if len(metrics_list) == 0:
                fb = self._basic_fallback_spec(df, target, problemtype)
                self.bus.set_state("model_spec", fb)

                for mcfg in fb["models"]:
                    try:
                        met = self._fit_one_model(fb, mcfg, df)
                        met["stepid"] = f"fallback_model_{mcfg.get('tag', '')}"
                        metrics_list.append(met)
                    except Exception as e:
                        errors.append({"step": "fallback_modeling", "model": mcfg.get("name"), "tag": mcfg.get("tag"), "error": repr(e)})

                if metrics_list:
                    best = self._select_best(fb, metrics_list)
                    if best is not None:
                        self.bus.set_state("best_model_summary", best)
                else:
                    errors.append({"step": "modeling", "error": "LLM ModelSpec produced no successful models AND fallback training also failed."})

        else:
            errors.append(
                {
                    "note": "Modeling disabled by planner or target missing in sample; modeling skipped.",
                    "debug": {
                        "need_modeling": bool(need_modeling),
                        "target_raw": target,
                        "target_resolved": self._resolve_column_name(df, target),
                        "df_columns": list(df.columns[:30]),
                        "analysis_plan_keys": list(plan.keys()) if isinstance(plan, dict) else [],
                    },
                }
            )

        results = {"metrics": metrics_list, "errors": errors}
        self.bus.set_state("analysis_results", results)

        self.bus.publish(
            self.name,
            {
                "ok": True,
                "nsamplerows": int(len(df)),
                "need_modeling": bool(need_modeling),
                "targetused": target,
                "nmodelstrained": len(metrics_list),
                "nerrors": len(errors),
            },
        )
        return results
