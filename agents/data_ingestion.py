# agents/data_ingestion.py
import json
import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from .base import Agent


DATA_PROFILE_SYSTEM = """
You are a data profiling agent for tabular datasets.

Input: a JSON payload with:
- n_rows_total: int
- sample_shape: [int,int]
- columns: list of column stats and sample values

Output: ONLY valid JSON (one object), no prose, no markdown.

Required output schema:
{
  "n_rows": int,
  "n_cols": int,
  "columns": [
    {"name": str, "inferred_type": "numeric"|"categorical"|"text", "role": "id"|"feature"|"target"}
  ],
  "issues": [str, ...],
  "target_recommendation": {
    "target": str,
    "problem_type": "classification"|"regression",
    "confidence": float,   # MUST be between 0.0 and 1.0
    "reason": str
  }
}

Rules:
- Pick exactly ONE target recommendation; if unsure, choose the most "label-like" column.
- Do NOT pick obvious IDs as target (very high uniqueness, column name like id, uuid, key).
- If a column is categorical with small/medium number of classes and not ID-like, it's a strong classification target candidate.
- If a column is numeric and not ID-like, it may be a regression target.
- confidence must be in [0,1].
"""


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _infer_type_from_series(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"

    if s.dtype == "object":
        ss = s.dropna().astype(str)
        if len(ss) == 0:
            return "categorical"
        avg_len = float(ss.head(500).str.len().mean())
        nun = int(ss.nunique())
        if avg_len >= 20 and nun >= 50:
            return "text"
        return "categorical"

    return "categorical"


def _col_stats(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    n = len(df)

    for c in df.columns:
        s = df[c]
        nonnull = int(s.notna().sum())
        nun = int(s.nunique(dropna=True))
        unique_ratio = float(nun / nonnull) if nonnull else 0.0

        if s.dtype == "object":
            ss = s.dropna().astype(str).head(500)
            avg_len = float(ss.str.len().mean()) if len(ss) else 0.0
            num_parse = pd.to_numeric(s.astype(str).replace({"\\N": np.nan}), errors="coerce")
            numeric_parse_rate = float(num_parse.notna().mean())
        else:
            avg_len = 0.0
            numeric_parse_rate = 1.0 if pd.api.types.is_numeric_dtype(s) else 0.0

        out.append({
            "name": c,
            "dtype": str(s.dtype),
            "missing_ratio": float(s.isna().mean()),
            "nunique": nun,
            "unique_ratio": unique_ratio,
            "avg_str_len": avg_len,
            "numeric_parse_rate": numeric_parse_rate,
            "sample_values": s.dropna().astype(str).head(5).tolist(),
            "n_rows_sample": int(n),
        })
    return out


def _heuristic_target_guess(df: pd.DataFrame) -> Tuple[Optional[str], str, float, str]:
    if df is None or len(df.columns) == 0:
        return None, "classification", 0.0, "Empty dataframe."

    preferred = ["target", "label", "y", "class", "species", "outcome"]
    lower_map = {c.lower(): c for c in df.columns}
    for key in preferred:
        if key in lower_map:
            col = lower_map[key]
            pt = "regression" if pd.api.types.is_numeric_dtype(df[col]) else "classification"
            return col, pt, 0.85, f"Column name suggests target ('{key}')."

    best = None
    best_score = -1.0
    for c in df.columns:
        s = df[c]
        nonnull = int(s.notna().sum())
        nun = int(s.nunique(dropna=True))
        unique_ratio = float(nun / nonnull) if nonnull else 0.0
        name_low = c.lower()

        if "id" in name_low or "uuid" in name_low or "key" in name_low:
            continue
        if unique_ratio >= 0.98:
            continue

        if s.dtype == "object":
            if 2 <= nun <= 50:
                score = 2.0 + (1.0 - abs(nun - 3) / 50.0)
            else:
                score = 0.2
            if score > best_score:
                best = (c, "classification", 0.65, "Likely label: categorical with limited classes.")
                best_score = score
        elif pd.api.types.is_numeric_dtype(s):
            score = 0.5
            if score > best_score:
                best = (c, "regression", 0.55, "Numeric column could be a regression target.")
                best_score = score

    if best is None:
        c = df.columns[-1]
        pt = "regression" if pd.api.types.is_numeric_dtype(df[c]) else "classification"
        return c, pt, 0.4, "Fallback to last column."

    return best


def _read_csv_chunked(
    csv_path: str,
    chunksize: int,
    low_memory: bool,
    encoding: str,
    encoding_errors: Optional[str],
) -> Any:
    """
    Generator wrapper around pd.read_csv(..., chunksize=...).
    Handles pandas versions that may not support encoding_errors.
    """
    kwargs = dict(low_memory=low_memory, chunksize=chunksize, encoding=encoding)
    if encoding_errors is not None:
        kwargs["encoding_errors"] = encoding_errors
    try:
        return pd.read_csv(csv_path, **kwargs)
    except TypeError:
        # Older pandas: no encoding_errors keyword
        kwargs.pop("encoding_errors", None)
        return pd.read_csv(csv_path, **kwargs)


def _read_csv_sample_and_count_random(
    csv_path: str,
    max_sample_rows: int = 30000,
    chunksize: int = 200000,
    low_memory: bool = False,
    seed: int = 42,
) -> Tuple[pd.DataFrame, int]:
    """
    Robust CSV reader:
    - Tries multiple encodings to avoid UnicodeDecodeError.
    - Never shrinks small datasets; for large datasets keeps a bounded random sample.
    """
    rng = np.random.default_rng(seed)

    # Try UTF-8 first, then common Windows encodings (cp1252/latin1)
    encoding_trials = [
        ("utf-8", "strict"),
        ("utf-8-sig", "strict"),
        ("cp1252", "replace"),
        ("latin1", "replace"),
    ]

    last_err = None
    for enc, enc_errs in encoding_trials:
        total = 0
        sample_parts: List[pd.DataFrame] = []
        try:
            reader = _read_csv_chunked(
                csv_path=csv_path,
                chunksize=chunksize,
                low_memory=low_memory,
                encoding=enc,
                encoding_errors=enc_errs,
            )

            for chunk in reader:
                total += len(chunk)
                sample_parts.append(chunk)

                cur = pd.concat(sample_parts, ignore_index=True)
                if len(cur) > max_sample_rows:
                    idx = rng.choice(len(cur), size=max_sample_rows, replace=False)
                    cur = cur.iloc[idx].reset_index(drop=True)
                    sample_parts = [cur]

            sample_df = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
            if sample_df is None or len(sample_df) == 0:
                raise RuntimeError(f"Empty dataframe after reading with encoding={enc}")

            return sample_df, int(total)

        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            # If a non-encoding issue occurs, remember it but still try next encoding once
            last_err = e
            continue

    raise RuntimeError(f"Failed to read CSV with tried encodings. Last error: {repr(last_err)}")


class DataIngestionAgent(Agent):
    """
    Loads CSV, builds a sample dataframe for downstream agents, and asks the LLM to
    profile columns and recommend a target.
    """
    def __init__(self, name, llm_client, bus, max_sample_rows: int = 30000):
        super().__init__(name, llm_client, bus)
        self.max_sample_rows = int(max_sample_rows)

    def run(self, csv_path: str) -> Dict[str, Any]:
        t0 = time.time()

        sample_df, n_total = _read_csv_sample_and_count_random(
            csv_path,
            max_sample_rows=self.max_sample_rows,
            chunksize=200000,
            low_memory=False,
            seed=42,
        )

        if sample_df is None or len(sample_df) == 0:
            raise RuntimeError("DataIngestion: empty dataframe after reading CSV.")

        columns = _col_stats(sample_df)
        payload = {
            "n_rows_total": int(n_total),
            "sample_shape": [int(sample_df.shape[0]), int(sample_df.shape[1])],
            "columns": columns,
            "notes": "Recommend ONE target column and problem type. Confidence must be in [0,1].",
        }

        try:
            raw, _dur = self.llm.chat(DATA_PROFILE_SYSTEM, json.dumps(payload, indent=2), max_tokens=900)
            obj = json.loads(raw) if raw.strip().startswith("{") else json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            profile = obj
        except Exception:
            tgt, pt, conf, reason = _heuristic_target_guess(sample_df)
            cols_out = []
            for c in sample_df.columns:
                inferred = _infer_type_from_series(sample_df[c])
                role = "feature"
                if isinstance(tgt, str) and c == tgt:
                    role = "target"
                if "id" in c.lower() and role != "target":
                    role = "id"
                cols_out.append({"name": c, "inferred_type": inferred, "role": role})

            profile = {
                "n_rows": int(n_total),
                "n_cols": int(sample_df.shape[1]),
                "columns": cols_out,
                "issues": ["LLM profiling failed; used heuristic target guess."],
                "target_recommendation": {
                    "target": tgt,
                    "problem_type": pt,
                    "confidence": conf,
                    "reason": reason,
                }
            }

        profile.setdefault("n_rows", int(n_total))
        profile.setdefault("n_cols", int(sample_df.shape[1]))
        profile.setdefault("issues", [])
        profile.setdefault("columns", [])

        tr = profile.get("target_recommendation") or {}
        tgt = tr.get("target")
        if not isinstance(tgt, str) or tgt not in sample_df.columns:
            ht, hpt, hconf, hreason = _heuristic_target_guess(sample_df)
            profile["issues"].append("LLM target was missing/invalid; replaced using heuristic.")
            tr["target"] = ht
            tr["problem_type"] = hpt
            tr["confidence"] = hconf
            tr["reason"] = hreason

        tr["confidence"] = _clamp01(tr.get("confidence"))
        if tr.get("problem_type") not in ("classification", "regression"):
            tr["problem_type"] = "classification"
        profile["target_recommendation"] = tr

        # Ensure columns roles/types exist and are consistent
        out_cols = []
        for c in sample_df.columns:
            inferred = _infer_type_from_series(sample_df[c])
            role = "feature"
            if isinstance(tr.get("target"), str) and c == tr["target"]:
                role = "target"
            else:
                name_low = c.lower()
                s = sample_df[c]
                nonnull = int(s.notna().sum())
                nun = int(s.nunique(dropna=True))
                unique_ratio = float(nun / nonnull) if nonnull else 0.0
                if ("id" in name_low or "uuid" in name_low or "key" in name_low) and unique_ratio >= 0.95:
                    role = "id"
            out_cols.append({"name": c, "inferred_type": inferred, "role": role})
        profile["columns"] = out_cols

        duration = time.time() - t0

        self.bus.set_state("df", sample_df)
        self.bus.set_state("data_profile", profile)

        self.bus.publish(self.name, {
            "duration": float(duration),
            "raw_llm_response": json.dumps(profile, indent=2),
            "profile": profile
        })

        return profile
