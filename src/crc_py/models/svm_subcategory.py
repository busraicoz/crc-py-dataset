"""
Subcategory classifier (LinearSVC) with multi-view TF-IDF features and weak signals.
Category is derived from a fixed sub->top mapping provided by the caller.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Any
from collections import Counter
import json
import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack, csr_array
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

NUMERIC_COLS = [
    "code_plus_lines", "code_minus_lines", "code_has_hunk",
    "has_qmark", "has_nit", "has_docs", "has_lgtm"
]

class MultiViewVectorizer:
    def __init__(self, min_df_comment: int = 3, min_df_code: int = 3, max_features: int = 40000):
        self.comment_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 3),
            min_df=min_df_comment, max_features=max_features,
            lowercase=True, strip_accents="unicode",
        )
        self.code_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            min_df=min_df_code, max_features=max_features // 2,
            lowercase=False,
        )
        self.path_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, max_features=20000,
        )
        self.meta_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 1), min_df=1, max_features=2000,
        )

    def fit_transform(self, df: pd.DataFrame) -> csr_array[Any, tuple[int, int]]:
        c = self.comment_vec.fit_transform(df["comment"])
        d = self.code_vec.fit_transform(df["code"])
        p = self.path_vec.fit_transform(df["file_path"])
        m = self.meta_vec.fit_transform((df["owner"] + " " + df["repo"]).tolist())
        return hstack([c, d, p, m], format="csr")

    def transform_one(self, comment: str, code: str, file_path: str, owner: str, repo: str) -> csr_array[
        Any, tuple[int, int]]:
        c = self.comment_vec.transform([comment or ""]) 
        d = self.code_vec.transform([code or ""]) 
        p = self.path_vec.transform([file_path or ""]) 
        m = self.meta_vec.transform(["{owner or ''} {repo or ''}"])
        return hstack([c, d, p, m], format="csr")

    def dump(self, path: Path) -> None:
        joblib.dump({
            "comment_vec": self.comment_vec,
            "code_vec": self.code_vec,
            "path_vec": self.path_vec,
            "meta_vec": self.meta_vec,
        }, path)

    @staticmethod
    def load(path: Path) -> "MultiViewVectorizer":
        payload = joblib.load(path)
        mv = MultiViewVectorizer()
        mv.comment_vec = payload["comment_vec"]
        mv.code_vec = payload["code_vec"]
        mv.path_vec = payload["path_vec"]
        mv.meta_vec = payload["meta_vec"]
        return mv


def _prob_from_model(model, X):
    if hasattr(model, 'predict_proba'):
        import numpy as np
        probs = model.predict_proba(X)
        idx = int(np.argmax(probs, axis=1)[0])
        score = float(np.max(probs, axis=1)[0])
        return idx, score
    dec = model.decision_function(X)
    import numpy as np
    dec = np.array(dec)
    if dec.ndim == 1:
        p = 1.0/(1.0+np.exp(-dec))
        probs = np.vstack([1.0-p, p]).T
    else:
        m = dec - dec.max(axis=1, keepdims=True)
        e = np.exp(m)
        probs = e / e.sum(axis=1, keepdims=True)
    idx = int(np.argmax(probs, axis=1)[0])
    score = float(np.max(probs, axis=1)[0])
    return idx, score


class SubcategoryClassifier:
    def __init__(self, model_dir: str = "out_svm_mv"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "fe": self.model_dir / "fe_bundle.joblib",
            "clf": self.model_dir / "svm_subcategory_model.joblib",
            "enc": self.model_dir / "subcategory_encoder.joblib",
            "meta": self.model_dir / "model_meta.json",
        }

    @staticmethod
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in ["comment", "code", "file_path", "owner", "repo"]:
            if c not in df.columns:
                df[c] = ""
            df[c] = df[c].fillna("").astype(str)
        df["code_plus_lines"] = df["code"].str.count(r"\+")
        df["code_minus_lines"] = df["code"].str.count(r"\-")
        df["code_has_hunk"] = df["code"].str.contains("@@").astype(int)
        df["has_qmark"] = df["comment"].str.strip().str.endswith("?").astype(int)
        df["has_nit"] = df["comment"].str.contains(r"nit", case=False).astype(int)
        df["has_docs"] = df["comment"].str.contains("doc", case=False).astype(int)
        df["has_lgtm"] = df["comment"].str.contains("lgtm", case=False).astype(int)
        return df

    @staticmethod
    def _can_calibrate(y, min_per_class: int = 3, min_folds: int = 3) -> int:
        counts = Counter(y)
        if len(counts) < 2:
            return 0
        min_cnt = min(counts.values())
        cv = min(5, max(min_cnt // min_per_class, 0))
        return cv if cv >= min_folds else 0

    def train(self, train_json_path: str) -> None:
        raw = json.load(open(train_json_path, 'r', encoding='utf-8'))
        if isinstance(raw, dict):
            for v in raw.values():
                if isinstance(v, list):
                    raw = v
                    break
        df = pd.DataFrame(raw)
        if not {"comment", "subcategory"} <= set(df.columns):
            raise ValueError("Training data must include 'comment' and 'subcategory'.")
        df = df[df["comment"].notna() & df["subcategory"].notna()].copy()
        df["subcategory"] = df["subcategory"].astype(str)
        df = self._prep(df)

        enc = LabelEncoder().fit(df["subcategory"])
        y = enc.transform(df["subcategory"])  

        mv = MultiViewVectorizer()
        X_text = mv.fit_transform(df)
        X_num = csr_matrix(df[NUMERIC_COLS].to_numpy(dtype=float))
        X = hstack([X_text, X_num], format='csr')

        base = LinearSVC(C=0.5, max_iter=10000)
        cv = self._can_calibrate(y)
        if cv > 0:
            clf = CalibratedClassifierCV(base, cv=cv, method='sigmoid')
        else:
            clf = base
        clf.fit(X, y)

        mv.dump(self.paths['fe'])
        joblib.dump(clf, self.paths['clf'])
        joblib.dump(enc, self.paths['enc'])
        json.dump({"n_features": int(X.shape[1])}, open(self.paths['meta'], 'w', encoding='utf-8'), indent=2)

    def infer(self, comment: str, *, code: str = "", file_path: str = "", owner: str = "", repo: str = "") -> Tuple[str, float]:
        mv = MultiViewVectorizer.load(self.paths['fe'])
        model = joblib.load(self.paths['clf'])
        enc: LabelEncoder = joblib.load(self.paths['enc'])
        X_text = mv.transform_one(comment, code, file_path, owner, repo)
        num = [code.count("+"), code.count("-"), 1.0 if "@@" in code else 0.0,
               1.0 if comment.strip().endswith("?") else 0.0,
               1.0 if 'nit' in comment.lower() else 0.0,
               1.0 if 'doc' in comment.lower() else 0.0,
               1.0 if 'lgtm' in comment.lower() else 0.0]
        X_num = csr_matrix([num], dtype=float)
        X = hstack([X_text, X_num], format='csr')
        idx, score = _prob_from_model(model, X)
        sub = enc.inverse_transform([idx])[0]
        return sub, score
