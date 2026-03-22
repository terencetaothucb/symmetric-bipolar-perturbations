# -*- coding: utf-8 -*-
"""
Model_Benchmark_FaiIrrevReff_AllInOne.py

Benchmark runner for PulseBat Step4 outputs.
Available feature groups are reduced to:
- fai_irrev
- Reff

Default data root:
E:/Datasets/PulseBat_all/Data/Data_Process_Output/Step4_SelectedFeatures

Default output root:
E:/Datasets/PulseBat_all/Data/Model_Output
"""

from __future__ import annotations

import re
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


# ============================================================
# 0) Torch-backed models
# ============================================================

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


# ============================================================
# 1) Feature groups + combos + builder
# ============================================================

FEATURE_GROUPS: Dict[str, List[str]] = {
    "fai_irrev": [
        "fai_irrev_0.5C", "fai_irrev_1C", "fai_irrev_1.5C", "fai_irrev_2C", "fai_irrev_2.5C",
    ],
    "Reff": [
        "Reff_p_0.5C", "Reff_n_0.5C", "Reff_p_1C", "Reff_n_1C", "Reff_p_1.5C", "Reff_n_1.5C",
        "Reff_p_2C", "Reff_n_2C", "Reff_p_2.5C", "Reff_n_2.5C",
    ],
}

FEATURE_COMBOS: Dict[str, List[str]] = {
    "fai_irrev_Only": ["fai_irrev"],
    "Reff_Only": ["Reff"],
    "fai_irrev_Reff": ["fai_irrev", "Reff"],
}

def _unique_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def build_feature_list(
    *,
    combo: Optional[str] = None,
    include_groups: Optional[List[str]] = None,
    exclude_groups: Optional[List[str]] = None,
    add_features: Optional[List[str]] = None,
    drop_features: Optional[List[str]] = None,
    keep_regex: Optional[str] = None,
) -> List[str]:
    feats: List[str] = []

    if combo is not None:
        if combo not in FEATURE_COMBOS:
            raise KeyError(f"Unknown combo '{combo}'. Available combos: {list(FEATURE_COMBOS.keys())}")
        for g in FEATURE_COMBOS[combo]:
            if g not in FEATURE_GROUPS:
                raise KeyError(f"Combo '{combo}' references unknown group '{g}'")
            feats.extend(FEATURE_GROUPS[g])

    if include_groups:
        for g in include_groups:
            if g not in FEATURE_GROUPS:
                raise KeyError(f"Unknown group '{g}'. Available groups: {list(FEATURE_GROUPS.keys())}")
            feats.extend(FEATURE_GROUPS[g])

    feats = _unique_preserve_order(feats)

    if exclude_groups:
        exclude_set = set()
        for g in exclude_groups:
            if g not in FEATURE_GROUPS:
                raise KeyError(f"Unknown group '{g}'. Available groups: {list(FEATURE_GROUPS.keys())}")
            exclude_set.update(FEATURE_GROUPS[g])
        feats = [f for f in feats if f not in exclude_set]

    if add_features:
        feats.extend(add_features)
        feats = _unique_preserve_order(feats)

    if drop_features:
        drop_set = set(drop_features)
        feats = [f for f in feats if f not in drop_set]

    if keep_regex:
        rgx = re.compile(keep_regex)
        feats = [f for f in feats if rgx.search(f)]

    return feats

def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def _safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-\.\s]+", "_", str(s))
    s = s.strip().replace(" ", "_")
    return s

def feature_tag_from_spec(spec: Dict[str, Any]) -> str:
    parts = []
    if spec.get("name"):
        parts.append(str(spec["name"]))
    if spec.get("combo"):
        parts.append(f"combo-{spec['combo']}")
    if spec.get("include_groups"):
        parts.append("inc-" + "_".join(spec["include_groups"]))
    if spec.get("exclude_groups"):
        parts.append("exc-" + "_".join(spec["exclude_groups"]))
    if spec.get("keep_regex"):
        parts.append("keep")
    if spec.get("add_features"):
        parts.append(f"add{len(spec['add_features'])}")
    if spec.get("drop_features"):
        parts.append(f"drop{len(spec['drop_features'])}")
    if not parts:
        return "features"
    tag = _safe_name("__".join(parts))
    return tag[:80]


# ============================================================
# 2) CONFIG (edit here only)
# ============================================================

CONFIG: Dict[str, Any] = dict(
    data_root=r"E:\Datasets\PulseBat_all\Data\Data_Process_Output\Step4_SelectedFeatures",

    material="20Ah LFP",
    width_ms=5000,
    soc=50,
    label_col="SOH",

    feature_spec=dict(
        name="fai_irrev_Only",
        combo="fai_irrev_Only",
        include_groups=None,
        exclude_groups=None,
        add_features=None,
        drop_features=None,
        keep_regex=None,
    ),

    test_size=0.2,
    seeds=list(range(100)),

    # Models to benchmark in each run.
    models=["linear", "ridge", "lasso", "en", "svm", "rf", "xgb", "gpr", "mlp", "transformer", "informer"],

    standardize=True,
    perm_repeats=10,
    topk_featimp=50,

    # Torch model training hyperparameters.
    torch_device="auto",        # "auto" | "cpu" | "cuda"
    torch_epochs=200,
    torch_batch_size=64,
    torch_lr=1e-3,
    torch_weight_decay=1e-6,
    torch_patience=30,
    torch_seed_offset=12345,    # Makes torch initialization depend on the run seed deterministically.

    out_dir=r"E:\Datasets\PulseBat_all\Data\Model_Output",
    show_progress=True,
)


# ============================================================
# 3) Utils / paths
# ============================================================

def script_dir() -> Path:
    return Path(__file__).resolve().parent

def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(30):
        if (cur / "Fts").is_dir() and (cur / "Model").is_dir():
            return cur
        if (cur / "Fts").is_dir():
            return cur
        if (cur / "Data").is_dir() and (cur / "Model_Codes").is_dir():
            return cur
        cur = cur.parent
    return start.resolve().parent

def resolve_data_root(cfg: Dict[str, Any], project_root: Path) -> Path:
    if cfg.get("data_root"):
        p = Path(cfg["data_root"]).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"CONFIG['data_root'] is not a directory: {p}")
        return p

    candidates = [
        (project_root / "Data" / "Data_Process_Output" / "Step4_SelectedFeatures"),
        (project_root / "Fts" / "Fts-For-Model"),
        (project_root / "Fts" / "Fts_For_Model"),
        (project_root / "Fts-For-Model"),
        (project_root / "Fts_For_Model"),
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()

    tried = "\n  - " + "\n  - ".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(
        "Cannot auto-detect data_root.\n"
        "Please set CONFIG['data_root'] to the absolute path of your Step4_SelectedFeatures folder.\n"
        f"Tried:{tried}"
    )

def resolve_run_dirs(cfg: Dict[str, Any], run_tag: str) -> Tuple[Path, Path, Path]:
    """
    Returns: (run_dir, summary_dir, raw_dir)
    """
    out = Path(cfg["out_dir"]).expanduser()
    if not out.is_absolute():
        out = (script_dir() / out).resolve()
    else:
        out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    run_dir = out / run_tag
    summary_dir = run_dir / "summary"
    raw_dir = run_dir / "raw"
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, summary_dir, raw_dir

def ensure_raw_subdirs(raw_dir: Path, model: str) -> Tuple[Path, Path]:
    p_pred = raw_dir / "predictions" / model
    p_imp = raw_dir / "featimp" / model
    p_pred.mkdir(parents=True, exist_ok=True)
    p_imp.mkdir(parents=True, exist_ok=True)
    return p_pred, p_imp


# ============================================================
# 4) Excel IO
# ============================================================

def find_xlsx(data_root: Path, material: str, width_ms: int) -> Path:
    material_dir = data_root / material
    pattern = f"*_W_{int(width_ms)}.xlsx"

    if material_dir.is_dir():
        hits = sorted(material_dir.glob(pattern))
        if hits:
            hits = sorted(hits, key=lambda p: (len(p.name), p.name))
            return hits[0]

    hits2 = []
    for p in data_root.rglob(pattern):
        if material in p.parts:
            hits2.append(p)
    if hits2:
        hits2 = sorted(hits2, key=lambda p: (len(p.as_posix()), p.as_posix()))
        return hits2[0]

    raise FileNotFoundError(
        f"Cannot find any '{pattern}' for material='{material}' under data_root='{data_root}'.\n"
        f"Expected example: {data_root}/{material}/{pattern}"
    )

def detect_soc_sheet(xlsx_path: Path, soc: int) -> str:
    import openpyxl  # noqa
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheets = wb.sheetnames
    wb.close()

    targets = [
        f"SOC{int(soc)}", f"SOC_{int(soc)}",
        f"soc{int(soc)}", f"soc_{int(soc)}",
        f"{int(soc)}",
    ]
    for t in targets:
        if t in sheets:
            return t
    lower_map = {s.lower(): s for s in sheets}
    for t in targets:
        if t.lower() in lower_map:
            return lower_map[t.lower()]
    raise KeyError(f"Cannot find SOC sheet for soc={soc} in '{xlsx_path.name}'. Available: {sheets[:30]}")

def detect_label_col(df: pd.DataFrame, hint: str = "SOH") -> str:
    cols = list(df.columns)
    if hint in cols:
        return hint
    low = {str(c).lower(): c for c in cols}
    if hint.lower() in low:
        return low[hint.lower()]
    for cand in ["soh", "label", "y", "target"]:
        if cand in low:
            return low[cand]
    raise KeyError(f"Cannot detect label column. Columns: {cols[:30]}")

def map_feature_list(df: pd.DataFrame, feature_list: List[str]) -> List[str]:
    cols = list(df.columns)
    low = {str(c).lower(): c for c in cols}
    mapped = []
    missing = []
    for c in feature_list:
        if c in cols:
            mapped.append(c)
        elif str(c).lower() in low:
            mapped.append(low[str(c).lower()])
        else:
            missing.append(c)
    if missing:
        raise KeyError(
            f"Some requested features are missing.\n"
            f"Missing examples: {missing[:20]}{' ...' if len(missing) > 20 else ''}\n"
            f"Available preview: {cols[:40]}{' ...' if len(cols) > 40 else ''}"
        )
    return mapped

def to_numeric_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def load_xy(
    xlsx_path: Path,
    soc: int,
    label_hint: str,
    feature_list: List[str]
) -> Tuple[pd.DataFrame, pd.Series, List[str], str, str]:
    sheet = detect_soc_sheet(xlsx_path, soc)
    df = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")

    label_col = detect_label_col(df, label_hint)
    mapped_features = map_feature_list(df, feature_list)

    Xdf = to_numeric_frame(df, mapped_features)
    y = pd.to_numeric(df[label_col], errors="coerce")

    mask = (~y.isna()) & (~Xdf.isna().any(axis=1))
    Xdf = Xdf.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    return Xdf, y, mapped_features, sheet, label_col


# ============================================================
# 5) Metrics + efficiency
# ============================================================

def metric_mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def metric_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def metric_mape(y_true, y_pred, eps: float = 1e-9) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def summarize_metrics(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model"):
        r = {"model": model}
        for c in cols:
            r[f"{c}_median"] = float(np.median(g[c].values))
            r[f"{c}_std"] = float(np.std(g[c].values, ddof=1)) if len(g) > 1 else 0.0
        rows.append(r)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


# ============================================================
# 6) Feature importance
# ============================================================

def native_importance(est, model_name: str, feature_names: List[str]) -> Optional[pd.DataFrame]:
    m = model_name.lower()
    # linear family
    if m in ["linear", "ridge", "lasso", "en", "elasticnet"] and hasattr(est, "coef_"):
        coef = np.asarray(est.coef_).reshape(-1)
        df = pd.DataFrame({"feature": feature_names, "importance": np.abs(coef), "signed": coef})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    # RF family
    if m == "rf" and hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_).reshape(-1)
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    # xgb gain
    if m == "xgb":
        try:
            booster = est.get_booster()
            score = booster.get_score(importance_type="gain")
            imp = np.zeros(len(feature_names), dtype=float)
            for k, v in score.items():
                mm = re.match(r"f(\d+)", k)
                if mm:
                    j = int(mm.group(1))
                    if 0 <= j < len(imp):
                        imp[j] = float(v)
            df = pd.DataFrame({"feature": feature_names, "importance": imp})
            return df.sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception:
            return None
    return None

def permutation_importance_df(model, X_test, y_test, feature_names, n_repeats, seed) -> pd.DataFrame:
    def _neg_mae(est, X, y):
        pred = est.predict(X)
        return -metric_mae(y, pred)

    res = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=_neg_mae,
        n_repeats=int(n_repeats),
        random_state=int(seed),
        n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": res.importances_mean,
        "importance_std": res.importances_std
    })
    return df.sort_values("importance", ascending=False).reset_index(drop=True)

def summarize_featimp(all_seed_imp: List[pd.DataFrame], topk: Optional[int]) -> pd.DataFrame:
    big = []
    for i, d in enumerate(all_seed_imp):
        dd = d[["feature", "importance"]].copy()
        dd["seed_idx"] = i
        big.append(dd)
    all_df = pd.concat(big, axis=0, ignore_index=True)
    agg = all_df.groupby("feature")["importance"].agg(["median", "std", "mean"]).reset_index()
    agg = agg.sort_values("median", ascending=False).reset_index(drop=True)
    agg = agg.rename(columns={"median": "importance_median", "std": "importance_std", "mean": "importance_mean"})
    if topk is not None:
        agg = agg.head(int(topk)).reset_index(drop=True)
    return agg


# ============================================================
# 7) Torch deep models (Transformer / Informer-lite)
# ============================================================

def _resolve_torch_device(cfg: Dict[str, Any]) -> str:
    mode = str(cfg.get("torch_device", "auto")).lower()
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        return "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    # Auto-select CUDA when available.
    return "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

class _TabTransformerNet(nn.Module):
    """
    Treat each feature as a token; project scalar -> d_model, add positional embedding, TransformerEncoder -> pooled -> regression head.
    """
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 3, dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.value_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_features, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        x = x.unsqueeze(-1)  # (B, F, 1)
        h = self.value_proj(x) + self.pos_emb  # (B, F, d)
        h = self.encoder(h)                   # (B, F, d)
        h = h.mean(dim=1)                     # (B, d)
        y = self.head(h).squeeze(-1)          # (B,)
        return y

class _InformerLiteNet(nn.Module):
    """
    A lightweight Informer-like encoder:
    - same tokenization as TabTransformer
    - uses stacked encoder blocks with dropout + FFN
    (Not full ProbSparse attention; kept simple and stable for tabular features)
    """
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 4, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.value_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_features, d_model))

        # Use TransformerEncoder as a practical proxy for Informer encoder stack
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)  # (B, F, 1)
        h = self.value_proj(x) + self.pos_emb
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)

class TorchRegressor:
    """
    sklearn-like wrapper: fit/predict for torch models
    """
    def __init__(self, net: nn.Module, cfg: Dict[str, Any], seed: int):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch not available.")
        self.cfg = cfg
        self.seed = int(seed)
        self.device = _resolve_torch_device(cfg)
        self.net = net.to(self.device)

        self.epochs = int(cfg.get("torch_epochs", 200))
        self.batch_size = int(cfg.get("torch_batch_size", 64))
        self.lr = float(cfg.get("torch_lr", 1e-3))
        self.weight_decay = float(cfg.get("torch_weight_decay", 1e-6))
        self.patience = int(cfg.get("torch_patience", 30))
        self.seed_offset = int(cfg.get("torch_seed_offset", 12345))

        self._is_fit = False

    def _set_seed(self):
        s = self.seed + self.seed_offset
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._set_seed()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        # Split the training fold again to create an internal validation set.
        n = len(y)
        idx = np.arange(n)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(idx)
        n_val = max(1, int(0.15 * n))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))

        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=self.batch_size, shuffle=False, drop_last=False)

        opt = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        bad = 0

        self.net.train()
        for epoch in range(self.epochs):
            self.net.train()
            for xb, yb in dl_tr:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            # Validation pass.
            self.net.eval()
            with torch.no_grad():
                vals = []
                for xb, yb in dl_va:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = self.net(xb)
                    vals.append(loss_fn(pred, yb).item())
                val_loss = float(np.mean(vals)) if vals else float("inf")

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self._is_fit = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("TorchRegressor not fit yet.")
        X = np.asarray(X, dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(X))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.net.eval()
        outs = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(self.device)
                pred = self.net(xb).detach().cpu().numpy()
                outs.append(pred)
        return np.concatenate(outs, axis=0)


# ============================================================
# 8) Model builder
# ============================================================

def build_model(name: str, random_state: int, n_features: int, cfg: Dict[str, Any]):
    name = name.lower().strip()

    # Classical regression models
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=random_state)
    if name == "lasso":
        return Lasso(alpha=1e-3, random_state=random_state, max_iter=20000)
    if name in ["en", "elasticnet"]:
        return ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=random_state, max_iter=20000)
    if name in ["svm", "svr"]:
        return SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.01)

    # Tree-based and boosted models
    if name == "rf":
        return RandomForestRegressor(n_estimators=600, random_state=random_state, n_jobs=-1, min_samples_leaf=1)

    # Kernel and neural-network baselines
    if name == "gpr":
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=random_state)
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-5,
            batch_size=64,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=5000,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=50,
            validation_fraction=0.15,
        )

    # XGBoost
    if name == "xgb":
        try:
            import xgboost as xgb  # noqa
        except Exception as e:
            raise ImportError("xgboost not installed. `pip install xgboost` or remove 'xgb' from models.") from e
        return xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            objective="reg:squarederror",
        )

    # Torch-backed models
    if name == "transformer":
        if not _TORCH_AVAILABLE:
            raise ImportError("torch not installed; cannot run transformer.")
        net = _TabTransformerNet(n_features=n_features, d_model=64, nhead=4, num_layers=3, dim_ff=128, dropout=0.1)
        return TorchRegressor(net=net, cfg=cfg, seed=random_state)

    if name == "informer":
        if not _TORCH_AVAILABLE:
            raise ImportError("torch not installed; cannot run informer.")
        net = _InformerLiteNet(n_features=n_features, d_model=64, nhead=4, num_layers=4, dim_ff=256, dropout=0.1)
        return TorchRegressor(net=net, cfg=cfg, seed=random_state)

    raise ValueError(f"Unknown model: {name}")


# ============================================================
# 9) Progress / feature spec
# ============================================================

def prog(msg: str, enabled: bool = True) -> None:
    if enabled:
        print(msg, flush=True)

def feature_list_from_config(cfg: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    spec = cfg.get("feature_spec", None)
    if spec is None or not isinstance(spec, dict):
        raise ValueError("CONFIG['feature_spec'] must be a dict.")

    resolved = dict(spec)
    feats = build_feature_list(
        combo=spec.get("combo", None),
        include_groups=spec.get("include_groups", None),
        exclude_groups=spec.get("exclude_groups", None),
        add_features=spec.get("add_features", None),
        drop_features=spec.get("drop_features", None),
        keep_regex=spec.get("keep_regex", None),
    )
    if not feats:
        raise ValueError("Feature spec resolved to an empty feature list.")
    resolved["n_features_requested"] = len(feats)
    return feats, resolved


# ============================================================
# 10) Main run
# ============================================================

def run(cfg: Dict[str, Any]) -> Path:
    this_dir = script_dir()
    project_root = find_project_root(this_dir)
    data_root = resolve_data_root(cfg, project_root)

    material = str(cfg["material"])
    width_ms = int(cfg["width_ms"])
    soc = int(cfg["soc"])
    show_progress = bool(cfg.get("show_progress", True))

    feat_list, feat_spec_resolved = feature_list_from_config(cfg)
    feat_tag = feature_tag_from_spec(feat_spec_resolved)

    xlsx_path = find_xlsx(data_root, material, width_ms)

    run_tag = f"{_now_ts()}__{_safe_name(material)}__W{width_ms}__SOC{soc}__{feat_tag}"
    run_dir, summary_dir, raw_dir = resolve_run_dirs(cfg, run_tag)

    # Save a snapshot of the resolved configuration for reproducibility.
    cfg_dump = dict(cfg)
    cfg_dump["project_root_resolved"] = str(project_root)
    cfg_dump["data_root_resolved"] = str(data_root)
    cfg_dump["xlsx_path_resolved"] = str(xlsx_path)
    cfg_dump["feature_spec_resolved"] = feat_spec_resolved
    (summary_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2, ensure_ascii=False), encoding="utf-8")

    # Load the selected Step4 workbook, SOC sheet, and feature columns.
    Xdf, yser, feature_cols, sheet_name, label_col = load_xy(
        xlsx_path=xlsx_path,
        soc=soc,
        label_hint=str(cfg.get("label_col", "SOH")),
        feature_list=feat_list,
    )

    data_summary = dict(
        xlsx=str(xlsx_path),
        sheet=sheet_name,
        label_col=label_col,
        n_samples=int(len(yser)),
        n_features=int(len(feature_cols)),
        features=feature_cols,
        feature_spec=feat_spec_resolved,
        y_min=float(np.min(yser.values)) if len(yser) else None,
        y_max=float(np.max(yser.values)) if len(yser) else None,
        y_mean=float(np.mean(yser.values)) if len(yser) else None,
        y_std=float(np.std(yser.values)) if len(yser) else None,
        torch_available=_TORCH_AVAILABLE,
    )
    (summary_dir / "data_summary.json").write_text(json.dumps(data_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    X = Xdf.values.astype(float)
    y = yser.values.astype(float)

    seeds = list(cfg["seeds"])
    models = [str(m).lower().strip() for m in cfg["models"]]
    test_size = float(cfg["test_size"])
    standardize = bool(cfg.get("standardize", True))
    perm_repeats = int(cfg.get("perm_repeats", 10))
    topk_featimp = cfg.get("topk_featimp", 50)

    prog(f"[INFO] Data: {material} | W={width_ms} ms | SOC={soc} | sheet={sheet_name} | N={len(y)} | d={X.shape[1]}", show_progress)
    prog(f"[INFO] FeatureSpec: {feat_spec_resolved}", show_progress)
    prog(f"[INFO] Output dir: {run_dir}", show_progress)
    if ("transformer" in models or "informer" in models) and (not _TORCH_AVAILABLE):
        prog("[WARN] torch not available: transformer/informer will be skipped.", show_progress)

    metrics_rows = []
    pred_rows = []
    featimp_long = []

    total_jobs = len(models) * len(seeds)
    done = 0

    for model_name in models:
        # Validate model dependencies before entering the seed loop.
        try:
            _ = build_model(model_name, random_state=0, n_features=X.shape[1], cfg=cfg)
        except Exception as e:
            warnings.warn(f"[SKIP] {model_name}: {e}")
            prog(f"[SKIP] {model_name}: {e}", show_progress)
            continue

        featimp_by_seed = []

        for seed in seeds:
            done += 1
            prog(f"[{done:03d}/{total_jobs:03d}] Train/Test -> model={model_name} | seed={seed}", show_progress)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, shuffle=True
            )

            base = build_model(model_name, random_state=seed, n_features=X.shape[1], cfg=cfg)

            # Standardize the feature matrix for models that benefit from scaling.
            need_scaler = standardize and (model_name in ["linear", "ridge", "lasso", "en", "elasticnet", "svm", "svr", "gpr", "mlp", "transformer", "informer"])
            if need_scaler and (model_name not in ["rf", "xgb"]):
                model = Pipeline([("scaler", StandardScaler()), ("est", base)])
            else:
                model = base

            # Measure training time and test-set inference time for each run.
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            t1 = time.perf_counter()

            y_pred_tr = model.predict(X_train)

            t2 = time.perf_counter()
            y_pred_te = model.predict(X_test)
            t3 = time.perf_counter()

            fit_time_s = float(t1 - t0)
            pred_time_test_s = float(t3 - t2)

            fit_ms_per_train_sample = fit_time_s / max(len(y_train), 1) * 1000.0
            pred_us_per_test_sample = pred_time_test_s / max(len(y_test), 1) * 1e6

            # Record train/test accuracy and efficiency metrics.
            mrow = dict(
                feature_tag=feat_tag,
                feature_name=feat_spec_resolved.get("name", ""),
                model=model_name,
                seed=seed,
                n_train=len(y_train),
                n_test=len(y_test),

                mae_train=metric_mae(y_train, y_pred_tr),
                rmse_train=metric_rmse(y_train, y_pred_tr),
                mape_train=metric_mape(y_train, y_pred_tr),

                mae_test=metric_mae(y_test, y_pred_te),
                rmse_test=metric_rmse(y_test, y_pred_te),
                mape_test=metric_mape(y_test, y_pred_te),

                fit_time_s=fit_time_s,
                pred_time_test_s=pred_time_test_s,
                fit_ms_per_train_sample=fit_ms_per_train_sample,
                pred_us_per_test_sample=pred_us_per_test_sample,
            )
            metrics_rows.append(mrow)

            # Save raw predictions with per-sample error columns.
            def _pred_df(split: str, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
                y_true = np.asarray(y_true).reshape(-1)
                y_pred = np.asarray(y_pred).reshape(-1)
                err = y_pred - y_true
                abs_err = np.abs(err)
                ape = abs_err / np.maximum(np.abs(y_true), 1e-9) * 100.0
                return pd.DataFrame({
                    "feature_tag": feat_tag,
                    "feature_name": feat_spec_resolved.get("name", ""),
                    "model": model_name,
                    "seed": seed,
                    "split": split,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "err": err,
                    "abs_err": abs_err,
                    "ape_%": ape,
                })

            df_tr = _pred_df("train", y_train, y_pred_tr)
            df_te = _pred_df("test", y_test, y_pred_te)
            df_seed_pred = pd.concat([df_tr, df_te], ignore_index=True)

            raw_pred_dir, raw_imp_dir = ensure_raw_subdirs(raw_dir, model_name)
            df_seed_pred.to_csv(raw_pred_dir / f"predictions__{model_name}__seed{seed}.csv", index=False)
            pred_rows.append(df_seed_pred)

            # Prefer native feature importance when the estimator exposes it,
            # otherwise fall back to permutation importance.
            core = model.named_steps["est"] if hasattr(model, "named_steps") else model
            fi = native_importance(core, model_name, feature_cols)
            if fi is None:
                fi = permutation_importance_df(model, X_test, y_test, feature_cols, perm_repeats, seed)

            fi["feature_tag"] = feat_tag
            fi["feature_name"] = feat_spec_resolved.get("name", "")
            fi["model"] = model_name
            fi["seed"] = seed
            featimp_long.append(fi.copy())

            fi_out = fi.copy()
            if topk_featimp is not None:
                fi_out = fi_out.head(int(topk_featimp))
            fi_out.to_csv(raw_imp_dir / f"featimp__{model_name}__seed{seed}.csv", index=False)
            featimp_by_seed.append(fi)

            prog(
                f"        test: MAE={mrow['mae_test']:.6g} | RMSE={mrow['rmse_test']:.6g} | "
                f"MAPE={mrow['mape_test']:.4g}% | "
                f"fit={mrow['fit_ms_per_train_sample']:.3f} ms/sample | "
                f"pred={mrow['pred_us_per_test_sample']:.1f} us/sample",
                show_progress
            )

        # Aggregate feature importance across seeds for each model.
        if featimp_by_seed:
            fis = summarize_featimp(featimp_by_seed, topk=topk_featimp)
            fis.insert(0, "feature_tag", feat_tag)
            fis.insert(1, "feature_name", feat_spec_resolved.get("name", ""))
            fis.insert(2, "model", model_name)
            fis.to_csv(summary_dir / f"featimp_summary__{model_name}.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        raise RuntimeError("No models were run. Please check CONFIG['models'] and dependencies.")

    metrics_df.to_csv(summary_dir / "metrics_by_seed.csv", index=False)

    summary_cols = [
        "mae_test", "rmse_test", "mape_test",
        "mae_train", "rmse_train", "mape_train",
        "fit_time_s", "pred_time_test_s",
        "fit_ms_per_train_sample", "pred_us_per_test_sample",
    ]
    metrics_summary = summarize_metrics(metrics_df, summary_cols)
    metrics_summary.insert(0, "feature_tag", feat_tag)
    metrics_summary.insert(1, "feature_name", feat_spec_resolved.get("name", ""))
    metrics_summary.to_csv(summary_dir / "metrics_summary.csv", index=False)

    pred_all = pd.concat(pred_rows, ignore_index=True)
    pred_all.to_csv(summary_dir / "predictions_all.csv", index=False)

    featimp_all = pd.concat(featimp_long, ignore_index=True)
    featimp_all.to_csv(summary_dir / "featimp_all_long.csv", index=False)

    all_sum = []
    for p in sorted(summary_dir.glob("featimp_summary__*.csv")):
        all_sum.append(pd.read_csv(p))
    if all_sum:
        pd.concat(all_sum, ignore_index=True).to_csv(summary_dir / "featimp_summary_all_models.csv", index=False)

    prog("\n=== DONE ===", show_progress)
    prog(f"Run dir: {run_dir}", show_progress)
    prog("Saved (summary): metrics_by_seed.csv, metrics_summary.csv, predictions_all.csv, featimp_all_long.csv", show_progress)
    prog("Saved (raw): raw/predictions/<model>/..., raw/featimp/<model>/...", show_progress)

    return run_dir


def main():
    run(CONFIG)


if __name__ == "__main__":
    main()
