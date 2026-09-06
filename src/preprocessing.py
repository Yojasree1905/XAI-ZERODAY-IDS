"""Dataset loading and the fitted-once feature transformer.

The transformer is fitted on training data only and then reused verbatim at
inference, so a flow scored in the UI goes through exactly the same arithmetic
as a row scored during evaluation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    ALL_FEATURES,
    ATTACK_CLASSES,
    BENIGN_CLASS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
)


def load_dataset(path, with_attack_cat=True):
    """Load a UNSW-NB15 CSV and return (features, binary_label, attack_cat).

    Nothing is fitted here — the frame comes back exactly as stored so the
    transformer can be fitted on train and merely applied to test.
    """
    df = pd.read_csv(Path(path))
    df.columns = [c.strip().lstrip('﻿') for c in df.columns]

    required = set(ALL_FEATURES) | {'label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Dataset is missing required columns: {sorted(missing)}')

    features = df[ALL_FEATURES].copy()

    labels = pd.to_numeric(df['label'], errors='raise').astype('int8')
    if not labels.isin([0, 1]).all():
        raise ValueError('The label column must contain only 0 (Normal) and 1 (Attack).')

    attack_cat = None
    if with_attack_cat and 'attack_cat' in df.columns:
        attack_cat = df['attack_cat'].astype(str).str.strip()
        # The published CSVs spell this family two ways.
        attack_cat = attack_cat.replace({'Backdoors': 'Backdoor'})
        unknown = set(attack_cat.unique()) - set(ATTACK_CLASSES) - {BENIGN_CLASS}
        if unknown:
            raise ValueError(f'Unrecognised attack_cat values: {sorted(unknown)}')

    return features, labels, attack_cat


def _log1p_signed(X):
    """Compress heavy tails while tolerating the occasional negative reading.

    Flow counters span nine orders of magnitude (``dur`` in microseconds up to
    ``sload`` in gigabits/sec). Without compression the scaler is dominated by a
    handful of volumetric flows and the Isolation Forest becomes a byte-count
    detector rather than a structure detector.
    """
    X = np.asarray(X, dtype=np.float64)
    return np.sign(X) * np.log1p(np.abs(X))


def build_preprocessor():
    """Construct the (unfitted) feature transformer used by every model."""
    numeric_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('compress', FunctionTransformer(_log1p_signed, feature_names_out='one-to-one')),
        ('scale', StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(
                handle_unknown='infrequent_if_exist',
                min_frequency=0.001,
                sparse_output=False,
            ), CATEGORICAL_COLS),
            ('numeric', numeric_pipeline, NUMERICAL_COLS),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor):
    """Human-readable names for the transformed matrix, in column order."""
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        return []
    cleaned = []
    for name in names:
        for prefix in ('categorical__', 'numeric__', 'cat__', 'num__'):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        cleaned.append(name)
    return cleaned


def align_features(df):
    """Coerce an arbitrary frame into the exact 42-column model input.

    Missing columns are filled with the benign defaults below rather than zeros:
    a zero TTL or zero window size is itself a strong attack signal, so padding
    with zeros would silently poison uploaded logs that omit a column.
    """
    defaults = {
        'proto': 'tcp', 'service': '-', 'state': 'FIN',
        'dur': 0.05, 'spkts': 10, 'dpkts': 8, 'sbytes': 1000, 'dbytes': 1200,
        'rate': 300.0, 'sttl': 31, 'dttl': 29, 'sload': 150000.0, 'dload': 180000.0,
        'sloss': 0, 'dloss': 0, 'sinpkt': 5.0, 'dinpkt': 5.0, 'sjit': 10.0, 'djit': 10.0,
        'swin': 255, 'stcpb': 1234567, 'dtcpb': 7654321, 'dwin': 255, 'tcprtt': 0.01,
        'synack': 0.005, 'ackdat': 0.005, 'smean': 100, 'dmean': 150, 'trans_depth': 0,
        'response_body_len': 0, 'ct_srv_src': 2, 'ct_state_ttl': 1, 'ct_dst_ltm': 2,
        'ct_src_dport_ltm': 1, 'ct_dst_sport_ltm': 1, 'ct_dst_src_ltm': 2,
        'is_ftp_login': 0, 'ct_ftp_cmd': 0, 'ct_flw_http_mthd': 0, 'ct_src_ltm': 2,
        'ct_srv_dst': 2, 'is_sm_ips_ports': 0,
    }

    out = df.copy()
    out.columns = [c.strip().lstrip('﻿') for c in out.columns]
    for col in ALL_FEATURES:
        if col not in out.columns:
            out[col] = defaults[col]

    # UNSW-NB15 stores proto/service lower-case and connection state upper-case;
    # match that exactly or the one-hot encoder drops the value as unseen.
    for col in ('proto', 'service'):
        out[col] = out[col].astype(str).str.strip().str.lower().replace({'nan': '-', '': '-'})
    out['state'] = out['state'].astype(str).str.strip().str.upper().replace({'NAN': 'INT', '': 'INT'})
    for col in NUMERICAL_COLS:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(defaults[col])

    return out[ALL_FEATURES]


# Backwards-compatible alias used by older notebooks.
def load_and_preprocess(path):
    features, labels, _ = load_dataset(path)
    return features, labels
