import os
import joblib
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "notebooks" / "models"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data" / "raw"

# Expected 42 features
CATEGORICAL_COLS = ['proto', 'service', 'state']
NUMERICAL_COLS = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
    'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
    'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
]

ALL_FEATURES = CATEGORICAL_COLS + NUMERICAL_COLS

def load_models_and_preprocessor():
    """Loads preprocessor and 3 models (Isolation Forest, Random Forest, XGBoost)."""
    preprocessor = None
    iso_model = None
    rf_model = None
    xgb_model = None
    
    # Load preprocessor
    prep_path = MODEL_DIR / "preprocessor.pkl"
    if prep_path.exists():
        preprocessor = joblib.load(prep_path)
        
    # Load Isolation Forest
    iso_path = MODEL_DIR / "iso_model.pkl"
    if not iso_path.exists():
        iso_path = RESULTS_DIR / "isolation_forest.pkl"
    if iso_path.exists():
        iso_model = joblib.load(iso_path)
        
    # Load Random Forest
    rf_path = MODEL_DIR / "rf_model.pkl"
    if rf_path.exists():
        rf_model = joblib.load(rf_path)
        
    # Load XGBoost
    xgb_path = MODEL_DIR / "xgb_model.pkl"
    if not xgb_path.exists():
        xgb_path = RESULTS_DIR / "xgboost.pkl"
    if xgb_path.exists():
        xgb_model = joblib.load(xgb_path)
        
    return preprocessor, iso_model, rf_model, xgb_model

def load_clf_model():
    """Loads 10-class multi-class attack classifier model and class names."""
    clf_path = MODEL_DIR / "clf_model.pkl"
    cls_path = MODEL_DIR / "attack_classes.pkl"
    clf_model = None
    classes = None
    if clf_path.exists():
        clf_model = joblib.load(clf_path)
    if cls_path.exists():
        classes = joblib.load(cls_path)
    return clf_model, classes

def get_sample_test_data(n_samples=50):
    """Loads sample test dataset for fast demonstration and background explainer data."""
    csv_path = DATA_DIR / "UNSW_NB15_testing-set.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df_clean = df.drop(columns=['id', 'attack_cat', 'label'], errors='ignore')
        return df_clean.head(n_samples)
    else:
        sample_path = BASE_DIR / "sample_network_data.csv"
        df = pd.read_csv(sample_path)
        return prepare_full_features_dataframe(df)

def prepare_full_features_dataframe(df_input):
    """Ensures input DataFrame contains all 42 expected feature columns."""
    df_out = df_input.copy()
    
    # Defaults based on real UNSW-NB15 Normal HTTP connection baseline
    defaults = {
        'proto': 'tcp', 'service': 'http', 'state': 'FIN',
        'dur': 0.98, 'spkts': 10, 'dpkts': 8, 'sbytes': 816, 'dbytes': 1172,
        'rate': 17.27, 'sttl': 62, 'dttl': 252, 'sload': 5976.0, 'dload': 8342.0,
        'sloss': 2, 'dloss': 2, 'sinpkt': 109.3, 'dinpkt': 124.9, 'sjit': 5929.0, 'djit': 192.5,
        'swin': 255, 'stcpb': 794167371, 'dtcpb': 1624757001, 'dwin': 255, 'tcprtt': 0.206,
        'synack': 0.108, 'ackdat': 0.098, 'smean': 82, 'dmean': 147, 'trans_depth': 1,
        'response_body_len': 184, 'ct_srv_src': 2, 'ct_state_ttl': 1, 'ct_dst_ltm': 1,
        'ct_src_dport_ltm': 1, 'ct_dst_sport_ltm': 1, 'ct_dst_src_ltm': 2,
        'is_ftp_login': 0, 'ct_ftp_cmd': 0, 'ct_flw_http_mthd': 1, 'ct_src_ltm': 1,
        'ct_srv_dst': 3, 'is_sm_ips_ports': 0
    }
    
    for col in ALL_FEATURES:
        if col not in df_out.columns:
            df_out[col] = defaults.get(col, 0)
            
    # Dynamic feature derivations
    df_out['smean'] = df_out.apply(lambda r: int(r['sbytes'] / r['spkts']) if r['spkts'] > 0 else 82, axis=1)
    df_out['dmean'] = df_out.apply(lambda r: int(r['dbytes'] / r['dpkts']) if r['dpkts'] > 0 else 147, axis=1)
    df_out['sinpkt'] = df_out.apply(lambda r: float(r['dur'] * 1000.0 / r['spkts']) if r['spkts'] > 0 else 109.3, axis=1)
    
    # Protocol / Service specific flags
    df_out['trans_depth'] = df_out.apply(lambda r: 1 if r['service'] == 'http' else 0, axis=1)
    df_out['response_body_len'] = df_out.apply(lambda r: 184 if r['service'] == 'http' else 0, axis=1)
    df_out['ct_flw_http_mthd'] = df_out.apply(lambda r: 1 if r['service'] == 'http' else 0, axis=1)
    df_out['is_ftp_login'] = df_out.apply(lambda r: 1 if r['service'] == 'ftp' else 0, axis=1)
    df_out['ct_ftp_cmd'] = df_out.apply(lambda r: 1 if r['service'] == 'ftp' else 0, axis=1)
    
    df_out['ct_srv_dst'] = df_out['ct_srv_src']
    df_out['ct_src_ltm'] = df_out['ct_dst_ltm']

    # Ensure correct column order
    return df_out[ALL_FEATURES]
