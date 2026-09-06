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
    
    # Defaults
    defaults = {
        'proto': 'tcp', 'service': 'http', 'state': 'FIN',
        'dur': 0.05, 'spkts': 10, 'dpkts': 8, 'sbytes': 1000, 'dbytes': 1200,
        'rate': 300.0, 'sttl': 62, 'dttl': 62, 'sload': 150000.0, 'dload': 180000.0,
        'sloss': 0, 'dloss': 0, 'sinpkt': 5.0, 'dinpkt': 5.0, 'sjit': 10.0, 'djit': 10.0,
        'swin': 255, 'stcpb': 100000, 'dtcpb': 100000, 'dwin': 255, 'tcprtt': 0.01,
        'synack': 0.005, 'ackdat': 0.005, 'smean': 100, 'dmean': 150, 'trans_depth': 0,
        'response_body_len': 0, 'ct_srv_src': 2, 'ct_state_ttl': 1, 'ct_dst_ltm': 2,
        'ct_src_dport_ltm': 1, 'ct_dst_sport_ltm': 1, 'ct_dst_src_ltm': 2,
        'is_ftp_login': 0, 'ct_ftp_cmd': 0, 'ct_flw_http_mthd': 0, 'ct_src_ltm': 2,
        'ct_srv_dst': 2, 'is_sm_ips_ports': 0
    }
    
    for col in ALL_FEATURES:
        if col not in df_out.columns:
            df_out[col] = defaults.get(col, 0)
            
    # Ensure correct column order
    return df_out[ALL_FEATURES]
