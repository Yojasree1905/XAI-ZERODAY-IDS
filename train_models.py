import json
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.hybrid import hybrid_predict
from src.model import train_isolation_forest, train_random_forest, train_xgboost
from src.preprocessing import build_preprocessor, load_dataset

project_root = Path(__file__).resolve().parent
model_dir = project_root / 'notebooks' / 'models'
results_dir = project_root / 'results'
model_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

X_train, y_train = load_dataset(project_root / 'data' / 'raw' / 'UNSW_NB15_training-set.csv')
X_test, y_test = load_dataset(project_root / 'data' / 'raw' / 'UNSW_NB15_testing-set.csv')

preprocessor = build_preprocessor()
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

iso_model, iso_metrics = train_isolation_forest(X_train_trans, y_train, X_test_trans, y_test)
rf_model, rf_metrics = train_random_forest(X_train_trans, y_train, X_test_trans, y_test)
xgb_model, xgb_metrics = train_xgboost(X_train_trans, y_train, X_test_trans, y_test)

hybrid_preds = hybrid_predict(xgb_model, rf_model, iso_model, X_test_trans)

hybrid_metrics = {
    "accuracy": accuracy_score(y_test, hybrid_preds),
    "precision": precision_score(y_test, hybrid_preds, zero_division=0),
    "recall": recall_score(y_test, hybrid_preds, zero_division=0),
    "f1_score": f1_score(y_test, hybrid_preds, zero_division=0)
}

metrics = {
    'Isolation Forest': iso_metrics,
    'Random Forest': rf_metrics,
    'XGBoost': xgb_metrics,
    'Hybrid Model': hybrid_metrics,
    'train_rows': len(X_train),
    'test_rows': len(X_test),
    'normal_label': 0,
    'attack_label': 1,
}
with (results_dir / 'metrics.json').open('w', encoding='utf-8') as file:
    json.dump(metrics, file, indent=2)

joblib.dump(preprocessor, model_dir / 'preprocessor.pkl')
joblib.dump(iso_model, model_dir / 'iso_model.pkl')
joblib.dump(rf_model, model_dir / 'rf_model.pkl')
joblib.dump(xgb_model, model_dir / 'xgb_model.pkl')

cm = confusion_matrix(y_test, hybrid_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Hybrid Model Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.tight_layout()
plt.savefig(results_dir / 'hybrid_confusion_matrix.png')
print(json.dumps(metrics, indent=2))