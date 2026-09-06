"""The three estimators behind the detection engine.

  1. Detector       - binary XGBoost: is this flow hostile at all?
  2. Classifier     - 9-class XGBoost: which attack family is it?
  3. Novelty        - Isolation Forest fitted on benign traffic only: is this
                      flow structurally unlike anything we were trained on?

Only (1) and (2) see labels. (3) never sees an attack during training, which is
what lets it flag families that did not exist when the model was built.
"""
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RANDOM_STATE = 42

DETECTOR_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    min_child_weight=1,
    tree_method='hist',
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

CLASSIFIER_PARAMS = dict(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    objective='multi:softprob',
    tree_method='hist',
    eval_metric='mlogloss',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def binary_metrics(y_true, y_pred, y_score=None):
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    y_true = np.asarray(y_true)
    negatives = (y_true == 0).sum()
    metrics['false_positive_rate'] = (
        float(((np.asarray(y_pred) == 1) & (y_true == 0)).sum() / negatives) if negatives else 0.0
    )
    if y_score is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_score))
        metrics['pr_auc'] = float(average_precision_score(y_true, y_score))
    return metrics


def train_detector(X_train, y_train):
    """Binary attack detector.

    ``scale_pos_weight`` rebalances the 68/32 attack-heavy training mix to an
    even prior. Without it the model inherits the corpus prior and floods a
    real sensor — where benign traffic dominates — with false positives.
    """
    y_train = np.asarray(y_train)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError('Detector training needs both benign and attack rows.')

    model = xgb.XGBClassifier(scale_pos_weight=n_neg / n_pos, **DETECTOR_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_reference_forest(X_train, y_train):
    """Random Forest kept as an independent second opinion / baseline."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, np.asarray(y_train))
    return model


def train_attack_classifier(X_attacks, y_family, n_classes):
    """Attack-family classifier, trained on hostile flows only.

    Square-root class weighting is a deliberate middle ground: full inverse
    weighting collapses the head classes to chase 44 Worms rows, no weighting
    ignores the rare families entirely. Square-root lifts macro-F1 without
    giving up the head classes.
    """
    y_family = np.asarray(y_family)
    counts = np.bincount(y_family, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = np.sqrt(counts.sum() / counts)
    weights /= weights.mean()

    model = xgb.XGBClassifier(num_class=n_classes, **CLASSIFIER_PARAMS)
    model.fit(X_attacks, y_family, sample_weight=weights[y_family])
    return model


def train_novelty_detector(X_benign):
    """Isolation Forest fitted purely on benign traffic.

    ``max_samples`` is capped so each tree sees a subsample: full-sample trees
    memorise the benign set and stop isolating anything.
    """
    if X_benign.shape[0] == 0:
        raise ValueError('Novelty detector requires benign training rows.')
    model = IsolationForest(
        n_estimators=300,
        max_samples=min(8192, X_benign.shape[0]),
        contamination='auto',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_benign)
    return model


def novelty_score(model, X):
    """Higher means more unlike the benign baseline."""
    return -model.score_samples(X)


def select_threshold_by_alert_budget(y_val, scores_val, budget):
    """Pick the detector cut-off from benign validation scores alone.

    Choosing the threshold on the benign class fixes the false-positive rate
    rather than the F1 of one particular attack mix, so the operating point
    holds when the live benign/attack ratio differs from the corpus — which it
    always does.
    """
    benign_scores = np.asarray(scores_val)[np.asarray(y_val) == 0]
    if benign_scores.size == 0:
        return 0.5
    return float(np.quantile(benign_scores, 1.0 - budget))


def select_threshold_by_f1(y_val, scores_val):
    """Best-F1 cut-off on validation, reported for comparison."""
    grid = np.linspace(0.02, 0.98, 97)
    scores = [f1_score(y_val, (np.asarray(scores_val) >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])
