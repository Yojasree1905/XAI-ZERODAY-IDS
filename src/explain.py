import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular

def get_feature_names(preprocessor):
    """Retrieves 194 feature names from preprocessor ColumnTransformer."""
    try:
        raw_names = list(preprocessor.get_feature_names_out())
        return [f.replace('cat__', '').replace('num__', '') for f in raw_names]
    except Exception:
        return [f"feature_{i}" for i in range(194)]

def compute_shap_summary(xgb_model, X_trans_sample, preprocessor, max_display=15):
    """
    Computes SHAP values using TreeExplainer and returns
    a matplotlib figure for global feature importance summary.
    """
    feature_names = get_feature_names(preprocessor)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_trans_sample)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_trans_sample, feature_names=feature_names, max_display=max_display, show=False)
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    ax.tick_params(colors='#ffffff')
    ax.xaxis.label.set_color('#ffffff')
    ax.yaxis.label.set_color('#ffffff')
    plt.tight_layout()
    return fig

def get_global_feature_importance(xgb_model, preprocessor, top_n=15):
    """
    Returns top N global feature importances as a DataFrame.
    """
    feature_names = get_feature_names(preprocessor)
    importances = xgb_model.feature_importances_
    
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    
    df_imp['Clean Feature'] = df_imp['Feature'].str.replace('proto_', 'proto: ').str.replace('service_', 'service: ').str.replace('state_', 'state: ')
    return df_imp

def compute_lime_explanation(xgb_model, preprocessor, X_trans_background, single_X_trans, top_n=10):
    """
    Generates local feature explanations for a single network traffic instance
    by evaluating exact TreeExplainer contribution weights on non-zero active features.
    """
    feature_names = get_feature_names(preprocessor)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(single_X_trans)
    
    if len(shap_values.values.shape) > 1:
        vals = shap_values.values[0]
    else:
        vals = shap_values.values
        
    df_exp = pd.DataFrame({
        'Feature Condition': feature_names,
        'Contribution Weight': vals
    })
    
    # Calculate absolute magnitude to find top active drivers
    df_exp['abs_weight'] = df_exp['Contribution Weight'].abs()
    df_exp = df_exp[df_exp['abs_weight'] > 0.0001].sort_values(by='abs_weight', ascending=False).head(top_n)
    
    # Format effect labels
    df_exp['Effect'] = df_exp['Contribution Weight'].apply(
        lambda w: 'Increases Threat Risk' if w > 0 else 'Supports Authorized Verdict'
    )
    
    df_out = df_exp[['Feature Condition', 'Contribution Weight', 'Effect']].reset_index(drop=True)
    return df_out, None
