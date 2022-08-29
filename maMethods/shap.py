import shap
  
def applySHAP(model,df_train):
  explainer = shap.Explainer(model)
  shap_values = explainer(df_train)
  shap.plots.bar(shap_values)