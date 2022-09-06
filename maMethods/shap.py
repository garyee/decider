import shap

from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import ExplanationScope, ResultTypes, TaskType
  

class SHAP(PostHocBase):
  NAME='PDP'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.BOTH
  RESULTS=[ResultTypes.VIS,ResultTypes.VIS]
  CONSTRAINTS={
    'heterogeneity':None,
    'col_count':None,
    'corr_det':None,
    'multicollinearity':None,
    'linearity':None,
    'monotonicity':None,
    'interactivity':None,
  }

def applySHAP(model,df_train):
  explainer = shap.Explainer(model)
  shap_values = explainer(df_train)
  shap.plots.bar(shap_values)