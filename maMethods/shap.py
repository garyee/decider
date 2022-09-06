import shap

from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import ColumCount, Complexity, Correlation, ExplanationScope, Heterogeneity, Interactivity, Linearity, Monotonicity, Multicollinearity, ResultTypes, TaskType
  

class SHAP(PostHocBase):
  NAME='SHAP'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.BOTH
  RESULTS=[ResultTypes.VIS,ResultTypes.VIS]
  CONSTRAINTS={
    'complexity': Complexity.MEDIUM,
    'heterogeneity':Heterogeneity.BOTH,
    'col_count':ColumCount.BAD,
    'corr_det':Correlation.BAD,
    'multicollinearity':Multicollinearity.NONE,
    'linearity':Linearity.NO,
    'monotonicity':Monotonicity.YES,
    'interactivity':Interactivity.YES,
  }


def applySHAP(model,df_train):
  explainer = shap.Explainer(model)
  shap_values = explainer(df_train)
  shap.plots.bar(shap_values)