from alepython import ale_plot

from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import ExplanationScope, ResultTypes, TaskType

class ALE(PostHocBase):
  NAME='ALE'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.GLOBAL
  RESULTS=[ResultTypes.VIS]
  COMPLEXITY=Complexity.HIGH
  CONSTRAINTS={
    'heterogeneity':None,
    'col_count':None,
    'corr_det':None,
    'multicollinearity':None,
    'linearity':None,
    'monotonicity':None,
    'interactivity':None,
  }
  
  def apply(model, df_train, number_of_feature_to_display=0):
    nameOfFeature=df_train.columns.values.tolist()[number_of_feature_to_display]
    ale = ale_plot(model, df_train, nameOfFeature, monte_carlo=True)