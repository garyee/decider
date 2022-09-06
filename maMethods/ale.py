from alepython import ale_plot

from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import ColumCount, Complexity, Correlation, ExplanationScope, Heterogeneity, Interactivity, Linearity, Monotonicity, Multicollinearity, ResultTypes, TaskType

class ALE(PostHocBase):
  NAME='ALE'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.GLOBAL
  RESULTS=[ResultTypes.VIS]
  CONSTRAINTS={
    'complexity': Complexity.HIGH,
    'heterogeneity':Heterogeneity.NUMMERICAL,
    'col_count':ColumCount.BAD,
    'corr_det':Correlation.NONE,
    'multicollinearity':Multicollinearity.BAD,
    'linearity':Linearity.NO,
    'monotonicity':Monotonicity.NO,
    'interactivity':Interactivity.NO,
  }
  
  def apply(model, df_train, number_of_feature_to_display=0):
    nameOfFeature=df_train.columns.values.tolist()[number_of_feature_to_display]
    ale = ale_plot(model, df_train, nameOfFeature, monte_carlo=True)